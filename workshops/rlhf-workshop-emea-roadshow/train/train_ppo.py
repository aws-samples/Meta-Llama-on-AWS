
import os
import boto3
import bitsandbytes as bnb
import json
import torch
import transformers
import warnings
import time
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    set_seed
)
from trl import (
    ModelConfig, 
    RewardConfig, 
    PPOConfig, 
    PPOTrainer, 
    RewardTrainer, 
    AutoModelForCausalLMWithValueHead, 
    get_kbit_device_map, 
    get_peft_config, 
    get_quantization_config
)
from trl.core import LengthSampler
from accelerate import Accelerator
from peft import (
    AutoPeftModelForCausalLM, 
    AutoPeftModelForSequenceClassification, 
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from tqdm import tqdm


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--train_ds", type=str, required=True, help="S3 path to training dataset")
    parser.add_argument("--rm_adapter", type=str, required=True, help="Path to reward model adapter")
    parser.add_argument("--s3_output_path", type=str, default=None, help="S3 path to save model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_hub_repo_id", type=str, default=None)
    parser.add_argument("--range_train", type=int, default=None)
    parser.add_argument("--use_safetensors", type=bool, default=True)
    parser.add_argument("--use_score_scaling", type=bool, default=False)
    parser.add_argument("--use_score_norm", type=bool, default=False)
    parser.add_argument("--score_clip", type=float, default=None)
    parser.add_argument("--log_with", type=str, default=None)
    parser.add_argument("--merge_weights", type=bool, default=True)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize Accelerator object handling distributed training
    accelerator = Accelerator()
    
    # Login to HuggingFace 
    if args.hf_token is not None:
        login(args.hf_token)
        
    # Load tokenizer. Padding side is "left" because focus needs to be on completion
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')

    # Set tokenizer's pad Token
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id  
    
    # Load data from S3
    dataset = load_from_disk(args.train_ds)
    
    # Allow for partial dataset training
    if args.range_train:
        train_dataset = dataset["train"].select(range(args.range_train))
    else: 
        train_dataset = dataset["train"]
    
    # Specify LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Specify quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        peft_config=lora_config,
        quantization_config=bnb_config,
        reward_adapter=args.rm_adapter,
        use_safetensors=args.use_safetensors,
    )
    
    # Set model pad token id
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    # Trainable parameters
    print_trainable_parameters(model)    

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # Specify PPO training config
    config = PPOConfig(
        args.model_name,
        log_with=args.log_with,
        learning_rate=1e-5,
        batch_size=args.per_device_train_batch_size,
        mini_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        seed=args.seed,
        use_score_scaling=args.use_score_scaling,
        use_score_norm=args.use_score_norm,
        score_clip=args.score_clip,
    )

    # Initialize PPOTrainer object handling training
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    # Specifying inference params
    generation_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 32,
    }
    
    step = 0

    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        
        question_tensors = batch["input_ids"]
        
        # Inference through model being fine-tuned
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        
        # Decode response
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Concat query and response
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        # Tokenize query - response pair
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
        
        # Compute reward score
        raw_rewards = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).compute_reward_score(**inputs)
        rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        step = step + 1      

    if args.merge_weights:
        if accelerator.is_main_process:
            output_dir = "/tmp/model"
            ppo_trainer.save_pretrained(output_dir, safe_serialization=True)
       
            # clear memory
            del model
            del ppo_trainer

            torch.cuda.empty_cache()

            # load PEFT model
            model = AutoPeftModelForCausalLM.from_pretrained(
                output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=True,
                cache_dir="/tmp/.cache",
            )

            # Merge LoRA and base model and save
            model = model.merge_and_unload()
            model.save_pretrained(
                os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                safe_serialization=True,
                max_shard_size="2GB"
            )
            #if args.model_hub_repo_id is not None:
            #    model.push_to_hub(repo_id=args.model_hub_repo_id)

            tokenizer.save_pretrained(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

            #if args.model_hub_repo_id is not None:
            #    tokenizer.push_to_hub(repo_id=args.model_hub_repo_id)
        accelerator.wait_for_everyone()
    else:
        if accelerator.is_main_process:
            ppo_trainer.model.module.save_pretrained(
                os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                safe_serialization=True
            )
    
            #if args.model_hub_repo_id is not None:
            #    ppo_trainer.push_to_hub(repo_id=args.model_hub_repo_id)
    
            tokenizer.save_pretrained(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    
            #if args.model_hub_repo_id is not None:
            #    tokenizer.push_to_hub(repo_id=args.model_hub_repo_id)

        accelerator.wait_for_everyone()

    # Upload the model files to S3 if output path specified
    if accelerator.is_main_process and args.s3_output_path:
        # Get the S3 output path from the environment variables
        # SageMaker automatically sets these environment variables
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        
        print(f"Uploading model from {model_dir} to {args.s3_output_path}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Extract bucket name and prefix from S3 URI
        s3_uri_parts = args.s3_output_path.replace("s3://", "").split("/")
        bucket_name = s3_uri_parts[0]
        prefix = "/".join(s3_uri_parts[1:]) if len(s3_uri_parts) > 1 else ""
        
        # Walk through all files in the model directory and upload them
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Create relative path to maintain directory structure
                relative_path = os.path.relpath(local_path, model_dir)
                s3_key = os.path.join(prefix, relative_path)
                
                print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
                try:
                    s3_client.upload_file(local_path, bucket_name, s3_key)
                except Exception as e:
                    print(f"Failed to upload {local_path} to S3: {e}")
        
        print("Model upload to S3 completed")

    # Wait for all processes to complete
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
