
import os
import boto3
import bitsandbytes as bnb
import json
import torch
import transformers
import warnings
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig, 
    set_seed
)
from trl import RewardConfig, RewardTrainer
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

def find_all_linear_names(hf_model):
    lora_module_names = set()
    for name, module in hf_model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--train_ds", type=str, required=True, help="S3 path to training dataset")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Scaling factor for the LoRA updates")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_hub_repo_id", type=str, default=None)
    parser.add_argument("--range_train", type=int, default=None)
    parser.add_argument("--range_eval", type=int, default=None)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize Accelerator object handling distributed training
    accelerator = Accelerator()

    # Login to HuggingFace if token provided
    if args.hf_token:
        login(args.hf_token)

    # Load tokenizer. Padding side is "left" because focus needs to be on completion
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

    # Set tokenizer's pad Token
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    # Load data from S3 or local path
    dataset = load_from_disk(args.train_ds)  
    
    # Allow for partial dataset training
    if args.range_train:
        train_dataset = dataset["train"].select(range(args.range_train))
    else: 
        train_dataset = dataset["train"]
  
    if args.range_eval:
        eval_dataset = dataset["test"].select(range(args.range_eval))
    else:
        eval_dataset = dataset["test"]

    # Specify quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        quant_storage_dtype=torch.bfloat16
    )
    
    # Load model with classification head for reward
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        use_cache=False if args.gradient_checkpointing else True,
    )
    
    # Pre-LoRA trainable parameters
    print_trainable_parameters(model)     
    
    # Set model pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare model for quantized training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")
    
    # Specify LoRA config
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    # Make sure to not train for CLM
    if config.task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
        )
    
    # Create PeftModel
    model = get_peft_model(model, config)
    
    # Post-LoRA trainable parameters
    print_trainable_parameters(model)     
    
    # Specify training config
    reward_config = RewardConfig(
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    per_device_eval_batch_size=args.per_device_eval_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    gradient_checkpointing=args.gradient_checkpointing,
                    logging_strategy="steps",
                    logging_steps=100,
                    log_on_each_node=False,
                    num_train_epochs=args.num_train_epochs,
                    learning_rate=args.learning_rate,
                    bf16=True,
                    ddp_find_unused_parameters=False,
                    save_strategy="no",
                    output_dir=os.environ["SM_MODEL_DIR"],
                    max_length=512, 
                    remove_unused_columns=False,
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
    
    # Initialize RewardTrainer object handling training
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    
    # Save model to SM_MODEL_DIR (SageMaker's standard output location)
    trainer.model.save_pretrained(os.environ["SM_MODEL_DIR"], safe_serialization=True)
    
    # Push to HuggingFace Hub if repo ID is provided
    if args.model_hub_repo_id:
        trainer.model.push_to_hub(repo_id=args.model_hub_repo_id)

    with accelerator.main_process_first():
        tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])
        if args.model_hub_repo_id:
            tokenizer.push_to_hub(repo_id=args.model_hub_repo_id)

if __name__ == "__main__":
    main()
