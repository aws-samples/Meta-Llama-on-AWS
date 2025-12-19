from datasets import load_dataset
from dataclasses import dataclass, field
import os
import random
import re
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, util
from trl import GRPOConfig, GRPOTrainer
from trl.scripts.utils import TrlParser

@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset, e.g. /opt/ml/input/data/train/"},
    )
    test_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset, e.g. /opt/ml/input/data/test/"},
    )
    model_id: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "Model ID to use for GRPO training"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "The maximum sequence length for GRPO Trainer"}
    )
    max_prompt_length: int = field(
        default=512, metadata={"help": "The maximum prompt length for GRPO Trainer"}
    )

def merge_and_save_model(model_id, adapter_dir, output_dir):
    from peft import PeftModel

    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)
    
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]



def accuracy_reward(completions, prompts=None, label=None,  **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    similarity_reward_multiplier = 2 #To put higher emphasis on the content of the summary, added a 2x modifier on the similarity score
    reward_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    completions = [completion[0]["content"] for completion in completions]
    rewards = []
    for prompt, completion in zip(prompts, completions):
        prompt_embedding = reward_model.encode(prompt)
        completion_embedding = reward_model.encode(completion)
        if len(completion_embedding) != 0:
            try:
                similarity = util.cos_sim(prompt_embedding, completion_embedding)
                similarity *= similarity_reward_multiplier
                rewards.append(similarity.diagonal())
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path, "test_dataset.json"),
        split="train",
    )
    ################
    # Model & Tokenizer
    ################
    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ## For 4 bit quantization
    quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,)
    model = AutoModelForCausalLM.from_pretrained(script_args.model_id,
                                             quantization_config=quantization_config,
                                             device_map="auto")
    #Lora Config
    lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    
    ################
    # Training
    ################
    training_args = GRPOConfig(
        learning_rate = training_args.learning_rate,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = training_args.lr_scheduler_type,
        optim = training_args.optim,
        logging_steps = 1,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        gradient_accumulation_steps = training_args.gradient_accumulation_steps, 
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = script_args.max_prompt_length,
        max_completion_length = script_args.max_seq_length - script_args.max_prompt_length,
        num_train_epochs = training_args.num_train_epochs, # Set to 1 for a full training run
        max_steps = training_args.max_steps,
        save_steps = training_args.max_steps,
        max_grad_norm = training_args.max_grad_norm,
        report_to = "none", # Can use Weights & Biases
        output_dir = "/opt/ml/code/outputs",
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model("/opt/ml/code/outputs")
    del model
    del trainer
    torch.cuda.empty_cache()  # Clears the cache
    # load and merge
    if training_args.distributed_state.is_main_process:
        merge_and_save_model(
            script_args.model_id, "/opt/ml/code/outputs", "/opt/ml/model"
        )
        tokenizer.save_pretrained("/opt/ml/model")
    training_args.distributed_state.wait_for_everyone()

    

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)