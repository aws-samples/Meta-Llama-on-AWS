import logging
from dataclasses import dataclass, field
import os
import random
import torch
import tarfile
import json
from multiprocessing import Pool
import pkg_resources
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig


from trl import SFTTrainer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)
def print_library_versions():
    libraries = [
        "datasets",
        "transformers",
        "accelerate",
        "evaluate",
        "bitsandbytes",
        "huggingface_hub",
        "trl",
        "peft",
        "torch",
        "scikit-learn"
    ]
    
    logger.info(f"Python version: {sys.version}")
    logger.info("Library versions:")
    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            logger.info(f"{lib}: {version}")
        except pkg_resources.DistributionNotFound:
            logger.info(f"{lib}: Not installed")

# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

def add_to_tar(args):
    tar_path, file_path, arcname = args
    with tarfile.open(tar_path, 'w:') as tar:
        tar.add(file_path, arcname=arcname)

def create_tarfile_parallel(source_dir, output_tar, num_processes=4):
    files_to_add = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, source_dir)
            files_to_add.append((output_tar, full_path, arcname))

    with Pool(num_processes) as pool:
        pool.map(add_to_tar, files_to_add)


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
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
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
    base_model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)    

    print("Model artifacts saved and packaged successfully.")


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path, "dataset.json"),
        split="train",
    )

    ################
    # Model & Tokenizer
    ################

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # template dataset
    def template_dataset(examples):
        return {
            "text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)
        }

    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])

    # print random sample on rank 0
    if training_args.distributed_state.is_main_process:
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

    # Model
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        #quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=(
            False if training_args.gradient_checkpointing else True
        ),  # this is needed for gradient checkpointing
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    #########################################
    # SAVE ADAPTER AND CONFIG FOR SAGEMAKER
    #########################################
    # save adapter
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    del model
    del trainer
    torch.cuda.empty_cache()  # Clears the cache
    # load and merge
    if training_args.distributed_state.is_main_process:
        merge_and_save_model(
            script_args.model_id, training_args.output_dir, "/opt/ml/model"
        )
        tokenizer.save_pretrained("/opt/ml/model")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print



if __name__ == "__main__":
    print_library_versions()
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)

    print("Create a tar.gz file")

    source_directory = '/opt/ml/model'
    output_tarfile = os.path.join('/opt/ml/model', 'model.tar.gz')
    create_tarfile_parallel(source_directory, output_tarfile)

    print(f"Model saved as tar.gz in '{output_tarfile}'")
