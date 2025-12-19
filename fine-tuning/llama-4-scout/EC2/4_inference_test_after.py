from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Llama4ForConditionalGeneration
import os

script_path = os.path.abspath(__file__)

# Get the directory name of the script
basedir = os.path.dirname(script_path)


#TODO: update it to your chosen epoch
merged_model_path = "{}/final_merged_weights".format(basedir)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# load the fine-tuned model
model = Llama4ForConditionalGeneration.from_pretrained(
                    merged_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )


# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Who gave the UN the land in NY to build their HQ: '"
print("Fine tuned model output:", generate_text(model, tokenizer, prompt))
