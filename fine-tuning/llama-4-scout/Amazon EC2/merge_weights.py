from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Llama4ForConditionalGeneration
from safetensors.torch import save_file
import torch
import sys
import os

adapter_dir = sys.argv[1]
model_id = sys.argv[2]
output_dir = sys.argv[3]

#Get the latest epoch dir
adap_epoch_dir = max([os.path.join(adapter_dir,d) for d in os.listdir(adapter_dir) if d.startswith("epoch")], key=os.path.getmtime)

def merge_and_save_model(model_id, adapter_epoch_dir, output_dir):

    print("Trying to load a Peft model. It might take a while without feedback")
    
    base_model = Llama4ForConditionalGeneration.from_pretrained(
                    model_id,
                    attn_implementation="flex_attention",
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )

    peft_model = PeftModel.from_pretrained(base_model, adapter_epoch_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)
    print(f"Saved the merged model to {output_dir}")

def convert_pt_to_safetensors(adap_epoch_dir):
    pt_path = "{}/adapter_model.pt".format(adap_epoch_dir)
    st_path = "{}/adapter_model.safetensors".format(adap_epoch_dir)
    print(f"Converting pt {pt_path} to safetensors in {adap_epoch_dir}")
    state_dict = torch.load(pt_path)
    save_file(state_dict, st_path)
    print(f"Successfully converted to safetensors and saved as {st_path}")
   
#Convert pt to safetensors
convert_pt_to_safetensors(adap_epoch_dir)

#Merge adaqter with base model weights and save
merge_and_save_model(model_id, adap_epoch_dir, output_dir)
