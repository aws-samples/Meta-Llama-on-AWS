#!/bin/bash

##Update the following variables as needed
MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct" #Huggingface model id
CONFIG="config.yaml" #Config path for fine tuning recipe - Don't update unless you change the filename
MAX_SEQ_LEN=2048 #Max sequence length
BATCH_SIZE=2 #Batch size
#MODEL_NAME=`echo $MODEL_ID|gawk -F / '{print $2}'`
DOWNLOAD_DIR="/tmp/Llama-4-Scout-17B-16E-Instruct" #HF weights download location
ADAPTER_DIR="/tmp/l4_output" #Lora adapter weights save location

cd /opt/ml/code
#Install dependencies
pip install -r requirements.txt --force

#Download HF weights
tune download $MODEL_ID --output-dir $DOWNLOAD_DIR

#Run fine tuning
PYTHONPATH=$PWD:$PYTHONPATH tune run --nproc_per_node 8 lora_finetune_distributed --config $CONFIG batch_size=$BATCH_SIZE fsdp_cpu_offload=True output_dir=$ADAPTER_DIR

#Merge and save final weights
python3 merge_weights.py $ADAPTER_DIR $DOWNLOAD_DIR

#Copy tokenizer and other json to final model
cp $DOWNLOAD_DIR/tokenizer* /opt/ml/model/
cp $DOWNLOAD_DIR/processor_config.json /opt/ml/model/
cp $DOWNLOAD_DIR/preprocessor_config.json /opt/ml/model/
cp $DOWNLOAD_DIR/special_tokens_map.json /opt/ml/model/