#!/bin/bash

set -e  # Exit on any error

#cd to CWD
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR

#Install dependencies
pip install -r requirements.txt

#create dir for dataset
mkdir -p dataset
#Login to huggingface with your token
huggingface-cli login

MODEL_ID="meta-llama/Llama-4-Scout-17B-16E" #Huggingface model id
DOWNLOAD_DIR="$SCRIPT_DIR/Llama-4-Scout-17B-16E"

echo "Downloading HF weights for model_id $MODEL_ID"

#Download HF weights
tune download $MODEL_ID --output-dir $DOWNLOAD_DIR

echo "Model weights downloaded and saved in $DOWNLOAD_DIR successfully"


#Download dolly-15k dataset
echo "Downloading dolly-15k dataset"
wget --no-check-certificate https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
