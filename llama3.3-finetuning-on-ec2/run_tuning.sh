#!/bin/bash

ACCELERATE_USE_FSDP=1 torchrun --nproc_per_node=8 ./finetune_llama70b.py --config config.yaml --output_dir ./llama-3-70b-dolly
