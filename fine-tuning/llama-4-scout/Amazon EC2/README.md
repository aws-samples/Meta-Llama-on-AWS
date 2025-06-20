# Instructions for fine-tuning LLama4-Scout model on Amazon EC2 p5.48xlarge (or) p4de.24xlarge instance 

## Overview <a name="overview2"></a>

This tutoral shows step-by-step instructions on fine-tuning LLama4-Scout model on Amazon EC2 p5.48xlarge (or) p4de.24xlarge instance. In this example, the [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) model will undergo fine-tuning using the opensource dataset: [Hugging face databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Note: You can increase the `max_seq_length`, `batch_size` or tune other hyperparameters in `config.yaml` or `3_run_l4_tune.sh`(override) according to the chosen instance type.

## 1. Setup Environment <a name="ec2Instance"></a>

## Prequisites

### 1.1 Launch Amazon EC2 instance 

In your chosen region (for ex: us-west-2), use the AWS Console or AWS CLI to launch an instance with the following configuration:

* **Instance Type:** p5.48xlarge (or) p4de.24xlarge
* **AMI:** Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)
* **Key pair name:** (choose a key pair that you have access to) 
* **Auto-assign public IP:** Enabled
* **Storage:** 1024 GiB root volume

#### Log into your g6e.48xlarge instance as follows:

* Connect to your instance via the AWS Console using [EC2 Instance Connect](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Connect-using-EC2-Instance-Connect.html)
* SSH to your instance's public IP using the key pair you specified above.
  * Ex: `ssh -i KEYPAIR.pem ec2-user@INSTANCE_PUBLIC_IP_ADDRESS`

### 1.2. Go to  [HF llama4 page](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and reuquest access.

### 1.3. create a access token in HF by following this instruction https://huggingface.co/docs/transformers.js/en/guides/private

### 1.4 install required python packages.  
```
sudo yum install python3-devel -y
```

### 1.5. install requried gcc package  
```
sudo yum install gcc -y
```

### 1.6. start a tmux session- this will make sure you can get back it even if your connection to p5 terminates.  
```
tmux
```


## 2. Install dependencies and download HF weights
```
sh 0_install_deps_download_hf_weghts.sh
```

Enter your HF token when prompted.  
choose n to continue for this question -> Add token as git credential? (Y/n)   


## 3. Download and prepare dataset
```
python3 1_prepare_dataset.py
```

### 3.1 Timstamp for logging

```
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
```

## 4. Inference test before fine tuning 
```
python3 2_inference_test_before.py 2>&1 | tee l4_lora_output/logs/inference_before_ft_output_${TIMESTAMP}.log
```

## 5. Run tuning, merge and save weights
```
sh 3_run_l4_tune.sh 2>&1 | tee l4_lora_output/logs/l4_tune_${TIMESTAMP}.log
```

## 4. Inference test after fine tuning 
```
python3 4_inference_test_after.py 2>&1 | tee l4_lora_output/logs/inference_after_ft_output_${TIMESTAMP}.log
```

## 5. Monitoring

While the job is running you can monitor the log `l4_tune_${TIMESTAMP}.log` and GPU utlization using `nvidia-smi`. 


To report any bugs, raise an issue via the GitHub Issues feature.


