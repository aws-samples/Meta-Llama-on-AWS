# Instructions for fine-tuning LLama3.3 70B model on Amazon EC2 g6e.48xlarge instance 

## Overview <a name="overview2"></a>

This tutoral shows step-by-step instructions on fine-tuning LLama3.3 70B model on Amazon EC2 [g6e.48xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) instance. In this example, the [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) model will undergo fine-tuning using the opensource dataset: [Hugging face databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Note: You can also run the same script on p4 or p5 instances. You can increase the `max_seq_length`, `per_device_train_batch_size` or tune other hyperparametersin `config.yaml` according to the chosen instance type.

## 1. Setup Environment <a name="ec2Instance"></a>

### 1.1 Launch Amazon EC2 instance - g6e.48xlarge

In your chosen region (for ex: us-west-2), use the AWS Console or AWS CLI to launch an instance with the following configuration:

* **Instance Type:** g6e.48xlarge
* **AMI:** Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
* **Key pair name:** (choose a key pair that you have access to) 
* **Auto-assign public IP:** Enabled
* **Storage:** 500 GiB root volume

#### Log into your g6e.48xlarge instance as follows:

* Connect to your instance via the AWS Console using [EC2 Instance Connect](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Connect-using-EC2-Instance-Connect.html)
* SSH to your instance's public IP using the key pair you specified above.
  * Ex: `ssh -i KEYPAIR.pem ec2-user@INSTANCE_PUBLIC_IP_ADDRESS`

### 1.2 Install dependencies

Follow the below instructions to install/update the dependencies
```
sudo apt-get update

export PATH=/home/ubuntu/.local/bin:$PATH

git clone https://github.com/aws-samples/Meta-Llama-on-AWS

cd Meta-Llama-on-AWS/llama3.3-finetuning-on-ec2

pip3 install -r requirements.txt
```

## 2. Run Fine Tuning

### 2.1 Login to huggingface using your tokens

```
huggingface-cli login --token <YOUR_TOKEN>
```

### 2.2 Download and prepare Dolly Dataset

```
wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
```

### 2.2 Prepare Dataset
The following script will format the dataset and split train/test data.

```
python3 prepare_dataset.py
```

### 2.3 Run fine-tuning

```
nohup ./run_tuning.sh > tuning_job.log 2>&1&
```

## 3. Monitoring

While the job is running you can monitor the log `tuning_job.log` and GPU utlization using `nvidia-smi`. You can also install tensorboard and monitor the tensorboard logs collected in the same working directory under a sub-directory named `runs`.



To report any bugs, raise an issue via the GitHub Issues feature.


