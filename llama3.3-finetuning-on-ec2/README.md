# Instructions for fine-tuning LLama3.3 70B model on Amazon EC2 `g6e.48xlarge` instance 

## Overview <a name="overview2"></a>

This tutoral shows step-by-step instructions on fine-tuning LLama3.3 70B model on Amazon EC2 [g6e.48xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) instance. In this example, the [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) model will undergo fine-tuning using the opensource dataset: [Hugging face databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Note: You can also run the same script on p4 or p5 instances. You can increase the `max_seq_length`, `per_device_train_batch_size` or tune other hyperparametersin `config.yaml` according to the chosen instance type.

## 1. Setup Environment <a name="ec2Instance"></a>

### 1.1 Launch Amazon EC2 instance - `g6e.48xlarge`

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

### 1.1 Clone the Github respository

```
sudo apt-get update

export PATH=/home/ubuntu/.local/bin:$PATH

git clone https://github.com/aws-samples/Meta-Llama-on-AWS

cd Meta-Llama-on-AWS/llama3.3-finetuning-on-ec2
```

### 1.2 Install `uv`

Follow the instructions below to install `uv` and the required python packages to run the finetuning job. 

```
# Install UV if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure UV is in the PATH
export PATH="$HOME/.local/bin:$PATH"

# Create a virtual environment
uv venv 

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Set environment variable for UV
export UV_PROJECT_ENVIRONMENT=.venv

python -m ipykernel install --user --name=.venv --display-name="Python (uv env)"
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

## 4. Results

This training run also produces results in the form of EC2 metrics. The results can be found in the `results` directory. The results directory contains the following two files:

1. `ec2_metrics.csv`: This file contains instance utilization metrics. For example, the mean GPU utilization, CPU utilization and the memory used up by the instance. These metrics are only collected during the training process. View an example of some of the metrics generated below:

  ```{.csv}
  timestamp,cpu_percent_mean,memory_percent_mean,memory_used_mean,gpu_utilization_mean,gpu_memory_used_mean,gpu_memory_free_mean,gpu_memory_total_mean
    2025-02-21 19:42:37,0.0,2.9,7.651355743408203,0.0,16002.375,29370.312500000004,46068.0
    2025-02-21 19:42:42,1.9079216328354662,3.321088803677962,7.926143636541902,72.13185182101137,16602.57081138804,28770.116622093094,46068.0
    2025-02-21 19:42:47,2.003390286757495,3.410012950457253,7.992933634798297,82.83819711742942,16809.281850116146,28563.40561512593,46068.0
    ```
  
1. `training_stats.txt`: This file logs some of the trainer stats, such as the number of global steps it took to get to a specific training loss, the train runtime, samples per second, steps per second, etc. View an example below:

    ```{.txt}
    {'train_runtime': 6179.1633, 'train_samples_per_second': 0.508, 'train_steps_per_second': 0.008, 'train_loss': 2.6074918508529663, 'epoch': 2.85}
    ```

To report any bugs, raise an issue via the GitHub Issues feature.


