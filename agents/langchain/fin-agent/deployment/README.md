# SageMaker Endpoint Deployment Guide - LMI Container

This guide covers deploying Llama models (8B and 70B) to AWS SageMaker with function calling support using AWS's LMI (Large Model Inference) container with vLLM backend.

## Why LMI Container?

We use the **LMI (Large Model Inference) container** because:
- Native support for function calling with `llama3_json` parser
- Built-in vLLM backend for optimized inference
- Automatic tool call parsing and formatting
- Better performance for tool-based workflows

**Note**: The standard HuggingFace TGI container does not support the function calling features required for this agent.

## Model Comparison

| Feature | Llama 3.1 8B | Llama 3.1/3.3 70B |
|---------|-------------|-------------------|
| Context Window | 8,192 tokens | Up to 128K tokens |
| Tool Calling | Inconsistent on complex queries | Reliable, multi-step support |
| Min Instance | `ml.g5.2xlarge` (1 GPU) | `ml.g5.48xlarge` (8 GPUs) |
| Min Cost/Hour | ~$1.52 | ~$7.09 (quantized) / ~$20.36 (FP16) |
| Deploy Time | ~5-10 min | ~10-25 min |
| Best For | Learning, demos, simple queries | Production, complex multi-tool queries |
| Deploy Script | `deploy_llama3_lmi.py` | `deploy_llama3_70b.py` |

Both models use the same LMI container, vLLM backend, and OpenAI-compatible chat completion API. The agent code works with either model — switch by changing only the `SAGEMAKER_ENDPOINT_NAME` environment variable:

```bash
# Use the 8B model (lower cost, development)
export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"

# Use the 70B model (better quality, production)
export SAGEMAKER_ENDPOINT_NAME="llama3-70b-lmi-agent"

# Then run the agent as usual
python fin-agent-sagemaker-v2.py
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deploying the 8B Model](#deploying-the-8b-model)
- [Deploying the 70B Model](#deploying-the-70b-model)
  - [Instance Type Selection](#instance-type-selection)
  - [70B Quick Start](#70b-quick-start)
  - [S3 Model Pre-Download](#s3-model-pre-download)
- [IAM Role Setup](#iam-role-setup)
- [Configuration Reference](#configuration-reference)
- [Validation](#validation)
- [Monitoring](#monitoring)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)
- [Additional Resources](#additional-resources)

## Prerequisites

### 1. AWS Account Setup

1. **AWS Account**: Active AWS account with billing enabled
2. **AWS CLI**: Installed and configured
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and default region
   ```

3. **Python Environment**: Python 3.10 or higher
   ```bash
   python --version  # Should be 3.10+
   ```

4. **Required Python Packages**:
   ```bash
   pip install boto3
   # or use uv
   uv pip install boto3
   ```

### 2. HuggingFace Token Setup

The deployment requires access to Meta's Llama models:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the Llama license:
   - 8B: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - 70B: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

### 3. Check Service Quotas

Check your SageMaker service quotas for GPU instances:

```bash
# For 8B deployment (ml.g5.2xlarge)
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --region us-west-2 \
  --query "Quotas[?contains(QuotaName, 'g5.2xlarge') && contains(QuotaName, 'endpoint')].{Name:QuotaName,Value:Value,Code:QuotaCode}" \
  --output table

# For 70B deployment (ml.g5.48xlarge)
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --region us-west-2 \
  --query "Quotas[?contains(QuotaName, 'g5.48xlarge') && contains(QuotaName, 'endpoint')].{Name:QuotaName,Value:Value,Code:QuotaCode}" \
  --output table
```

If you need more quota, request an increase through the AWS Console:
1. Go to Service Quotas → AWS Services → Amazon SageMaker
2. Find the quota for your target instance type
3. Request quota increase if needed

---

## Deploying the 8B Model

The 8B model is ideal for development, learning agent patterns, and cost-conscious experimentation. It runs on a single GPU instance.

### Quick Start

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Run deployment
python deploy_llama3_lmi.py
```

The script will:
- Create or use existing SageMaker execution role
- Deploy Llama 3.1 8B with LMI container and vLLM backend
- Configure tool calling support with `llama3_json` parser
- Wait for endpoint to be ready (~5-10 minutes)

### 8B Default Configuration

```python
ENDPOINT_NAME = "llama3-lmi-agent"
INSTANCE_TYPE = "ml.g5.2xlarge"  # 1x A10G GPU (24GB)
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_ENV = {
    "HF_MODEL_ID": HF_MODEL_ID,
    "HF_TOKEN": HF_TOKEN,
    "OPTION_ROLLING_BATCH": "vllm",
    "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",
    "OPTION_TOOL_CALL_PARSER": "llama3_json",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
    "OPTION_MAX_MODEL_LEN": "8192",
    "OPTION_DTYPE": "fp16",
    "TENSOR_PARALLEL_DEGREE": "1"
}
```

### 8B Instance Options

| Instance Type | GPU | vCPU | Memory | Use Case | Cost/Hour |
|---------------|-----|------|--------|----------|-----------|
| ml.g5.2xlarge | 1 | 8 | 32 GB | Development, low traffic | ~$1.52 |
| ml.g5.4xlarge | 1 | 16 | 64 GB | Production, medium traffic | ~$2.03 |
| ml.g5.12xlarge | 4 | 48 | 192 GB | High traffic, low latency | ~$7.09 |

### 8B Expected Output

```
================================================================================
  Llama 3.1 8B Model Deployment (LMI with vLLM)
================================================================================

Endpoint Name: llama3-lmi-agent
Instance Type: ml.g5.2xlarge
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Container: LMI (Large Model Inference) with vLLM backend
Tool Calling: Enabled with llama3_json parser
Region: us-west-2

Proceed with deployment? (yes/no): yes

🔍 Looking for SageMaker execution role...
✅ Found existing role: arn:aws:iam::123456789:role/SageMakerExecutionRole
...
================================================================================
✅ Endpoint deployed successfully!

Endpoint Name: llama3-lmi-agent
Status: InService
================================================================================

📋 Next Steps:
1. export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"
2. python test_multiple_parallel_tools.py
3. python fin-agent-sagemaker-v2.py
```

### 8B Known Limitations

- **Inconsistent tool calling format**: May fail on complex multi-step queries
- **Context window limit (8,192 tokens)**: Agent can make 5-6 tool calls before hitting the limit
- **Sequential tool calling only**: SageMaker endpoint processes one tool at a time

For production workloads, consider upgrading to the 70B model.

---

## Deploying the 70B Model

The 70B model offers significantly improved tool calling reliability, a 128K token context window, and better reasoning for complex multi-tool queries. It requires a multi-GPU instance with tensor parallelism to fit its ~140GB FP16 weights in GPU memory.

### Instance Type Selection

The deployment script includes an interactive instance selector. Run the script and choose from the menu:

```bash
python deployment/deploy_llama3_70b.py
```

```
========================================================================
  SELECT INSTANCE TYPE
========================================================================

  [1] ml.g5.48xlarge ← current default
      GPUs: 8x A10G  |  VRAM: 192 GB  |  TP: 8-way
      Context: 16384 tokens  |  Batch: 4
      Dtype: fp16  |  GPU Mem: 0.95
      Cost: ~$20.36/hr
      Default. FP16, no quantization needed.

  [2] ml.p4d.24xlarge
      GPUs: 8x A100  |  VRAM: 320 GB  |  TP: 8-way
      Context: 32768 tokens  |  Batch: 8
      Dtype: fp16  |  GPU Mem: 0.90
      Cost: ~$25.25/hr
      Higher throughput, lower latency. Supports 32K+ context.

  [3] ml.g5.12xlarge
      GPUs: 4x A10G  |  VRAM: 96 GB  |  TP: 4-way
      Context: 8192 tokens  |  Batch: 4
      Dtype: auto  |  GPU Mem: 0.90
      Cost: ~$7.09/hr
      Budget. Requires AWQ/GPTQ 4-bit quantized model.
      Model: hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

  [4] ml.p4d.24xlarge:max
      GPUs: 8x A100  |  VRAM: 320 GB  |  TP: 8-way
      Context: 65536 tokens  |  Batch: 16
      Dtype: fp16  |  GPU Mem: 0.95
      Cost: ~$25.25/hr
      Max context. 70B FP16 uses ~140GB, leaves ~180GB for KV cache.

  [0] Cancel
```

To skip the interactive selector in scripted deployments, use `--instance-type`:

```bash
python deployment/deploy_llama3_70b.py --instance-type ml.p4d.24xlarge
python deployment/deploy_llama3_70b.py --instance-type "ml.p4d.24xlarge:max"
```

**Choosing an instance type:**
- **g5.48xlarge (default)**: Best balance of cost and performance for FP16 inference. 192GB total VRAM across 8 GPUs, but each A10G only has 24GB — the 70B model shard uses ~16.5GB per GPU, leaving ~5.5GB per GPU for KV cache. Context window is limited to 16K tokens by default.
- **p4d.24xlarge**: Use when you need higher throughput or lower per-token latency. 320GB VRAM allows larger batch sizes and 32K context. Only ~$5/hr more than g5.48xlarge.
- **g5.12xlarge**: Budget option requiring 4-bit quantization (AWQ or GPTQ). 96GB VRAM fits the quantized model (~35GB) with ample KV cache room. The deployment script auto-selects the quantized model ID for this profile.
- **p4d.24xlarge:max**: Same hardware as p4d.24xlarge but tuned for maximum context (65K tokens) and throughput (batch 16, GPU mem 0.95). The 70B FP16 model uses ~140GB, leaving ~180GB for KV cache across 8 A100s.

| Instance Type | GPUs | Total VRAM | TP Degree | Est. Cost/hr | Deploy Time (S3) | Notes |
|---------------|------|------------|-----------|--------------|------------------|-------|
| `ml.g5.48xlarge` | 8x A10G | 192 GB | 8 | ~$20.36 | ~10-15 min | **Default** — FP16, no quantization needed |
| `ml.p4d.24xlarge` | 8x A100 | 320 GB | 8 | ~$25.25 | ~15-25 min | Higher throughput, lower latency, 32K context |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | 4 | ~$7.09 | ~10-15 min | Requires AWQ/GPTQ 4-bit quantization |
| `ml.p4d.24xlarge:max` | 8x A100 | 320 GB | 8 | ~$25.25 | ~15-25 min | Max context (65K), batch 16, GPU mem 0.95 |

### 70B Quick Start

**Default deployment (interactive instance selector):**

```bash
export HF_TOKEN="your_token_here"
python deployment/deploy_llama3_70b.py
```

**Faster deployment with S3 pre-downloaded weights:**

```bash
# Step 1: Pre-download model to S3 (one-time, ~140GB)
python deployment/download_model_to_s3.py

# Step 2: Deploy from S3 (significantly faster)
python deployment/deploy_llama3_70b.py \
  --model-s3-uri s3://my-bucket/Meta-Llama-3.1-70B-Instruct/
```

**Quantized model deployment (g5.12xlarge — budget option):**

```bash
# Step 1: Download the AWQ-INT4 quantized model to S3
python deployment/download_model_to_s3.py \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

# Step 2: Deploy on g5.12xlarge with the quantized weights
python deployment/deploy_llama3_70b.py \
  --instance-type ml.g5.12xlarge \
  --model-s3-uri s3://my-bucket/Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
```

> **⚠️ S3 Prefix Collision Warning**: The download script auto-derives the S3 prefix from the model ID, so FP16 and quantized models get separate prefixes. If you previously used `--s3-prefix llama-70b-instruct/` for both, they would overwrite each other.

> **⚠️ FP16 + g5.12xlarge Mismatch**: If you select `ml.g5.12xlarge` and provide `--model-s3-uri` pointing to FP16 weights (~140GB), the deployment will fail because FP16 weights don't fit in 96GB VRAM. Either use the quantized model or pick a larger instance.

### 70B Default Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_MODEL_ID` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Model to load (or S3 URI when using pre-download) |
| `OPTION_ROLLING_BATCH` | `vllm` | vLLM backend for PagedAttention-based memory management |
| `OPTION_ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable automatic tool calling for agent workflows |
| `OPTION_TOOL_CALL_PARSER` | `llama3_json` | Llama 3 JSON parser for structured tool call extraction |
| `OPTION_MAX_ROLLING_BATCH_SIZE` | `4` | Max concurrent requests (lower than 8B's 32 due to memory) |
| `OPTION_MAX_MODEL_LEN` | `16384` | Max sequence length — 16K tokens (use p4d.24xlarge for 32K+) |
| `OPTION_DTYPE` | `fp16` | Full FP16 precision for best quality |
| `OPTION_GPU_MEMORY_UTILIZATION` | `0.95` | Allocate 95% of GPU VRAM to vLLM, maximizes KV cache room |
| `TENSOR_PARALLEL_DEGREE` | `8` | Distribute model across all 8 GPUs |

**Additional tunable parameters (not set by default):**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPTION_ENABLE_CHUNKED_PREFILL` | not set | Set to `"true"` to overlap prefill and decode phases for better throughput |
| `OPTION_SPECULATIVE_DECODING` | not set | Enable speculative decoding with a draft model (experimental) |

### S3 Model Pre-Download

Pre-downloading the 70B model weights to S3 avoids repeated ~140GB downloads from HuggingFace during development and testing, and significantly speeds up SageMaker endpoint creation.

> **⚠️ Disk Space Requirement**: The script downloads the model to a local temp directory before uploading to S3. You need at least **~150GB of free disk space** for FP16 or **~40GB** for quantized models. The temp files are automatically cleaned up after the upload completes. Check available space with `df -h .` before running.

```bash
# FP16 model (default) — stored at s3://bucket/Meta-Llama-3.1-70B-Instruct/
python deployment/download_model_to_s3.py

# Quantized AWQ-INT4 model — stored at s3://bucket/Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
python deployment/download_model_to_s3.py \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

# Specify a custom bucket (prefix still auto-derived):
python deployment/download_model_to_s3.py --s3-bucket my-model-bucket
```

**S3 prefix auto-derivation examples:**

| `--model-id` | Auto-derived `--s3-prefix` |
|---|---|
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | `Meta-Llama-3.1-70B-Instruct/` |
| `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | `Meta-Llama-3.1-70B-Instruct-AWQ-INT4/` |
| `meta-llama/Llama-3.3-70B-Instruct` | `Llama-3.3-70B-Instruct/` |

The script will:
1. Validate S3 bucket access
2. Check if model already exists at the derived prefix (skip if found, use `--force` to re-upload)
3. Download model artifacts from HuggingFace Hub to a local temp directory
4. Upload all files to S3 with multipart upload (100MB chunks, 10 concurrent threads)
5. Display progress with estimated time remaining
6. Print the S3 URI for use with the deployment script

---

## IAM Role Setup

The deployment scripts automatically handle IAM role creation. If you need to create it manually:

**Option A: Using AWS Console**

1. Go to IAM → Roles → Create role
2. Select "AWS service" → "SageMaker"
3. Select "SageMaker - Execution"
4. Click "Next"
5. Attach policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or more restrictive S3 policy)
6. Name the role: `SageMakerLlamaExecutionRole`
7. Create role
8. Copy the Role ARN (e.g., `arn:aws:iam::123456789:role/SageMakerLlamaExecutionRole`)

**Option B: Using AWS CLI**

```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name SageMakerLlamaExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SageMakerLlamaExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Get role ARN
aws iam get-role \
  --role-name SageMakerLlamaExecutionRole \
  --query 'Role.Arn' \
  --output text
```

### Minimum Required Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerModelManagement",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:DescribeModel",
        "sagemaker:DeleteModel"
      ],
      "Resource": "arn:aws:sagemaker:*:*:model/*"
    },
    {
      "Sid": "SageMakerEndpointManagement",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateEndpointConfig",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": [
        "arn:aws:sagemaker:*:*:endpoint-config/*",
        "arn:aws:sagemaker:*:*:endpoint/*"
      ]
    },
    {
      "Sid": "ECRAccess",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    }
  ]
}
```

### S3 Permissions (70B only)

If using S3 model pre-download, the user running the download script needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-model-bucket",
        "arn:aws:s3:::my-model-bucket/*"
      ]
    }
  ]
}
```

The SageMaker execution role also needs read access to the S3 model artifacts:

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::my-model-bucket/*"
}
```

---

## Configuration Reference

### Inference Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_new_tokens` | Maximum tokens to generate | 512 | 1-4096 |
| `temperature` | Sampling temperature (higher = more random) | 0.1 | 0.0-2.0 |
| `top_p` | Nucleus sampling threshold | 0.9 | 0.0-1.0 |
| `do_sample` | Enable sampling (required for temperature) | true | true/false |
| `repetition_penalty` | Penalty for repetition | 1.0 | 1.0-2.0 |

**Note**: The LMI container uses different parameter names than TGI. Use `do_sample=true` instead of `return_full_text=false`.

### Container Environment Variables (8B)

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_MODEL_ID` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Model to load |
| `OPTION_ROLLING_BATCH` | `vllm` | Use vLLM backend for inference |
| `OPTION_ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable automatic tool calling |
| `OPTION_TOOL_CALL_PARSER` | `llama3_json` | Use Llama 3 JSON parser for function calls |
| `OPTION_MAX_ROLLING_BATCH_SIZE` | `32` | Maximum batch size for rolling batch |
| `OPTION_MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `OPTION_DTYPE` | `fp16` | Use FP16 precision for faster inference |
| `TENSOR_PARALLEL_DEGREE` | `1` | Number of GPUs for tensor parallelism |

### Container Environment Variables (70B)

See [70B Default Configuration](#70b-default-configuration) above.

### Timeout Configuration

| Timeout | Default | Purpose |
|---------|---------|---------|
| `model_data_download_timeout` | 600s | Time to download model |
| `container_startup_health_check_timeout` | 600s | Time for health checks |
| `deployment_timeout` | 1800s | Total deployment timeout |

---

## Validation

### Automatic Validation

Both deployment scripts automatically validate the endpoint by:

1. **Connectivity Test**: Verifies endpoint is reachable
2. **Response Structure Test**: Checks response format
3. **Generation Test**: Validates text generation works

### Manual Validation

Test the endpoint manually:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

test_payload = {
    "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSay hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9
    }
}

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',  # llama3-lmi-agent or llama3-70b-lmi-agent
    ContentType='application/json',
    Accept='application/json',
    Body=json.dumps(test_payload)
)

result = json.loads(response['Body'].read().decode('utf-8'))
print("✓ Endpoint is functional")
print(f"Response: {result['generated_text']}")
```

### Health Check

```bash
aws sagemaker describe-endpoint \
  --endpoint-name your-endpoint-name \
  --region us-west-2 \
  --query 'EndpointStatus' \
  --output text
```

Expected output: `InService`

---

## Monitoring

### CloudWatch Metrics

Monitor these key metrics in CloudWatch:

1. **Invocations**: Number of inference requests
2. **ModelLatency**: Time for model inference (ms)
3. **OverheadLatency**: Time for request/response processing (ms)
4. **Invocation4XXErrors**: Client errors
5. **Invocation5XXErrors**: Server errors

### View Metrics in Console

1. Go to CloudWatch → Metrics → All metrics
2. Select "AWS/SageMaker"
3. Select "Endpoint Metrics"
4. Choose your endpoint name

### CloudWatch Logs

```bash
# View logs (replace with your endpoint name)
aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name \
  --follow \
  --region us-west-2
```

### Set Up Alarms

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name sagemaker-endpoint-errors \
  --alarm-description "Alert on endpoint errors" \
  --metric-name Invocation5XXErrors \
  --namespace AWS/SageMaker \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=EndpointName,Value=your-endpoint-name \
  --region us-west-2
```

---

## Cleanup

When you're done, delete the endpoint to stop incurring charges.

> **⚠️ Important**: The 70B endpoint costs ~$20.36/hour. Always delete when not in use during development.

**Option A: Using AWS Console**

1. Go to SageMaker → Endpoints
2. Select your endpoint
3. Click "Delete"
4. Confirm deletion

**Option B: Using AWS CLI**

```bash
# Delete 8B endpoint
aws sagemaker delete-endpoint --endpoint-name llama3-lmi-agent --region us-west-2

# Delete 70B endpoint
aws sagemaker delete-endpoint --endpoint-name llama3-70b-lmi-agent --region us-west-2

# Also clean up endpoint configuration and model
aws sagemaker delete-endpoint-config --endpoint-config-name your-config-name --region us-west-2
aws sagemaker delete-model --model-name your-model-name --region us-west-2
```

**Option C: Using the cleanup script**

```bash
python cleanup_project.py
```

### Verify Deletion

```bash
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name --region us-west-2
```

Expected: `ResourceNotFound` error

---

## Troubleshooting

### ResourceLimitExceeded (Quota)

**Error:**
```
ResourceLimitExceeded: The account-level service limit 'ml.g5.Xxlarge for endpoint usage' is 0 Instances
```

**Solution:**
1. Request quota increase (see [Prerequisites](#3-check-service-quotas))
2. Try a different instance type
3. Deploy in a different region
4. For 70B, quota increases for GPU instances may take 1-3 business days

### Endpoint Stuck in Creating Status

**Symptoms:** Endpoint status remains "Creating" for > 15 minutes (8B) or > 30 minutes (70B)

**Solutions:**
1. Check CloudWatch logs:
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name --follow
   ```
2. Verify the model ID is correct and accessible
3. Check if the container image is available in your region
4. Increase timeout and retry

**70B-specific:** The 70B endpoint takes 10-15 minutes on G5 instances and 15-25 minutes on p4d.24xlarge. This is normal.

**Why p4d.24xlarge takes longer:**

| Factor | Details |
|--------|---------|
| Instance provisioning | A100 GPUs are less common, AWS takes longer to allocate |
| GPU initialization | 8x A100s have complex memory hierarchies (HBM2e) and NVLink interconnects |
| Model sharding | vLLM must shard ~140GB across 8 GPUs and configure NCCL communication rings |
| Health check warmup | First inference pass on A100s with large context/batch configs takes longer |

### Validation Fails / ModelError

**Solutions:**
1. Check endpoint logs for model loading errors
2. Verify instance type has enough memory for the model
3. Check if model requires authentication (HuggingFace token)
4. Try with a smaller model first to isolate the issue

### IAM Permission Denied

**Solutions:**
1. Verify IAM role has required permissions (see [IAM Role Setup](#iam-role-setup))
2. Check trust relationship allows SageMaker service
3. Ensure you're using the correct role ARN
4. Wait a few minutes for IAM changes to propagate

### High Latency

**Solutions:**
1. Use a larger instance type with more GPUs
2. Reduce `max_new_tokens` to generate shorter responses
3. Enable auto-scaling for load distribution
4. Consider using multiple endpoints with load balancing

### Out of Memory (CUDA OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**8B Solutions:**
1. Use a larger instance type (more GPU memory)
2. Reduce `OPTION_MAX_MODEL_LEN`
3. Reduce `max_new_tokens` in inference parameters

**70B Solutions:**
1. The default `OPTION_MAX_MODEL_LEN` is `16384` — do not increase beyond this on g5.48xlarge
2. Reduce `OPTION_MAX_ROLLING_BATCH_SIZE` (e.g., from `4` to `2`)
3. `OPTION_GPU_MEMORY_UTILIZATION` is `0.95` by default — do not lower it on g5.48xlarge
4. Upgrade to `ml.p4d.24xlarge` (320 GB VRAM) for longer context windows (32K+)
5. Use a quantized model variant on `ml.g5.12xlarge` with `OPTION_DTYPE="auto"`

**70B KV Cache Error:**
```
ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache
```

On `ml.g5.48xlarge`, each A10G GPU has 24GB VRAM. The 70B FP16 model shard uses ~16.5GB per GPU, leaving only ~5.5GB for KV cache. Reduce `OPTION_MAX_MODEL_LEN` to `16384` or `8192`.

### Instance Capacity Errors (70B)

**Error:**
```
Unable to provision requested ML compute capacity due to InsufficientInstanceCapacity error.
```

This means AWS didn't have the requested GPU instance available in your region. Common with `ml.p4d.24xlarge` (A100 GPUs).

**Solutions:**
1. Clean up the failed endpoint and retry later (capacity fluctuates, try off-peak hours)
2. Fall back to `ml.g5.48xlarge` (more abundant)
3. Use `ml.g5.12xlarge` with AWQ quantization as a budget fallback

### 70B Memory Requirements

The 70B model in FP16 requires approximately:
- **~140 GB** for model weights
- **~20-50 GB** additional for KV cache (depends on `OPTION_MAX_MODEL_LEN` and batch size)
- **Total: ~160-192 GB** GPU VRAM

This is why `ml.g5.48xlarge` (192 GB across 8x A10G GPUs) is the minimum recommended instance for FP16 inference.

---

## Cost Estimation

### Hourly Costs (us-west-2)

| Instance Type | Model | Cost/Hour |
|---------------|-------|-----------|
| ml.g5.2xlarge | 8B | ~$1.52 |
| ml.g5.4xlarge | 8B | ~$2.03 |
| ml.g5.12xlarge | 8B or 70B (quantized) | ~$7.09 |
| ml.g5.48xlarge | 70B (FP16) | ~$20.36 |
| ml.p4d.24xlarge | 70B (FP16) | ~$25.25 |

### Monthly Cost Examples

**8B — Development (8 hrs/day, 20 days/month, ml.g5.2xlarge):**
- Hours: 160/month → ~$243/month

**8B — Production (24/7, ml.g5.2xlarge):**
- Hours: 720/month → ~$1,094/month

**70B — Development (8 hrs/day, 20 days/month, ml.g5.48xlarge):**
- Hours: 160/month → ~$3,258/month

**70B — Production (24/7, ml.g5.48xlarge):**
- Hours: 720/month → ~$14,659/month

> **⚠️ Cost Warning**: The 70B deployment on `ml.g5.48xlarge` costs roughly **13x more** than the 8B on `ml.g5.2xlarge`. Always delete the endpoint when not in use during development.

### Cost Optimization Tips

1. **Delete endpoints when not in use** (especially 70B during development)
2. **Use auto-scaling** to match demand
3. **Set up CloudWatch alarms** for unexpected usage
4. **Use the 8B model** for development and testing, 70B for production
5. **Consider quantized 70B** on `ml.g5.12xlarge` (~$7.09/hr) for budget-conscious production
6. **Batch requests** when possible to maximize throughput

---

## Next Steps

After successful deployment:

1. **Set endpoint variable**: `export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"`
2. **Test with tools**: `python test_multiple_parallel_tools.py`
3. **Run the agent**: `python fin-agent-sagemaker-v2.py`
4. **Monitor performance**: Set up CloudWatch dashboards and alarms
5. **Optimize costs**: Configure auto-scaling based on usage patterns
6. **Review security**: Ensure IAM policies follow least privilege principle

## Additional Resources

- [AWS SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [AWS LMI Container Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Service Quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
