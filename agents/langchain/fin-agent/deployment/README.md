# SageMaker Endpoint Deployment Guide - LMI Container

This guide provides step-by-step instructions for deploying the Llama 3.1 8B Instruct model to AWS SageMaker with function calling support using AWS's LMI (Large Model Inference) container with vLLM backend.

## Why LMI Container?

We use the **LMI (Large Model Inference) container** because:
- Native support for function calling with `llama3_json` parser
- Built-in vLLM backend for optimized inference
- Automatic tool call parsing and formatting
- Better performance for tool-based workflows

**Note**: The standard HuggingFace TGI container does not support the function calling features required for this agent.

## Quick Start


The script will:
- Create or use existing SageMaker execution role
- Deploy Llama 3.1 8B with **LMI container and vLLM backend**
- Configure tool calling support with **llama3_json parser**
- Wait for endpoint to be ready (~5-10 minutes)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Script](#deployment-script)
- [Configuration](#configuration)
- [IAM Role Setup](#iam-role-setup)
- [Validation](#validation)
- [Monitoring](#monitoring)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)
- [Llama 70B Deployment](#llama-70b-deployment)
  - [70B Interactive Instance Selector](#70b-interactive-instance-selector)
  - [70B Instance Type Comparison](#70b-instance-type-comparison)
  - [70B Quick Start](#70b-quick-start)
  - [70B Configuration](#70b-configuration)
  - [S3 Model Pre-Download](#s3-model-pre-download)
  - [70B Cost Estimates](#70b-cost-estimates)
  - [70B Troubleshooting](#70b-troubleshooting)
  - [Switching Between 8B and 70B](#switching-between-8b-and-70b)

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

The deployment requires access to Meta's Llama 3.1 model:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   # or
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   ```

### 3. Check Service Quotas

Check your SageMaker service quotas for GPU instances:

If you need more quota, request an increase through the AWS Console webpage:
1. Go to Service Quotas → AWS Services → Amazon SageMaker
2. Find "ml.g5.2xlarge for endpoint usage"
3. Request quota increase if needed

```bash
# List SageMaker endpoint quotas and search for your instance type
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --region us-west-2 \
  --query "Quotas[?contains(QuotaName, 'g5.2xlarge') && contains(QuotaName, 'endpoint')].{Name:QuotaName,Value:Value,Code:QuotaCode}" \
  --output table
```

## Run Deployment Script

Configure and run the deploy_llama3_lmi.py script that automates the entire deployment process:

### Features
- **Automatic IAM Role Management**: Creates or uses existing SageMaker execution role
- **LMI Container**: Uses AWS's Large Model Inference container with vLLM backend
- **Tool Calling Support**: Configures llama3_json parser for function calling
- **Progress Monitoring**: Real-time deployment status updates
- **Error Handling**: Clear error messages and troubleshooting guidance

### Configuration

The script uses these default settings (can be modified in the script):

```python
# Endpoint Configuration
ENDPOINT_NAME = "llama3-lmi-agent"
AWS_REGION = "us-west-2"
INSTANCE_TYPE = "ml.g5.2xlarge"  # 1x A10G GPU (24GB)
INSTANCE_COUNT = 1

# Model Configuration
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# LMI Container Environment Variables
# These configure the vLLM backend for optimal function calling
MODEL_ENV = {
    "HF_MODEL_ID": HF_MODEL_ID,
    "HF_TOKEN": HF_TOKEN,  # From environment variable
    "OPTION_ROLLING_BATCH": "vllm",  # Use vLLM backend
    "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",  # Enable function calling
    "OPTION_TOOL_CALL_PARSER": "llama3_json",  # Use Llama 3 JSON parser
    "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
    "OPTION_MAX_MODEL_LEN": "8192",
    "OPTION_DTYPE": "fp16",
    "TENSOR_PARALLEL_DEGREE": "1"
}
```

### Running the Script

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Run deployment
python deploy_llama3_lmi.py
```

The script will:
1. Check for existing endpoint
2. Create or use existing SageMaker execution role
3. Create SageMaker model with LMI container
4. Create endpoint configuration
5. Deploy endpoint
6. Monitor deployment progress (~5-10 minutes)
7. Display success message with next steps

### Expected Output

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

--------------------------------------------------------------------------------
  Step 1: Creating Model
--------------------------------------------------------------------------------
Creating model: llama3-lmi-agent-model-1234567890
✅ Model created

--------------------------------------------------------------------------------
  Step 2: Creating Endpoint Configuration
--------------------------------------------------------------------------------
Creating endpoint configuration: llama3-lmi-agent-config-1234567890
✅ Endpoint configuration created

--------------------------------------------------------------------------------
  Step 3: Creating Endpoint
--------------------------------------------------------------------------------
Creating endpoint: llama3-lmi-agent
⏳ This will take 5-10 minutes...
✅ Endpoint creation initiated

--------------------------------------------------------------------------------
  Waiting for Endpoint Deployment
--------------------------------------------------------------------------------
[0s] Status: Creating
[30s] Status: Creating
[300s] Status: InService

================================================================================
✅ Endpoint deployed successfully in 300 seconds!

Endpoint Name: llama3-lmi-agent
Status: InService
Container: LMI with vLLM backend
Tool Calling: Enabled (llama3_json parser)
================================================================================

📋 Next Steps:
1. Set environment variable:
   export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"

2. Test tool calling:
   python test_multiple_parallel_tools.py

3. Run agent:
   python fin-agent-sagemaker-v2.py
```

## IAM Role Setup

The deployment script automatically handles IAM role creation. If you need to create it manually:

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

### Step 2: Configure Deployment

The deployment script uses sensible defaults configured for the LMI container. You can modify these in `deploy_llama3_lmi.py`:

```python
# Endpoint settings
ENDPOINT_NAME = "llama3-lmi-agent"
INSTANCE_TYPE = "ml.g5.2xlarge"  # 1x A10G GPU (24GB)

# LMI container settings
OPTION_ROLLING_BATCH = "vllm"  # Use vLLM backend
OPTION_ENABLE_AUTO_TOOL_CHOICE = "true"  # Enable function calling
OPTION_TOOL_CALL_PARSER = "llama3_json"  # Llama 3 JSON parser
```

**Instance Type Selection**:

| Instance Type | GPU | vCPU | Memory | Use Case | Cost/Hour |
|---------------|-----|------|--------|----------|-----------|
| ml.g5.2xlarge | 1 | 8 | 32 GB | Development, low traffic | ~$1.52 |
| ml.g5.4xlarge | 1 | 16 | 64 GB | Production, medium traffic | ~$2.03 |
| ml.g5.12xlarge | 4 | 48 | 192 GB | High traffic, low latency | ~$7.09 |

### Step 3: Run Deployment Script

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Run deployment
python deploy_llama3_lmi.py
```

The script will:
1. Check for existing endpoint
2. Create or use existing SageMaker execution role
3. Create SageMaker model with LMI container
4. Create endpoint configuration
5. Deploy endpoint
6. Monitor deployment progress (~5-10 minutes)
7. Display success message with next steps

### Step 4: Save Endpoint Name

```bash
# The endpoint name is shown in the deployment output
export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"

# Add to your shell profile for persistence
echo 'export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"' >> ~/.bashrc
```

### Step 5: Test the Endpoint

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Test prompt with LMI format
test_payload = {
    "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSay hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9
    }
}

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='llama3-lmi-agent',
    ContentType='application/json',
    Accept='application/json',
    Body=json.dumps(test_payload)
)

# Parse response from LMI container
result = json.loads(response['Body'].read().decode('utf-8'))
print(result['generated_text'])
```

## IAM Role Setup

### Minimum Required Permissions

Create a custom policy with minimum required permissions:

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

### Trust Relationship

Ensure your role has this trust relationship:

```json
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
```

## Configuration

### Deployment Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_new_tokens` | Maximum tokens to generate | 512 | 1-4096 |
| `temperature` | Sampling temperature (higher = more random) | 0.1 | 0.0-2.0 |
| `top_p` | Nucleus sampling threshold | 0.9 | 0.0-1.0 |
| `do_sample` | Enable sampling (required for temperature) | true | true/false |
| `repetition_penalty` | Penalty for repetition | 1.0 | 1.0-2.0 |

**Note**: The LMI container uses different parameter names than TGI. Use `do_sample=true` instead of `return_full_text=false`.

### Container Environment Variables

The deployment script configures these environment variables in the LMI container:

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

### Timeout Configuration

| Timeout | Default | Purpose |
|---------|---------|---------|
| `model_data_download_timeout` | 600s | Time to download model |
| `container_startup_health_check_timeout` | 600s | Time for health checks |
| `deployment_timeout` | 1800s | Total deployment timeout |

## Validation

### Automatic Validation

The deployment script automatically validates the endpoint by:

1. **Connectivity Test**: Verifies endpoint is reachable
2. **Response Structure Test**: Checks response format
3. **Generation Test**: Validates text generation works

### Manual Validation

Test the endpoint manually:

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Test prompt with LMI format
test_payload = {
    "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSay hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9
    }
}

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Accept='application/json',
    Body=json.dumps(test_payload)
)

# Parse response from LMI container
result = json.loads(response['Body'].read().decode('utf-8'))
print("✓ Endpoint is functional")
print(f"Response: {result['generated_text']}")
```

### Health Check

Check endpoint status:

```bash
aws sagemaker describe-endpoint \
  --endpoint-name your-endpoint-name \
  --region us-west-2 \
  --query 'EndpointStatus' \
  --output text
```

Expected output: `InService`

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

View endpoint logs:

```bash
# List log streams
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/Endpoints/your-endpoint-name \
  --region us-west-2

# View logs
aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name \
  --follow \
  --region us-west-2
```

### Set Up Alarms

Create CloudWatch alarms for monitoring:

```bash
# Alarm for high error rate
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

## Cleanup

### Delete Endpoint

When you're done, delete the endpoint to stop incurring charges:

**Option A: Using AWS Console**

1. Go to SageMaker → Endpoints
2. Select your endpoint
3. Click "Delete"
4. Confirm deletion

**Option B: Using AWS CLI**

```bash
# Delete endpoint
aws sagemaker delete-endpoint \
  --endpoint-name "llama3-lmi-agent" \
  --region us-west-2

# Delete endpoint configuration
aws sagemaker delete-endpoint-config \
  --endpoint-config-name your-endpoint-config-name \
  --region us-west-2

# Delete model
aws sagemaker delete-model \
  --model-name your-model-name \
  --region us-west-2
```

### Verify Deletion

```bash
aws sagemaker describe-endpoint \
  --endpoint-name your-endpoint-name \
  --region us-west-2
```

Expected: `ResourceNotFound` error

## Troubleshooting

### Issue: Deployment Fails with ResourceLimitExceeded

**Error Message**:
```
ResourceLimitExceeded: The account-level service limit 'ml.g5.2xlarge for endpoint usage' is 0 Instances
```

**Solution**:
1. Request quota increase (see Prerequisites section)
2. Or try a different instance type
3. Or deploy in a different region

### Issue: Endpoint Stuck in Creating Status

**Symptoms**:
- Endpoint status remains "Creating" for > 15 minutes
- No error messages in logs

**Solutions**:
1. Check CloudWatch logs for errors:
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name --follow
   ```

2. Verify the model ID is correct and accessible

3. Check if the container image is available in your region

4. Increase timeout and retry:
   ```python
   wait_for_endpoint(endpoint_name, timeout=3600)  # 1 hour
   ```

### Issue: Validation Fails

**Error Message**:
```
Validation failed: Endpoint invocation failed: ModelError
```

**Solutions**:
1. Check endpoint logs for model loading errors
2. Verify instance type has enough memory for the model
3. Check if model requires authentication (HuggingFace token)
4. Try with a smaller model first to isolate the issue

### Issue: IAM Permission Denied

**Error Message**:
```
AccessDeniedException: User: arn:aws:iam::123456789:user/myuser is not authorized to perform: sagemaker:CreateEndpoint
```

**Solutions**:
1. Verify IAM role has required permissions
2. Check trust relationship allows SageMaker service
3. Ensure you're using the correct role ARN
4. Wait a few minutes for IAM changes to propagate

### Issue: High Latency

**Symptoms**:
- Response times > 5 seconds
- ModelLatency metric is high

**Solutions**:
1. Use a larger instance type with more GPUs
2. Reduce `max_new_tokens` to generate shorter responses
3. Enable auto-scaling for load distribution
4. Consider using multiple endpoints with load balancing

### Issue: Out of Memory Errors

**Error Message**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use a larger instance type (more GPU memory)
2. Reduce `MAX_INPUT_LENGTH` in container config
3. Reduce `max_new_tokens` in inference parameters
4. Use a smaller model variant

## Cost Estimation

### Hourly Costs

| Instance Type | Cost/Hour (us-west-2) |
|---------------|----------------------|
| ml.g5.2xlarge | $1.52 |
| ml.g5.4xlarge | $2.03 |
| ml.g5.12xlarge | $7.09 |

### Monthly Cost Examples

**Development (8 hours/day, 20 days/month)**:
- Instance: ml.g5.2xlarge
- Hours: 160/month
- Cost: ~$243/month

**Production (24/7, single instance)**:
- Instance: ml.g5.2xlarge
- Hours: 720/month
- Cost: ~$1,094/month

**Production (24/7, auto-scaling 1-3 instances, avg 1.5)**:
- Instance: ml.g5.2xlarge
- Hours: 1,080/month
- Cost: ~$1,642/month

### Cost Optimization Tips

1. **Delete endpoints when not in use** (development)
2. **Use auto-scaling** to match demand
3. **Set up CloudWatch alarms** for unexpected usage
4. **Use smaller instance types** for low-traffic scenarios
5. **Batch requests** when possible to maximize throughput

## Next Steps

After successful deployment:

1. **Configure your application**: Set `SAGEMAKER_ENDPOINT_NAME` environment variable
2. **Test with your tools**: Run the financial agent with SageMaker endpoint
3. **Monitor performance**: Set up CloudWatch dashboards and alarms
4. **Optimize costs**: Configure auto-scaling based on usage patterns
5. **Review security**: Ensure IAM policies follow least privilege principle

---

## Llama 70B Deployment

The Llama 3.1 70B Instruct model is a production-ready alternative to the 8B model, offering significantly improved tool calling reliability, a 128K token context window, and better reasoning for complex multi-tool queries. The 70B model requires a multi-GPU instance with tensor parallelism to fit its ~140GB FP16 weights in GPU memory.

The existing agent code works with the 70B endpoint without modification — switch between 8B and 70B by changing only the `SAGEMAKER_ENDPOINT_NAME` environment variable.

### 70B Interactive Instance Selector

The deployment script includes an interactive instance selector that displays all available profiles with their full specs. Run the script and choose from the numbered menu:

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

### 70B Instance Type Comparison

| Instance Type | GPUs | Total VRAM | TP Degree | Est. Cost/hr | Deploy Time (S3) | Notes |
|---------------|------|------------|-----------|--------------|------------------|-------|
| `ml.g5.48xlarge` | 8x A10G | 192 GB | 8 | ~$20.36 | ~10-15 min | **Default** — FP16, no quantization needed |
| `ml.p4d.24xlarge` | 8x A100 | 320 GB | 8 | ~$25.25 | ~15-25 min | Higher throughput, lower latency, 32K context |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | 4 | ~$7.09 | ~10-15 min | Requires AWQ/GPTQ 4-bit quantization |
| `ml.p4d.24xlarge:max` | 8x A100 | 320 GB | 8 | ~$25.25 | ~15-25 min | Max context (65K), batch 16, GPU mem 0.95 |

**Choosing an instance type:**
- **g5.48xlarge (default)**: Best balance of cost and performance for FP16 inference. 192GB total VRAM across 8 GPUs, but each A10G only has 24GB — the 70B model shard uses ~16.5GB per GPU, leaving ~5.5GB per GPU for KV cache. Context window is limited to 16K tokens by default.
- **p4d.24xlarge**: Use when you need higher throughput or lower per-token latency. 320GB VRAM allows larger batch sizes and 32K context. Only ~$5/hr more than g5.48xlarge.
- **g5.12xlarge**: Budget option requiring 4-bit quantization (AWQ or GPTQ). 96GB VRAM fits the quantized model (~35GB) with ample KV cache room. The deployment script auto-selects the quantized model ID for this profile.
- **p4d.24xlarge:max**: Same hardware as p4d.24xlarge but tuned for maximum context (65K tokens) and throughput (batch 16, GPU mem 0.95). The 70B FP16 model uses ~140GB, leaving ~180GB for KV cache across 8 A100s.

### 70B Quick Start

**Default deployment (interactive instance selector):**

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Deploy 70B model — the interactive selector will prompt for instance type
python deployment/deploy_llama3_70b.py
```

**Faster deployment with S3 pre-downloaded weights:**

```bash
# Step 1: Pre-download model to S3 (one-time, ~140GB)
# The S3 prefix is auto-derived from the model ID:
#   meta-llama/Meta-Llama-3.1-70B-Instruct → Meta-Llama-3.1-70B-Instruct/
python deployment/download_model_to_s3.py

# Step 2: Deploy from S3 (significantly faster)
python deployment/deploy_llama3_70b.py \
  --model-s3-uri s3://my-bucket/Meta-Llama-3.1-70B-Instruct/
```

**Quantized model deployment (g5.12xlarge — budget option):**

```bash
# Step 1: Download the AWQ-INT4 quantized model to S3
# Auto-derives a SEPARATE S3 prefix: Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
python deployment/download_model_to_s3.py \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

# Step 2: Deploy on g5.12xlarge with the quantized weights
python deployment/deploy_llama3_70b.py \
  --instance-type ml.g5.12xlarge \
  --model-s3-uri s3://my-bucket/Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
```

> **⚠️ S3 Prefix Collision Warning**: The download script auto-derives the S3 prefix from the model ID, so FP16 and quantized models get separate prefixes. If you previously used `--s3-prefix llama-70b-instruct/` for both FP16 and quantized models, they would overwrite each other. The default behavior avoids this by using the model name as the prefix (e.g., `Meta-Llama-3.1-70B-Instruct/` vs `Meta-Llama-3.1-70B-Instruct-AWQ-INT4/`).

> **⚠️ FP16 + g5.12xlarge Mismatch**: If you select the `ml.g5.12xlarge` profile and provide `--model-s3-uri` pointing to FP16 weights (~140GB), the deployment will fail because FP16 weights don't fit in 96GB VRAM. The script warns about this and prompts for confirmation. Either use the quantized model or pick a larger instance.

The deployment script will:
1. Show the interactive instance selector (or use `--instance-type` if provided)
2. Apply the selected profile (TP degree, context length, batch size, dtype, model ID)
3. Verify your instance quota
4. Display estimated hourly cost and prompt for confirmation
5. Create or use existing SageMaker execution role
6. Create SageMaker model with LMI container and selected configuration
7. Create endpoint configuration and deploy
8. Monitor deployment progress (~10-15 minutes for 70B)
9. Display endpoint name and cleanup instructions

### 70B Configuration

The 70B deployment uses these LMI container environment variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_MODEL_ID` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Model to load (or S3 URI when using pre-download) |
| `OPTION_ROLLING_BATCH` | `vllm` | vLLM backend for PagedAttention-based memory management |
| `OPTION_ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable automatic tool calling for agent workflows |
| `OPTION_TOOL_CALL_PARSER` | `llama3_json` | Llama 3 JSON parser for structured tool call extraction |
| `OPTION_MAX_ROLLING_BATCH_SIZE` | `4` | Max concurrent requests (lower than 8B's 32 due to memory) |
| `OPTION_MAX_MODEL_LEN` | `16384` | Max sequence length — 16K tokens (fits A10G KV cache at GPU_MEMORY_UTILIZATION=0.95; use p4d.24xlarge for 32K+) |
| `OPTION_DTYPE` | `fp16` | Full FP16 precision for best quality |
| `OPTION_GPU_MEMORY_UTILIZATION` | `0.95` | Allocate 95% of GPU VRAM to vLLM (default 0.90), maximizes KV cache room |
| `TENSOR_PARALLEL_DEGREE` | `8` | Distribute model across all 8 GPUs |

**Additional tunable parameters (not set by default):**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPTION_ENABLE_CHUNKED_PREFILL` | not set | Set to `"true"` to overlap prefill and decode phases for better throughput |
| `OPTION_SPECULATIVE_DECODING` | not set | Enable speculative decoding with a draft model (experimental) |

### S3 Model Pre-Download

Pre-downloading the 70B model weights to S3 avoids repeated ~140GB downloads from HuggingFace during development and testing, and significantly speeds up SageMaker endpoint creation.

The download script auto-derives the S3 prefix from the model ID, so different model variants (FP16 vs quantized) are stored at separate S3 paths and don't collide.

> **⚠️ Disk Space Requirement**: The script downloads the model to a local temp directory before uploading to S3. You need at least **~150GB of free disk space** for FP16 or **~40GB** for quantized models. The temp files are automatically cleaned up after the upload completes. Check available space with `df -h .` before running.

**Workflow:**

```bash
# FP16 model (default) — stored at s3://bucket/Meta-Llama-3.1-70B-Instruct/
python deployment/download_model_to_s3.py

# Quantized AWQ-INT4 model — stored at s3://bucket/Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
python deployment/download_model_to_s3.py \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

# Specify a custom bucket (prefix still auto-derived):
python deployment/download_model_to_s3.py --s3-bucket my-model-bucket

# Override the S3 prefix manually (not recommended):
python deployment/download_model_to_s3.py --s3-prefix my-custom-prefix/
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

**IAM Permissions for S3 Bucket (your AWS user/role running the download script):**

The user or role executing `download_model_to_s3.py` needs write access to the target S3 bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-model-bucket",
        "arn:aws:s3:::my-model-bucket/llama-70b-instruct/*"
      ]
    }
  ]
}
```

**IAM Permissions for SageMaker Execution Role (to read model from S3 at deploy time):**

The SageMaker execution role needs read access to the S3 model artifacts:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::my-model-bucket/llama-70b-instruct/*"
    }
  ]
}
```

### 70B Cost Estimates

**Hourly Costs (us-west-2 on-demand pricing):**

| Instance Type | Cost/Hour |
|---------------|-----------|
| `ml.g5.48xlarge` | ~$20.36 |
| `ml.p4d.24xlarge` | ~$25.25 |
| `ml.g5.12xlarge` | ~$7.09 |

**Monthly Cost Examples (ml.g5.48xlarge):**

| Usage Pattern | Hours/Month | Est. Monthly Cost |
|---------------|-------------|-------------------|
| Development (8 hrs/day, 20 days) | 160 | ~$3,258 |
| Production (24/7) | 720 | ~$14,659 |

> **⚠️ Cost Warning**: The 70B deployment on `ml.g5.48xlarge` costs ~$20.36/hour — roughly **13x more** than the 8B deployment on `ml.g5.2xlarge` (~$1.52/hour). Always delete the endpoint when not in use during development.

**Cleanup commands:**

```bash
aws sagemaker delete-endpoint --endpoint-name llama3-70b-lmi-agent --region us-west-2
```

Or use the project cleanup script:

```bash
python cleanup_project.py
```

### 70B Troubleshooting

#### Quota Errors

**Error:**
```
ResourceLimitExceeded: The account-level service limit 'ml.g5.48xlarge for endpoint usage' is 0 Instances
```

**Solution:**
1. Go to **Service Quotas → Amazon SageMaker** in the AWS Console
2. Search for `ml.g5.48xlarge for endpoint usage`
3. Request a quota increase to at least 1 instance
4. Quota increases for GPU instances may take 1-3 business days to approve

```bash
# Check your current quota for g5.48xlarge
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --region us-west-2 \
  --query "Quotas[?contains(QuotaName, 'g5.48xlarge') && contains(QuotaName, 'endpoint')].{Name:QuotaName,Value:Value,Code:QuotaCode}" \
  --output table
```

#### Memory Requirements

The 70B model in FP16 requires approximately:
- **~140 GB** for model weights
- **~20-50 GB** additional for KV cache (depends on `OPTION_MAX_MODEL_LEN` and batch size)
- **Total: ~160-192 GB** GPU VRAM

This is why `ml.g5.48xlarge` (192 GB across 8x A10G GPUs) is the minimum recommended instance for FP16 inference. If you see memory-related failures, do not attempt to use a smaller instance without quantization.

#### Instance Capacity Errors

**Error:**
```
Unable to provision requested ML compute capacity due to InsufficientInstanceCapacity error.
Please retry using a different ML instance type or after some time.
```

This means AWS did not have the requested GPU instance available in your region at that moment. This is especially common with `ml.p4d.24xlarge` (A100 GPUs) which are in high demand and limited supply. G5 instances (A10G) are generally more available.

**What to do:**
1. Clean up the failed endpoint (the deploy script will offer to do this automatically on the next run):
   ```bash
   aws sagemaker delete-endpoint --endpoint-name llama3-70b-lmi-agent --region us-west-2
   ```
2. Retry later — capacity fluctuates, especially during off-peak hours (late evening / early morning Pacific time)
3. Fall back to a different instance type:
   - `ml.g5.48xlarge` — more abundant, 16K context, FP16
   - `ml.g5.12xlarge` — budget option with AWQ quantization, 8K context

> **Note**: This is an AWS infrastructure limitation, not a problem with your configuration. GPU instances are a shared resource and availability varies by region and time of day.

#### Deployment Time

The 70B endpoint takes **10-15 minutes** to become `InService` on G5 instances, compared to ~5-10 minutes for the 8B model. On `ml.p4d.24xlarge`, expect **15-25 minutes** even when loading from S3.

**Why p4d.24xlarge takes longer than G5 instances:**

| Factor | Details |
|--------|---------|
| Instance provisioning | A100 GPU instances are less common in the fleet than A10G instances, so AWS takes longer to allocate one |
| GPU initialization | 8x A100s have more complex memory hierarchies (HBM2e) and NVLink interconnects that require additional setup |
| Model sharding | vLLM must shard ~140GB across 8 GPUs, configure NCCL communication rings, and pre-allocate KV cache — larger configs (`:max` profile with 65K context, batch 16) increase this overhead |
| Health check warmup | The first inference pass on A100s with large context/batch configs takes longer to complete |

The S3 model loading itself is fast (~2-3 minutes). The extra time is all hardware provisioning and initialization.

Using `--model-s3-uri` with pre-downloaded weights still helps significantly — it eliminates the ~140GB HuggingFace download, which can add 10+ minutes on top of the provisioning time.

If the endpoint stays in `Creating` status for more than 30 minutes, check CloudWatch logs:

```bash
aws logs tail /aws/sagemaker/Endpoints/llama3-70b-lmi-agent --follow --region us-west-2
```

#### OOM (Out of Memory) / KV Cache Errors

**Error:**
```
RuntimeError: CUDA out of memory
```
or
```
ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache
```

**Root cause:** On `ml.g5.48xlarge`, each A10G GPU has 24GB VRAM. The 70B FP16 model shard uses ~16.5GB per GPU, leaving only ~5.5GB for KV cache. A `max_model_len` of 32768 requires far more KV cache than available.

**Solutions:**
1. The default `OPTION_MAX_MODEL_LEN` is set to `16384` to balance context capacity and memory
2. If you increased it beyond `16384` and hit OOM, reduce it back to `16384` or `8192`
3. Reduce `OPTION_MAX_ROLLING_BATCH_SIZE` (e.g., from `4` to `2`)
4. `OPTION_GPU_MEMORY_UTILIZATION` is set to `0.95` by default — do not lower it on g5.48xlarge
5. Upgrade to `ml.p4d.24xlarge` (320 GB VRAM) for longer context windows (32K+)
6. Use a quantized model variant on `ml.g5.12xlarge` with `OPTION_DTYPE="auto"`

### Switching Between 8B and 70B

The agent code uses the `SAGEMAKER_ENDPOINT_NAME` environment variable to determine which endpoint to call. No code changes are needed to switch between models:

```bash
# Use the 8B model (default, lower cost)
export SAGEMAKER_ENDPOINT_NAME="llama3-lmi-agent"

# Use the 70B model (better quality, higher cost)
export SAGEMAKER_ENDPOINT_NAME="llama3-70b-lmi-agent"

# Then run the agent as usual
python fin-agent-sagemaker-v2.py
```

Both endpoints use the same LMI container with vLLM backend and the same OpenAI-compatible chat completion API, so the content handler and agent wrapper work identically with either model.

## Additional Resources

- [AWS SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [AWS LMI Container Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Service Quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
