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

```bash
# 1. Set HuggingFace token
export HF_TOKEN="your_huggingface_token"

# 2. Run deployment script
python deploy_llama3_lmi.py
```

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

### 3. Service Quotas

Check your SageMaker service quotas for GPU instances:

```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-E1EAAA6F \
  --region us-west-2
```

If you need more quota, request an increase through the AWS Console:
1. Go to Service Quotas → AWS Services → Amazon SageMaker
2. Find "ml.g5.2xlarge for endpoint usage"
3. Request quota increase if needed

## Deployment Script

The `deploy_llama3_lmi.py` script automates the entire deployment process:

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
from deployment.deploy_endpoint import validate_endpoint

result = validate_endpoint(
    endpoint_name="your-endpoint-name",
    region="us-west-2"
)

if result['valid']:
    print("✓ Endpoint is functional")
    print(f"Response: {result['response']}")
else:
    print("✗ Validation failed")
    print(f"Error: {result['message']}")
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

**Option A: Using Python**

```python
from deployment.deploy_endpoint import delete_endpoint

result = delete_endpoint(
    endpoint_name="your-endpoint-name",
    region="us-west-2",
    delete_model=True,
    delete_endpoint_config=True
)

print(result['message'])
```

**Option B: Using AWS CLI**

```bash
# Delete endpoint
aws sagemaker delete-endpoint \
  --endpoint-name your-endpoint-name \
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

## Additional Resources

- [AWS SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [AWS LMI Container Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Service Quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
