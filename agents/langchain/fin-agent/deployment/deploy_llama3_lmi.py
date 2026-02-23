#!/usr/bin/env python3
"""
Deploy Llama 3 8B model to SageMaker using LMI (Large Model Inference) container.

This script deploys Meta Llama 3.1 8B Instruct using AWS's LMI container with vLLM backend,
which is specifically designed for agent workflows with native tool calling support.

Key Features:
- Native tool calling support via vLLM
- Configurable tool call parser (llama3_json)
- Multi-turn conversation support
- Better performance than TGI for agent workflows

Usage:
    python deploy_llama3_lmi.py

Requirements:
    - AWS credentials configured
    - Sufficient quota for ml.g5.2xlarge instances
    - SageMaker execution role with necessary permissions
    - Hugging Face token for model access
"""

import boto3
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory or parent directories
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

# Configuration
AWS_REGION = "us-west-2"
ENDPOINT_NAME = "llama3-lmi-agent"

# LMI Container Image (latest version with vLLM support)
# This is the official AWS Deep Learning Container for LMI
LMI_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126"

# Hugging Face model ID - Using Llama 3.1 for better tool calling
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Instance configuration
# ml.g5.2xlarge: 1x A10G GPU (24GB), perfect for 8B models
INSTANCE_TYPE = "ml.g5.2xlarge"
INSTANCE_COUNT = 1


def get_hf_token():
    """Get HuggingFace token from environment, only when needed."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        raise ValueError(
            "HuggingFace token not found! Please set one of these environment variables:\n"
            "  export HF_TOKEN='your_token_here'\n"
            "  export HUGGING_FACE_HUB_TOKEN='your_token_here'\n"
            "\nOr add to .env file:\n"
            "  HF_TOKEN=your_token_here\n"
            "\nGet your token from: https://huggingface.co/settings/tokens"
        )
    
    return hf_token


def get_model_env():
    """Build LMI environment variables with HF token."""
    return {
        # Model Configuration
        "HF_MODEL_ID": HF_MODEL_ID,
        "HF_TOKEN": get_hf_token(),
        
        # Tool Calling Configuration (CRITICAL for agent workflows)
        # Note: Parallel tool calling is configured at the REQUEST level via the
        # parallel_tool_calls parameter in the payload, not at the server level.
        # The content handler sets this parameter when tools are present.
        "OPTION_ROLLING_BATCH": "vllm",  # Use vLLM backend for tool calling
        "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",  # Enable automatic tool calling
        "OPTION_TOOL_CALL_PARSER": "llama3_json",  # Use Llama 3 JSON parser
        
        # Performance Configuration
        "OPTION_MAX_ROLLING_BATCH_SIZE": "32",  # Max concurrent requests
        "OPTION_MAX_MODEL_LEN": "8192",  # Max sequence length
        "OPTION_DTYPE": "fp16",  # Use FP16 for faster inference
        
        # GPU Configuration
        "TENSOR_PARALLEL_DEGREE": "1",  # Single GPU
    }


def print_separator(title="", char="="):
    """Print a visual separator."""
    if title:
        print(f"\n{char * 80}")
        print(f"  {title}")
        print(f"{char * 80}\n")
    else:
        print(f"\n{char * 80}\n")


def get_or_create_sagemaker_role():
    """Get or create SageMaker execution role."""
    from botocore.exceptions import ClientError
    
    iam_client = boto3.client('iam', region_name=AWS_REGION)
    
    print("🔍 Looking for SageMaker execution role...")
    
    common_role_names = [
        "SageMakerExecutionRole",
        "AmazonSageMaker-ExecutionRole",
        "sagemaker-execution-role"
    ]
    
    for role_name in common_role_names:
        try:
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"✅ Found existing role: {role_arn}")
            return role_arn
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                continue
            raise
    
    print("📝 Creating new SageMaker execution role...")
    
    role_name = "SageMakerExecutionRole"
    
    trust_policy = {
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
    
    try:
        create_role_response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker execution role for LMI deployment"
        )
        
        role_arn = create_role_response['Role']['Arn']
        
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        
        print(f"✅ Created new role: {role_arn}")
        print("⏳ Waiting 10 seconds for role to propagate...")
        time.sleep(10)
        
        return role_arn
        
    except Exception as e:
        print(f"❌ Error creating role: {e}")
        print("\nPlease create a SageMaker execution role manually:")
        print("1. Go to IAM Console → Roles → Create role")
        print("2. Select 'SageMaker' as the service")
        print("3. Attach 'AmazonSageMakerFullAccess' policy")
        print("4. Name it 'SageMakerExecutionRole'")
        raise


def check_endpoint_exists(sagemaker_client, endpoint_name):
    """Check if endpoint already exists."""
    from botocore.exceptions import ClientError
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"⚠️  Endpoint '{endpoint_name}' already exists with status: {status}")
        
        if status == 'InService':
            print("✅ Endpoint is already deployed and ready to use!")
            return True
        elif status in ['Creating', 'Updating']:
            print("⏳ Endpoint is being created/updated. Waiting for completion...")
            return 'wait'
        else:
            print(f"❌ Endpoint is in {status} state. Please delete it first.")
            return False
    except ClientError as e:
        if 'Could not find endpoint' in str(e):
            return None
        raise


def deploy_model():
    """Deploy Llama 3.1 8B model using LMI container."""
    
    print_separator("Llama 3.1 8B Model Deployment (LMI with vLLM)", "=")
    print(f"Endpoint Name: {ENDPOINT_NAME}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print(f"Model: {HF_MODEL_ID}")
    print(f"Container: LMI (Large Model Inference) with vLLM backend")
    print(f"Tool Calling: Enabled with llama3_json parser")
    print(f"Region: {AWS_REGION}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("", "=")
    
    print("\n📋 LMI Configuration:")
    print(f"  - Backend: vLLM (optimized for agent workflows)")
    print(f"  - Tool Call Parser: llama3_json")
    print(f"  - Auto Tool Choice: Enabled")
    print(f"  - Max Batch Size: 32")
    print(f"  - Max Sequence Length: 8192")
    print(f"  - Data Type: FP16")
    
    # Initialize clients
    sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Check if endpoint exists
    endpoint_status = check_endpoint_exists(sagemaker_client, ENDPOINT_NAME)
    
    if endpoint_status is True:
        print("\n✅ Deployment complete! Endpoint is ready to use.")
        print(f"\nSet environment variable:")
        print(f'export SAGEMAKER_ENDPOINT_NAME="{ENDPOINT_NAME}"')
        return ENDPOINT_NAME
    elif endpoint_status == 'wait':
        return wait_for_endpoint(sagemaker_client, ENDPOINT_NAME)
    elif endpoint_status is False:
        print("\n❌ Please delete the existing endpoint first:")
        print(f'aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {AWS_REGION}')
        return None
    
    # Get or create execution role
    try:
        execution_role = get_or_create_sagemaker_role()
    except Exception as e:
        print(f"\n❌ Error with execution role: {e}")
        return None
    
    print_separator("Step 1: Creating Model", "-")
    
    model_name = f"{ENDPOINT_NAME}-model-{int(time.time())}"
    
    try:
        print(f"Creating model: {model_name}")
        print(f"Using LMI container: {LMI_IMAGE}")
        print(f"Model ID: {HF_MODEL_ID}")
        print(f"Tool calling: Enabled with vLLM backend")
        
        create_model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': LMI_IMAGE,
                'Environment': get_model_env(),  # Get env with token at runtime
                'Mode': 'SingleModel'
            },
            ExecutionRoleArn=execution_role,
        )
        
        print(f"✅ Model created: {model_name}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if you have access to the LMI container image")
        print("2. Verify your execution role has necessary permissions")
        print("3. Check if the Hugging Face token is valid")
        print("4. Ensure the model ID is correct")
        return None
    
    print_separator("Step 2: Creating Endpoint Configuration", "-")
    
    endpoint_config_name = f"{ENDPOINT_NAME}-config-{int(time.time())}"
    
    try:
        print(f"Creating endpoint configuration: {endpoint_config_name}")
        print(f"Instance type: {INSTANCE_TYPE}")
        print(f"Instance count: {INSTANCE_COUNT}")
        
        create_endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': INSTANCE_COUNT,
                    'InstanceType': INSTANCE_TYPE,
                    'InitialVariantWeight': 1.0,
                }
            ],
        )
        
        print(f"✅ Endpoint configuration created: {endpoint_config_name}")
        
    except Exception as e:
        print(f"❌ Error creating endpoint configuration: {e}")
        print("\nTroubleshooting:")
        print("1. Check if you have quota for the instance type")
        print("2. Try a different instance type (e.g., ml.g5.xlarge)")
        print("3. Request quota increase in AWS Service Quotas")
        return None
    
    print_separator("Step 3: Creating Endpoint", "-")
    
    try:
        print(f"Creating endpoint: {ENDPOINT_NAME}")
        print("⏳ This will take 5-10 minutes...")
        print("   The LMI container needs to:")
        print("   - Download the model from Hugging Face")
        print("   - Initialize vLLM backend")
        print("   - Configure tool calling support")
        print("   - Load model into GPU memory")
        
        create_endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )
        
        print(f"✅ Endpoint creation initiated")
        
    except Exception as e:
        print(f"❌ Error creating endpoint: {e}")
        return None
    
    return wait_for_endpoint(sagemaker_client, ENDPOINT_NAME)


def wait_for_endpoint(sagemaker_client, endpoint_name):
    """Wait for endpoint to be in service."""
    
    print_separator("Waiting for Endpoint Deployment", "-")
    print("⏳ Endpoint is being deployed. This typically takes 5-10 minutes.")
    print("   You can monitor progress in the AWS Console:")
    print(f"   SageMaker → Endpoints → {endpoint_name}")
    print_separator("", "-")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Status: {status}")
                last_status = status
            
            if status == 'InService':
                elapsed = int(time.time() - start_time)
                print_separator("", "=")
                print(f"✅ Endpoint deployed successfully in {elapsed} seconds!")
                print(f"\nEndpoint Name: {endpoint_name}")
                print(f"Status: {status}")
                print(f"Container: LMI with vLLM backend")
                print(f"Tool Calling: Enabled (llama3_json parser)")
                print_separator("", "=")
                
                print("\n📋 Next Steps:")
                print(f'1. Set environment variable:')
                print(f'   export SAGEMAKER_ENDPOINT_NAME="{endpoint_name}"')
                print(f'\n2. Test tool calling:')
                print(f'   python test_lmi_tool_calling.py')
                print(f'\n3. Run multi-tool calling test:')
                print(f'   python test_multiple_tool_calls.py')
                print(f'\n4. Compare with TGI results!')
                
                print("\n💡 Key Differences from TGI:")
                print("  - Native tool calling support via vLLM")
                print("  - Configurable tool call parser (llama3_json)")
                print("  - Designed for multi-turn agent workflows")
                print("  - Better performance with PagedAttention")
                print("  - Can return multiple tool calls per response")
                
                return endpoint_name
            
            elif status == 'Failed':
                print(f"\n❌ Endpoint deployment failed!")
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                return None
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Deployment interrupted by user")
            print("   The endpoint will continue deploying in the background.")
            print(f"   Check status: aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
            return None
        except Exception as e:
            print(f"\n❌ Error checking endpoint status: {e}")
            return None


def main():
    """Main entry point."""
    
    print("\n" + "=" * 80)
    print("  Llama 3.1 8B Model Deployment (LMI with vLLM)")
    print("=" * 80)
    print("\nThis script deploys Llama 3.1 8B using AWS's LMI container with vLLM backend.")
    print("LMI is specifically designed for agent workflows with native tool calling support.")
    print("\n🎯 Key Features:")
    print("  - Native tool calling via vLLM backend")
    print("  - Configurable tool call parser (llama3_json)")
    print("  - Multi-turn conversation support")
    print("  - Better performance than TGI for agents")
    print("  - Can return multiple tool calls per response")
    print("\n💰 COST INFO:")
    print(f"   Instance Type: {INSTANCE_TYPE}")
    print(f"   Estimated Cost: ~$1.52/hour")
    print(f"   Same cost as TGI but better for agents!")
    print("=" * 80)
    
    response = input("\nProceed with deployment? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Deployment cancelled")
        return 1
    
    endpoint_name = deploy_model()
    
    if endpoint_name:
        print("\n✅ Deployment successful!")
        return 0
    else:
        print("\n❌ Deployment failed")
        return 1


if __name__ == "__main__":
    exit(main())
