#!/usr/bin/env python3
"""
Deploy Llama 3.1 70B Instruct model to SageMaker using LMI (Large Model Inference) container.

This script deploys Meta Llama 3.1 70B Instruct (or Llama 3.3 70B Instruct) using AWS's
LMI container with vLLM backend. The 70B model provides significantly improved tool calling
reliability, a 128K token context window, and better reasoning compared to the 8B variant.

The 70B model requires a multi-GPU instance with tensor parallelism. The default configuration
uses ml.g5.48xlarge (8x A10G GPUs, 192GB total VRAM) which can host the full FP16 model
(~140GB weights) with room for KV cache.

Key Features:
    - Native tool calling support via vLLM with llama3_json parser
    - 8-way tensor parallelism across all GPUs
    - 32K token context window (configurable up to 128K)
    - PagedAttention-based memory management for efficient batching
    - Compatible with existing agent code — switch via SAGEMAKER_ENDPOINT_NAME

Instance Type Comparison:
    ┌───────────────────┬──────────┬────────────┬───────────┬──────────────┬──────────────────────────────────┐
    │ Instance Type      │ GPUs     │ Total VRAM │ TP Degree │ Est. Cost/hr │ Notes                            │
    ├───────────────────┼──────────┼────────────┼───────────┼──────────────┼──────────────────────────────────┤
    │ ml.g5.48xlarge     │ 8x A10G  │ 192 GB     │ 8         │ ~$20.36      │ Default, FP16, no quantization   │
    │ ml.p4d.24xlarge    │ 8x A100  │ 320 GB     │ 8         │ ~$25.25      │ Higher throughput, lower latency  │
    │ ml.g5.12xlarge     │ 4x A10G  │ 96 GB      │ 4         │ ~$7.09       │ Requires AWQ/GPTQ 4-bit quant    │
    └───────────────────┴──────────┴────────────┴───────────┴──────────────┴──────────────────────────────────┘

Tunable Inference Parameters:
    TENSOR_PARALLEL_DEGREE (default: "8")
        Number of GPUs to distribute the model across. Must match the GPU count of the
        selected instance type. Use "8" for g5.48xlarge/p4d.24xlarge, "4" for g5.12xlarge.
        Range: 1–8 (must evenly divide model layers).

    OPTION_MAX_MODEL_LEN (default: "8192")
        Maximum sequence length (input + output tokens). The 70B model natively supports
        up to 128K tokens, but higher values require more GPU memory for KV cache.
        On g5.48xlarge (A10G GPUs), the 70B FP16 model leaves limited VRAM for KV cache,
        so 8192 is the safe default. Use ml.p4d.24xlarge for longer context windows.
        Range: 4096–131072. Recommended: 8192 for g5.48xlarge, 32768+ for p4d.24xlarge.

    OPTION_MAX_ROLLING_BATCH_SIZE (default: "4")
        Maximum number of concurrent requests processed via continuous batching. Lower
        than the 8B default (32) due to the 70B model's larger memory footprint per request.
        Range: 1–16. Recommended: 2–4 for g5.48xlarge, 4–8 for p4d.24xlarge.

    OPTION_DTYPE (default: "fp16")
        Model weight precision. FP16 provides full quality; use "auto" to let vLLM choose
        based on model config. For quantized models (AWQ/GPTQ), set to "auto".
        Options: "fp16", "bf16", "auto".

    OPTION_ROLLING_BATCH (default: "vllm")
        Inference backend. vLLM provides PagedAttention for efficient memory management
        and native tool calling support. Do not change unless testing alternatives.
        Options: "vllm".

    OPTION_ENABLE_AUTO_TOOL_CHOICE (default: "true")
        Enables the model to autonomously decide when to call tools based on the
        conversation context. Required for agent workflows.
        Options: "true", "false".

    OPTION_TOOL_CALL_PARSER (default: "llama3_json")
        Parser for extracting structured tool calls from model output. Must match the
        model family. Use "llama3_json" for Llama 3.x models.
        Options: "llama3_json", "hermes", "mistral".

    OPTION_ENABLE_CHUNKED_PREFILL (default: not set)
        When set to "true", enables chunked prefill to overlap prefill and decode phases,
        improving throughput for long-context requests. Experimental in some LMI versions.
        Options: "true", "false".

    OPTION_SPECULATIVE_DECODING (default: not set)
        When configured, enables speculative decoding with a smaller draft model to
        accelerate generation. Requires additional configuration for the draft model.
        Experimental; consult LMI documentation for setup.

Usage:
    # Default: download from HuggingFace at deploy time
    python deployment/deploy_llama3_70b.py

    # With pre-downloaded S3 weights (faster deployment)
    python deployment/deploy_llama3_70b.py --model-s3-uri s3://my-bucket/llama-70b/

Requirements:
    - AWS credentials configured
    - Sufficient quota for ml.g5.48xlarge instances (or selected instance type)
    - SageMaker execution role with necessary permissions
    - HuggingFace token for model access (not needed when using --model-s3-uri)

Cost Warning:
    The ml.g5.48xlarge instance costs ~$20.36/hour, significantly more than the 8B
    deployment on ml.g5.2xlarge (~$1.52/hour). Remember to delete the endpoint when
    not in use to avoid unnecessary charges.
"""

import argparse
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

# =============================================================================
# 70B Model Configuration
# =============================================================================

# AWS Region
AWS_REGION = "us-west-2"

# Endpoint name — distinct from 8B to allow coexistence
ENDPOINT_NAME = "llama3-70b-lmi-agent"

# LMI Container Image (same version as 8B deployment)
# Official AWS Deep Learning Container for Large Model Inference
LMI_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126"

# HuggingFace model ID — Llama 3.1 70B Instruct
# Can also use "meta-llama/Llama-3.3-70B-Instruct" for the 3.3 variant
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Instance configuration
# ml.g5.48xlarge: 8x A10G GPUs (192GB total VRAM)
# Required for 70B FP16 (~140GB model weights + KV cache overhead)
INSTANCE_TYPE = "ml.g5.48xlarge"
INSTANCE_COUNT = 1

# =============================================================================
# LMI / vLLM Inference Parameters (70B-optimized)
# =============================================================================

# Tensor parallelism — distribute model across all 8 GPUs
TENSOR_PARALLEL_DEGREE = "8"

# Context window — 16K tokens for g5.48xlarge (A10G GPUs)
# The 70B FP16 model uses ~16.5GB per GPU shard out of 24GB, leaving ~5.5GB
# for KV cache per GPU. 16384 tokens uses ~2x the KV cache of 8192 and fits
# within the available headroom at GPU_MEMORY_UTILIZATION=0.95.
# 32768 causes OOM on A10G; use ml.p4d.24xlarge (A100) for 32K+ context.
OPTION_MAX_MODEL_LEN = "16384"

# Concurrent request batch size — lower than 8B (32) due to larger memory footprint
OPTION_MAX_ROLLING_BATCH_SIZE = "4"

# Model precision — FP16 for full quality on g5.48xlarge
OPTION_DTYPE = "fp16"

# GPU memory utilization — raise from default 0.90 to 0.95 to maximize KV cache room
# On g5.48xlarge each A10G has 24GB; the 70B model shard uses ~16.5GB per GPU,
# so every extra percent of VRAM allocated to vLLM helps KV cache capacity.
OPTION_GPU_MEMORY_UTILIZATION = "0.95"

# =============================================================================
# Instance Type Comparison (for reference)
# =============================================================================
#
# | Instance Type    | GPUs     | Total VRAM | TP Degree | Est. Cost/hr | Notes                          |
# |------------------|----------|------------|-----------|--------------|--------------------------------|
# | ml.g5.48xlarge   | 8x A10G  | 192 GB     | 8         | ~$20.36      | Default, FP16, no quant needed |
# | ml.p4d.24xlarge  | 8x A100  | 320 GB     | 8         | ~$25.25      | Higher throughput, lower latency|
# | ml.g5.12xlarge   | 4x A10G  | 96 GB      | 4         | ~$7.09       | Requires AWQ/GPTQ 4-bit quant  |
#
# Choosing an instance type:
#   - g5.48xlarge (default): Best balance of cost and performance for FP16 inference.
#     192GB VRAM fits the ~140GB model with ~52GB remaining for KV cache.
#   - p4d.24xlarge: Use when you need maximum throughput or lower per-token latency.
#     320GB VRAM allows larger batch sizes and longer context windows.
#   - g5.12xlarge: Budget option requiring 4-bit quantization (AWQ or GPTQ).
#     96GB VRAM fits quantized model (~35GB) with ample KV cache room.
#     Change HF_MODEL_ID to a quantized variant and set OPTION_DTYPE="auto".
#
# =============================================================================

# Estimated hourly costs by instance type (us-west-2 on-demand pricing)
INSTANCE_COST_PER_HOUR = {
    "ml.g5.48xlarge": 20.36,
    "ml.p4d.24xlarge": 25.25,
    "ml.g5.12xlarge": 7.09,
}

# =============================================================================
# Instance Profiles — each profile contains all tunable params for that instance
# =============================================================================
INSTANCE_PROFILES = {
    "ml.g5.48xlarge": {
        "gpus": "8x A10G",
        "vram": "192 GB",
        "tp_degree": "8",
        "max_model_len": "16384",
        "max_rolling_batch_size": "4",
        "dtype": "fp16",
        "gpu_memory_utilization": "0.95",
        "cost_per_hour": 20.36,
        "notes": "Default. FP16, no quantization needed.",
        "model_id": None,  # None = use default HF_MODEL_ID
    },
    "ml.p4d.24xlarge": {
        "gpus": "8x A100",
        "vram": "320 GB",
        "tp_degree": "8",
        "max_model_len": "32768",
        "max_rolling_batch_size": "8",
        "dtype": "fp16",
        "gpu_memory_utilization": "0.90",
        "cost_per_hour": 25.25,
        "notes": "Higher throughput, lower latency. Supports 32K+ context.",
        "model_id": None,
    },
    "ml.g5.12xlarge": {
        "gpus": "4x A10G",
        "vram": "96 GB",
        "tp_degree": "4",
        "max_model_len": "8192",
        "max_rolling_batch_size": "4",
        "dtype": "auto",
        "gpu_memory_utilization": "0.90",
        "cost_per_hour": 7.09,
        "notes": "Budget. Requires AWQ/GPTQ 4-bit quantized model.",
        "model_id": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    },
    "ml.p4d.24xlarge:max": {
        "gpus": "8x A100",
        "vram": "320 GB",
        "tp_degree": "8",
        "max_model_len": "65536",
        "max_rolling_batch_size": "16",
        "dtype": "fp16",
        "gpu_memory_utilization": "0.95",
        "cost_per_hour": 25.25,
        "notes": "Max context. 70B FP16 uses ~140GB, leaves ~180GB for KV cache.",
        "model_id": None,
        "instance_type": "ml.p4d.24xlarge",  # actual AWS instance type
    },
}


def select_instance_type():
    """Interactive terminal selector for instance type.

    Displays a numbered menu of available instance profiles and lets the user
    pick one. Returns the selected profile key or None if cancelled.
    """
    profiles = list(INSTANCE_PROFILES.items())

    print("\n" + "=" * 72)
    print("  SELECT INSTANCE TYPE")
    print("=" * 72)
    print()

    for i, (instance_type, profile) in enumerate(profiles, 1):
        default_marker = " ← current default" if instance_type == INSTANCE_TYPE else ""
        print(f"  [{i}] {instance_type}{default_marker}")
        print(f"      GPUs: {profile['gpus']}  |  VRAM: {profile['vram']}  |  TP: {profile['tp_degree']}-way")
        print(f"      Context: {profile['max_model_len']} tokens  |  Batch: {profile['max_rolling_batch_size']}")
        print(f"      Dtype: {profile['dtype']}  |  GPU Mem: {profile['gpu_memory_utilization']}")
        print(f"      Cost: ~${profile['cost_per_hour']:.2f}/hr")
        print(f"      {profile['notes']}")
        if profile.get("model_id"):
            print(f"      Model: {profile['model_id']}")
        print()

    print(f"  [0] Cancel")
    print()

    while True:
        try:
            choice = input(f"  Select instance [1-{len(profiles)}] (0 to cancel): ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                selected_type, selected_profile = profiles[idx]
                print(f"\n  ✅ Selected: {selected_type}")
                return selected_type
            else:
                print(f"  ⚠️  Enter a number between 0 and {len(profiles)}")
        except (ValueError, EOFError):
            print(f"  ⚠️  Enter a number between 0 and {len(profiles)}")


def apply_instance_profile(instance_type):
    """Apply the selected instance profile to global configuration variables.

    Updates all globals that vary by instance type so deploy_model() uses
    the correct parameters.

    Args:
        instance_type: Key from INSTANCE_PROFILES.
    """
    global INSTANCE_TYPE, TENSOR_PARALLEL_DEGREE, OPTION_MAX_MODEL_LEN
    global OPTION_MAX_ROLLING_BATCH_SIZE, OPTION_DTYPE, OPTION_GPU_MEMORY_UTILIZATION
    global HF_MODEL_ID

    profile = INSTANCE_PROFILES[instance_type]
    # Some profiles use a different key than the actual AWS instance type
    # (e.g. "ml.p4d.24xlarge:max" maps to "ml.p4d.24xlarge")
    INSTANCE_TYPE = profile.get("instance_type", instance_type)
    TENSOR_PARALLEL_DEGREE = profile["tp_degree"]
    OPTION_MAX_MODEL_LEN = profile["max_model_len"]
    OPTION_MAX_ROLLING_BATCH_SIZE = profile["max_rolling_batch_size"]
    OPTION_DTYPE = profile["dtype"]
    OPTION_GPU_MEMORY_UTILIZATION = profile["gpu_memory_utilization"]
    if profile.get("model_id"):
        HF_MODEL_ID = profile["model_id"]

    print(f"\n  📋 Applied profile: {instance_type}")
    print(f"     AWS Instance: {INSTANCE_TYPE}")
    print(f"     TP Degree: {TENSOR_PARALLEL_DEGREE}")
    print(f"     Max Context: {OPTION_MAX_MODEL_LEN} tokens")
    print(f"     Batch Size: {OPTION_MAX_ROLLING_BATCH_SIZE}")
    print(f"     Dtype: {OPTION_DTYPE}")
    print(f"     GPU Memory: {OPTION_GPU_MEMORY_UTILIZATION}")
    print(f"     Model: {HF_MODEL_ID}")


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



def get_model_env(model_s3_uri=None):
    """Build LMI environment variables for the 70B model.

    When model_s3_uri is provided, the model is loaded from S3 instead of
    HuggingFace Hub, and HF_TOKEN is omitted.

    Args:
        model_s3_uri: Optional S3 URI of pre-downloaded model weights
                      (e.g., "s3://my-bucket/llama-70b/").

    Returns:
        dict: Environment variables for the LMI container.

    Raises:
        ValueError: If model_s3_uri is provided but has an invalid format.
    """
    env = {
        "OPTION_ROLLING_BATCH": "vllm",
        "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",
        "OPTION_TOOL_CALL_PARSER": "llama3_json",
        "OPTION_MAX_ROLLING_BATCH_SIZE": OPTION_MAX_ROLLING_BATCH_SIZE,
        "OPTION_MAX_MODEL_LEN": OPTION_MAX_MODEL_LEN,
        "OPTION_DTYPE": OPTION_DTYPE,
        "OPTION_GPU_MEMORY_UTILIZATION": OPTION_GPU_MEMORY_UTILIZATION,
        "TENSOR_PARALLEL_DEGREE": TENSOR_PARALLEL_DEGREE,
    }

    if model_s3_uri is not None:
        # Validate S3 URI format
        if not model_s3_uri.startswith("s3://"):
            raise ValueError(
                f"Invalid S3 URI: '{model_s3_uri}'. "
                "S3 URI must start with 's3://' (e.g., 's3://my-bucket/llama-70b/')."
            )
        # Ensure there's at least a bucket name after the prefix
        path_after_prefix = model_s3_uri[len("s3://"):]
        if not path_after_prefix or path_after_prefix.startswith("/"):
            raise ValueError(
                f"Invalid S3 URI: '{model_s3_uri}'. "
                "S3 URI must include a bucket name after 's3://' "
                "(e.g., 's3://my-bucket/llama-70b/')."
            )
        env["HF_MODEL_ID"] = model_s3_uri
    else:
        env["HF_MODEL_ID"] = HF_MODEL_ID
        env["HF_TOKEN"] = get_hf_token()

    return env



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
    """Check if endpoint already exists.

    Returns:
        True if endpoint is InService, 'wait' if Creating/Updating,
        'cleaned' if a failed endpoint was deleted and we can proceed,
        False if in a bad state and user declined cleanup, None if not found.
    """
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
        elif status == 'Failed':
            print(f"\n   The previous deployment failed.")
            try:
                cleanup = input("   Delete the failed endpoint and retry? (yes/no): ").strip().lower()
            except EOFError:
                cleanup = "no"
            if cleanup in ['yes', 'y']:
                print(f"   🗑️  Deleting failed endpoint '{endpoint_name}'...")
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                # Also try to clean up the old endpoint config and model
                try:
                    config_name = response.get('EndpointConfigName')
                    if config_name:
                        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                except Exception:
                    pass  # Best-effort cleanup
                print("   ⏳ Waiting for endpoint deletion...")
                _wait_for_endpoint_deletion(sagemaker_client, endpoint_name)
                print("   ✅ Failed endpoint cleaned up. Proceeding with fresh deployment.")
                return 'cleaned'
            else:
                print(f"❌ Endpoint is in {status} state. Please delete it first.")
                return False
        else:
            print(f"❌ Endpoint is in {status} state. Please delete it first.")
            return False
    except ClientError as e:
        if 'Could not find endpoint' in str(e):
            return None
        raise


def _wait_for_endpoint_deletion(sagemaker_client, endpoint_name, timeout=300):
    """Wait for an endpoint to be fully deleted.

    Args:
        sagemaker_client: boto3 SageMaker client.
        endpoint_name: Name of the endpoint being deleted.
        timeout: Max seconds to wait (default 5 minutes).
    """
    from botocore.exceptions import ClientError

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = resp['EndpointStatus']
            if status == 'Deleting':
                time.sleep(10)
                continue
            # If it's in some other state, something went wrong
            break
        except ClientError as e:
            if 'Could not find endpoint' in str(e):
                return  # Successfully deleted
            raise
    # Timeout — proceed anyway, SageMaker may still be cleaning up



def wait_for_endpoint(sagemaker_client, endpoint_name):
    """Wait for endpoint to be in service, streaming CloudWatch logs for progress."""

    print_separator("Waiting for Endpoint Deployment", "-")
    print("⏳ Endpoint is being deployed. This typically takes 10-15 minutes for 70B.")
    print("   The LMI container needs to:")
    print("   - Download the 70B model (~140GB) from HuggingFace (or load from S3)")
    print(f"   - Initialize vLLM backend with {TENSOR_PARALLEL_DEGREE}-way tensor parallelism")
    print("   - Configure tool calling support")
    print(f"   - Load model shards into GPU memory across {TENSOR_PARALLEL_DEGREE} GPUs")
    print(f"   Monitor in AWS Console: SageMaker → Endpoints → {endpoint_name}")
    print_separator("", "-")

    logs_client = boto3.client("logs", region_name=AWS_REGION)
    log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    last_log_token = None
    log_streams_found = False

    start_time = time.time()
    last_status = None

    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            elapsed = int(time.time() - start_time)

            if status != last_status:
                print(f"\n[{elapsed}s] Status: {status}")
                last_status = status

            # Stream CloudWatch logs for real-time progress
            try:
                if not log_streams_found:
                    streams_resp = logs_client.describe_log_streams(
                        logGroupName=log_group,
                        orderBy="LastEventTime",
                        descending=True,
                        limit=3,
                    )
                    if streams_resp.get("logStreams"):
                        log_streams_found = True

                if log_streams_found:
                    get_kwargs = {
                        "logGroupName": log_group,
                        "startFromHead": False,
                        "limit": 20,
                    }
                    if last_log_token:
                        get_kwargs["nextToken"] = last_log_token

                    events_resp = logs_client.filter_log_events(**get_kwargs)
                    for event in events_resp.get("events", []):
                        msg = event.get("message", "").strip()
                        if msg:
                            # Show key progress lines, skip noisy ones
                            msg_lower = msg.lower()
                            if any(kw in msg_lower for kw in [
                                "loading", "download", "model", "vllm",
                                "tensor", "gpu", "memory", "error", "exception",
                                "warning", "ready", "serving", "health",
                                "worker", "engine", "shard", "complete",
                                "starting", "initialized", "torch",
                            ]):
                                print(f"  📋 {msg[:200]}")
                    if events_resp.get("nextToken"):
                        last_log_token = events_resp["nextToken"]

            except logs_client.exceptions.ResourceNotFoundException:
                if elapsed > 120:
                    print(f"  ⏳ [{elapsed}s] Waiting for container logs...")
            except Exception:
                pass  # Don't let log errors break the wait loop

            if status == 'InService':
                print_separator("", "=")
                print(f"✅ Endpoint deployed successfully in {elapsed} seconds!")
                print(f"\nEndpoint Name: {endpoint_name}")
                print(f"Status: {status}")
                print(f"Instance Type: {INSTANCE_TYPE}")
                print(f"Container: LMI with vLLM backend")
                print(f"Tool Calling: Enabled (llama3_json parser)")
                print(f"Tensor Parallelism: {TENSOR_PARALLEL_DEGREE}-way")
                print(f"Max Context Length: {OPTION_MAX_MODEL_LEN} tokens")
                print_separator("", "=")

                print("\n📋 Next Steps:")
                print(f'1. Set environment variable:')
                print(f'   export SAGEMAKER_ENDPOINT_NAME="{endpoint_name}"')
                print(f'\n2. Run the finance agent:')
                print(f'   python fin-agent-sagemaker-v2.py')
                print(f'\n3. Test tool calling:')
                print(f'   python test_lmi_tool_calling.py')

                cost = INSTANCE_COST_PER_HOUR.get(INSTANCE_TYPE, 0)
                print(f"\n⚠️  COST REMINDER: {INSTANCE_TYPE} costs ~${cost:.2f}/hour.")
                print("   Delete the endpoint when not in use:")
                print(f"   aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {AWS_REGION}")
                print(f"   aws sagemaker delete-endpoint-config --endpoint-config-name <config-name> --region {AWS_REGION}")
                print(f"   aws sagemaker delete-model --model-name <model-name> --region {AWS_REGION}")

                return endpoint_name

            elif status == 'Failed':
                print(f"\n❌ Endpoint deployment failed!")
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                print("\n📋 Recent container logs:")
                try:
                    events_resp = logs_client.filter_log_events(
                        logGroupName=log_group,
                        startFromHead=False,
                        limit=30,
                    )
                    for event in events_resp.get("events", [])[-15:]:
                        msg = event.get("message", "").strip()
                        if msg:
                            print(f"  {msg[:250]}")
                except Exception:
                    print(f"  Check logs: aws logs tail {log_group} --region {AWS_REGION}")
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




def check_instance_quota(instance_type):
    """Check if the AWS account has sufficient quota for the given instance type.

    Calls the Service Quotas API to verify the user has quota for the selected
    SageMaker instance type. Prints a clear error with instructions to request
    a quota increase if insufficient.

    Uses ``list_service_quotas`` and searches for the instance type string in
    the quota name (pattern: ``<instance_type> for endpoint usage``). This is
    more robust than maintaining a static mapping of quota codes.

    Args:
        instance_type: SageMaker instance type (e.g., "ml.g5.48xlarge").

    Returns:
        bool: True if quota is sufficient (>= 1), False otherwise.
    """
    try:
        sq_client = boto3.client("service-quotas", region_name=AWS_REGION)

        # Search through applied (account-level) quotas for SageMaker.
        # Quota names follow the pattern "<instance_type> for endpoint usage".
        search_string = f"{instance_type} for endpoint usage"
        quota_value = None

        paginator = sq_client.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode="sagemaker"):
            for quota in page.get("Quotas", []):
                if search_string.lower() in quota.get("QuotaName", "").lower():
                    quota_value = quota.get("Value", 0)
                    break
            if quota_value is not None:
                break

        # If we didn't find an applied quota, check the AWS default quotas.
        if quota_value is None:
            default_paginator = sq_client.get_paginator(
                "list_aws_default_service_quotas"
            )
            for page in default_paginator.paginate(ServiceCode="sagemaker"):
                for quota in page.get("Quotas", []):
                    if search_string.lower() in quota.get("QuotaName", "").lower():
                        quota_value = quota.get("Value", 0)
                        break
                if quota_value is not None:
                    break

        if quota_value is None:
            print(
                f"\n⚠️  Could not find quota information for {instance_type}. "
                "Proceeding with deployment — SageMaker will reject the request "
                "if quota is insufficient."
            )
            return True

        if quota_value < 1:
            print(f"\n❌ Insufficient quota for {instance_type}.")
            print(f"   Current quota value: {int(quota_value)}")
            print("\n   To request a quota increase:")
            print("   1. Open the AWS Console → Service Quotas → Amazon SageMaker")
            print(f'   2. Search for "{instance_type} for endpoint usage"')
            print("   3. Click 'Request quota increase' and request at least 1")
            print(
                f"   Direct link: https://{AWS_REGION}.console.aws.amazon.com/"
                "servicequotas/home/services/sagemaker/quotas"
            )
            return False

        print(f"✅ Quota check passed: {instance_type} quota = {int(quota_value)}")
        return True

    except Exception as e:
        # Don't block deployment if the service-quotas API is unavailable
        # (e.g., region not supported, permissions issue).
        print(
            f"\n⚠️  Could not verify quota for {instance_type}: {e}\n"
            "   Proceeding with deployment — SageMaker will reject the request "
            "if quota is insufficient."
        )
        return True




def deploy_model(model_s3_uri=None):
    """Deploy Llama 3.1 70B model using LMI container.

    Creates a SageMaker model, endpoint configuration, and endpoint using the
    70B-specific configuration. Supports loading model weights from either
    HuggingFace Hub or a pre-downloaded S3 location.

    Args:
        model_s3_uri: Optional S3 URI of pre-downloaded model weights.

    Returns:
        str or None: Endpoint name if successful, None on failure.
    """

    print_separator("Llama 3.1 70B Model Deployment (LMI with vLLM)", "=")
    print(f"Endpoint Name: {ENDPOINT_NAME}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print(f"Model: {HF_MODEL_ID}")
    print(f"Container: LMI (Large Model Inference) with vLLM backend")
    print(f"Tool Calling: Enabled with llama3_json parser")
    print(f"Region: {AWS_REGION}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if model_s3_uri:
        print(f"Model Source: S3 ({model_s3_uri})")
    else:
        print(f"Model Source: HuggingFace Hub ({HF_MODEL_ID})")
    print_separator("", "=")

    print("\n📋 LMI Configuration:")
    print(f"  - Backend: vLLM (optimized for agent workflows)")
    print(f"  - Tool Call Parser: llama3_json")
    print(f"  - Auto Tool Choice: Enabled")
    print(f"  - Max Batch Size: {OPTION_MAX_ROLLING_BATCH_SIZE}")
    print(f"  - Max Context Length: {OPTION_MAX_MODEL_LEN} tokens")
    print(f"  - Data Type: {OPTION_DTYPE}")
    print(f"  - Tensor Parallelism: {TENSOR_PARALLEL_DEGREE}-way")

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
    # 'cleaned' means a failed endpoint was deleted — fall through to create a new one

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
        if model_s3_uri:
            print(f"Model source: S3 ({model_s3_uri})")
        else:
            print(f"Model ID: {HF_MODEL_ID}")
        print(f"Tool calling: Enabled with vLLM backend")

        create_model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': LMI_IMAGE,
                'Environment': get_model_env(model_s3_uri),
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
        if model_s3_uri:
            print("3. Verify the S3 URI is correct and the execution role has s3:GetObject access")
            print("4. Ensure the S3 bucket is in the same region as SageMaker")
        else:
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
                    'ModelDataDownloadTimeoutInSeconds': 1800,
                    'ContainerStartupHealthCheckTimeoutInSeconds': 1800,
                }
            ],
        )

        print(f"✅ Endpoint configuration created: {endpoint_config_name}")

    except Exception as e:
        print(f"❌ Error creating endpoint configuration: {e}")
        print("\nTroubleshooting:")
        print(f"1. Check if you have quota for {INSTANCE_TYPE}")
        print("2. Request quota increase in AWS Service Quotas")
        print(f"   Search for '{INSTANCE_TYPE} for endpoint usage'")
        return None

    print_separator("Step 3: Creating Endpoint", "-")

    try:
        print(f"Creating endpoint: {ENDPOINT_NAME}")
        print("⏳ This will take 10-15 minutes for the 70B model...")
        print("   The LMI container needs to:")
        if model_s3_uri:
            print("   - Load the 70B model (~140GB) from S3")
        else:
            print("   - Download the 70B model (~140GB) from HuggingFace")
        print("   - Initialize vLLM backend with tensor parallelism")
        print("   - Configure tool calling support")
        print(f"   - Load model shards into GPU memory across {TENSOR_PARALLEL_DEGREE} GPUs")

        create_endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )

        print(f"✅ Endpoint creation initiated")

    except Exception as e:
        print(f"❌ Error creating endpoint: {e}")
        return None

    return wait_for_endpoint(sagemaker_client, ENDPOINT_NAME)




def main():
    """Main entry point with argparse for --model-s3-uri."""

    parser = argparse.ArgumentParser(
        description="Deploy Llama 3.1 70B Instruct to SageMaker using LMI container with vLLM backend."
    )
    parser.add_argument(
        "--model-s3-uri",
        type=str,
        default=None,
        help="S3 URI of pre-downloaded model weights (e.g., s3://my-bucket/llama-70b/). "
             "If not provided, the model is downloaded from HuggingFace Hub at deploy time.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=None,
        choices=list(INSTANCE_PROFILES.keys()),
        help="Instance type to deploy on. If not provided, an interactive selector is shown.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  Llama 3.1 70B Model Deployment (LMI with vLLM)")
    print("=" * 80)
    print("\nThis script deploys Llama 3.1 70B Instruct using AWS's LMI container with vLLM backend.")
    print("The 70B model provides significantly improved tool calling reliability, a 128K token")
    print("context window, and better reasoning compared to the 8B variant.")

    # --- Instance type selection ---
    if args.instance_type:
        # CLI flag provided — skip interactive selector
        selected = args.instance_type
    else:
        selected = select_instance_type()
        if selected is None:
            print("\n❌ Deployment cancelled")
            return 1

    apply_instance_profile(selected)

    print(f"\n🎯 Model: {HF_MODEL_ID}")
    if args.model_s3_uri:
        print(f"📦 Model Source: S3 ({args.model_s3_uri})")
    else:
        print(f"📦 Model Source: HuggingFace Hub")
    print(f"🔧 Instance Type: {INSTANCE_TYPE}")
    print(f"🧠 Tensor Parallelism: {TENSOR_PARALLEL_DEGREE}-way")
    print(f"📏 Max Context Length: {OPTION_MAX_MODEL_LEN} tokens")

    # Validate: warn if --model-s3-uri is used with a profile that requires a quantized model
    # but the S3 URI doesn't look like it contains the expected quantized weights
    profile = INSTANCE_PROFILES[selected]
    if args.model_s3_uri and profile.get("model_id"):
        expected_model_name = profile["model_id"].split("/")[-1]  # e.g. "Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        if expected_model_name.lower() not in args.model_s3_uri.lower():
            print(f"\n⚠️  WARNING: You selected a profile that requires a quantized model:")
            print(f"   Expected model: {profile['model_id']}")
            print(f"   But --model-s3-uri doesn't appear to contain '{expected_model_name}':")
            print(f"   {args.model_s3_uri}")
            print(f"   If your S3 bucket contains FP16 weights (~140GB), they will NOT fit")
            print(f"   on {INSTANCE_TYPE} ({profile['vram']} VRAM).")
            print()
            override = input("   Continue anyway? (yes/no): ").strip().lower()
            if override not in ['yes', 'y']:
                print("\n   Tip: Run without --model-s3-uri to download the quantized model from HuggingFace.")
                print("   Or upload the quantized model to S3 first:")
                print(f"   python deployment/download_model_to_s3.py --model-id {profile['model_id']}")
                print("\n❌ Deployment cancelled")
                return 1
            # User chose to continue — they may have quantized weights in S3

    # Display estimated hourly cost
    cost = INSTANCE_COST_PER_HOUR.get(INSTANCE_TYPE, 0)
    print(f"\n💰 ESTIMATED COST:")
    print(f"   Instance Type: {INSTANCE_TYPE}")
    print(f"   Estimated Cost: ~${cost:.2f}/hour")

    # Display cost warning comparing 70B vs 8B
    print(f"\n⚠️  COST WARNING:")
    print(f"   The 70B model on {INSTANCE_TYPE} costs ~${cost:.2f}/hour,")
    print(f"   which is ~{cost / 1.52:.1f}x more than the 8B model on ml.g5.2xlarge (~$1.52/hour).")
    print(f"   Remember to delete the endpoint when not in use!")
    print("=" * 80)

    # Check instance quota
    if not check_instance_quota(INSTANCE_TYPE):
        return 1

    response = input("\nProceed with deployment? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("\n❌ Deployment cancelled")
        return 1

    endpoint_name = deploy_model(args.model_s3_uri)

    if endpoint_name:
        print("\n✅ Deployment successful!")
        print(f"\n🧹 Cleanup Instructions (run when done):")
        print(f"   aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {AWS_REGION}")
        print(f"   aws sagemaker delete-endpoint-config --endpoint-config-name $(aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME} --region {AWS_REGION} --query 'EndpointConfigName' --output text) --region {AWS_REGION}")
        print(f"   aws sagemaker delete-model --model-name $(aws sagemaker list-models --region {AWS_REGION} --name-contains {ENDPOINT_NAME} --query 'Models[0].ModelName' --output text) --region {AWS_REGION}")
        return 0
    else:
        print("\n❌ Deployment failed")
        return 1



if __name__ == "__main__":
    exit(main())
