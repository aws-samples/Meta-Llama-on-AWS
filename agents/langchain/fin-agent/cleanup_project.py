#!/usr/bin/env python3
"""
Cleanup script for SageMaker environment resources.

Detects and offers to delete known SageMaker endpoints, endpoint
configurations, and models created by the fin-agent deployment scripts.
No local files are modified or removed.
"""

import boto3
from botocore.exceptions import ClientError

# Known SageMaker endpoint names to check during cleanup
KNOWN_ENDPOINT_NAMES = [
    "llama3-lmi-agent",
    "llama3-70b-lmi-agent",
]

def cleanup_sagemaker_endpoints():
    """Detect and offer to delete known SageMaker endpoints (8B and 70B)."""
    sagemaker_client = boto3.client("sagemaker", region_name="us-west-2")

    found_any = False
    for endpoint_name in KNOWN_ENDPOINT_NAMES:
        try:
            resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                # Endpoint does not exist
                print(f"  ℹ️  Endpoint '{endpoint_name}' not found — skipping.")
                continue
            raise

        found_any = True
        status = resp.get("EndpointStatus", "Unknown")

        # Retrieve instance type and model name from the endpoint config
        instance_type = "Unknown"
        model_name = ""
        config_name = resp.get("EndpointConfigName", "")
        if config_name:
            try:
                cfg = sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=config_name
                )
                variants = cfg.get("ProductionVariants", [])
                if variants:
                    instance_type = variants[0].get("InstanceType", "Unknown")
                    model_name = variants[0].get("ModelName", "")
            except ClientError:
                pass

        print(f"\n  🔍 Found endpoint: {endpoint_name}")
        print(f"     Status: {status}")
        print(f"     Instance type: {instance_type}")

        confirm = input(f"\n  Delete endpoint '{endpoint_name}' and its resources? (yes/no): ").strip().lower()
        if confirm != "yes":
            print(f"  ⏭️  Skipping '{endpoint_name}'.")
            continue

        # Delete endpoint
        try:
            print(f"  Deleting endpoint '{endpoint_name}'...")
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"  ✓ Endpoint deleted.")
        except ClientError as e:
            print(f"  ✗ Error deleting endpoint: {e}")

        # Delete endpoint configuration
        if config_name:
            try:
                print(f"  Deleting endpoint config '{config_name}'...")
                sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"  ✓ Endpoint config deleted.")
            except ClientError as e:
                print(f"  ✗ Error deleting endpoint config: {e}")

        # Delete model
        if model_name:
            try:
                print(f"  Deleting model '{model_name}'...")
                sagemaker_client.delete_model(ModelName=model_name)
                print(f"  ✓ Model deleted.")
            except ClientError as e:
                print(f"  ⚠️  Could not delete model: {e}")

    if not found_any:
        print("  ✅ No active SageMaker endpoints found.")


def main():
    """Clean up SageMaker environment resources (endpoints, configs, models)."""

    print("=" * 80)
    print("SAGEMAKER ENDPOINT CLEANUP")
    print("=" * 80)
    check_endpoints = (
        input("\nCheck for SageMaker endpoints to clean up? (yes/no): ")
        .strip()
        .lower()
    )
    if check_endpoints == "yes":
        cleanup_sagemaker_endpoints()
    else:
        print("  Skipping SageMaker endpoint check.\n")

    print("\n✅ Environment cleanup complete.")

if __name__ == "__main__":
    main()
