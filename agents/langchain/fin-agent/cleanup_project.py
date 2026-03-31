#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files and keep only essential ones.

ESSENTIAL FILES TO KEEP:
- Main agents:
  * fin-agent-sagemaker-v2.py - Uses SageMaker endpoint + src/ directory
  * fin-agent-llama-api.py - Uses Llama API directly (no SageMaker/src needed)
- Deployment: deploy_llama3_lmi.py (only for SageMaker agent)
- Tests mentioned in README: test_multiple_parallel_tools.py, test_multi_step_detailed.py
- Source code: src/ directory (only used by fin-agent-sagemaker-v2.py)
- Configuration: pyproject.toml, uv.lock, .gitignore, .python-version
- Documentation: README.md, README_SAGEMAKER.md
"""

import os
import shutil
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Files to KEEP
KEEP_FILES = {
    # Main scripts
    "fin-agent-sagemaker-v2.py",
    "fin-agent-llama-api.py",  # Alternative agent using Llama API
    "deploy_llama3_lmi.py",
    
    # Tests mentioned in README_SAGEMAKER.md
    "test_multiple_parallel_tools.py",
    "test_multi_step_detailed.py",
    
    # Configuration files
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".python-version",
    
    # Documentation
    "README.md",
    "README_SAGEMAKER.md",
    "ESSENTIAL_FILES.md",
    "AGENT_COMPARISON.md",
    "SECURITY_IMPROVEMENTS.md",
    "PROJECT_STATUS.md",
    
    # Cleanup script itself
    "cleanup_project.py",
}

# Directories to KEEP
KEEP_DIRS = {
    "src",           # Source code
    "deployment",    # Deployment scripts
    ".venv",         # Virtual environment
    ".kiro",         # Kiro configuration
    ".git",          # Git repository
    "__pycache__",   # Python cache (will be regenerated)
    ".pytest_cache", # Pytest cache (will be regenerated)
    ".hypothesis",   # Hypothesis cache (will be regenerated)
    "fin_agent.egg-info",  # Package info (will be regenerated)
}

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
    """Remove unnecessary files while keeping essential ones."""
    
    current_dir = Path(".")

    # --- SageMaker endpoint cleanup ---
    print("=" * 80)
    print("SAGEMAKER ENDPOINT CLEANUP")
    print("=" * 80)
    check_endpoints = input("\nCheck for SageMaker endpoints to clean up? (yes/no): ").strip().lower()
    if check_endpoints == "yes":
        cleanup_sagemaker_endpoints()
    else:
        print("  Skipping SageMaker endpoint check.\n")
    
    # Lists to track what we're doing
    files_to_remove = []
    files_to_keep = []
    
    # Scan all files in current directory
    for item in current_dir.iterdir():
        if item.is_file():
            if item.name in KEEP_FILES:
                files_to_keep.append(item.name)
            else:
                files_to_remove.append(item.name)
        elif item.is_dir():
            if item.name not in KEEP_DIRS:
                # Check if it's a directory we should remove
                if item.name not in {".git", ".venv"}:  # Never remove these
                    print(f"⚠️  Directory to review: {item.name}")
    
    # Show summary
    print("=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"\n✅ Files to KEEP ({len(files_to_keep)}):")
    for f in sorted(files_to_keep):
        print(f"   - {f}")
    
    print(f"\n🗑️  Files to REMOVE ({len(files_to_remove)}):")
    for f in sorted(files_to_remove):
        print(f"   - {f}")
    
    print("\n" + "=" * 80)
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response == "yes":
        removed_count = 0
        for filename in files_to_remove:
            try:
                os.remove(filename)
                removed_count += 1
                print(f"✓ Removed: {filename}")
            except Exception as e:
                print(f"✗ Error removing {filename}: {e}")
        
        print(f"\n✅ Cleanup complete! Removed {removed_count} files.")
        print("\nEssential files remaining:")
        print("  - fin-agent-sagemaker-v2.py (main agent - SageMaker)")
        print("  - fin-agent-llama-api.py (alternative agent - Llama API)")
        print("  - deploy_llama3_lmi.py (deployment)")
        print("  - test_multiple_parallel_tools.py (parallel tool test)")
        print("  - test_multi_step_detailed.py (multi-step test)")
        print("  - src/ (source code)")
        print("  - README.md, README_SAGEMAKER.md (documentation)")
    else:
        print("\n❌ Cleanup cancelled.")

if __name__ == "__main__":
    main()
