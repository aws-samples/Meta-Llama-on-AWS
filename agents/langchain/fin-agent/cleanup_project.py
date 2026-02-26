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
    ".venv",         # Virtual environment
    ".kiro",         # Kiro configuration
    ".git",          # Git repository
    "__pycache__",   # Python cache (will be regenerated)
    ".pytest_cache", # Pytest cache (will be regenerated)
    ".hypothesis",   # Hypothesis cache (will be regenerated)
    "fin_agent.egg-info",  # Package info (will be regenerated)
}

def main():
    """Remove unnecessary files while keeping essential ones."""
    
    current_dir = Path(".")
    
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
