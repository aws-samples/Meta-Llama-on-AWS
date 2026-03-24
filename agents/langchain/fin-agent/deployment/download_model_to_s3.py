#!/usr/bin/env python3
"""
Download a HuggingFace model and upload it to S3 for faster SageMaker deployment.

This utility downloads model weights from HuggingFace Hub to a local temp directory,
then uploads all artifacts to a specified S3 bucket with multipart upload support.
Pre-staging model weights in S3 avoids repeated ~140GB downloads from HuggingFace
during development and testing, and significantly speeds up SageMaker endpoint creation.

Usage:
    python deployment/download_model_to_s3.py \\
        --s3-bucket my-model-bucket \\
        --model-id meta-llama/Meta-Llama-3.1-70B-Instruct \\
        --hf-token hf_xxxxx

    # Quantized model (auto-derives separate S3 prefix):
    python deployment/download_model_to_s3.py \\
        --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

    # Minimal with explicit bucket:
    python deployment/download_model_to_s3.py --s3-bucket my-model-bucket

    # Use SageMaker default bucket (no --s3-bucket needed):
    python deployment/download_model_to_s3.py

    # Override S3 prefix (not recommended — prefer auto-derived):
    python deployment/download_model_to_s3.py --s3-prefix my-custom-prefix/

Requirements:
    - AWS credentials configured with S3 write access
    - HuggingFace token with access to the gated model (via --hf-token or $HF_TOKEN)
    - Sufficient local disk space for the model download (~140GB for 70B FP16)
    - huggingface_hub package installed
    - sagemaker package installed (only needed when using default bucket)
"""

import argparse
import os
import sys
import time
import tempfile

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


# Multipart upload config: 100MB chunks, 10 concurrent threads
S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    multipart_chunksize=100 * 1024 * 1024,
    max_concurrency=10,
)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
DEFAULT_S3_PREFIX = None  # Auto-derived from model ID when not specified


def _format_size(size_bytes):
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _derive_s3_prefix(model_id):
    """Derive a unique S3 prefix from a HuggingFace model ID.

    Extracts the model name from the HuggingFace ID (org/model-name) and
    converts it to a clean S3 prefix. This ensures different model variants
    (e.g., FP16 vs AWQ-INT4) get separate S3 prefixes and don't collide.

    Examples:
        "meta-llama/Meta-Llama-3.1-70B-Instruct"
            → "Meta-Llama-3.1-70B-Instruct/"
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
            → "Meta-Llama-3.1-70B-Instruct-AWQ-INT4/"

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct").

    Returns:
        S3 prefix string ending with '/'.
    """
    # Use the model name part (after the org/), or the whole string if no slash
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    # Ensure trailing slash
    return f"{name}/"


def _format_time(seconds):
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace with model_id, s3_bucket, s3_prefix, and hf_token.
    """
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model and upload it to S3.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Target S3 bucket name. If omitted, uses the SageMaker session default bucket.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help="S3 key prefix. If omitted, auto-derived from --model-id "
             "(e.g., 'Meta-Llama-3.1-70B-Instruct/' or 'Meta-Llama-3.1-70B-Instruct-AWQ-INT4/').",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token (falls back to $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download and re-upload even if model already exists in S3",
    )

    args = parser.parse_args(argv)

    # Resolve HF token: CLI arg > HF_TOKEN env > HUGGING_FACE_HUB_TOKEN env > cached token file
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
    if args.hf_token is None:
        cached_token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(cached_token_path):
            with open(cached_token_path) as f:
                token = f.read().strip()
            if token:
                args.hf_token = token
                print(f"ℹ️  Using cached HuggingFace token from {cached_token_path}")

    # Resolve S3 bucket: CLI arg > SageMaker default bucket
    if args.s3_bucket is None:
        args.s3_bucket = _resolve_sagemaker_default_bucket()

    # Resolve S3 prefix: CLI arg > auto-derive from model ID
    if args.s3_prefix is None:
        args.s3_prefix = _derive_s3_prefix(args.model_id)
        print(f"ℹ️  Auto-derived S3 prefix from model ID: {args.s3_prefix}")

    return args


def _resolve_sagemaker_default_bucket():
    """Resolve the SageMaker session default bucket.

    Returns:
        The default bucket name (e.g. 'sagemaker-us-west-2-123456789012').

    Raises:
        SystemExit: If the sagemaker SDK is not installed or bucket resolution fails.
    """
    try:
        import sagemaker
    except ImportError:
        print(
            "❌ The 'sagemaker' package is required to resolve the default bucket.\n"
            "   Install it with: pip install sagemaker\n"
            "   Or provide --s3-bucket explicitly."
        )
        sys.exit(1)

    try:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        print(f"ℹ️  No --s3-bucket provided. Using SageMaker default bucket: {bucket}")
        return bucket
    except Exception as e:
        print(f"❌ Failed to resolve SageMaker default bucket: {e}")
        print("   Provide --s3-bucket explicitly or check your AWS credentials.")
        sys.exit(1)


def validate_s3_access(s3_client, bucket, prefix, role_arn=None):
    """Verify the S3 bucket exists and is writable.

    Performs a head_bucket call to check existence, then a small put_object
    test to verify write access. The test object is deleted after verification.

    Args:
        s3_client: A boto3 S3 client.
        bucket: S3 bucket name.
        prefix: S3 key prefix.
        role_arn: Optional IAM role ARN (unused, reserved for future STS assume-role).

    Returns:
        True if the bucket exists and is writable.

    Raises:
        SystemExit: If the bucket does not exist or is not writable.
    """
    # Check bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404" or error_code == "NoSuchBucket":
            print(f"❌ S3 bucket '{bucket}' does not exist.")
        elif error_code == "403":
            print(f"❌ Access denied to S3 bucket '{bucket}'. Check your IAM permissions.")
        else:
            print(f"❌ Error accessing S3 bucket '{bucket}': {e}")
        return False

    # Verify write access with a small test object
    test_key = f"{prefix}.write_test"
    try:
        s3_client.put_object(Bucket=bucket, Key=test_key, Body=b"write_test")
        s3_client.delete_object(Bucket=bucket, Key=test_key)
    except ClientError as e:
        print(f"❌ Cannot write to s3://{bucket}/{prefix} — check IAM permissions: {e}")
        return False

    print(f"✅ S3 bucket '{bucket}' is accessible and writable.")
    return True

def check_existing_model(s3_client, bucket, prefix):
    """Check whether model artifacts already exist at the S3 prefix.

    Lists up to 5 objects under the given prefix. If any non-hidden files
    are found, the model is considered already uploaded.

    Args:
        s3_client: A boto3 S3 client.
        bucket: S3 bucket name.
        prefix: S3 key prefix (e.g. 'llama-70b-instruct/').

    Returns:
        A tuple (exists: bool, file_count: int, total_size: int).
        exists is True if at least one object was found under the prefix.
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        file_count = 0
        total_size = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Skip hidden files and the write-test marker
                basename = key.rsplit("/", 1)[-1]
                if basename.startswith(".") or basename == ".write_test":
                    continue
                file_count += 1
                total_size += obj.get("Size", 0)

        return file_count > 0, file_count, total_size
    except ClientError:
        # If we can't list, assume no existing model — the upload will
        # fail later with a clearer permission error.
        return False, 0, 0



def download_and_upload(model_id, s3_bucket, s3_prefix, hf_token):
    """Download model from HuggingFace Hub and upload to S3.

    Uses huggingface_hub.snapshot_download() to download the model to a
    temporary directory, then uploads each file to S3 using multipart
    upload for large files.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct").
        s3_bucket: Target S3 bucket name.
        s3_prefix: S3 key prefix for uploaded files.
        hf_token: HuggingFace access token (can be None for public models).

    Returns:
        The full S3 URI (e.g. "s3://my-bucket/llama-70b-instruct/").
    """
    from huggingface_hub import snapshot_download

    s3_client = boto3.client("s3")

    # --- Download from HuggingFace ---
    print(f"\n📥 Downloading model '{model_id}' from HuggingFace Hub...")
    print("   This may take a while for large models (~140GB for 70B FP16).\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=tmp_dir,
            token=hf_token,
        )

        # --- Collect files to upload ---
        files_to_upload = []
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir)
                # Skip hidden files/dirs (e.g. .cache, .git)
                if any(part.startswith(".") for part in rel_path.split(os.sep)):
                    continue
                file_size = os.path.getsize(local_path)
                s3_key = f"{s3_prefix}{rel_path}"
                files_to_upload.append((local_path, s3_key, file_size))

        total_files = len(files_to_upload)
        total_size = sum(f[2] for f in files_to_upload)

        print(f"📤 Uploading {total_files} files ({_format_size(total_size)}) to s3://{s3_bucket}/{s3_prefix}")
        print()

        # --- Upload to S3 with progress ---
        uploaded_bytes = 0
        start_time = time.time()

        for idx, (local_path, s3_key, file_size) in enumerate(files_to_upload, 1):
            file_name = os.path.basename(local_path)
            elapsed = time.time() - start_time

            # Estimate time remaining
            if uploaded_bytes > 0 and elapsed > 0:
                speed = uploaded_bytes / elapsed
                remaining_bytes = total_size - uploaded_bytes
                eta = _format_time(remaining_bytes / speed) if speed > 0 else "unknown"
            else:
                eta = "calculating..."

            print(
                f"  [{idx}/{total_files}] Uploading {file_name} "
                f"({_format_size(uploaded_bytes)} / {_format_size(total_size)}) "
                f"ETA: {eta}"
            )

            s3_client.upload_file(
                local_path,
                s3_bucket,
                s3_key,
                Config=S3_TRANSFER_CONFIG,
            )

            uploaded_bytes += file_size

    elapsed_total = time.time() - start_time
    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"

    print(f"\n✅ Upload complete in {_format_time(elapsed_total)}.")
    print(f"   Total: {_format_size(total_size)} across {total_files} files.")
    print(f"\n   S3 URI: {s3_uri}")

    return s3_uri


def main():
    """Main entry point. Returns 0 on success, 1 on failure."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("  HuggingFace → S3 Model Uploader")
    print("=" * 70)
    print(f"  Model ID  : {args.model_id}")
    print(f"  S3 Bucket : {args.s3_bucket}")
    print(f"  S3 Prefix : {args.s3_prefix}")
    print(f"  HF Token  : {'provided' if args.hf_token else 'not set (public models only)'}")
    print("=" * 70 + "\n")

    s3_client = boto3.client("s3")

    if not validate_s3_access(s3_client, args.s3_bucket, args.s3_prefix):
        return 1

    # Check if model already exists in S3
    s3_uri = f"s3://{args.s3_bucket}/{args.s3_prefix}"
    if not args.force:
        exists, file_count, total_size = check_existing_model(
            s3_client, args.s3_bucket, args.s3_prefix,
        )
        if exists:
            print(f"✅ Model already exists at {s3_uri}")
            print(f"   Found {file_count} files ({_format_size(total_size)}).")
            print(f"\n   Use with the 70B deployment script:")
            print(f"   python deployment/deploy_llama3_70b.py --model-s3-uri {s3_uri}")
            print(f"\n   To re-upload, run again with --force.")
            return 0

    try:
        s3_uri = download_and_upload(
            model_id=args.model_id,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            hf_token=args.hf_token,
        )
    except Exception as e:
        print(f"\n❌ Error during download/upload: {e}")
        return 1

    print(f"\n🎉 Model artifacts are ready at: {s3_uri}")
    print(f"\n   Use with the 70B deployment script:")
    print(f"   python deployment/deploy_llama3_70b.py --model-s3-uri {s3_uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
