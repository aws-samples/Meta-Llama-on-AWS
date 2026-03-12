#!/usr/bin/env python3
"""Upload policy documents to S3 for Bedrock Knowledge Base ingestion.

This script uploads all markdown files from the docs/policies/ directory to S3,
preserving filenames and adding appropriate metadata. It handles errors gracefully
and provides a summary of uploaded and failed files.

After uploading, it can trigger a Knowledge Base sync job and poll for completion.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import DsConfig, EnvSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_markdown_files(docs_dir: str) -> List[Path]:
    """List all markdown files in the specified directory.
    
    Args:
        docs_dir: Path to directory containing policy documents
        
    Returns:
        List of Path objects for markdown files
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    if not docs_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {docs_dir}")
    
    # Find all .md files (case-insensitive by checking all files)
    # glob patterns are case-sensitive on some systems, so we check manually
    md_files = [
        f for f in docs_path.iterdir()
        if f.is_file() and f.suffix.lower() == '.md'
    ]
    
    logger.info(f"Found {len(md_files)} markdown files in {docs_dir}")
    return md_files


def upload_file_to_s3(
    s3_client: Any,
    file_path: Path,
    bucket_name: str,
    s3_prefix: str = "policies"
) -> None:
    """Upload a single file to S3 with proper content type and metadata.
    
    Args:
        s3_client: Boto3 S3 client
        file_path: Path to local file
        bucket_name: S3 bucket name
        s3_prefix: S3 key prefix (default: "policies")
        
    Raises:
        ClientError: If S3 upload fails
    """
    # Construct S3 key with prefix
    s3_key = f"{s3_prefix}/{file_path.name}"
    
    # Upload with markdown content type and metadata
    s3_client.upload_file(
        str(file_path),
        bucket_name,
        s3_key,
        ExtraArgs={
            'ContentType': 'text/markdown',
            'Metadata': {
                'source': 'sre-poc-policy-documents',
                'original-filename': file_path.name
            }
        }
    )
    
    logger.info(f"✅ Uploaded: {file_path.name} -> s3://{bucket_name}/{s3_key}")


def upload_documents(
    docs_dir: str = "docs/policies",
    bucket_name: str | None = None,
    region: str | None = None,
    s3_prefix: str = "policies"
) -> Dict[str, Any]:
    """Upload all markdown files from docs directory to S3.
    
    This function:
    1. Lists all markdown files in the specified directory
    2. Uploads each file to S3 with proper content type
    3. Handles errors gracefully, continuing with remaining files
    4. Returns a summary of uploaded and failed files
    
    Args:
        docs_dir: Local directory containing policy documents (default: "docs/policies")
        bucket_name: S3 bucket name (from config if not provided)
        region: AWS region (from config if not provided)
        s3_prefix: S3 key prefix for uploaded files (default: "policies")
        
    Returns:
        Dictionary containing:
            - uploaded: List of successfully uploaded filenames
            - failed: List of tuples (filename, error_message)
            - bucket: S3 bucket name used
            - region: AWS region used
            - total: Total number of files uploaded successfully
            
    Raises:
        FileNotFoundError: If docs_dir doesn't exist
        NoCredentialsError: If AWS credentials are not configured
    """
    # Use config values if not provided
    bucket_name = bucket_name or DsConfig.S3_BUCKET_NAME
    region = region or EnvSettings.ACCOUNT_REGION
    
    logger.info(f"Starting document upload to s3://{bucket_name}/{s3_prefix}")
    logger.info(f"Source directory: {docs_dir}")
    logger.info(f"AWS Region: {region}")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=region)
    except NoCredentialsError as e:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    
    # List all markdown files
    try:
        md_files = list_markdown_files(docs_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    
    if not md_files:
        logger.warning(f"No markdown files found in {docs_dir}")
        return {
            "uploaded": [],
            "failed": [],
            "bucket": bucket_name,
            "region": region,
            "total": 0
        }
    
    # Track results
    uploaded_files: List[str] = []
    failed_files: List[Tuple[str, str]] = []
    
    # Upload each file
    for md_file in md_files:
        try:
            upload_file_to_s3(s3_client, md_file, bucket_name, s3_prefix)
            uploaded_files.append(md_file.name)
        except ClientError as e:
            error_msg = f"S3 error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
            failed_files.append((md_file.name, error_msg))
            logger.error(f"❌ Failed to upload {md_file.name}: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            failed_files.append((md_file.name, error_msg))
            logger.error(f"❌ Failed to upload {md_file.name}: {error_msg}")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Upload Summary:")
    logger.info(f"  Total files found: {len(md_files)}")
    logger.info(f"  Successfully uploaded: {len(uploaded_files)}")
    logger.info(f"  Failed: {len(failed_files)}")
    logger.info(f"  Bucket: s3://{bucket_name}/{s3_prefix}")
    logger.info(f"{'='*60}\n")
    
    if failed_files:
        logger.warning("Failed uploads:")
        for filename, error in failed_files:
            logger.warning(f"  - {filename}: {error}")
    
    return {
        "uploaded": uploaded_files,
        "failed": failed_files,
        "bucket": bucket_name,
        "region": region,
        "total": len(uploaded_files)
    }


def trigger_sync_job(
    knowledge_base_id: str,
    data_source_id: str,
    region: str | None = None
) -> Dict[str, Any]:
    """Trigger Knowledge Base ingestion job after document upload.
    
    This function starts a sync/ingestion job that indexes all documents
    in the S3 data source into the Knowledge Base vector store.
    
    Args:
        knowledge_base_id: Bedrock Knowledge Base ID
        data_source_id: Data Source ID within the Knowledge Base
        region: AWS region (from config if not provided)
        
    Returns:
        Dictionary containing:
            - ingestion_job_id: ID of the started ingestion job
            - status: Initial status of the job
            - started_at: Timestamp when job was started
            
    Raises:
        ClientError: If the API call fails
        
    Requirements: 2.3
    """
    region = region or EnvSettings.ACCOUNT_REGION
    
    logger.info(f"Triggering sync job for Knowledge Base: {knowledge_base_id}")
    logger.info(f"Data Source ID: {data_source_id}")
    
    try:
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id
        )
        
        ingestion_job = response['ingestionJob']
        
        logger.info(f"✅ Sync job started successfully")
        logger.info(f"  Job ID: {ingestion_job['ingestionJobId']}")
        logger.info(f"  Status: {ingestion_job['status']}")
        
        return {
            "ingestion_job_id": ingestion_job['ingestionJobId'],
            "status": ingestion_job['status'],
            "started_at": ingestion_job.get('startedAt'),
            "knowledge_base_id": knowledge_base_id,
            "data_source_id": data_source_id
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        logger.error(f"❌ Failed to start sync job: {error_code} - {error_msg}")
        raise


def poll_sync_job_status(
    knowledge_base_id: str,
    data_source_id: str,
    ingestion_job_id: str,
    region: str | None = None,
    poll_interval: int = 10,
    max_wait_time: int = 600
) -> Dict[str, Any]:
    """Poll sync job status until completion or timeout.
    
    This function polls the ingestion job status at regular intervals
    until the job completes (SUCCESS or FAILED) or the maximum wait
    time is exceeded.
    
    Args:
        knowledge_base_id: Bedrock Knowledge Base ID
        data_source_id: Data Source ID within the Knowledge Base
        ingestion_job_id: ID of the ingestion job to poll
        region: AWS region (from config if not provided)
        poll_interval: Seconds to wait between status checks (default: 10)
        max_wait_time: Maximum seconds to wait for completion (default: 600)
        
    Returns:
        Dictionary containing:
            - status: Final status (COMPLETE, FAILED, or IN_PROGRESS if timeout)
            - ingestion_job_id: ID of the ingestion job
            - statistics: Job statistics (documents processed, failed, etc.)
            - failure_reasons: List of failure reasons if job failed
            
    Raises:
        ClientError: If the API call fails
        TimeoutError: If max_wait_time is exceeded
        
    Requirements: 2.3
    """
    region = region or EnvSettings.ACCOUNT_REGION
    
    logger.info(f"Polling sync job status: {ingestion_job_id}")
    logger.info(f"Poll interval: {poll_interval}s, Max wait: {max_wait_time}s")
    
    bedrock_agent = boto3.client('bedrock-agent', region_name=region)
    
    start_time = time.time()
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            response = bedrock_agent.get_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=ingestion_job_id
            )
            
            ingestion_job = response['ingestionJob']
            status = ingestion_job['status']
            
            logger.info(f"  Status: {status} (elapsed: {int(elapsed_time)}s)")
            
            # Check if job is complete
            if status == 'COMPLETE':
                logger.info("✅ Sync job completed successfully")
                
                # Extract statistics
                stats = ingestion_job.get('statistics', {})
                logger.info(f"  Documents scanned: {stats.get('numberOfDocumentsScanned', 0)}")
                logger.info(f"  Documents indexed: {stats.get('numberOfNewDocumentsIndexed', 0)}")
                logger.info(f"  Documents modified: {stats.get('numberOfModifiedDocumentsIndexed', 0)}")
                logger.info(f"  Documents deleted: {stats.get('numberOfDocumentsDeleted', 0)}")
                logger.info(f"  Documents failed: {stats.get('numberOfDocumentsFailed', 0)}")
                
                return {
                    "status": status,
                    "ingestion_job_id": ingestion_job_id,
                    "statistics": stats,
                    "failure_reasons": ingestion_job.get('failureReasons', [])
                }
            
            elif status == 'FAILED':
                logger.error("❌ Sync job failed")
                failure_reasons = ingestion_job.get('failureReasons', [])
                for reason in failure_reasons:
                    logger.error(f"  Failure reason: {reason}")
                
                return {
                    "status": status,
                    "ingestion_job_id": ingestion_job_id,
                    "statistics": ingestion_job.get('statistics', {}),
                    "failure_reasons": failure_reasons
                }
            
            # Job still in progress, wait and retry
            time.sleep(poll_interval)
            elapsed_time = time.time() - start_time
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"❌ Error polling sync job: {error_code} - {error_msg}")
            raise
    
    # Timeout exceeded
    logger.warning(f"⚠️  Sync job still in progress after {max_wait_time}s")
    raise TimeoutError(
        f"Sync job did not complete within {max_wait_time} seconds. "
        f"Job ID: {ingestion_job_id}"
    )


def verify_documents_indexed(
    knowledge_base_id: str,
    data_source_id: str,
    ingestion_job_id: str,
    expected_document_count: int,
    region: str | None = None
) -> bool:
    """Verify that all documents are indexed successfully.
    
    This function checks the ingestion job statistics to verify that
    the expected number of documents were indexed without failures.
    
    Args:
        knowledge_base_id: Bedrock Knowledge Base ID
        data_source_id: Data Source ID within the Knowledge Base
        ingestion_job_id: ID of the completed ingestion job
        expected_document_count: Number of documents expected to be indexed
        region: AWS region (from config if not provided)
        
    Returns:
        True if all documents indexed successfully, False otherwise
        
    Raises:
        ClientError: If the API call fails
        
    Requirements: 2.4
    """
    region = region or EnvSettings.ACCOUNT_REGION
    
    logger.info(f"Verifying document indexing for job: {ingestion_job_id}")
    logger.info(f"Expected documents: {expected_document_count}")
    
    try:
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        response = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            ingestionJobId=ingestion_job_id
        )
        
        ingestion_job = response['ingestionJob']
        stats = ingestion_job.get('statistics', {})
        
        # Calculate total indexed documents
        new_docs = stats.get('numberOfNewDocumentsIndexed', 0)
        modified_docs = stats.get('numberOfModifiedDocumentsIndexed', 0)
        total_indexed = new_docs + modified_docs
        
        failed_docs = stats.get('numberOfDocumentsFailed', 0)
        
        logger.info(f"  New documents indexed: {new_docs}")
        logger.info(f"  Modified documents indexed: {modified_docs}")
        logger.info(f"  Total indexed: {total_indexed}")
        logger.info(f"  Failed: {failed_docs}")
        
        # Verify all documents indexed successfully
        if failed_docs > 0:
            logger.error(f"❌ {failed_docs} documents failed to index")
            failure_reasons = ingestion_job.get('failureReasons', [])
            for reason in failure_reasons:
                logger.error(f"  Failure reason: {reason}")
            return False
        
        if total_indexed < expected_document_count:
            logger.warning(
                f"⚠️  Only {total_indexed} of {expected_document_count} "
                f"documents were indexed"
            )
            return False
        
        logger.info(f"✅ All {total_indexed} documents indexed successfully")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        logger.error(f"❌ Error verifying documents: {error_code} - {error_msg}")
        raise


def main():
    """Main entry point for the upload script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload policy documents to S3 for Bedrock Knowledge Base"
    )
    parser.add_argument(
        "--docs-dir",
        default="docs/policies",
        help="Directory containing policy documents (default: docs/policies)"
    )
    parser.add_argument(
        "--bucket",
        help=f"S3 bucket name (default: {DsConfig.S3_BUCKET_NAME})"
    )
    parser.add_argument(
        "--region",
        help=f"AWS region (default: {EnvSettings.ACCOUNT_REGION})"
    )
    parser.add_argument(
        "--prefix",
        default="policies",
        help="S3 key prefix (default: policies)"
    )
    parser.add_argument(
        "--trigger-sync",
        action="store_true",
        help="Trigger Knowledge Base sync job after upload"
    )
    parser.add_argument(
        "--kb-id",
        help="Knowledge Base ID (required if --trigger-sync is used)"
    )
    parser.add_argument(
        "--data-source-id",
        help="Data Source ID (required if --trigger-sync is used)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between sync job status checks (default: 10)"
    )
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=600,
        help="Maximum seconds to wait for sync completion (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Validate sync job arguments
    if args.trigger_sync:
        if not args.kb_id or not args.data_source_id:
            parser.error("--kb-id and --data-source-id are required when --trigger-sync is used")
    
    try:
        # Upload documents
        result = upload_documents(
            docs_dir=args.docs_dir,
            bucket_name=args.bucket,
            region=args.region,
            s3_prefix=args.prefix
        )
        
        # Exit with error code if any uploads failed
        if result["failed"]:
            logger.error("Some documents failed to upload")
            sys.exit(1)
        
        logger.info("✅ All documents uploaded successfully!")
        
        # Trigger sync job if requested
        if args.trigger_sync:
            logger.info("\n" + "="*60)
            logger.info("Starting Knowledge Base sync job...")
            logger.info("="*60 + "\n")
            
            # Trigger sync
            sync_result = trigger_sync_job(
                knowledge_base_id=args.kb_id,
                data_source_id=args.data_source_id,
                region=args.region
            )
            
            # Poll for completion
            poll_result = poll_sync_job_status(
                knowledge_base_id=args.kb_id,
                data_source_id=args.data_source_id,
                ingestion_job_id=sync_result['ingestion_job_id'],
                region=args.region,
                poll_interval=args.poll_interval,
                max_wait_time=args.max_wait_time
            )
            
            # Verify all documents indexed
            if poll_result['status'] == 'COMPLETE':
                verified = verify_documents_indexed(
                    knowledge_base_id=args.kb_id,
                    data_source_id=args.data_source_id,
                    ingestion_job_id=sync_result['ingestion_job_id'],
                    expected_document_count=result['total'],
                    region=args.region
                )
                
                if verified:
                    logger.info("\n✅ All documents uploaded and indexed successfully!")
                    sys.exit(0)
                else:
                    logger.error("\n❌ Document indexing verification failed")
                    sys.exit(1)
            else:
                logger.error(f"\n❌ Sync job failed with status: {poll_result['status']}")
                sys.exit(1)
        else:
            sys.exit(0)
            
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
