import boto3
import json
import random
import time
from botocore.exceptions import ClientError

def check_job_status(bedrock, job_name):
    """
    Check the status of a model import job
    Returns: (status, model_arn, error_message)
    """
    response = bedrock.list_model_import_jobs(
        nameContains=job_name,
        sortBy='CreationTime',
        sortOrder='Descending'
    )
    
    if response['modelImportJobSummaries']:
        job_summary = response['modelImportJobSummaries'][0]
        status = job_summary['status']
        model_arn = job_summary.get('importedModelArn')
        error_message = job_summary.get('failureReason', '')
        return status, model_arn, error_message
    return None, None, None

def lambda_handler(event, context):
    REGION_NAME = 'us-west-2'
    bedrock = boto3.client(service_name='bedrock',
                       region_name=REGION_NAME)
    # Generate a unique job name with timestamp and random number
    timestamp = int(time.time())
    random_number = random.randint(1000, 9999)
    JOB_NAME = f"meta3-import-model-{timestamp}-{random_number}"

    # Get parameters from the event
    ROLE_ARN = event.get('role_arn')
    if not ROLE_ARN:
        raise ValueError("role_arn must be provided in the event")

    IMPORTED_MODEL_NAME = event.get('model_name', "llama3_sagemaker")
    S3_URI = event.get('model_uri')
    if not S3_URI:
        raise ValueError("model_uri must be provided in the event")

    try:
        # Create Model Import Job
        create_job_response = bedrock.create_model_import_job(
            jobName=JOB_NAME,
            importedModelName=IMPORTED_MODEL_NAME,
            roleArn=ROLE_ARN,
            modelDataSource={
                "s3DataSource": {
                    "s3Uri": S3_URI
                }
            },
        )
        job_arn = create_job_response.get("jobArn")
        print(f"Job created with ARN: {job_arn}")
        
        return {
            'statusCode': 200,
            'job_name': JOB_NAME,
            'job_arn': job_arn
        }
    
    except ClientError as e:
        error_message = str(e)
        print(f"An error occurred: {error_message}")
        raise e
