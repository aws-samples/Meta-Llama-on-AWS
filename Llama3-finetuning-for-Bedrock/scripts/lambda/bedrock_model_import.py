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
        
        # Calculate end time (context.get_remaining_time_in_millis() - 10 seconds buffer)
        end_time = time.time() + (context.get_remaining_time_in_millis() / 1000) - 10
        
        # Wait for the job to complete or fail
        while time.time() < end_time:
            status, model_arn, error_message = check_job_status(bedrock, JOB_NAME)
            print(f"Current status: {status}")
            
            if status == 'Completed':
                return {
                    'statusCode': 200,
                    'body': json.dumps('Model import job completed successfully'),
                    'model_arn': model_arn,
                    'job_name': JOB_NAME
                }
            elif status == 'Failed':
                return {
                    'statusCode': 500,
                    'body': json.dumps(f'Model import job failed: {error_message}'),
                    'job_name': JOB_NAME,
                    'error': error_message
                }
            elif status == 'Stopped':
                return {
                    'statusCode': 500,
                    'body': json.dumps('Model import job was stopped'),
                    'job_name': JOB_NAME
                }
            
            # Wait before next check
            time.sleep(30)
        
        # If we're about to timeout, return job information for follow-up
        return {
            'statusCode': 202,  # Accepted
            'body': json.dumps('Job still in progress - Lambda timeout approaching'),
            'job_name': JOB_NAME,
            'job_arn': job_arn,
            'requires_follow_up': True
        }
    
    except ClientError as e:
        error_message = str(e)
        print(f"An error occurred: {error_message}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {error_message}'),
            'job_name': JOB_NAME,
            'error': error_message
        }
    except Exception as e:
        error_message = str(e)
        print(f"Unexpected error: {error_message}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Unexpected error: {error_message}'),
            'job_name': JOB_NAME,
            'error': error_message
        }
