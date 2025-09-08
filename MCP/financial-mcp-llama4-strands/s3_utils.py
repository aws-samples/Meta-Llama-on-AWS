import boto3
import os
import json
from datetime import datetime

def get_sagemaker_bucket():
    """Get SageMaker default bucket name"""
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = boto3.Session().region_name or "us-west-2"
    return f"sagemaker-{region}-{account_id}"

def download_from_s3(s3_uri):
    """Download data from S3 URI"""
    s3_parts = s3_uri.replace('s3://', '').split('/', 1)
    bucket = s3_parts[0]
    key = s3_parts[1]
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def upload_json_to_s3(data, ticker, data_source):
    """Upload JSON data to S3 with ticker organization"""
    bucket = get_sagemaker_bucket()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    s3_key = f"financial-data/{ticker}/{timestamp}/{data_source}_data.json"
    
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket, 
        Key=s3_key, 
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )
    return f"s3://{bucket}/{s3_key}"

