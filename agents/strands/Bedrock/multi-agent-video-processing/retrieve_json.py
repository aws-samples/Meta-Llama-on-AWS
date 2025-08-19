import json
import boto3
import os
from strands import Agent, tool
from strands.models import BedrockModel

# Get current default region from boto3 session
session = boto3.session.Session()
region = session.region_name

# Set environment variables based on detected region
if region:
    os.environ['AWS_DEFAULT_REGION'] = region
    os.environ['AWS_REGION'] = region
    boto3.setup_default_session(region_name=region)

@tool
def retrieve_json_from_s3(s3_uri: str) -> str:
    """
    Retrieve JSON document from given S3 URI.
    Validates the JSON content before returning.
    Deletes local visual analysis file 'visual_analysis_results.json' if it exists.
    
    Args:
        s3_uri (str): S3 URI in the format s3://bucket/key
    
    Returns:
        str: JSON content as string if valid,
             else error message string.
    """

    try:
        # Parse S3 URI
        s3_parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        # Download JSON from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_content = response['Body'].read().decode('utf-8')
        
        # Validate JSON content
        json.loads(json_content)  # will raise an exception if invalid
        
        # Delete local visual analysis file if exists
        local_file = "visual_analysis_results.json"
        if os.path.exists(local_file):
            os.remove(local_file)
        # print("THIS IS AN ERROR MESSAGING CONFIRMING json IS SENT", s3_uri)

        return json_content
    except Exception as e:
        return f"Error retrieving JSON from {s3_uri}: {str(e)}"

@tool
def extract_text_from_analysis(json_content: str) -> str:
    """Extract only the analysis text from the JSON, removing metadata"""
    try:
        data = json.loads(json_content)
        
        # Handle new format with 'analyses'
        if 'analyses' in data:
            analyses = data['analyses']
        # Handle old format with 'sessions'
        elif 'sessions' in data:
            analyses = [session['data'] for session in data['sessions'] if 'data' in session]
        else:
            return "Error: No 'analyses' or 'sessions' field found in JSON"
        
        text_only = []
        for analysis in analyses:
            if 'analysis' in analysis:
                text = analysis['analysis']
                if not text.startswith("Failed:"):
                    text_only.append(text)
        print("THIS IS AN ERROR MESSAGING CONFIRMING json extraction")

        return "\n".join(text_only)
    except Exception as e:
        return f"Error extracting text: {str(e)}"
@tool
def process_s3_analysis_json(s3_uri: str) -> str:
    """Retrieve JSON from S3 and extract only the analysis text"""
    try:
        # Parse S3 URI and download JSON
        s3_parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_content = response['Body'].read().decode('utf-8')
        
        # Parse and extract text
        data = json.loads(json_content)
        
        # Handle both formats
        if 'analyses' in data:
            analyses = data['analyses']
        elif 'sessions' in data:
            analyses = [session['data'] for session in data['sessions'] if 'data' in session]
        else:
            return "Error: No 'analyses' or 'sessions' field found"
        
        # Extract text only
        text_only = []
        for analysis in analyses:
            if 'analysis' in analysis:
                text = analysis['analysis']
                if not text.startswith("Failed:"):
                    text_only.append(text)
        
        # Clean up local file
        local_file = "visual_analysis_results.json"
        if os.path.exists(local_file):
            os.remove(local_file)
        
        return "\n".join(text_only)
    except Exception as e:
        return f"Error processing {s3_uri}: {str(e)}"


bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name=region,
    streaming=False,
    temperature=0
)  

retrieve_json_agent = Agent(
system_prompt="Call process_s3_analysis_json with the S3 URI. Your response must be the exact text output from the tool, nothing else.",
    model=bedrock_model,
    callback_handler=None,

    tools=[process_s3_analysis_json],
)


