import boto3
import os
import ssl
import json
from datetime import datetime
from strands import Agent, tool
from strands.models import BedrockModel
from s3_frame_extraction_agent import s3_frame_extraction_agent
from s_visual_analysis_agent import s_visual_analysis_agent
# from temporal_analysis_agent import temporal_analysis_agent
from summary_generation_agent import summary_generation_agent
from retrieve_json import retrieve_json_agent
from c_temporal_analysis_agent import c_temporal_analysis_agent
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['AWS_REGION'] = 'us-east-1'
boto3.setup_default_session(region_name='us-east-1')

bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name='us-east-1',
    streaming=False
)

@tool
def run_frame_extraction(instruction: str) -> str:
    """Run the frame extraction agent with the given instruction"""
    return s3_frame_extraction_agent(instruction)

@tool
def run_visual_analysis(instruction: str) -> str:
    """Run the visual analysis agent with the given instruction"""
    return s_visual_analysis_agent(instruction)
@tool
def run_temporal_reasoning(instruction: str) -> str:
    """Run the temporal reasoning agent with the given instruction"""
    return c_temporal_analysis_agent(instruction)
@tool
def run_summary_generation(instruction: str) -> str:
    """Run the final summary generation agent with the given instruction"""
    return summary_generation_agent(instruction)

@tool
def retrieve_json_from_s3(instruction: str) -> str:
    """Retrieve the json analysis from the S3 URI"""
    return retrieve_json_agent(instruction)

# @tool
# def retrieve_json_from_s3(s3_uri: str) -> str:
#     """Retrieve JSON document from S3 URI and delete local visual analysis file"""
#     try:
#         # Parse S3 URI
#         s3_parts = s3_uri.replace('s3://', '').split('/', 1)
#         bucket = s3_parts[0]
#         key = s3_parts[1]
        
#         # Download JSON from S3
#         s3_client = boto3.client('s3')
#         response = s3_client.get_object(Bucket=bucket, Key=key)
#         json_content = response['Body'].read().decode('utf-8')
        
#         # Validate JSON content
#         json.loads(json_content)  # This will raise exception if invalid
        
#         # Delete local visual analysis file if JSON is valid
#         local_file = "visual_analysis_results.json"
#         if os.path.exists(local_file):
#             os.remove(local_file)
#         return json_content
#     except Exception as e:
#         return f"Error retrieving JSON from {s3_uri}: {str(e)}"
@tool
def print_s3_uri(s3_uri: str) -> str:
    """Print the S3 URI"""
    return f"S3 URI: {s3_uri}"

@tool
def upload_text_analysis_results(results: str, s3_video_path: str) -> str:
    """upload analysis results as TXT file to S3 bucket in videos folder"""
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Parse S3 path to get bucket and video folder
        s3_parts = s3_video_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        video_folder = s3_parts[-1]  # Get the main video folder name
        filename = f"{video_folder}_analysis_results_{timestamp}.txt"

        # Create S3 key for results
        s3_key = f"videos/{video_folder}/{filename}"
        
        # Prepare text data
        if isinstance(results, str):
            text_content = results
        else:
            text_content = str(results)
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=text_content,
            ContentType='text/plain'
        )
        
        return f"Analysis results saved to s3://{bucket}/{s3_key}"
    except Exception as e:
        return f"Error saving results: {str(e)}"

    
@tool
def upload_analysis_results(results: str, s3_video_path: str) -> str:
    """upload analysis results as JSON file to S3 bucket in videos folder"""
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.json"
    
    try:
        # Parse S3 path to get bucket and video folder
        s3_parts = s3_video_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        video_folder = s3_parts[-1]  # Get the main video folder name
        
        # Create S3 key for results
        s3_key = f"videos/{video_folder}/{filename}"
        
        # Prepare data
        if isinstance(results, str):
            try:
                data = json.loads(results)
            except json.JSONDecodeError:
                data = {"analysis_results": results, "timestamp": datetime.now().isoformat()}
        else:
            data = results
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        
        return f"Analysis results saved to s3://{bucket}/{s3_key}"
    except Exception as e:
        return f"Error saving results: {str(e)}"
llama4_coordinator_agent = Agent(
    system_prompt="""You are a video processing coordinator. Your job is to process videos step by step.

##When asked to process a video:
1. Extract frames from S3 video using run_frame_extraction
2. Use the frame location from step 1 to run_visual_analysis
3. WAIT for visual analysis to complete sending the json to s3
4. Use the retrieve_json agent to extract the json from step 3
5. Use the text result of retrieve_json_from_s3 by passing it to run_temporal_reasoning
6. Pass the result from temporal reasoning to run_summary_generation 
7. Upload analysis generated in run_summary_generation and return s3 location

##IMPORTANT:
- Call ONE tool at a time and wait for the result
- Use the EXACT result from the previous step as input to the next step
- Do NOT call multiple tools simultaneously
- Do NOT return raw JSON or function call syntax
""",
    model=bedrock_model,
     tools=[run_frame_extraction, run_visual_analysis, run_temporal_reasoning, run_summary_generation, upload_analysis_results, retrieve_json_from_s3],
)