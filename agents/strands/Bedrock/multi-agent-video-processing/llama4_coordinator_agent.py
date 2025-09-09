import boto3
import os
import ssl
import json
from datetime import datetime
from strands import Agent, tool
from strands.models import BedrockModel
from s3_frame_extraction_agent import s3_frame_extraction_agent
from s_visual_analysis_agent import s_visual_analysis_agent
from summary_generation_agent import summary_generation_agent
from retrieve_json import retrieve_json_agent
from c_temporal_analysis_agent import c_temporal_analysis_agent

ssl._create_default_https_context = ssl._create_unverified_context

# Get current default region from boto3 session
session = boto3.session.Session()
region = session.region_name

# Set environment variables based on detected region
if region:
    os.environ['AWS_DEFAULT_REGION'] = region
    os.environ['AWS_REGION'] = region
    boto3.setup_default_session(region_name=region)

bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name=region,
    streaming=False
)

# ----------------- TOOLS -----------------
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

@tool
def print_s3_uri(s3_uri: str) -> str:
    """Print the S3 URI"""
    return f"S3 URI: {s3_uri}"

@tool
def upload_text_analysis_results(results: str, s3_video_path: str) -> str:
    """Upload analysis results as TXT file to S3"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        s3_parts = s3_video_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        video_folder = s3_parts[-1]
        filename = f"{video_folder}_analysis_results_{timestamp}.txt"
        s3_key = f"videos/{video_folder}/{filename}"

        text_content = results if isinstance(results, str) else str(results)
        boto3.client('s3').put_object(
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
    """Upload analysis results as JSON file to S3"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.json"
    try:
        s3_parts = s3_video_path.replace('s3://', '').split('/')
        bucket = s3_parts[0]
        video_folder = s3_parts[-1]
        s3_key = f"videos/{video_folder}/{filename}"

        if isinstance(results, str):
            try:
                data = json.loads(results)
            except json.JSONDecodeError:
                data = {"analysis_results": results, "timestamp": datetime.now().isoformat()}
        else:
            data = results

        boto3.client('s3').put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        return f"Analysis results saved to s3://{bucket}/{s3_key}"
    except Exception as e:
        return f"Error saving results: {str(e)}"

# ----------------- FACTORY FUNC -----------------
def llama4_coordinator_agent() -> Agent:
    """
    Factory constructor: creates a NEW agent instance with a fresh conversation history.
    Use this per video request for clean isolation.
    """
    return Agent(
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
- Use the EXACT result from the previous step as input
- Do NOT call multiple tools simultaneously
- Do NOT return raw JSON or function call syntax
""",
        model=bedrock_model,
        tools=[
            run_frame_extraction,
            run_visual_analysis,
            run_temporal_reasoning,
            run_summary_generation,
            upload_analysis_results,
            retrieve_json_from_s3,
        ],
    )
