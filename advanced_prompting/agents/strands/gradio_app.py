import boto3
import os
import ssl
import json
from datetime import datetime
from strands import Agent, tool
from strands.models import BedrockModel
from s3_frame_extraction_agent import s3_frame_extraction_agent
from s_visual_analysis_agent import s_visual_analysis_agent
from c_temporal_analysis_agent import c_temporal_analysis_agent
from summary_generation_agent import summary_generation_agent
from retrieve_json import retrieve_json_agent
import gradio as gr

ssl._create_default_https_context = ssl._create_unverified_context

##** User Changes:**##
## The following bucket name to your own bucket in your s3

bucket=""
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
os.environ['AWS_REGION'] = 'us-west-2'
boto3.setup_default_session(region_name='us-west-2')

bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name='us-west-2',
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
def upload_to_sagemaker_bucket(local_video_path, base_folder="videos/"):
    sagemaker = boto3.client('sagemaker')
    s3 = boto3.client('s3')

    # Get default SageMaker bucket
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = boto3.Session().region_name
    bucket_name = f"sagemaker-{region}-{account_id}"
    # Get filename and create subfolder name
    filename = os.path.basename(local_video_path)
    filename_without_ext = os.path.splitext(filename)[0]
    # Create the full S3 path: videos/filename_without_ext/filename
    s3_key = os.path.join(base_folder, filename_without_ext, filename)
    # Upload file
    s3.upload_file(local_video_path, bucket_name, s3_key)  
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    print(f"Uploaded to {s3_uri}")  

    s3_folder_path = os.path.join(base_folder, filename_without_ext)
    s3_folder_uri = f"s3://{bucket_name}/{s3_folder_path}"

    return s3_folder_uri

def get_latest_analysis(video_file):
    """Get most recent analysis file for specific video and return its contents"""
    if video_file is None:
        return None
        
    s3 = boto3.client('s3')
    
    # Extract base name from video file (remove extension)
    filename = os.path.basename(video_file.name)
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Search for analysis files in the specific video folder
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=f"videos/{base_name}/"
    )
    
    if 'Contents' not in response:
        return None
    
    # Find analysis files for this video with filename prefix
    analysis_files = [
        obj for obj in response['Contents'] 
        if f'{base_name}_analysis_results_' in obj['Key'] and obj['Key'].endswith('.txt')
    ]
    
    if not analysis_files:
        return None
    
    latest_file = max(analysis_files, key=lambda x: x['LastModified'])
    
    # Download and return text file contents
    obj = s3.get_object(Bucket=bucket, Key=latest_file['Key'])
    return obj['Body'].read().decode('utf-8')




def process_video_with_gradio(video_file):
    """Process uploaded video file and return summary"""
    try:
        # Create fresh coordinator agent
        llama4_coordinator_agent = Agent(
        system_prompt="""You are a video processing coordinator. Your job is to process videos step by step.

##When asked to process a video:
1. Extract frames from S3 video using run_frame_extraction
2. Use the frame location from step 1 to run_visual_analysis
3. WAIT for visual analysis to complete sending the json to s3
4. Use the retrieve_json agent to extract the json from step 3
5. Use the text result of retrieve_json_from_s3 by passing it to run_temporal_reasoning
6. Pass both the text result from temporal reasoning result to run_summary_generation for comprehensive analysis
7. Upload analysis generated in run_summary_generation and return s3 location

##IMPORTANT:
- Call ONE tool at a time and wait for the result
- Use the EXACT result from the previous step as input to the next step
- Do NOT call multiple tools simultaneously
- Do NOT return raw JSON or function call syntax
""",
        model=bedrock_model,
        tools=[run_frame_extraction, run_visual_analysis, run_temporal_reasoning, run_summary_generation, upload_text_analysis_results, retrieve_json_from_s3],
        )
        
        # Clear conversation history
        #llama4_coordinator_agent.conversation_history = []
        
        # Upload video to S3
        s3_video_uri = upload_to_sagemaker_bucket(video_file.name)
        print(s3_video_uri)
        # Process with coordinator agent
        video_instruction = f"Process a video from {s3_video_uri}"
        response = llama4_coordinator_agent(video_instruction)

        # Get analysis results
        analysis_data = get_latest_analysis(video_file)
        
        return analysis_data if analysis_data else "Processing failed - no summary generated"
        
    except Exception as e:
        return f"Error processing video: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=process_video_with_gradio,
    inputs=gr.File(label="Upload Video File", file_types=["video"]),
    outputs=gr.Textbox(label="Llama4 Video Analysis Summary", lines=15),
    title="🎬 Video Analysis with Llama4 AI",
    description="Upload a video file to get an AI-generated analysis and summary using Llama4 agents"
)

if __name__ == "__main__":
    interface.launch(share=True)