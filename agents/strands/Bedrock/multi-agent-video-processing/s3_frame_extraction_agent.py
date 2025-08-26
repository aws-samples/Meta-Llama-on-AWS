import boto3
import os
import cv2
import ssl
import io
from concurrent.futures import ThreadPoolExecutor
from strands import Agent, tool
from strands.models import BedrockModel
import logging

##** User Changes:**##
## The following number of frames to decide for your video use case. Ex: 10-30 frames

number_of_frames_selected=15

cv2.setLogLevel(0)  # Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('strands').setLevel(logging.WARNING)
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

def upload_frame_to_s3(frame_data, bucket_name, s3_key):
    """Upload frame data to S3"""
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=frame_data)
    return f"s3://{bucket_name}/{s3_key}"

@tool
def extract_frames_from_s3(bucket_name: str, folder_path: str, max_frames: int = number_of_frames_selected) -> str:
    """Extract frames from video in S3 bucket using optimized seeking and parallel uploads"""
    s3 = boto3.client('s3')
    
    # List videos in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    video_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        return f"No video files found in s3://{bucket_name}/{folder_path}"
    
    video_key = video_files[0]
    
    # Download full video
    local_video = "temp_video.mp4"
    s3.download_file(bucket_name, video_key, local_video)
    
    # Get video metadata
    cap = cv2.VideoCapture(local_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Calculate target frame positions
    if total_frames <= max_frames:
        target_frames = list(range(total_frames))
    else:
        interval = total_frames // max_frames
        target_frames = [i * interval for i in range(max_frames)]
    
    # Extract frames using seeking
    cap = cv2.VideoCapture(local_video)
    frame_data_list = []
    
    for i, frame_pos in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            # Encode frame to JPEG in memory
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_data = buffer.tobytes()
            frame_data_list.append((f"frame_{i+1}.jpg", frame_data))
    
    cap.release()
    os.remove(local_video)
    
    # Upload frames to S3 sequentially
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_folder = f"{folder_path}_frames_{date_str}"
    
    uploaded_frames = []
    for frame_name, frame_data in frame_data_list:
        s3_key = f"{frames_folder}/{frame_name}"
        frame_url = upload_frame_to_s3(frame_data, bucket_name, s3_key)
        uploaded_frames.append(frame_url)
    
    # Return only the S3 folder path
    return f"s3://{bucket_name}/{frames_folder}"

@tool
def get_frames_folder_path(extraction_result: str) -> str:
    """Extract the S3 folder path from frame extraction result"""
    try:
        import json
        data = json.loads(extraction_result)
        return f"s3://{data['bucket']}/{data['frames_folder']}"
    except:
        # If it's already a path, return as-is
        if extraction_result.startswith('s3://'):
            return extraction_result
        return f"Error parsing result: {extraction_result}"

s3_frame_extraction_agent = Agent(
    system_prompt="""
Your role:
- Use extract_frames_from_s3 tool to extract frames
- Return ONLY the S3 folder path result from the tool
- Do NOT add any additional text or formatting

CRITICAL: Call each tool only ONCE. Do not retry tools that return valid results.


""",
    model=bedrock_model,
    callback_handler=None,
    tools=[extract_frames_from_s3],
)