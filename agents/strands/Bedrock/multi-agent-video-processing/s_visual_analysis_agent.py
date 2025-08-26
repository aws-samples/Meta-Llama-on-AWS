from datetime import datetime
from random import randint, random
import boto3
import os
import base64
import ssl
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from strands import Agent, tool
from strands.models import BedrockModel
from urllib.parse import urlparse

ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    streaming=False,
    verify=False
)

@tool
def list_s3_frames(s3_folder_path: str) -> str:
    """List all frames in an S3 folder path (format: s3://bucket-name/folder/path/)"""
    try:
        parsed = urlparse(s3_folder_path)
        if parsed.scheme != 's3':
            return "Error: Path must be an S3 URL starting with s3://"
            
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        # print(f"Listing objects in bucket={bucket}, prefix={prefix}")
        
        s3 = boto3.client('s3', verify=False)
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            # print(f"No contents found in {bucket}/{prefix}")
            return json.dumps({"bucket": bucket, "frame_urls": []})
        
        # print(f"Found {len(response['Contents'])} objects")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        frame_urls = [
            f"s3://{bucket}/{obj['Key']}"
            for obj in response['Contents']
            if obj['Key'].lower().endswith(image_extensions)
        ]
        
        import re
        def natural_sort_key(url):
            match = re.search(r'frame_([0-9]+)', url)
            return int(match.group(1)) if match else 0
        
        frame_urls.sort(key=natural_sort_key)
        
        # print(f"Filtered to {len(frame_urls)} image files (sorted by frame number)")
        
        return json.dumps({"bucket": bucket, "frame_urls": frame_urls})
        
    except Exception as e:
        return f"Error listing frames: {str(e)} - Type: {type(e).__name__}"

@tool
def analyze_image(image_url: str, max_retries: int = 3) -> str:
    """Analyze a single image from S3 URL with exponential backoff for throttling"""
    for attempt in range(max_retries + 1):
        try:
            parsed = urlparse(image_url)
            if parsed.scheme != 's3':
                return "Error: Path must be an S3 URL starting with s3://"
            
            bucket = parsed.netloc
            s3_key = parsed.path.lstrip('/')
            # print(f"Downloading {bucket}/{s3_key} (attempt {attempt + 1})")
            
            s3 = boto3.client('s3', verify=False)
            local_file = f"temp_frame_{randint(1000, 9999)}.jpg"
            try:
                s3.download_file(bucket, s3_key, local_file)
            except Exception as s3_error:
                return f"Error: Cannot download {image_url}: {str(s3_error)}"
            
            # Validate file exists and has content
            if not os.path.exists(local_file) or os.path.getsize(local_file) == 0:
                if os.path.exists(local_file):
                    os.remove(local_file)
                return f"Error: Downloaded file is empty or corrupted for {image_url}"
            
            try:
                with Image.open(local_file) as img:
                    img.load()  # Force load to catch truncated images
                    width, height = img.size
                    current_pixels = width * height
                    max_pixels = 262144
                    
                    if current_pixels > max_pixels:
                        # print(f"Resizing image from {width}x{height} ({current_pixels} pixels)")
                        ratio = (max_pixels / current_pixels) ** 0.5
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        
                        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        resized_img.save(local_file, "JPEG", quality=85, optimize=True)
                        # print(f"Resized to {new_width}x{new_height} ({new_width * new_height} pixels)")
            except Exception as img_error:
                if os.path.exists(local_file):
                    os.remove(local_file)
                if "truncated" in str(img_error).lower():
                    return f"Error: Image file is corrupted/truncated for {image_url}"
                return f"Error: Cannot process image file for {image_url}: {str(img_error)}"
            
            bedrock = boto3.client('bedrock-runtime', region_name='us-east-1', verify=False)
            with open(local_file, 'rb') as f:
                file_bytes = f.read()
                if len(file_bytes) == 0:
                    return f"Error: Image file is empty after processing for {image_url}"
                image_data = base64.b64encode(file_bytes).decode('utf-8')
            
            import re
            frame_match = re.search(r'frame_([0-9]+)', image_url)
            frame_number = int(frame_match.group(1)) if frame_match else 0
            
            response = bedrock.converse(
                modelId='us.meta.llama4-maverick-17b-instruct-v1:0',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'text': "Describe this image in detail. What objects, people, actions, and settings do you see? What is the overall context or scene depicted? Limit response to 2 sentences."},
                         {
                            'image': {
                                'format': 'jpeg',
                                'source': {'bytes': base64.b64decode(image_data)}
                            }
                        }
                    ]
                }]
            )
            
            raw_analysis = response['output']['message']['content'][0]['text']
            analysis = f"{frame_number}. {raw_analysis}"
            
            frame_match = re.search(r'frame_([0-9]+)', image_url)
            frame_number = int(frame_match.group(1)) if frame_match else 0
            
            result = json.dumps({
                "image_url": image_url,
                "analysis": analysis,
                "frame_number": frame_number
            })
            
            # Clean up temp file after successful analysis
            if os.path.exists(local_file):
                os.remove(local_file)
            
            return result
            
        except Exception as e:
            if os.path.exists(local_file):
                os.remove(local_file)
            
            if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
                if attempt < max_retries:
                    delay = (2 ** attempt) + (attempt * 0.5)
                    # print(f"Throttling detected, waiting {delay:.1f}s before retry {attempt + 2}")
                    time.sleep(delay)
                    continue
                else:
                    return f"Error: Max retries exceeded due to throttling for {image_url}"
            else:
                return f"Error analyzing image {image_url}: {str(e)} - Type: {type(e).__name__}"
    
    return f"Error: Unexpected failure after {max_retries} retries for {image_url}"

async def analyze_image_async(image_url: str, semaphore: asyncio.Semaphore, executor: ThreadPoolExecutor) -> dict:
    """Async wrapper for image analysis with concurrency control"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analyze_image, image_url)
        
        if result.startswith("Error"):
            return {"image_url": image_url, "analysis": f"Failed: {result}"}
        else:
            return json.loads(result)

async def process_frames_async(frame_urls: list, max_concurrent: int = 2) -> list:
    """Process multiple frames asynchronously with controlled concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        tasks = [
            analyze_image_async(frame_url, semaphore, executor)
            for frame_url in frame_urls
        ]
        
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            # print(f"Completed {i+1}/{len(frame_urls)} frames")
            
            if i < len(frame_urls) - 1:
                await asyncio.sleep(0.5)
        
        return results

@tool
def analyze_all_frames(s3_folder_path: str, max_concurrent: int = 2) -> str:
    """Analyze all frames in an S3 folder asynchronously to prevent throttling"""
    try:
        frames_result = list_s3_frames(s3_folder_path)
        if frames_result.startswith("Error"):
            return frames_result
        
        frames_data = json.loads(frames_result)
        if not frames_data["frame_urls"]:
            return "No frames found in the specified S3 folder."
        
        # print(f"Processing {len(frames_data['frame_urls'])} frames with max {max_concurrent} concurrent requests")
        
        results = asyncio.run(process_frames_async(frames_data["frame_urls"], max_concurrent))
        
        # Sort results by frame number and save
        results.sort(key=lambda x: x.get('frame_number', 0))
        final_results = {"folder": s3_folder_path, "analyses": results}
        save_analysis_to_local_json(json.dumps(final_results))
        
        return json.dumps(final_results)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def analyze_frames_batch(s3_folder_path: str, batch_size: int = 3, delay_between_batches: float = 5.0) -> str:
    """Process frames in batches to minimize throttling risk"""
    try:
        frames_result = list_s3_frames(s3_folder_path)
        if frames_result.startswith("Error"):
            return frames_result
        
        frames_data = json.loads(frames_result)
        if not frames_data["frame_urls"]:
            return "No frames found in the specified S3 folder."
        
        frame_urls = frames_data["frame_urls"]
        all_results = []
        
        for i in range(0, len(frame_urls), batch_size):
            batch = frame_urls[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(frame_urls) + batch_size - 1) // batch_size
            
            try:
                batch_results = asyncio.run(process_frames_async(batch, max_concurrent=2))
            except Exception as async_error:
                return f"Error in async processing: {str(async_error)}"
            all_results.extend(batch_results)
            
            if i + batch_size < len(frame_urls):
                # print(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        # Sort results by frame number and save
        all_results.sort(key=lambda x: x.get('frame_number', 0))
        final_results = {"folder": s3_folder_path, "analyses": all_results}
        save_analysis_to_local_json(json.dumps(final_results))
        
        return json.dumps(final_results)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def save_analysis_to_local_json(results: str, filename: str = "visual_analysis_results.json") -> str:
    """Save analysis results to local JSON file"""
    try:
        if isinstance(results, str):
            try:
                data = json.loads(results)
            except json.JSONDecodeError:
                data = {"analysis_results": results, "timestamp": datetime.now().isoformat()}
        else:
            data = results
        
        # Add timestamp to data
        data["timestamp"] = datetime.now().isoformat()
        
        # Save data directly (no session appending)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Analysis results saved to {filename}"
    except Exception as e:
        return f"Error saving results: {str(e)}"

@tool
def upload_local_json_to_s3(s3_video_path: str, local_filename: str = "visual_analysis_results.json") -> str:
    """Upload local JSON file to S3 bucket in video folder"""
    try:
        s3_parts = [part for part in s3_video_path.replace('s3://', '').split('/') if part]
        bucket = s3_parts[0]
        video_folder = s3_parts[-1]
        
        if '_' in video_folder:
            base_video_name = video_folder.split('_')[0]
        else:
            base_video_name = video_folder
        random_num = randint(1000, 9999)
        
        s3_key = f"videos/{base_video_name}/{random_num}_{local_filename}"
        
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_filename, bucket, s3_key)
        
        return f"s3://{bucket}/{s3_key}"
    except Exception as e:
        return f"Error uploading file: {str(e)}"

s_visual_analysis_agent = Agent(
    system_prompt="""You are an image analysis agent that processes frames from S3 buckets.

Your workflow:
1. Use the available tools to analyze images
2. Use the video path folder to place the analysis results

IMPORTANT:
- Do NOT generate, write, or return any code
- Focus on describing what you see in the images
- Images are automatically resized if too large
- Put numbered labels in front of each image description (e.g., "1. ", "2. ", etc.)
- Always save analysis results locally first, then upload to S3

Return Format:
The uri from the upload_local_json_to_s3 tool""",
    model=bedrock_model,
    callback_handler=None,

    tools=[list_s3_frames, analyze_image, analyze_all_frames, analyze_frames_batch, upload_local_json_to_s3],
)