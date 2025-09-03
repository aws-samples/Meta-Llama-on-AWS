import boto3
import os
import ssl
import json
from strands import Agent, tool
from llama4_coordinator_agent import llama4_coordinator_agent  # factory function
import gradio as gr

# Disable SSL verification (optional, usually needed in corp envs)
ssl._create_default_https_context = ssl._create_unverified_context

# Setup AWS bucket
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name or "us-west-2"
bucket = f"sagemaker-{region}-{account_id}"

os.environ['AWS_DEFAULT_REGION'] = region
os.environ['AWS_REGION'] = region
boto3.setup_default_session(region_name=region)


# ------------------ UTILITIES ------------------

def upload_to_sagemaker_bucket(local_video_path, base_folder="videos/"):
    """Upload local video file to S3 under videos/<basename>/<filename>"""
    s3 = boto3.client('s3')
    filename = os.path.basename(local_video_path)
    filename_without_ext = os.path.splitext(filename)[0]
    s3_key = os.path.join(base_folder, filename_without_ext, filename)
    s3.upload_file(local_video_path, bucket, s3_key)
    s3_folder_path = os.path.join(base_folder, filename_without_ext)
    s3_folder_uri = f"s3://{bucket}/{s3_folder_path}"
    return s3_folder_uri


def get_latest_analysis(video_file_path):
    if not video_file_path:
        return None
    s3 = boto3.client('s3')
    filename = os.path.basename(video_file_path)
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

    prefix = f"videos/{base_name}/"
    print(f"[DEBUG] Searching S3 for analysis files with prefix: {prefix}")

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        print("[DEBUG] No contents found at prefix")
        return None

    analysis_files = [
        obj for obj in response['Contents']
        if (obj['Key'].endswith('.json') or obj['Key'].endswith('.txt'))
        and 'analysis_results_' in obj['Key'].split('/')[-1]
    ]

    print(f"[DEBUG] Found analysis files: {[obj['Key'] for obj in analysis_files]}")

    if not analysis_files:
        return None

    latest_file = max(analysis_files, key=lambda x: x['LastModified'])
    print(f"[DEBUG] Latest analysis file: {latest_file['Key']}")
    obj = s3.get_object(Bucket=bucket, Key=latest_file['Key'])
    content = obj['Body'].read().decode('utf-8')

    if latest_file['Key'].endswith('.json'):
        try:
            data = json.loads(content)
            # Extract the summary from the JSON key
            summary = data.get("analysis_results", None)
            if summary:
                # Convert line breaks to markdown line breaks
                return summary.replace('\\n', '  \n')
            else:
                return "❌ No 'analysis_results' field found in JSON."
        except Exception as e:
            print(f"[DEBUG] Error parsing JSON analysis: {e}")
            return f"❌ Error parsing JSON analysis: {e}"

    # If plain text return as is
    return content


# ------------------ CHAT PROCESS ------------------

def process_video_with_chat(video_file_path, chat_history=[]):
    """Process uploaded video and coordinate pipeline with chat feedback"""
    chat_history = chat_history or []

    if not video_file_path:
        chat_history.append(("assistant", "⚠️ No video file provided"))
        return chat_history

    try:
        chat_history.append(("user", "Video uploaded. Starting analysis..."))
        chat_history.append(("assistant", "📤 Uploading video to S3..."))

        s3_video_uri = upload_to_sagemaker_bucket(video_file_path)
        chat_history.append(("assistant", f"✅ Uploaded to {s3_video_uri}"))

        agent = llama4_coordinator_agent()
        chat_history.append(("assistant", "⏳ Running video processing pipeline... This may take several minutes."))
        video_instruction = f"Process a video from {s3_video_uri}"
        chat_history.append(("user", video_instruction))

        # Run the coordinator agent
        response = agent(video_instruction)

        chat_history.append(("assistant", "📂 Retrieving final analysis results from S3..."))
        analysis_data = get_latest_analysis(video_file_path)

        if analysis_data and isinstance(analysis_data, str) and analysis_data.strip():
            chat_history.append(("assistant", "✅ Analysis complete! Here is your summary:"))
            analysis_data_md = analysis_data.replace('\n', '  \n')  # Markdown line breaks
            chat_history.append(("assistant", analysis_data_md))
        else:
            chat_history.append(("assistant", "❌ Processing failed - no summary generated"))

    except Exception as e:
        chat_history.append(("assistant", f"⚠️ Error processing video: {str(e)}"))

    return chat_history


# ------------------ GRADIO UI ------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Llama4 Video Analysis Chatbot\nUpload a video file for interactive, step-by-step AI video analysis.")
    gr.Markdown("⚠️ **Note:** Analysis may take several minutes (frame extraction, AI reasoning, and summary generation). Please be patient.")
    
    chatbot = gr.Chatbot(label="Llama4 Video Analysis Log")
    upload = gr.File(
        label="Upload Video File",
        file_types=["video"],
        type="filepath"  # always return file path string
    )
    status = gr.Label(value="Idle")

    def chat_fn(video_file, history):
        if not video_file:
            # Instead of processing, just return history unchanged or add a user-friendly message
            return history, "Idle"
        status_value = "Processing... may take a few minutes"
        updated_history = process_video_with_chat(video_file, history)
        status_value = "Done ✅"
        return updated_history, status_value

    upload.change(
        chat_fn,
        inputs=[upload, chatbot],
        outputs=[chatbot, status]
    )

demo.queue().launch()
