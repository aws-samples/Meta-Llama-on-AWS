# Instructions for deploying multi-agent video processing solution using AWS Strands and Llama 4

## Recipe Overview <a name="overview"></a>

This tutorial shows a step-by-step solution on deploying a multi-agent video processing application on a Gradio UI using AWS Strands and Llama 4. This solution will leverage a llama4_coordinator agent that interacts with other agents to extract frames out of videos, upload the frames to s3 and understand the temporal analysis of each of the extracted frames and summarize it. 

## 

## Requirements

- AWS Strands
- Python 3.9+
- An AWS Account with Bedrock Access

## Installation
1. Clone the Meta-Llama-on-AWS Github Repo

- git clone https://github.com/aws-samples/Meta-Llama-on-AWS.git
- cd agents/strands/Multi-agent-video-processing-app

In terminal install correct dependencies:
pip install pip boto3==1.39.0 opencv-python==4.11.0.86 strands pillow==11.2.1 urllib3 gradio strands-agents==0.1.9 strands-agents-builder==0.1.4 strands-agents-tools==0.1.7

2. In the gradio_app.py file, add your s3 bucket that you will use to store and process your videos.

Run the Gradio UI:

python3 gradio_app.py

This will provide a local URL: http://127.0.0.1:7861 or a public URL like: https://9053c244410d26f679.gradio.live that will allow you to input a solution.

Once you have the gradio app up and running, you can now test your own videos or use sample videos provided in the repo!

## Detailed Outline for running notebook: 

In this notebook, you will need to import all the necessary libaries and tools that were created via AWS Strands.

Once all libraries are imported, you will need to:
- set your AWS region
- load videos from the local path
- upload the video to SageMaker default bucket

## Agentic Workflow

To start the agentic workflow, you will leverage the llama4_coordinater_agent, which is designed to manage and automate a multi-step video analyssi workflow using several specialized agents and tools.

- llama4_coordinator_agent: orchestrates the workflow
- s_visual_analysis_agent: analyzes the images and returns it in json
- c_temporal_analysis_agent: analyzes the temporal sequences in video frames
- retrieve_json_agent: helps with additional retrieval from s3 to give to temporal analysis agent
- summary_generation_agent: takes the temporal analysis input and generates summary

## Step-by-step workflow

**Frame Extraction:**
It starts by extracting video frames from the specified S3 video using the run_frame_extraction tool.

**Visual Analysis:**
The agent then takes the location of the extracted frames and performs visual analysis using the run_visual_analysis tool.

Wait for JSON Output:
It waits for the visual analysis to complete and store the results as a JSON file in S3.

**Fetch Visual Analysis Data:**
The agent retrieves the JSON analysis file from S3 using the retrieve_json_from_s3 tool.

**Temporal Reasoning:**
The retrieved JSON results are input into a temporal reasoning process (using the run_temporal_reasoning tool) which analyzes sequences and changes over time within the video.

**Summary Generation:**
The result of the temporal reasoning step is then used to generate a final summary using the run_summary_generation tool.

**Upload Final Analysis:**
Finally, the generated summary is uploaded as a JSON file back to the S3 bucket using the upload_analysis_results tool, and the agent returns the location of the analysis result in S3.

