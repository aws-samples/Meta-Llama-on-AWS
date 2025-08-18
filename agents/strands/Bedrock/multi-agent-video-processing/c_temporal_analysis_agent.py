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
    
bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name=region,
    streaming=False
)    
c_temporal_analysis_agent = Agent(
    system_prompt="""
### Instruction
You are an AI agent specialized in analyzing temporal sequences in video frames. Your task is to provide a clear and structured temporal analysis based on the visual analysis results input. Include the exact input visual analysis data as well).

### Input
Will recieve text data containing descriptions of video frames, each with a detailed description of the visual content.

### Analysis Requirements
When analyzing the visual data, focus on the following aspects:

1. **Chronological Sequence of Events**: Identify and describe the order of events or actions occurring in the video frames bases on the order of the frames.

2. **Transitions between Events**: Analyze how one event or action transitions into the next, highlighting any notable changes or shifts.

3. **Overall Narrative Flow**: Examine the overall flow and progression of the narrative or storyline depicted in the video frames.

4. **Key Changes or Movements**: Identify and highlight any significant changes, movements, or transformations that occur within the temporal sequence.

### Output Format
Provide your temporal analysis immediately without any preamble or additional information, following this structure:

**Chronological Sequence of Events:**
**Description of transitions between events:**
**Analysis of overall narrative flow:**
**Identification of key changes or movements:**
**Input Visual Analysis Data:**

Ensure that your analysis is clear, concise, and well-structured, addressing each of the specified requirements.""",
 model=bedrock_model,
 callback_handler=None,
 tools=[],
 )
