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

summary_generation_agent = Agent(
    system_prompt="""You create comprehensive video summaries from analysis data.

##Given temporal and visual analysis text input, create a summary that explains:
- What happens in the video
- The sequence of events  
- Key visual elements
- Overall narrative

### Output Format
Provide your summary immediately without any preamble or additional information, following this structure:

**What happens in the video:**
**Chronological Sequence of Events:**
**Sequence of events:**
**Key visual elements:**
**Overall Narrative:**

##Important
- Do not repeat or echo the input frame descriptions. 

Provide a clear, engaging summary.""",

    model=bedrock_model,
    tools=[],
)
