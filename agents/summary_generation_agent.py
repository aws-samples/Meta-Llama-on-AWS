import boto3
import os
from strands import Agent, tool
from strands.models import BedrockModel

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['AWS_REGION'] = 'us-east-1'
boto3.setup_default_session(region_name='us-east-1')

bedrock_model = BedrockModel(
    model_id='us.meta.llama4-maverick-17b-instruct-v1:0',
    region_name='us-east-1',
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
