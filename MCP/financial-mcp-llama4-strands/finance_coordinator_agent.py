import boto3
import os
from strands.models import BedrockModel
from strands import Agent
from config import AWS_CONFIG

# Setup AWS
session = boto3.session.Session()
region = session.region_name or AWS_CONFIG["region"]
os.environ['AWS_DEFAULT_REGION'] = region
os.environ['AWS_REGION'] = region
boto3.setup_default_session(region_name=region)

bedrock_model = BedrockModel(
    model_id=AWS_CONFIG["model_id"],
    region_name=region,
    streaming=False,
    timeout=120
)

def new_finance_coordinator_agent() -> Agent:
    """Factory function to create a new finance coordinator agent"""
    return Agent(
        model=bedrock_model,
        system_prompt="You are a financial analysis coordinator.",
        tools=[]
    )

