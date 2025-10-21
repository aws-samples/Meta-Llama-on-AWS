import os
import re
import json
from strands import Agent, tool
import boto3
from strands.models import BedrockModel
from pydantic import BaseModel, Field
from typing import Optional, List
import ast
from typing import List, Dict

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)


TP_SYSTEM_PROMPT = """
Task: Travel Claim Processing

You are a travel claim processing agent responsible for checking receipts against the provided travel policy and identifying ALL policy violations.

## Systematic Checking Process

For EACH receipt, you MUST perform ALL applicable checks below:

### Hotel Receipt Checks:
1. Calculate nightly rate (total room charges ÷ number of nights)
2. Compare nightly rate against policy limits for appropriate city tier
3. Identify and sum ALL incidental charges (room service, minibar, laundry, movies, etc.)
4. Compare total incidentals against policy limit ($75/stay)

### Meal Receipt Checks:
1. Identify ALL alcohol items and calculate total alcohol cost
2. Calculate alcohol percentage of total meal cost (alcohol cost ÷ total meal cost × 100)
3. Check if alcohol percentage exceeds 20% limit
4. Compare total meal cost against daily meal allowance limits for appropriate tier
5. Perform BOTH alcohol and meal allowance checks - they are separate violations

### Transportation Receipt Checks:
1. Compare cost against daily limits for the transport type
2. Verify service type compliance (e.g., standard service only for taxi/rideshare)

## Output Requirements

- The "STATUS" field MUST be either "✅" (compliant) or "⚠️" (non-compliant).
- The "EXCEPTION" field MUST contain a detailed violation description if "STATUS" is "⚠️", otherwise empty string.
- Return the complete updated list of receipts in JSON format only.
- Do NOT return receipts without STATUS and EXCEPTION fields added.

## Critical Instructions

- Process receipts ONE BY ONE and perform ALL applicable checks from the systematic process above
- Calculate exact amounts and percentages - show your work in the exception description
- NEVER skip any check - even if one violation is found, continue checking for additional violations
- Multiple violations can exist in a single receipt - list ALL violations found
- Be consistent - apply the same checking methodology to every receipt of the same type
"""

boto_session = boto3.session.Session()
region = boto_session.region_name

bedrock_client = boto_session.client(
    service_name='bedrock-runtime',
    region_name=region
)

@tool
def make_messages(travel_policy, receipts):
    
    messages = [            
            {
                "role": "user",
                "content": [
                {                        
                    "text": f"{TP_SYSTEM_PROMPT} Travel Policy: {travel_policy} Receipts: {receipts}" 
                }
                ]
            }
        ]
    return messages

def read_text_from_file(file_path):
    """Reads the content of a text file and returns it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


@tool
def policy_compliant_check(query: str) -> str:
    """
    Validate if the receipt is compliant as per travel policy
    """
    # Format the query for the math agent with clear instructions
    policy = read_text_from_file(f"{script_dir}/travel_policy.txt")
    
    formatted_query = f"Please check if the given receipts are compliant as per travel policy: {policy} for {query}"
    try:
        print("***** Travel Compliance check - Started *****")        
        messages = make_messages(policy, query)
        MODEL_ID = "us.meta.llama4-maverick-17b-instruct-v1:0"
        #MODEL_ID = "us.meta.llama4-scout-17b-instruct-v1:0"
        tp_out = bedrock_client.converse(
                modelId=MODEL_ID, 
                messages=messages,
                inferenceConfig={
                "maxTokens": 8192,
                "temperature": 0,
                "topP": 0.95
                },        
            )    
        print("***** Travel Compliance check - Completed *****")
        return tp_out
        
    except Exception as e:
        return f"Error processing travel policy checker: {str(e)}"