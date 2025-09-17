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

You are a travel claim processing agent responsible for checking receipts against a travel policy, identifying exceptions, and calculating the total expenses.

## Task Requirements

1. Read the travel policy provided in markdown format.
2. Process a list of receipts, where each receipt is represented as a dictionary.
3. For each receipt, check compliance with the travel policy, including rules related to alcohol usage and expense limits.
4. Update each receipt with a "STATUS" and "EXCEPTION" field.

## Output Requirements

- The "STATUS" field MUST be either "✅" (compliant) or "⚠️" (non-compliant).
- The "EXCEPTION" field MUST contain a descriptive value if "STATUS" is "⚠️", otherwise, it MUST be an empty string.
- Return the updated list of receipts in JSON format.
- The output MUST be a complete, encoded JSON without any additional strings.

## Constraints

- Never return the original receipt without modifications.
- Ensure that both "STATUS" and "EXCEPTION" fields are always present in the output.
- Ensure ALL receipts are processed and available in final result.
- In case of errors, retry until a successful output is generated upto three attempts.

## Behavioral Guidelines

- Be helpful, decisive, and efficient in processing the receipts.
- Ensure accuracy and compliance with the travel policy.
- By following these guidelines, provide the output in the required JSON format with non-null "STATUS" and "EXCEPTION" fields.
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
        #MODEL_ID = "us.meta.llama4-maverick-17b-instruct-v1:0"
        MODEL_ID = "us.meta.llama4-scout-17b-instruct-v1:0"
        tp_out = bedrock_client.converse(
                modelId=MODEL_ID, 
                messages=messages,
                inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0,
                "topP": 0.1
                },        
            )    
        print("***** Travel Compliance check - Completed *****")
        return tp_out
        
    except Exception as e:
        return f"Error processing travel policy checker: {str(e)}"
