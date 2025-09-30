from strands import Agent, tool
from strands.models import BedrockModel
import boto3
from typing import List, Optional, Any, Dict
import os
import json
import re
import ast
import argparse

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import date, time


class TravelClaimOut(BaseModel):
    """Final processed travel claim in structured format"""
    Name: str = Field(description="Employee Name")
    Employee_Number: int = Field(description="Employee Number")
    Cost_Center: int = Field(description="Cost Center")
    Division: str = Field(description="Division")
    Expenses: List[Dict[str, Any]] | Any = Field(description="List of expenses")
    Status: str = 'Pending Approval'

boto_session = boto3.session.Session()
region = boto_session.region_name

bedrock_scout_model = BedrockModel(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    streaming=False,
    max_tokens=4096,
    temperature=0.1,
    top_p=0.1,
    region=region
)

@tool
def run_final_sum(emp_name, emp_num, cost_center, division, expenses):
    print("***** Final structured travel claim - Started *****")
    claim_agent = Agent(model=bedrock_scout_model)
    resp = claim_agent.structured_output(TravelClaimOut, f"Extract Info: {emp_name}, {emp_num}, {cost_center}, {division}, {expenses}")
    print("***** Final structured travel claim - Completed *****")
    return resp

                           
    