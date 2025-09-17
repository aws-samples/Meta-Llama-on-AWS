from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
import boto3
from ocr_agent import ocr
from cur_convert import convert_currency_to_usd
from policy_checker import policy_compliant_check
from structured_out import run_final_sum
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import os
import json
import re
import ast
import argparse
import pprint

app = BedrockAgentCoreApp()

boto_session = boto3.session.Session()
region = boto_session.region_name
s3_resource = boto3.resource('s3')
bedrock_maverick_model = BedrockModel(
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
    streaming=False,
    max_tokens=4096,
    temperature=0.,
    top_p=0.1,
    region=region
)

instruction = """
Task: Process invoices or receipts to extract and create expense reports.

Step-by-Step Instructions:
Extract Text: Use the 'ocr' tool to extract text from the provided image.
Convert Currency: Use the 'convert_currency_to_usd' tool to convert all currencies to USD. Ensure all currencies are converted and rename "total_due" and "totalAmount" to "total".
Policy Compliance Check: Use the 'policy_compliant_check' tool to analyze each receipt for compliance with the given travel policy. Capture NON-COMPLIANT details.
Return Final Result: Provide the final result in an encoded, usable JSON format without any additional strings or markers.

Critical Guidelines:
Sequential Tool Usage: Use one tool at a time and wait for the result before proceeding to the next step.
Exact Result Usage: Use the exact result from the previous step as input for the next tool. Do not make assumptions or guesses.
No Simultaneous Tool Calls: Avoid calling multiple tools simultaneously.
JSON Output: Return a complete, encoded JSON without json or any additional strings.
Error Handling: Retry a tool up to 3 times in case of errors or failures. Transition directly to providing the final answer after successful tool execution.
Direct Transition: Move directly to providing the final answer after all tools have been executed.

Behavioral Expectations:
Tool Result Utilization: Use tool results to construct a comprehensive response.
No Re-queries: Avoid second-guessing or re-querying tools unnecessarily.
Currency Conversion: Always use the 'convert_currency_to_usd' tool for currency conversions.
Policy Compliance: Always use the 'policy_compliant_check' tool for policy compliance checks.
Final Response: Provide the final response only after running all the tools.

Output Requirements:
Usable JSON: Return a usable list of dictionaries in JSON format.
No Raw JSON or Scripts: Avoid returning raw JSON, function call syntax, function recommendations or Python scripts.

Agent Traits:
Helpful: Provide accurate and comprehensive responses.
Decisive: Make decisions based on tool results.
Efficient: Execute tasks in a sequential and timely manner.
"""

def calc_total(expenses):
    keys = ["total", "TOTAL_USD", "total_due", "total_USD"]
    total_sum = 0
    if type(expenses['Expenses']) is str:
        expenses['Expenses'] = json.loads(expenses['Expenses'])
    for entry in expenses['Expenses']:
        if "STATUS" in entry:
            del entry["STATUS"]
        for k in keys:
            if k in entry:
                if type(entry[k]) is str:
                    entr_k = float(entry[k].replace("$", "").replace(",", ""))
                    total_sum += entr_k
                else:
                    total_sum += entry[k]
                break  # Stop after the first matching key to avoid duplicates
    total_sum = round(total_sum, 2)
    
    except_keys = ["compliance_status", "EXCEPTION", "non_compliance_reason", "non_compliant_details"]
    exceptions = []
    for entry in expenses['Expenses']:
        for ek in except_keys:
            if ek in entry and entry[ek] != "":
                exceptions.append(entry[ek])      
    expenses["TOTAL_DUE_TO_COMPANY"] = total_sum
    expenses["EXCEPTIONS_SUMMARY"] = exceptions
    if len(expenses["EXCEPTIONS_SUMMARY"]) > 0:
        expenses["COMPLIANT_STATUS"] = "⚠️"
    else:
        expenses["COMPLIANT_STATUS"] = "✅"
    return expenses
    
def expense_processor(bucket_name, images, emp_name, emp_num, cost_center, division):
    agent = Agent(model=bedrock_maverick_model, tools=[ocr, convert_currency_to_usd, policy_compliant_check], system_prompt=instruction)
    query = f"Extract and process the given invoice or receipts using {bucket_name}, {images}"
    response = agent(query)
    agent_response = json.dumps(response.message["content"][0]["text"], ensure_ascii=False)
    structured_out = run_final_sum(emp_name, emp_num, cost_center, division, agent_response)
    structured_out_json = structured_out.model_dump()
    structured_out_json_with_total = calc_total(structured_out_json)
    # Pretty print the entire JSON structure with indentation
    print(json.dumps(structured_out_json_with_total, indent=4, ensure_ascii=False))
    return structured_out_json_with_total

@app.entrypoint
def run_expense_processor(payload):
    images_prefix = payload.get("images_prefix")
    bucket = payload.get("bucket")
    emp_first_name = payload.get("emp_first_name")
    emp_last_name = payload.get("emp_last_name")
    emp_name = f"{emp_first_name} {emp_last_name}"
    emp_num = payload.get("emp_num")
    cost_center = payload.get("cost_center")
    division = payload.get("division")
    bucket_name = s3_resource.Bucket(bucket)
    images = []
    for obj in bucket_name.objects.filter(Prefix=images_prefix):
        images.append(obj.key)
    images.pop(0)
    expenses_claim_report = expense_processor(bucket_name, images, emp_name, emp_num, cost_center, division)
    return expenses_claim_report

                           
if __name__ == "__main__":
    print("**************** Processing Expense Claim with Llama4 Scout and Maverick on Amazon Bedrock with AWS Strands ****************")
    app.run()
    