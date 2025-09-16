import json
import boto3
import os
from PIL import Image
import re
from strands import Agent, tool
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from cur_convert import convert_currency_to_usd
import ast
import json
import time
import io

boto_session = boto3.session.Session()
region = boto_session.region_name
bedrock_client = boto_session.client(
    service_name='bedrock-runtime',
    region_name=region
)

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

prompt = f"""
You are an intelligent OCR extraction agent capable of understanding and processing documents in multiple languages.
- For each invoice or receipt, classify the type of invoice/receipt between FLIGHT_TICKET, HOTEL, MEAL and TAXI; extract all relevant information in structured JSON format only.
- For each receipt/invoice, classify the currency with example value like USD, GBP, EUR, etc.,
- If any field cannot be found in the invoice, return it as null. Focus on clarity and accuracy, and ignore irrelevant text such as watermarks, headers, or decorative elements. 
- The final extracted result MUST be strictly in an encoded JSON without ```json``` and any additional strings.
"""


def make_multi_images_messages(prompt, bucket_name, prefixes):
    img_msg = []
    try:
        for img in prefixes:
            
            response = s3_client.get_object(Bucket=bucket_name, Key=img)
            image_data = response['Body'].read()
            image_stream = io.BytesIO(image_data)
            img_1 = Image.open(image_stream)
            imgformat = img_1.format
            imgformat = imgformat.lower()
            img_msg.append({"image": {
                "format": imgformat,
                        "source": {
                            "bytes": image_data
                        }
                    }
                })
    except FileNotFoundError:
        print(f"Image file not found at {image_paths}")
        image_data = None
        image_media_type = None
   
    messages = [            
            {
                "role": "user",
                "content": [
                {                        
                    "text": prompt
                },
                *img_msg
                ]
            }
        ]
    return messages

def split_list_input(original_list, chunk_size):
    if len(original_list) < chunk_size:
        return [original_list]
    
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]

def parse_out(out):
    objects = []
    for big_str in out:
        app = json.dumps(json.loads(json.dumps(big_str, ensure_ascii=False)), ensure_ascii=False)
        # 1. Unescape the JSON string to get a raw concatenation of objects
        cleaned = json.loads(app)  # removes all the backslashes
        
        #2. Replace \n
        cleaned_rep = cleaned.replace("\n", "")
        
        # 3. Insert a separator between boundaries `}{` if missing
        fixed = re.sub(r'}\s*{', '}\n{', cleaned_rep.strip())
        
        # 3. Split into individual JSON objects (now one per line roughly)
        for block in fixed.splitlines():
            #print(block)
            #print("\n")
            block = block.strip()
            
            if block:
                #print(json.loads(block))
                objects.append(json.loads(block))
    return objects


@tool
def ocr(bucket_name, images: List[str]):
    n_images_per_req = 3
    images_paths_list = split_list_input(images, n_images_per_req)
    MODELS=['us.meta.llama4-scout-17b-instruct-v1:0', 'us.meta.llama4-maverick-17b-instruct-v1:0']
    try:
        result = []
        for images_paths, MODEL_ID in zip(images_paths_list, MODELS):
            messages = make_multi_images_messages(prompt, bucket_name, images_paths)
            # Invoke the SageMaker endpoint
            print("***** OCR - Started *****")
            response = bedrock_client.converse(
                modelId=MODEL_ID, 
                messages=messages,
                inferenceConfig={
                "maxTokens": 2048,
                "temperature": 0,
                "topP": .1
                },        
            )           
            result.append(response['output']['message']['content'][0]['text'])
        
        ocr_response = parse_out(result)
        #print(ocr_response)
        print("***** OCR - Completed *****")
        return ocr_response
    except Exception as e:
        print(f"An error occurred while invoking the endpoint: {str(e)}")


