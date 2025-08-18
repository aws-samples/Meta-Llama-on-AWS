import boto3
import json
import sys
from botocore.config import Config

def call_bedrock(prompt):
    config = Config(
        region_name='us-west-2',
        retries = dict(max_attempts = 3)
    )

    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        config=config
    )

    payload = {
        "prompt": prompt,
        "max_gen_len": 4096,
        "temperature": 0.2,
        "top_p": 0.9
    }

    response = bedrock_runtime.invoke_model(
        modelId='us.meta.llama3-2-11b-instruct-v1:0',
        body=json.dumps(payload)
    )

    response_body = json.loads(response['body'].read())
    generation = response_body['generation']
    
    # Remove instruction tags first
    generation = generation.replace('[INST]', '').replace('[/INST]', '')
    
    # Split into lines and clean up while preserving structure
    lines = generation.split('\n')
    cleaned_lines = []
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            cleaned_lines.append(line)
            seen_lines.add(line)
            # Stop if we have enough content
            if len('\n'.join(cleaned_lines)) > 3000:
                break
    
    # Join lines back with proper spacing
    generation = '\n\n'.join(cleaned_lines)
    
    return generation

if __name__ == '__main__':
    prompt = sys.argv[1]
    result = call_bedrock(prompt)
    print(result)