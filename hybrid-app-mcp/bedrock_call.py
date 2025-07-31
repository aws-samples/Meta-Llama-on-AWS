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
        "max_gen_len": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = bedrock_runtime.invoke_model(
        modelId='us.meta.llama3-2-11b-instruct-v1:0',
        body=json.dumps(payload)
    )

    response_body = json.loads(response['body'].read())
    return response_body['generation']

if __name__ == '__main__':
    prompt = sys.argv[1]
    result = call_bedrock(prompt)
    print(result)