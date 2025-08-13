import os
from abc import ABC, abstractmethod
import boto3
import json
from typing import List, Dict

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        pass

class BedrockProvider(LLMProvider):
    def __init__(self, model_id: str = "us.meta.llama4-scout-17b-instruct-v1:0", region: str = "us-west-2"):
        self.model_id = model_id
        self.region = region
        self.client = boto3.client('bedrock-runtime', region_name=region)
    
    async def generate(self, prompt: str) -> str:
        try:
            payload = {
                "prompt": prompt,
                "max_gen_len": 2048,
                "temperature": 0.1,
                "top_p": 0.9
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )

            result = json.loads(response['body'].read().decode())
            return result['generation']

        except Exception as e:
            raise Exception(f"Bedrock generation failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        return [self.model_id]

class SageMakerProvider(LLMProvider):
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        self.endpoint_name = endpoint_name
        self.region = region
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    async def generate(self, prompt: str) -> str:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True
                }            
            }

            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read().decode())
            return result[0]['generated_text']

        except Exception as e:
            raise Exception(f"SageMaker generation failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        return [self.endpoint_name]

def get_llm_provider() -> LLMProvider:
    """Factory function to get the appropriate LLM provider based on environment"""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME")
        region = os.getenv("AWS_REGION", "us-east-1")

        if not endpoint_name:
            raise ValueError("SAGEMAKER_ENDPOINT_NAME must be set for production environment")

        return SageMakerProvider(endpoint_name, region)
    else:
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama4-scout-17b-instruct-v1:0")
        region = os.getenv("AWS_REGION", "us-west-2")

        return BedrockProvider(model_id, region)

