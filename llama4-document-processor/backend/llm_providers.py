import os
from abc import ABC, abstractmethod
import ollama
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

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
    
    async def generate(self, prompt: str) -> str:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama generation failed: {str(e)}")
    
    async def get_available_models(self) -> List[str]:
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            return [self.model_name]  # Fallback to configured model

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
        # Development environment - use Ollama
        model_name = os.getenv("OLLAMA_MODEL", "llama4")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return OllamaProvider(model_name, base_url)