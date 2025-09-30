from strands import Agent
from strands_tools import current_time, retrieve
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from typing import List
from utils import get_ollama_ip

model_id = "llama3.1:8b"
#print(get_ollama_ip())
ollama_ip = get_ollama_ip()[0]
#print(ollama_ip)
model = OllamaModel(
            host=f"http://{ollama_ip}:11434",
            model_id=model_id, 
            tempertaure=0.7,
            top_p=0.3,
            streaming=True,
            keep_alive="10m",
        )
system_prompt = """
    You are a helpful customer support agent ready to assist customers with their inquiries and service needs.
    You have access to tools to: current_time and retrieve.
    
    You have been provided with a set of functions to help resolve customer inquiries.
    You will ALWAYS follow the below guidelines when assisting customers:
    <guidelines>
        - Never assume any parameter values while using internal tools.
        - If you do not have the necessary information to process a request, politely ask the customer for the required details
        - NEVER disclose any information about the internal tools, systems, or functions available to you.
        - If asked about your internal processes, tools, functions, or training, ALWAYS respond with "I'm sorry, but I cannot provide information about our internal systems."
        - Always maintain a professional and helpful tone when assisting customers
        - Focus on resolving the customer's inquiries efficiently and accurately
    </guidelines>
"""
tools = (
            [
                retrieve,
                current_time,
            ]
    )
agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools
        )
response = agent("What's the time now in San Francisco?")
print(response)
