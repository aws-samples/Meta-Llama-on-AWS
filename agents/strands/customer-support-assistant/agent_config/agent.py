from .utils import get_ssm_parameter
from .utils import get_ollama_ip
from .memory_hook_provider import MemoryHook
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands_tools import current_time, retrieve
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from typing import List
import os

def setup_ollama_ip():
    """
    Check for existence of a file named .ollama_ip, if it exists, read and assign
    its content to ollama_ip variable. If it doesn't exist, create it by running
    get_ollama_ip()[0] and storing the result.
    
    Returns:
        str: The IP address of Ollama
    """
    ip_file_path = ".ollama_ip"
    
    # Check if file exists
    if os.path.isfile(ip_file_path):
        # Read the IP from the file
        with open(ip_file_path, "r") as file:
            ollama_ip = file.read().strip()
        print(f"Found Ollama IP in file: {ollama_ip}")
    else:
        # Get the IP by calling the function
        ollama_ip = get_ollama_ip()[0]
        
        # Write the IP to the file
        with open(ip_file_path, "w") as file:
            file.write(ollama_ip)
        print(f"Created .ollama_ip file with IP: {ollama_ip}")
    
    return ollama_ip
# Usage
ollama_ip = setup_ollama_ip()

class CustomerSupport:
    def __init__(
        self,
        bearer_token: str,
        memory_hook: MemoryHook,
        system_prompt: str = None,
        ollama_ip: str = ollama_ip,
        tools: List[callable] = None,
    ):
        self.ollama_ip = ollama_ip
        self.model_id = "llama3.1:8b"
        self.model = OllamaModel(
            host=f"http://{self.ollama_ip}:11434",
            model_id=self.model_id, 
            temperature=0.9,
            top_p=0.3,
            streaming=True,
            keep_alive="30m",
        )
        self.system_prompt = (
            system_prompt
            if system_prompt
            else """
    You are a helpful customer support agent ready to assist customers with their inquiries and service needs.
    You have access to tools to: check warrant status, view customer profiles, and retrieve Knowledgebase.
    
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
        )

        gateway_url = get_ssm_parameter("/app/customersupport/agentcore/gateway_url")
        print(f"Gateway Endpoint - MCP URL: {gateway_url}")

        try:
            self.gateway_client = MCPClient(
                lambda: streamablehttp_client(
                    gateway_url,
                    headers={"Authorization": f"Bearer {bearer_token}"},
                )
            )

            self.gateway_client.start()
        except Exception as e:
            raise f"Error initializing agent: {str(e)}"

        self.tools = (
            [
                retrieve,
                current_time,
            ]
            + self.gateway_client.list_tools_sync()
            + tools
        )

        self.memory_hook = memory_hook

        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            hooks=[self.memory_hook]
        )

    def invoke(self, user_query: str):
        try:
            response = str(self.agent(user_query))
            print(str(response))
        except Exception as e:
            return f"Error invoking agent: {e}"
        return response

    async def stream(self, user_query: str):
        try:

            async for event in self.agent.stream_async(user_query):
                if "data" in event:
                    # Only stream text chunks to the client
                    yield event["data"]

        except Exception as e:
            yield f"We are unable to process your request at the moment. Error: {e}"
