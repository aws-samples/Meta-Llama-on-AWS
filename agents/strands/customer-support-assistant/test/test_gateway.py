#!/usr/bin/python

import asyncio
import click
from bedrock_agentcore.identity.auth import requires_access_token
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from strands.models.ollama import OllamaModel
from utils import get_ollama_ip
from strands import Agent
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import get_ssm_parameter

gateway_access_token = None

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

ollama_ip = setup_ollama_ip()

model = OllamaModel(
    host=f"http://{ollama_ip}:11434",
    model_id="llama3.1:8b",
    temperature=0.9,
    top_p=0.3,
    streaming=True,
    keep_alive="15m"
)

@requires_access_token(
    provider_name=get_ssm_parameter("/app/customersupport/agentcore/cognito_provider"),
    scopes=[],  # Optional unless required
    auth_flow="M2M",
)
async def _get_access_token_manually(*, access_token: str):
    global gateway_access_token
    gateway_access_token = access_token
    return access_token


@click.command()
@click.option("--prompt", "-p", required=True, help="Prompt to send to the MCP agent")
def main(prompt: str):
    """CLI tool to interact with an MCP Agent using a prompt."""

    # Fetch access token
    asyncio.run(_get_access_token_manually(access_token=""))

    # Load gateway configuration from SSM parameters
    try:
        gateway_url = get_ssm_parameter("/app/customersupport/agentcore/gateway_url")
    except Exception as e:
        print(f"❌ Error reading gateway URL from SSM: {str(e)}")
        sys.exit(1)

    print(f"Gateway Endpoint - MCP URL: {gateway_url}")

    # Set up MCP client
    client = MCPClient(
        lambda: streamablehttp_client(
            gateway_url,
            headers={"Authorization": f"Bearer {gateway_access_token}"},
        )
    )

    with client:
        agent = Agent(model=model, tools=client.list_tools_sync())
        response = agent(prompt)
        print(str(response))


if __name__ == "__main__":
    main()
