# Finance Agent

## Introduction

This a finance agent showcasing capabilities of distilled Llama 8b using LangChain framework.

## Installation and running

Follow below steps:

1. Install uv

1. Set required environment variables explain on top of the file: 
`export TAVILY_API_KEY="MY_TAVILY_API_KEY"`

1. Run the agent. Dependencies would be installed if this is the first time running below command:
```console
uv run fin-agent-llama-api.py
```

To test MCP server ask the agent: 

`What is price of coffee today?`