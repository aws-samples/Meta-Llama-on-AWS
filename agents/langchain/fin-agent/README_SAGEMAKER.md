# SageMaker Llama 3 Function Calling - Financial Agent

Financial agent using Llama 3 8B on AWS SageMaker with parallel tool calling capabilities.

## Features

- **Parallel Tool Calling**: Call up to 5+ tools simultaneously (60-80% fewer API calls)
- **Multi-Step Reasoning**: Handle complex queries requiring sequential tool calls
- **LMI Container**: Purpose-built for agent workflows with vLLM backend
- **LangGraph Integration**: Compatible with existing LangChain patterns

## Quick Start

### 1. Get HuggingFace Token

The deployment requires access to Meta's Llama 3.1 model on HuggingFace:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set your HuggingFace token as environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   # or
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   ```

### 2. Deploy SageMaker Endpoint

```bash
python deployment/deploy_llama3_lmi.py
```

Creates endpoint `llama3-lmi-agent` (~5-10 minutes).

### 3. Set Environment Variables

```bash
export SAGEMAKER_ENDPOINT_NAME='llama3-lmi-agent'
export TAVILY_API_KEY='your-tavily-key'
export ALPHAVANTAGE_API_KEY='your-alphavantage-key'
export EDGAR_IDENTITY='your-email@example.com'
```

### 4. Run the Agent

```bash
uv run fin-agent-sagemaker-v2.py
```

Try: "Compare stock prices for Apple, Microsoft, and Google"

The agent calls 3 tools in parallel and synthesizes results.

## Performance

- **60-80% fewer API calls** with parallel tool calling
- **Response time**: 15-48 seconds for complex queries
- **Capacity**: Up to 20+ sequential tool calls
- **Success rate**: 100% on tested scenarios

## Configuration

**Instance**: `ml.g5.2xlarge` (1 GPU, 24GB)
- Cost: ~$1.52/hour (~$243/month for 8hrs/day)
- Handles: Development and moderate production workloads

**Environment Variables**:
- `SAGEMAKER_ENDPOINT_NAME` (required)
- `TAVILY_API_KEY` (optional - web search)
- `ALPHAVANTAGE_API_KEY` (optional - financial data)
- `EDGAR_IDENTITY` (optional - SEC filings)

## Testing

```bash
# Test parallel tool calling (2-5 tools)
uv run test_multiple_parallel_tools.py

# Test multi-step reasoning
uv run test_multi_step_detailed.py
```

## Key Improvements

### Parallel Tool Calling
- Enabled via `parallel_tool_calls=True` in content handler
- Reduces API calls by 60-80% for multi-tool queries
- Single agent invocation handles multiple tools

### Enhanced Response Generation
- 500 chars per tool result (vs 50 chars before)
- 512 max tokens (vs 128 before)
- System message for better response quality
- Comprehensive synthesis of tool results

## Important Notes

**HuggingFace Token Required**:
- You need a HuggingFace account and token to deploy the model
- Accept the Llama 3.1 license before deployment
- Update the token in `deployment/deploy_llama3_lmi.py` before running
- Never commit your token to version control

**Do NOT use quantized models** (AWQ, GPTQ) for tool calling:
- 4-bit quantized models hallucinate tool results
- Cannot maintain precision for structured outputs
- Always use unquantized models for agents

**Cost optimization**:
- Delete unused endpoints when not in use
- Use auto-scaling for variable workloads
- Monitor usage with CloudWatch alarms

## Resources

- [LMI Container Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

Follows the Llama 3 Community License Agreement.
