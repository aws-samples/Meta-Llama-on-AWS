# Finance Agent

## Introduction

This a finance agent showcasing capabilities of distilled Llama 8b using LangChain framework.

## Installation and running

Follow below steps:

1. Install uv

1. Run:

```bash
uv sync
```

### 1. Get HuggingFace Token

1. Create account at https://huggingface.co
2. Accept Llama 3.1 license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
3. Generate token: https://huggingface.co/settings/tokens
4. Set environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

### 2. Deploy SageMaker Endpoint

1. Setup AWS account by following "deployment/README.md".

1. Run the deployment script described there (deployment/deploy_llama3_lmi.py) which creates endpoint (llama3-lmi-agent) and takes ~5-10 minutes.

1. After you are done make sure you delete the endpoint so you do not continue incure charges. See section "Cleanup" in "deployment/README.md".

### 3. Set Environment Variables

```bash
export SAGEMAKER_ENDPOINT_NAME='llama3-lmi-agent'
export TAVILY_API_KEY='your-tavily-key'
export ALPHAVANTAGE_API_KEY='your-alphavantage-key'  # optional
export EDGAR_IDENTITY='your-email@example.com'        # optional
```

### 4. Run the Agent

```bash
uv run fin-agent-sagemaker-v2.py
```

Try: "What is Apple's stock price?"

After you are done make sure you delete the endpoint so you do not continue incure charges. See section "Cleanup" in "deployment/README.md".

## What Works Well ✅

- Simple queries with 1-2 tool calls
- Single-company stock price lookups
- Basic financial data retrieval
- Learning LangGraph agent patterns

## Known Limitations ⚠️

**Llama 3.1 8B has fundamental limitations for function calling:**

1. **Inconsistent Tool Calling Format**
   - Model sometimes generates tool calls in different formats
   - May fail on complex multi-step queries
   - Not a bug - this is expected behavior

2. **Context Window Limit (8,192 tokens)**
   - Agent can make 5-6 tool calls before hitting limit
   - Complex queries requiring many tools will fail

3. **Sequential Tool Calling Only**
   - SageMaker endpoint processes one tool at a time
   - Multi-company comparisons require multiple loops

### Recommendation for Production

**Upgrade to better models:**
- **Llama 3.3 70B**: Better function calling, 128K context window
- **GPT-4 or Claude**: Designed specifically for tool use
- **Amazon Bedrock**: Managed service with Claude 3.5 Sonnet

**This implementation is best for:**
- Learning agent architectures
- Testing LangGraph patterns
- Cost-conscious experimentation
- Simple demos

## Configuration

**Instance**: `ml.g5.2xlarge` (1 GPU, 24GB VRAM)
- Cost: ~$1.52/hour
- Suitable for: Development and testing

**Environment Variables**:
- `SAGEMAKER_ENDPOINT_NAME` (required)
- `TAVILY_API_KEY` (required - web search)
- `ALPHAVANTAGE_API_KEY` (optional - financial data)
- `EDGAR_IDENTITY` (optional - SEC filings)

## Testing

```bash
# Run the agent interactively
uv run fin-agent-sagemaker-v2.py
```

**Example Queries:**
- "What is Apple's stock price?" ✅
- "Get news for Microsoft" ✅
- "Compare Apple and Amazon stock prices" ⚠️ (may require multiple loops)


## Architecture

- **LangGraph**: Agent workflow orchestration
- **LangChain**: Tool integration and message handling
- **SageMaker**: Llama 3.1 8B model hosting
- **MCP**: Edgar (SEC filings) and AlphaVantage (financial data)

## Key Features

- Multi-tool financial research
- Sequential tool execution with validation
- Automatic error handling and recovery
- Support for Yahoo Finance, Tavily, Edgar, and AlphaVantage

## Files

- `fin-agent-sagemaker-v2.py` - Main agent implementation
- `src/sagemaker_with_tools.py` - SageMaker LangChain integration
- `src/content_handler.py` - Tool calling format handler
- `deployment/deploy_llama3_lmi.py` - Endpoint deployment script

## Troubleshooting

**"Input validation error: Input length exceeds maximum"**
- Query required too many tool calls
- Simplify the query or upgrade to a model with larger context

**"This model only supports single tool-calls at once"**
- This is expected - agent handles it automatically
- Tools are called sequentially, not in parallel

**Agent returns outdated information**
- Model may generate text instead of calling tools
- This is a known limitation of Llama 3.1 8B
- Try rephrasing the query or upgrade to a better model

## Cost Estimate

**SageMaker Endpoint** (ml.g5.2xlarge):
- $1.52/hour
- ~$36/day (24 hours)
- ~$243/month (8 hours/day, 20 days)

**API Costs** (per 1000 queries):
- Tavily: ~$1-2
- AlphaVantage: Free tier available
- Edgar: Free

## Next Steps

1. **Try simple queries** to understand agent behavior
2. **Review limitations** to set proper expectations
3. **For production**: Upgrade to Llama 3.3 70B or use Bedrock
4. **Explore LangGraph**: Learn agent patterns and workflows

## Additional Documentation

- `KNOWN_LIMITATIONS.md` - Detailed technical explanation of model limitations
- `deployment/README.md` - Deployment guide and troubleshooting

## License

This sample code is made available under the MIT-0 license. See the LICENSE file.