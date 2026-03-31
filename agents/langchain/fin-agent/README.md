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

Use `--debug` for verbose output (token counts, payload details, type coercion):

```bash
uv run fin-agent-sagemaker-v2.py --debug
```

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
- **Llama 3.1/3.3 70B**: Now supported — see `deployment/deploy_llama3_70b.py`. Better function calling, 128K context window
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

For production workloads, consider the 70B model on `ml.g5.48xlarge` (~$20.36/hour). See `deployment/README.md` for details.

### Switching Between 8B and 70B

```bash
# Use 8B endpoint
export SAGEMAKER_ENDPOINT_NAME='llama3-lmi-agent'

# Use 70B endpoint
export SAGEMAKER_ENDPOINT_NAME='llama3-70b-lmi-agent'
```

**Environment Variables**:
- `SAGEMAKER_ENDPOINT_NAME` (required)
- `TAVILY_API_KEY` (required - web search)
- `ALPHAVANTAGE_API_KEY` (optional - financial data)
- `EDGAR_IDENTITY` (optional - SEC filings)
- `DEBUG=1` (optional - enable debug output, same as `--debug` flag)

## Testing

```bash
# Run the agent interactively
uv run fin-agent-sagemaker-v2.py

# Run with debug output (token counts, payload details)
uv run fin-agent-sagemaker-v2.py --debug
```

### Test Prompts by Complexity

The table below documents test prompts, what they exercise, and expected behavior on each model.
#### Level 1: Single Tool Call (warm-up)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "What is Apple's current stock price?" | 1× `yahoo_stock_price` | ✅ | ✅ | ✅ |
| "Get me the latest news for NVDA" | 1× `yahoo_news` | ✅ | ✅ | ✅ |
| "Search for the latest Federal Reserve interest rate decision" | 1× `tavily_search` | ✅ | ✅ | ✅ |

#### Level 2: Two Sequential Tool Calls

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "What is Tesla's stock price and latest news?" | `yahoo_stock_price` + `yahoo_news` | ✅ | ✅ | ✅ |
| "Get Microsoft's stock price and search for their latest AI announcements" | `yahoo_stock_price` + `tavily_search` | ✅ | ✅ | ✅ |

#### Level 3: Multi-Company (3-5 sequential tool calls)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "Compare the stock prices of Apple, Microsoft, and Google" | 3× `yahoo_stock_price` | ⚠️ may loop | ✅ | ✅ |
| "Get the latest news for both NVDA and AMD" | 2× `yahoo_news` | ✅ | ✅ | ✅ |
| "What is Amazon's stock price and what recent news might be affecting it?" | `yahoo_stock_price` + `yahoo_news` + `tavily_search` | ⚠️ inconsistent | ✅ | ✅ |

#### Level 4: Multi-Tool Reasoning (5-7 tool calls)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "Compare Apple and Microsoft stock prices, and find recent news about their AI strategies" | 2× `yahoo_stock_price` + 2× `yahoo_news` + `tavily_search` | ❌ context overflow | ✅ | ✅ |
| "Get NVDA's stock price, latest news, and search for analyst price targets for Nvidia" | `yahoo_stock_price` + `yahoo_news` + `tavily_search` | ⚠️ may fail | ✅ | ✅ |

#### Level 5: Stress Test (7+ tool calls, pushes context window)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "Search for S&P 500 performance 2026, then get stock prices for AAPL MSFT NVDA, get news for each, and write a detailed market analysis" | 1× `tavily_search` + 3× `yahoo_stock_price` + 3× `yahoo_news` (7 total) | ❌ fails | ✅ ~8.5K peak tokens | ⚠️ tight — trimmer drops older results to fit 8K |
| "Compare stock prices for AAPL, MSFT, GOOGL, AMZN, NVDA, META, and TSLA and rank them" | 7× `yahoo_stock_price` | ❌ fails | ✅ | ✅ stock prices are small payloads |

#### Level 6: Maximum Stress (21+ tool calls)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "Get the stock price, latest news, and search for analyst price targets for each: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA. Write a comprehensive investment report." | 7× `yahoo_stock_price` + 7× `yahoo_news` + 7× `tavily_search` (21 total) | ❌ fails | ⚠️ may hit 16K token limit | ❌ exceeds 8K even with aggressive trimming |

#### With Edgar/AlphaVantage (optional MCP tools)

| Prompt | Tools Used | 8B | 70B (16K) | 70B-AWQ (8K) |
|--------|-----------|-----|-----------|--------------|
| "What is Apple's revenue and net income for the last year?" | `edgar_compare` | ⚠️ format issues | ✅ | ✅ |
| "Compare revenue growth between Apple and Microsoft" | `edgar_compare` | ❌ | ✅ | ✅ |
| "Get Tesla's stock price and their latest SEC filing highlights" | `yahoo_stock_price` + `edgar` | ❌ | ✅ | ✅ |

**Column key:**
- **70B (16K)**: FP16 model on `ml.g5.48xlarge` or `ml.p4d.24xlarge` with 16K+ context window
- **70B-AWQ (8K)**: Quantized AWQ-INT4 model on `ml.g5.12xlarge` with 8K context window

**Legend:** ✅ works reliably · ⚠️ inconsistent/may fail · ❌ expected to fail

**Token note:** Peak token counts reflect the full context at the final tool call (system prompt + all messages + tool schemas). The agent auto-detects the endpoint's context window and trims older tool results to fit. With 8K context, queries requiring 5+ tool calls with large payloads (news, search results) will lose older results to trimming.

**Key takeaway:** The 8B model handles 1-2 tool calls reliably. The 70B model handles 7+ tool calls with coherent multi-section responses, making it suitable for real agent workflows. The quantized 70B-AWQ on `ml.g5.12xlarge` is a cost-effective option (~$7/hr vs ~$20/hr) that handles most queries, but the 8K context window limits complex multi-tool scenarios.


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
- `deployment/deploy_llama3_70b.py` - 70B endpoint deployment script
- `deployment/download_model_to_s3.py` - S3 model pre-download utility

## Troubleshooting

**"Input validation error: Input length exceeds maximum"**
- Query required too many tool calls
- Simplify the query or upgrade to a model with larger context
- Run with `--debug` to see token breakdown per request

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