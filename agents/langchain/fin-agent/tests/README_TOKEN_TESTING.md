# Token Limit Testing Guide

This guide explains how to use the token limit testing script to measure the maximum token capacity of your SageMaker endpoint.

## Overview

The `test_prompts_token_limits.py` script tests various prompts and measures:
- **Input tokens**: Estimated tokens in the prompt
- **Output tokens**: Estimated tokens in the response
- **Total tokens**: Combined input + output
- **Tool calls**: Number of tools invoked
- **Response time**: Time to complete the request
- **Success rate**: Percentage of successful requests

## Quick Start

### 1. Basic Test (All Categories)

```bash
python tests/test_prompts_token_limits.py
```

This runs all test categories and generates a summary report.

### 2. Test Specific Category

```bash
# Test only stock price queries
python tests/test_prompts_token_limits.py --category stock_price_parallel

# Test only web search queries
python tests/test_prompts_token_limits.py --category web_search

# Test complex multi-step queries
python tests/test_prompts_token_limits.py --category complex_multi_step
```

### 3. Test Maximum Token Limits

```bash
python tests/test_prompts_token_limits.py --test-limits
```

This progressively increases prompt size to find the breaking point.

### 4. Verbose Mode

```bash
python tests/test_prompts_token_limits.py --verbose
```

Shows detailed output for each test.

### 5. Custom Output File

```bash
python tests/test_prompts_token_limits.py --save my_results.json
```

## Test Categories

### stock_price_single
Tests single stock price queries:
- "What's the current price of Apple stock?"
- "Show me Microsoft's stock price today"

**Expected**: 1 tool call, ~200-400 tokens

### stock_price_parallel
Tests parallel stock price queries:
- "Compare stock prices for Apple, Microsoft, and Google"
- "Show me prices for the top 5 tech stocks"

**Expected**: 3-5 parallel tool calls, ~800-1500 tokens

### stock_news
Tests stock news queries:
- "What's the latest news about Apple?"
- "What's the latest news for Apple and Microsoft?"

**Expected**: 1-2 tool calls, ~500-1000 tokens

### combined_price_news
Tests combined price + news queries:
- "Give me Apple's current stock price and latest news"
- "Compare prices and news for Apple and Tesla"

**Expected**: 2-4 parallel tool calls, ~1000-2000 tokens

### web_search
Tests general web search:
- "What are the latest developments in artificial intelligence?"
- "Search for recent news about electric vehicles"

**Expected**: 1 tool call, ~600-1200 tokens

### mixed_queries
Tests stock data + web search:
- "What's Apple's stock price and what are analysts saying?"
- "Compare Microsoft and Google prices, then search for their AI competition"

**Expected**: 2-3 tool calls, ~1200-2000 tokens

### complex_multi_step
Tests complex multi-step reasoning:
- "Find prices for Apple, Microsoft, and Google, then search for AI strategy info"
- "Get Tesla's price and news, then search for EV market trends"

**Expected**: 4-6 tool calls, ~2000-3500 tokens

## Understanding Results

### Token Estimates

The script uses a rough approximation: **1 token ≈ 4 characters**

This is approximate because:
- Actual tokenization depends on the model's tokenizer
- Special tokens (tool calls, formatting) add overhead
- Different languages have different token densities

### Token Limits

Based on the LMI configuration in `deploy_llama3_lmi.py`:
- **Max Model Length**: 8192 tokens (configured)
- **Practical Limit**: ~6000-7000 tokens (with safety margin)
- **Recommended**: Keep under 4000 tokens for reliability

### Color Coding

- **Green** (< 2000 tokens): Safe zone
- **Yellow** (2000-4000 tokens): ⚠️ Caution zone
- **Red** (> 4000 tokens): 🔴 High usage zone

## Sample Output

```
================================================================================
Testing Category: Stock Price Parallel
================================================================================

ℹ️ Testing: Compare stock prices for Apple, Microsoft, and Google...
✅ Success! Tokens: 1247 (312 in + 935 out), Tools: 3, Time: 4.23s

ℹ️ Testing: What are the current prices for Amazon, Netflix, and Meta?...
✅ Success! Tokens: 1189 (298 in + 891 out), Tools: 3, Time: 3.87s

================================================================================
Test Summary
================================================================================

Total Tests: 15
Successful: 15 (100.0%)
Failed: 0 (0.0%)

Token Usage:
  Average: 1456 tokens
  Maximum: 2847 tokens
  Total: 21840 tokens

Response Time:
  Average: 5.34s
  Total: 80.10s
```

## Interpreting Results

### Success Rate
- **100%**: Endpoint is stable and handling all queries
- **90-99%**: Occasional failures, may need investigation
- **< 90%**: Significant issues, check endpoint health

### Token Usage
- **Average < 2000**: Good, plenty of headroom
- **Average 2000-4000**: Moderate usage, monitor closely
- **Average > 4000**: High usage, consider optimization

### Response Time
- **< 5s**: Excellent performance
- **5-10s**: Good performance for complex queries
- **> 10s**: Slow, may indicate resource constraints

## Troubleshooting

### "Connection timeout" errors
- Endpoint may be cold starting
- Wait 30 seconds and retry
- Check endpoint status in AWS Console

### "Token limit exceeded" errors
- Reduce prompt complexity
- Use fewer parallel tool calls
- Truncate tool results

### High failure rate
- Check API keys (TAVILY_API_KEY)
- Verify endpoint is InService
- Check CloudWatch logs for errors

### Inconsistent results
- Model temperature is set to 0.1 (low randomness)
- Tool results may vary (live data)
- Network latency affects timing

## Advanced Usage

### Test Custom Prompts

Edit `TEST_PROMPTS` dictionary in the script:

```python
TEST_PROMPTS = {
    "my_custom_category": [
        "My custom prompt 1",
        "My custom prompt 2",
    ]
}
```

Then run:
```bash
python tests/test_prompts_token_limits.py --category my_custom_category
```

### Analyze Results Programmatically

```python
import json

with open('tests/test_results_token_limits.json') as f:
    data = json.load(f)

# Find highest token usage
max_result = max(data['results'], key=lambda x: x['total_tokens'])
print(f"Highest usage: {max_result['total_tokens']} tokens")
print(f"Prompt: {max_result['prompt']}")

# Calculate success rate by category
from collections import defaultdict
stats = defaultdict(lambda: {'total': 0, 'success': 0})

for result in data['results']:
    cat = result['category']
    stats[cat]['total'] += 1
    if result['success']:
        stats[cat]['success'] += 1

for cat, counts in stats.items():
    rate = counts['success'] / counts['total'] * 100
    print(f"{cat}: {rate:.1f}% success rate")
```

## Best Practices

1. **Start Small**: Test simple queries first
2. **Monitor Tokens**: Keep track of token usage trends
3. **Rate Limit**: Add delays between tests (script includes 1s delay)
4. **Save Results**: Always save results for analysis
5. **Compare Baselines**: Run tests regularly to detect regressions

## Cost Considerations

Each test invokes the SageMaker endpoint:
- **Cost**: ~$1.52/hour for ml.g5.2xlarge
- **Per Request**: ~$0.0004 (assuming 1 second per request)
- **Full Test Suite**: ~$0.01 (15-20 requests)

Running tests frequently is cost-effective for monitoring endpoint health.

## Next Steps

After running tests:

1. **Review Summary**: Check success rate and token usage
2. **Identify Limits**: Note where failures occur
3. **Optimize Prompts**: Reduce token usage if needed
4. **Monitor Production**: Use similar metrics in production
5. **Set Alerts**: Create CloudWatch alarms for high token usage

## Related Files

- `TEST_PROMPTS_NO_ALPHAVANTAGE.md`: Detailed prompt examples
- `test_multiple_parallel_tools.py`: Parallel tool calling tests
- `test_multi_step_detailed.py`: Multi-step reasoning tests
- `SEQUENCE_LENGTH_FINDINGS.md`: Previous token limit research
