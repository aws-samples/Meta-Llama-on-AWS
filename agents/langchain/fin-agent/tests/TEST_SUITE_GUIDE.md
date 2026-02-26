# Test Suite Guide

## Overview

This directory contains comprehensive tests for the financial agent project, organized into three categories:

1. **Unit Tests** - Fast, isolated tests with no external dependencies
2. **Integration Tests** - Tests requiring SageMaker endpoint and API keys
3. **Deployment Tests** - Tests for endpoint deployment and configuration

### Important: SageMaker Endpoint Requirement

**Integration tests require a deployed SageMaker endpoint to run.**

- **Without endpoint**: Integration tests skip gracefully (not a failure)
- **With endpoint**: Integration tests run and verify full system functionality
- **Unit tests**: Always run regardless of endpoint status (96 tests)

**To deploy the endpoint:**
```bash
python deployment/deploy_llama3_lmi.py
```

Wait ~5-10 minutes for endpoint to reach `InService` status.

**Cost**: ~$1.52/hour when endpoint is running (ml.g5.2xlarge instance)

## Quick Start

### Run All Tests
```bash
python run_all_tests.py
```

### Run Only Unit Tests (Fast)
```bash
python run_all_tests.py --unit-only
```

### Run Without Deployment Tests
```bash
python run_all_tests.py --skip-deployment
```

### Verbose Output
```bash
python run_all_tests.py --verbose
```

---

## Test Files

### Unit Tests (pytest)

These tests run quickly and don't require external services:

#### 1. `test_content_handler.py`
**Purpose**: Test LlamaFunctionCallingHandler for tool calling

**What it tests**:
- Tool schema formatting
- Request/response transformation
- Function calling format conversion
- Error handling

**Run individually**:
```bash
pytest tests/test_content_handler.py -v
```

**Expected**: All tests pass

---

#### 2. `test_tool_schema_formatter.py`
**Purpose**: Test tool schema formatting for Llama 3.1

**What it tests**:
- JSON schema generation
- Parameter type conversion
- Required/optional field handling
- Complex nested schemas

**Run individually**:
```bash
pytest tests/test_tool_schema_formatter.py -v
```

**Expected**: All tests pass

---

#### 3. `test_response_parser.py`
**Purpose**: Test tool call parsing from model responses

**What it tests**:
- JSON extraction from text
- Multiple tool call parsing
- Error handling for malformed responses
- Property-based testing with Hypothesis

**Run individually**:
```bash
pytest tests/test_response_parser.py -v
```

**Expected**: All tests pass

---

#### 4. `test_prompt_formatter.py`
**Purpose**: Test prompt formatting for tool calling

**What it tests**:
- System prompt generation
- Tool list formatting
- Message history handling
- Token counting

**Run individually**:
```bash
pytest tests/test_prompt_formatter.py -v
```

**Expected**: All tests pass

---

#### 5. `test_sagemaker_with_tools.py`
**Purpose**: Test ChatSagemakerWithTools wrapper (mocked)

**What it tests**:
- Tool binding
- Message formatting
- Response parsing
- Error handling

**Run individually**:
```bash
pytest tests/test_sagemaker_with_tools.py -v
```

**Expected**: All tests pass

---

#### 6. `test_error_diagnostics_demo.py`
**Purpose**: Demonstrate error handling and diagnostics

**What it tests**:
- Error message parsing
- Diagnostic information extraction
- Recovery strategies
- Logging

**Run individually**:
```bash
pytest tests/test_error_diagnostics_demo.py -v
```

**Expected**: All tests pass

---

### Integration Tests (async)

These tests require SageMaker endpoint and API keys.

**IMPORTANT**: Integration tests will **skip gracefully** if the SageMaker endpoint is not deployed. They will not fail - they will simply skip with a helpful message explaining how to deploy the endpoint.

#### 7. `test_all_tools.py`
**Purpose**: Test all built-in tools (Yahoo Finance, Tavily)

**What it tests**:
- Yahoo stock price fetching
- Yahoo news retrieval
- Tavily web search
- Tool error handling

**Requirements**:
- `SAGEMAKER_ENDPOINT_NAME` set
- `TAVILY_API_KEY` set
- Active SageMaker endpoint (InService status)

**Run individually**:
```bash
python tests/test_all_tools.py
```

**Behavior**:
- ✅ **Endpoint available**: Runs all tool tests
- ⏭️ **Endpoint not found**: Skips gracefully with instructions
- ⚠️ **Endpoint not InService**: Skips with status message

**Expected**: All tools work correctly (when endpoint is available)

**Duration**: ~2-3 minutes (when running)

---

#### 8. `test_mcp_coordination.py`
**Purpose**: Test MCP tool coordination and multi-tool workflows

**What it tests**:
- Agent initialization with MCP tools
- Multi-step tool calling
- Sequential tool coordination
- Edgar MCP integration
- Response quality

**Requirements**:
- `SAGEMAKER_ENDPOINT_NAME` set
- `TAVILY_API_KEY` set
- `EDGAR_IDENTITY` set (optional, for Edgar tests)
- Active SageMaker endpoint (InService status)

**Run individually**:
```bash
python tests/test_mcp_coordination.py
```

**Behavior**:
- ✅ **Endpoint available**: Runs all coordination tests
- ⏭️ **Endpoint not found**: Skips gracefully with instructions
- ⚠️ **Endpoint not InService**: Skips with status message

**Expected**: 9/11 scenarios pass (81.8%)
- 2 failures are expected (Edgar library bugs)

**Duration**: ~5-10 minutes (when running)

**Test scenarios**:
1. Multi-step stock analysis (Yahoo)
2. Stock + web search coordination
3. Comprehensive company analysis (3 tools)
4. Multi-company comparison
5. News aggregation
6. Web search then stock
7. SEC filing + stock (Edgar + Yahoo)
8. Company profile + news (Edgar + Yahoo)
9. Filing content + stock (Edgar + Yahoo)
10. Company comparison + search (Edgar + Tavily)
11. Ownership + stock (Edgar + Yahoo)

---

#### 9. `test_prompts_token_limits.py`
**Purpose**: Test token limit handling and context management

**What it tests**:
- Token counting accuracy
- Context truncation
- Long conversation handling
- Tool result summarization

**Requirements**:
- `SAGEMAKER_ENDPOINT_NAME` set
- `TAVILY_API_KEY` set
- Active SageMaker endpoint (InService status)

**Run individually**:
```bash
python tests/test_prompts_token_limits.py
```

**Behavior**:
- ✅ **Endpoint available**: Runs token limit tests
- ⏭️ **Endpoint not found**: Skips gracefully (this test doesn't check endpoint status yet, but will pass/fail based on actual endpoint availability)

**Expected**: All prompts within token limits

**Duration**: ~3-5 minutes (when running)

---

### Deployment Tests (pytest)

These tests check endpoint deployment:

#### 10. `test_deploy_endpoint.py`
**Purpose**: Test SageMaker endpoint deployment functions

**What it tests**:
- Endpoint creation
- Model configuration
- Deployment validation
- Cleanup procedures

**Requirements**:
- AWS credentials configured
- `HF_TOKEN` set
- Sufficient AWS permissions

**Run individually**:
```bash
pytest tests/test_deploy_endpoint.py -v
```

**Expected**: All tests pass (mocked)

**Note**: This tests deployment logic, not actual deployment

---

## Test Results Summary

### Expected Results

| Test File | Type | Expected Result | Duration | Endpoint Required |
|-----------|------|----------------|----------|-------------------|
| test_content_handler.py | Unit | ✅ All pass | <1s | No |
| test_tool_schema_formatter.py | Unit | ✅ All pass | <1s | No |
| test_response_parser.py | Unit | ✅ All pass | <1s | No |
| test_prompt_formatter.py | Unit | ✅ All pass | <1s | No |
| test_sagemaker_with_tools.py | Unit | ✅ All pass | <1s | No |
| test_error_diagnostics_demo.py | Unit | ✅ All pass | <1s | No |
| test_all_tools.py | Integration | ✅ Pass or ⏭️ Skip | 2-3 min | Yes |
| test_mcp_coordination.py | Integration | ✅ Pass or ⏭️ Skip | 5-10 min | Yes |
| test_prompts_token_limits.py | Integration | ✅ All pass | 3-5 min | Yes |
| test_deploy_endpoint.py | Deployment | ✅ All pass | <1s | No |

**Total Duration**: 
- Without endpoint: <10 seconds (unit tests only)
- With endpoint: ~10-20 minutes (full suite)

**Success Rate**:
- Without endpoint: 100% (integration tests skip gracefully)
- With endpoint: ~90% (Edgar bugs expected)

---

## Environment Requirements

### Required Variables
```bash
SAGEMAKER_ENDPOINT_NAME=llama3-lmi-agent
TAVILY_API_KEY=tvly-dev-***
HF_TOKEN=hf_***
```

### Optional Variables
```bash
EDGAR_IDENTITY=your.email@example.com  # For Edgar MCP tests
ALPHAVANTAGE_API_KEY=***               # Currently disabled
```

### Check Environment
```bash
# Verify all variables are set
env | grep -E "SAGEMAKER|TAVILY|EDGAR|HF_TOKEN"
```

---

## Troubleshooting

### Common Issues

#### 1. "Required environment variables not set"

**Solution**: Set variables in `.env` file:
```bash
cp .env.example .env
# Edit .env with your values
```

#### 2. "SageMaker endpoint not found" or Tests Skipping

**Symptom**: Integration tests show:
```
❌ Endpoint llama3-lmi-agent not found
⏭️  SKIPPING TESTS - Endpoint not available
```

**This is NOT a failure** - tests skip gracefully when endpoint isn't deployed.

**Solution**: Deploy endpoint first:
```bash
python deployment/deploy_llama3_lmi.py
```

Wait ~5-10 minutes for endpoint to reach InService status, then run tests again.

**Why tests skip instead of fail**:
- Integration tests require a live SageMaker endpoint
- Endpoint costs ~$1.52/hour when running
- Tests skip gracefully to avoid false failures
- Unit tests (96 tests) still run and pass without endpoint

#### 3. "Endpoint exists but status is not InService"

**Symptom**:
```
⚠️  Endpoint llama3-lmi-agent exists but status is: Creating
```

**Solution**: Wait for endpoint to finish deploying:
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name llama3-lmi-agent --region us-west-2
```

Endpoint must be in `InService` status to run tests.

#### 4. "Edgar MCP tests failing"

**Expected**: 2/5 Edgar tools fail due to library bugs
- `edgar_company` - AttributeError
- `edgar_compare` - AttributeError

This is documented and expected. See `MCP_RESEARCH_SUMMARY.md`.

#### 4. "Tests timing out"

**Solution**: 
- Check internet connection
- Verify SageMaker endpoint is running and InService
- Increase timeout in test runner

#### 5. "Import errors"

**Solution**: Install dependencies:
```bash
uv pip install -r requirements.txt
```

#### 6. "All tests pass but I expected some to run"

**Symptom**: Test suite shows 100% pass rate but integration tests didn't actually run.

**Explanation**: Integration tests skip gracefully when endpoint is not available. This is expected behavior.

**To verify**:
```bash
# Check if endpoint exists
aws sagemaker describe-endpoint --endpoint-name llama3-lmi-agent --region us-west-2

# If endpoint doesn't exist, deploy it
python deployment/deploy_llama3_lmi.py
```

**Test output when skipping**:
```
📝 Running: tests/test_all_tools.py
   ✅ PASSED  # <- Test passed because it skipped gracefully

📝 Running: tests/test_mcp_coordination.py
   ✅ PASSED  # <- Test passed because it skipped gracefully
```

**Test output when actually running**:
```
📝 Running: tests/test_all_tools.py
   ✅ PASSED  # <- Test ran and all tools worked

📝 Running: tests/test_mcp_coordination.py
   ✅ PASSED  # <- Test ran and coordination worked
```

To see the difference, run tests individually with verbose output.

---

## Test Development Guidelines

### Adding New Tests

1. **Unit tests**: Add to appropriate test file or create new one
2. **Integration tests**: Create new async test file
3. **Follow naming convention**: `test_*.py`
4. **Add to test runner**: Update `run_all_tests.py`

### Test Structure

```python
#!/usr/bin/env python3
"""
Test description and purpose.

What it tests:
- Feature 1
- Feature 2
- Feature 3
"""

import pytest  # For unit tests
import asyncio  # For integration tests

# Test implementation

if __name__ == "__main__":
    # For integration tests
    asyncio.run(main())
    
    # For unit tests
    pytest.main([__file__, "-v"])
```

### Best Practices

1. **Isolate tests**: Each test should be independent
2. **Mock external services**: For unit tests
3. **Use real services**: For integration tests
4. **Add delays**: Between API calls to avoid rate limits
5. **Document expected failures**: Like Edgar MCP bugs
6. **Provide clear output**: Use descriptive print statements

---

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: python run_all_tests.py --unit-only
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: python run_all_tests.py --skip-deployment
        env:
          SAGEMAKER_ENDPOINT_NAME: ${{ secrets.SAGEMAKER_ENDPOINT_NAME }}
          TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
```

---

## Performance Benchmarks

### Unit Tests
- Total: ~6 tests
- Duration: <10 seconds
- Success rate: 100%

### Integration Tests
- Total: ~3 tests
- Duration: 10-20 minutes
- Success rate: ~90% (Edgar bugs expected)

### Full Suite
- Total: ~10 tests
- Duration: 10-20 minutes
- Success rate: ~90%

---

## Related Documentation

- **MCP_RESEARCH_SUMMARY.md** - MCP tool findings
- **MCP_RATE_LIMITS_AND_CONFIG.md** - Detailed MCP configuration
- **MCP_QUICK_REFERENCE.md** - Quick reference for MCP tools
- **README_TOKEN_TESTING.md** - Token limit testing guide

---

## Support

For issues or questions:
1. Check this guide first
2. Review related documentation
3. Check test output for specific errors
4. Review MCP documentation for tool-specific issues

---

**Last Updated**: February 23, 2026
**Test Suite Version**: 1.1
**Expected Success Rate**: 100% (integration tests skip gracefully without endpoint)

## Recent Changes

### Version 1.1 (February 23, 2026)
- Added graceful skipping for integration tests when endpoint is not available
- Tests no longer fail when endpoint is missing - they skip with helpful instructions
- Updated documentation to clarify endpoint requirement
- Added troubleshooting section for endpoint-related issues
