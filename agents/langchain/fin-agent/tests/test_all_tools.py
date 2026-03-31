#!/usr/bin/env python3
"""
Comprehensive test for all MCP and built-in tools in fin-agent-sagemaker-v2.py

This test verifies that all tools work correctly:
- Built-in tools: yahoo_news, yahoo_stock_price, tavily_search
- MCP tools: Edgar SEC filing tools (AlphaVantage excluded due to API issues)

The test uses queries designed to trigger specific tools and avoid AlphaVantage.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

# Import the main agent
import importlib.util
spec = importlib.util.spec_from_file_location("fin_agent", "fin-agent-sagemaker-v2.py")
fin_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin_agent)


# Test queries designed to trigger specific tools (avoiding AlphaVantage)
TEST_QUERIES = {
    "yahoo_stock_price": {
        "query": "What is the current stock price of Apple (AAPL)?",
        "expected_tools": ["yahoo_stock_price"],
        "description": "Test Yahoo Finance stock price tool"
    },
    "yahoo_news": {
        "query": "What are the latest news headlines for Tesla (TSLA)?",
        "expected_tools": ["yahoo_news"],
        "description": "Test Yahoo Finance news tool"
    },
    "yahoo_combined": {
        "query": "Give me Microsoft's current stock price and latest news",
        "expected_tools": ["yahoo_stock_price", "yahoo_news"],
        "description": "Test multiple Yahoo Finance tools"
    },
    "tavily_search": {
        "query": "Search for recent developments in artificial intelligence",
        "expected_tools": ["tavily_search"],
        "description": "Test Tavily web search tool"
    },
    "edgar_company_info": {
        "query": "Get company information for Apple Inc from SEC filings",
        "expected_tools": ["edgar_company"],
        "description": "Test Edgar MCP tools for company information",
        "mcp_only": True
    },
    "edgar_filings": {
        "query": "Find recent 10-K filings for Microsoft",
        "expected_tools": ["edgar_search"],
        "description": "Test Edgar MCP tools for SEC filings",
        "mcp_only": True
    },
    "mixed_tools": {
        "query": "What is Amazon's stock price and search for information about their latest earnings report",
        "expected_tools": ["yahoo_stock_price", "tavily_search"],
        "description": "Test combination of built-in tools"
    }
}


async def test_single_query(graph, test_name: str, test_config: dict) -> dict:
    """Test a single query and verify tool usage."""
    
    query = test_config["query"]
    expected_tools = test_config["expected_tools"]
    description = test_config["description"]
    
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print(f"Query: {query}")
    print(f"Expected tools: {', '.join(expected_tools)}")
    
    try:
        # Run the query
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "revision_count": 0,
            "evaluation": None,
            "best_revision": None,
            "best_evaluation": None
        }
        
        result = await graph.ainvoke(initial_state)
        
        # Extract tools called
        tools_called = []
        final_response = ""
        
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    if tool_name not in tools_called:
                        tools_called.append(tool_name)
            elif hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                final_response = msg.content
        
        # Verify results
        success = True
        issues = []
        
        # Check if any expected tools were called
        tools_found = [tool for tool in expected_tools if tool in tools_called]
        
        if not tools_called:
            success = False
            issues.append("No tools were called")
        elif not tools_found:
            success = False
            issues.append(f"Expected tools {expected_tools} but got {tools_called}")
        
        if not final_response:
            success = False
            issues.append("No response generated")
        elif len(final_response) < 50:
            success = False
            issues.append("Response too short")
        
        # Display results
        print(f"\n✅ Tools called: {', '.join(tools_called) if tools_called else 'None'}")
        print(f"📏 Response length: {len(final_response)} chars (~{len(final_response)//4} tokens)")
        print(f"\n📄 Response preview (first 200 chars):")
        print(f"{final_response[:200]}{'...' if len(final_response) > 200 else ''}")
        
        if success:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED")
            for issue in issues:
                print(f"   - {issue}")
        
        return {
            "test_name": test_name,
            "success": success,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "response_length": len(final_response),
            "issues": issues
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "test_name": test_name,
            "success": False,
            "tools_called": [],
            "expected_tools": expected_tools,
            "response_length": 0,
            "issues": [f"Exception: {str(e)}"]
        }


async def check_endpoint_exists() -> bool:
    """Check if SageMaker endpoint exists before running tests."""
    import boto3
    from botocore.exceptions import ClientError
    
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "llama3-lmi-agent")
    
    try:
        client = boto3.client('sagemaker', region_name='us-west-2')
        response = client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        
        if status != 'InService':
            print(f"⚠️  Endpoint {endpoint_name} exists but status is: {status}")
            print("   Endpoint must be 'InService' to run tests")
            return False
        
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"❌ Endpoint {endpoint_name} not found")
            print("   Deploy the endpoint first: python deployment/deploy_llama3_lmi.py")
            return False
        raise


async def test_all_tools():
    """Run comprehensive tests for all tools."""
    
    print("=" * 80)
    print("🧪 Comprehensive Tool Testing for fin-agent-sagemaker-v2.py")
    print("=" * 80)
    
    # Check if endpoint exists
    print("\n🔍 Checking SageMaker endpoint...")
    if not await check_endpoint_exists():
        print("\n⏭️  SKIPPING TESTS - Endpoint not available")
        print("\nTo run these tests:")
        print("  1. Deploy endpoint: python deployment/deploy_llama3_lmi.py")
        print("  2. Wait for endpoint to be InService (~5-10 minutes)")
        print("  3. Run tests again: python tests/test_all_tools.py")
        return True  # Return success to not fail the test suite
    
    print("✅ Endpoint is available and InService")
    
    # Connect to MCP servers
    print("\n🔌 Connecting to MCP servers...")
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
    EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY")
    
    MCP_SERVERS = {}
    
    # Skip AlphaVantage due to API issues
    print("⚠️  Skipping AlphaVantage (API issues)")
    
    # Add Edgar if available
    if EDGAR_IDENTITY:
        MCP_SERVERS["edgar"] = {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "edgar.ai"],
            "env": {"EDGAR_IDENTITY": EDGAR_IDENTITY},
        }
    
    mcp_tools = []
    has_edgar = False
    
    if MCP_SERVERS:
        try:
            mcp_client = MultiServerMCPClient(MCP_SERVERS)
            mcp_tools = await mcp_client.get_tools()
            has_edgar = len(mcp_tools) > 0
            print(f"✅ Connected to MCP servers! Loaded {len(mcp_tools)} MCP tools")
            
            # List available MCP tools
            if mcp_tools:
                print("\n📋 Available MCP tools:")
                for tool in mcp_tools:
                    print(f"   - {tool.name}")
        except Exception as e:
            print(f"⚠️  Could not connect to MCP servers: {e}")
            print("   Continuing with built-in tools only")
    else:
        print("⚠️  No MCP servers configured, using built-in tools only")
    
    # Build the graph
    print("\n🔨 Building agent graph...")
    graph = fin_agent.build_graph(
        mcp_tools=mcp_tools,
        max_revisions=0,
        shortcut_quality_score=5.0,
        use_builtin_tools=True
    )
    print("✅ Agent graph built successfully")
    
    # Determine which tests to run
    tests_to_run = {}
    for test_name, test_config in TEST_QUERIES.items():
        # Skip MCP-only tests if Edgar is not available
        if test_config.get("mcp_only", False) and not has_edgar:
            print(f"\n⏭️  Skipping {test_name} (requires Edgar MCP)")
            continue
        tests_to_run[test_name] = test_config
    
    print(f"\n📊 Running {len(tests_to_run)} tests...")
    
    # Run all tests
    results = []
    for test_name, test_config in tests_to_run.items():
        result = await test_single_query(graph, test_name, test_config)
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal tests: {total_tests}")
    print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    
    # Tool coverage
    all_tools_called = set()
    for result in results:
        all_tools_called.update(result["tools_called"])
    
    print(f"\n🔧 Tools tested: {len(all_tools_called)}")
    print(f"   {', '.join(sorted(all_tools_called))}")
    
    # Built-in tools coverage
    builtin_tools = {"yahoo_stock_price", "yahoo_news", "tavily_search"}
    builtin_tested = builtin_tools & all_tools_called
    print(f"\n📦 Built-in tools coverage: {len(builtin_tested)}/3")
    for tool in sorted(builtin_tools):
        status = "✅" if tool in builtin_tested else "❌"
        print(f"   {status} {tool}")
    
    # MCP tools coverage
    if has_edgar:
        edgar_tools_tested = [t for t in all_tools_called if t not in builtin_tools]
        print(f"\n🔌 MCP tools tested: {len(edgar_tools_tested)}")
        for tool in sorted(edgar_tools_tested):
            print(f"   ✅ {tool}")
    
    # Failed tests details
    if failed_tests > 0:
        print(f"\n❌ Failed tests details:")
        for result in results:
            if not result["success"]:
                print(f"\n   Test: {result['test_name']}")
                print(f"   Expected: {result['expected_tools']}")
                print(f"   Got: {result['tools_called']}")
                for issue in result["issues"]:
                    print(f"   Issue: {issue}")
    
    print("\n" + "=" * 80)
    if failed_tests == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"⚠️  {failed_tests} TEST(S) FAILED")
    print("=" * 80)
    
    return passed_tests == total_tests


async def main():
    """Main entry point."""
    try:
        success = await test_all_tools()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
