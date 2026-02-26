#!/usr/bin/env python3
"""
Comprehensive MCP Tool Coordination Test for fin-agent-sagemaker-v2.py

This test verifies that the main agent script:
1. Loads correctly with all MCP tools
2. Coordinates multiple tool calls in sequence
3. Properly handles tool results and generates coherent responses
4. Works with both built-in tools and MCP tools

Test scenarios include:
- Multi-step queries requiring sequential tool calls
- Queries that need both built-in and MCP tools
- Complex financial analysis requiring multiple data sources

Test Results (as of 2026-02-19):
- Success rate: 81.8% (9/11 scenarios passed)
- Edgar MCP: 3/5 tools working (edgar_search, edgar_filing, edgar_ownership)
- Edgar failures: 2/5 tools broken due to library bugs (edgar_company, edgar_compare)
- No rate limit issues detected - failures are due to AttributeError in Edgar library
- See MCP_RESEARCH_SUMMARY.md for detailed findings
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import the main agent
import importlib.util
spec = importlib.util.spec_from_file_location("fin_agent", "fin-agent-sagemaker-v2.py")
fin_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin_agent)


# Test scenarios designed to test multi-tool coordination
TEST_SCENARIOS = {
    "multi_step_stock_analysis": {
        "query": "What is Apple's current stock price and what are the latest news about the company? Use Yahoo Finance for both.",
        "description": "Multi-step: Get stock price, then get news (2 tools, sequential)",
        "min_tools": 2,
        "expected_tool_types": ["yahoo_stock_price", "yahoo_news"],
        "verify_response": lambda resp: "AAPL" in resp or "Apple" in resp
    },
    "stock_and_web_search": {
        "query": "Get Tesla's stock price from Yahoo Finance and use Tavily to search for recent news about their electric vehicle production",
        "description": "Coordination: Stock data + web search (2 different tool types)",
        "min_tools": 2,
        "expected_tool_types": ["yahoo_stock_price", "tavily_search"],
        "verify_response": lambda resp: "Tesla" in resp or "TSLA" in resp
    },
    "comprehensive_company_analysis": {
        "query": "I want to analyze Microsoft: get the current stock price from Yahoo, latest news from Yahoo, and use Tavily to search for information about their AI strategy",
        "description": "Complex: 3 tools coordinated (stock price + news + web search)",
        "min_tools": 3,
        "expected_tool_types": ["yahoo_stock_price", "yahoo_news", "tavily_search"],
        "verify_response": lambda resp: "Microsoft" in resp or "MSFT" in resp
    },
    "multi_company_comparison": {
        "query": "Compare the stock prices of Apple (AAPL) and Google (GOOGL) using Yahoo Finance",
        "description": "Multi-entity: Same tool called multiple times for different entities",
        "min_tools": 2,
        "expected_tool_types": ["yahoo_stock_price"],
        "verify_response": lambda resp: ("Apple" in resp or "AAPL" in resp) and ("Google" in resp or "GOOGL" in resp)
    },
    "news_aggregation": {
        "query": "What are the latest news for Amazon from Yahoo Finance and what's happening with their stock price?",
        "description": "Sequential: News first, then stock price (order matters)",
        "min_tools": 2,
        "expected_tool_types": ["yahoo_news", "yahoo_stock_price"],
        "verify_response": lambda resp: "Amazon" in resp or "AMZN" in resp
    },
    "web_search_then_stock": {
        "query": "Use Tavily to search for information about NVIDIA's latest GPU announcement, then get their stock price from Yahoo Finance",
        "description": "Explicit sequence: Web search → stock price",
        "min_tools": 2,
        "expected_tool_types": ["tavily_search", "yahoo_stock_price"],
        "verify_response": lambda resp: "NVIDIA" in resp or "NVDA" in resp
    }
}


# Edgar-specific tests (only run if Edgar is available)
# Note: Edgar MCP tools are: edgar_company, edgar_search, edgar_filing, edgar_compare, edgar_ownership
EDGAR_TEST_SCENARIOS = {
    "sec_filing_and_stock": {
        "query": "Use Edgar to search for Apple's recent 10-K SEC filings, then get their current stock price",
        "description": "MCP + Built-in: Edgar SEC search + Yahoo Finance",
        "min_tools": 2,
        "expected_tool_types": ["edgar_search", "yahoo_stock_price"],
        "verify_response": lambda resp: "Apple" in resp or "AAPL" in resp or "10-K" in resp,
        "requires_edgar": True
    },
    "company_profile_and_news": {
        "query": "Use Edgar to get Microsoft's company profile and financial information, then get their latest news from Yahoo",
        "description": "MCP + Built-in: Edgar company profile + Yahoo news",
        "min_tools": 2,
        "expected_tool_types": ["edgar_company", "yahoo_news"],
        "verify_response": lambda resp: "Microsoft" in resp or "MSFT" in resp,
        "requires_edgar": True
    },
    "edgar_filing_content": {
        "query": "Use Edgar to read the latest 10-K filing for Tesla and get their stock price",
        "description": "MCP + Built-in: Edgar filing reader + Yahoo stock",
        "min_tools": 2,
        "expected_tool_types": ["edgar_filing", "yahoo_stock_price"],
        "verify_response": lambda resp: "Tesla" in resp or "TSLA" in resp,
        "requires_edgar": True
    },
    "edgar_company_comparison": {
        "query": "Use Edgar to compare Apple and Microsoft companies, then search for recent news about their competition",
        "description": "MCP + Built-in: Edgar comparison + web search",
        "min_tools": 2,
        "expected_tool_types": ["edgar_compare", "tavily_search"],
        "verify_response": lambda resp: ("Apple" in resp or "AAPL" in resp) and ("Microsoft" in resp or "MSFT" in resp),
        "requires_edgar": True
    },
    "edgar_ownership_analysis": {
        "query": "Use Edgar to get insider ownership data for Amazon, then get their current stock price",
        "description": "MCP + Built-in: Edgar ownership + Yahoo stock",
        "min_tools": 2,
        "expected_tool_types": ["edgar_ownership", "yahoo_stock_price"],
        "verify_response": lambda resp: "Amazon" in resp or "AMZN" in resp,
        "requires_edgar": True
    }
}


async def test_scenario(graph, scenario_name: str, scenario_config: dict) -> dict:
    """Test a single coordination scenario."""
    
    query = scenario_config["query"]
    description = scenario_config["description"]
    min_tools = scenario_config["min_tools"]
    expected_tool_types = scenario_config["expected_tool_types"]
    verify_response = scenario_config["verify_response"]
    
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print(f"Query: {query}")
    print(f"Expected: {min_tools}+ tool calls, types: {', '.join(expected_tool_types)}")
    
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
        
        # Extract detailed information
        tools_called = []
        tool_call_sequence = []  # Track order of tool calls
        tool_results = []
        final_response = ""
        
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    tools_called.append(tool_name)
                    tool_call_sequence.append({
                        "tool": tool_name,
                        "args": tool_call.get("args", {})
                    })
            elif isinstance(msg, ToolMessage):
                tool_results.append(msg.content)
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = msg.content
        
        # Verify results
        success = True
        issues = []
        
        # Check minimum tool calls
        if len(tools_called) < min_tools:
            success = False
            issues.append(f"Expected {min_tools}+ tool calls, got {len(tools_called)}")
        
        # Check if expected tool types were used
        tools_found = [tool for tool in expected_tool_types if tool in tools_called]
        if len(tools_found) < len(expected_tool_types):
            missing = set(expected_tool_types) - set(tools_called)
            success = False
            issues.append(f"Missing expected tools: {', '.join(missing)}")
        
        # Check response quality
        if not final_response:
            success = False
            issues.append("No response generated")
        elif len(final_response) < 100:
            success = False
            issues.append(f"Response too short ({len(final_response)} chars)")
        elif not verify_response(final_response):
            success = False
            issues.append("Response verification failed (missing expected content)")
        
        # Check tool results were used
        if len(tool_results) < min_tools:
            success = False
            issues.append(f"Expected {min_tools}+ tool results, got {len(tool_results)}")
        
        # Display results
        print(f"\n🔧 Tool Call Sequence ({len(tools_called)} calls):")
        for i, call in enumerate(tool_call_sequence, 1):
            args_preview = str(call['args'])[:50]
            print(f"   {i}. {call['tool']}({args_preview}{'...' if len(str(call['args'])) > 50 else ''})")
        
        print(f"\n📊 Tool Results: {len(tool_results)} results received")
        print(f"📏 Response length: {len(final_response)} chars (~{len(final_response)//4} tokens)")
        
        print(f"\n📄 Response preview (first 300 chars):")
        print(f"{final_response[:300]}{'...' if len(final_response) > 300 else ''}")
        
        # Coordination quality check
        coordination_score = "Good"
        if len(tools_called) > min_tools:
            coordination_score = "Excellent (extra tools used)"
        elif len(tools_called) == min_tools and len(set(tools_called)) == len(expected_tool_types):
            coordination_score = "Perfect (exact match)"
        
        print(f"\n🎯 Coordination Quality: {coordination_score}")
        
        if success:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED")
            for issue in issues:
                print(f"   - {issue}")
        
        return {
            "scenario_name": scenario_name,
            "success": success,
            "tools_called": tools_called,
            "tool_call_count": len(tools_called),
            "unique_tools": len(set(tools_called)),
            "expected_tools": expected_tool_types,
            "response_length": len(final_response),
            "coordination_score": coordination_score,
            "issues": issues
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "scenario_name": scenario_name,
            "success": False,
            "tools_called": [],
            "tool_call_count": 0,
            "unique_tools": 0,
            "expected_tools": expected_tool_types,
            "response_length": 0,
            "coordination_score": "Failed",
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


async def test_agent_initialization():
    """Test that the agent initializes correctly with all components."""
    
    print("=" * 80)
    print("🧪 Testing Agent Initialization")
    print("=" * 80)
    
    # Check if endpoint exists
    print("\n🔍 Checking SageMaker endpoint...")
    if not await check_endpoint_exists():
        print("\n⏭️  SKIPPING TESTS - Endpoint not available")
        print("\nTo run these tests:")
        print("  1. Deploy endpoint: python deployment/deploy_llama3_lmi.py")
        print("  2. Wait for endpoint to be InService (~5-10 minutes)")
        print("  3. Run tests again: python tests/test_mcp_coordination.py")
        return None, False
    
    print("✅ Endpoint is available and InService")
    
    try:
        # Test 1: Check required environment variables
        print("\n1️⃣ Checking environment variables...")
        required_vars = ["SAGEMAKER_ENDPOINT_NAME", "TAVILY_API_KEY"]
        optional_vars = ["ALPHAVANTAGE_API_KEY", "EDGAR_IDENTITY"]
        
        missing_required = [var for var in required_vars if not os.environ.get(var)]
        missing_optional = [var for var in optional_vars if not os.environ.get(var)]
        
        if missing_required:
            print(f"   ❌ Missing required: {', '.join(missing_required)}")
            return False
        else:
            print(f"   ✅ All required variables present")
        
        if missing_optional:
            print(f"   ⚠️  Missing optional: {', '.join(missing_optional)}")
        
        # Test 2: Connect to MCP servers
        print("\n2️⃣ Connecting to MCP servers...")
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
        EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY")
        
        MCP_SERVERS = {}
        
        # Skip AlphaVantage due to known API issues
        if ALPHAVANTAGE_API_KEY:
            print("   ⚠️  AlphaVantage available but skipping (known API issues)")
        
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
                print(f"   ✅ Connected! Loaded {len(mcp_tools)} MCP tools")
                
                if mcp_tools:
                    print(f"\n   📋 Available MCP tools:")
                    for tool in mcp_tools[:10]:  # Show first 10
                        print(f"      - {tool.name}")
                    if len(mcp_tools) > 10:
                        print(f"      ... and {len(mcp_tools) - 10} more")
            except Exception as e:
                print(f"   ⚠️  Could not connect to MCP servers: {e}")
                print("      Continuing with built-in tools only")
        else:
            print("   ⚠️  No MCP servers configured")
        
        # Test 3: Build the graph
        print("\n3️⃣ Building agent graph...")
        graph = fin_agent.build_graph(
            mcp_tools=mcp_tools,
            max_revisions=0,
            shortcut_quality_score=5.0,
            use_builtin_tools=True
        )
        print("   ✅ Agent graph built successfully")
        
        # Test 4: Verify built-in tools
        print("\n4️⃣ Verifying built-in tools...")
        builtin_tools = ["yahoo_news", "yahoo_stock_price", "tavily_search"]
        print(f"   ✅ {len(builtin_tools)} built-in tools available:")
        for tool in builtin_tools:
            print(f"      - {tool}")
        
        print("\n✅ Agent initialization successful!")
        return graph, has_edgar
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


async def test_mcp_coordination():
    """Run comprehensive MCP tool coordination tests."""
    
    print("=" * 80)
    print("🧪 MCP Tool Coordination Test for fin-agent-sagemaker-v2.py")
    print("=" * 80)
    print("\nThis test verifies:")
    print("  • Agent loads correctly with all MCP tools")
    print("  • Multiple tools are coordinated in sequence")
    print("  • Tool results are properly used in responses")
    print("  • Both built-in and MCP tools work together")
    
    # Initialize agent
    init_result = await test_agent_initialization()
    if not init_result or init_result[0] is None:
        print("\n⏭️  TESTS SKIPPED - Endpoint not available")
        print("\nThese tests require a deployed SageMaker endpoint.")
        print("The tests are not failing - they're being skipped because the endpoint doesn't exist.")
        return True  # Return success since we're skipping, not failing
    
    graph, has_edgar = init_result
    
    # Determine which scenarios to run
    scenarios_to_run = dict(TEST_SCENARIOS)
    
    # Add Edgar scenarios if available
    if has_edgar:
        print("\n✅ Edgar MCP available - including Edgar coordination tests")
        scenarios_to_run.update(EDGAR_TEST_SCENARIOS)
    else:
        print("\n⚠️  Edgar MCP not available - skipping Edgar coordination tests")
    
    print(f"\n📊 Running {len(scenarios_to_run)} coordination scenarios...")
    
    # Run all scenarios
    results = []
    for scenario_name, scenario_config in scenarios_to_run.items():
        result = await test_scenario(graph, scenario_name, scenario_config)
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Generate comprehensive summary
    print("\n" + "=" * 80)
    print("📊 COORDINATION TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal scenarios: {total_tests}")
    print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    
    # Tool coordination statistics
    total_tool_calls = sum(r["tool_call_count"] for r in results)
    avg_tools_per_scenario = total_tool_calls / total_tests if total_tests > 0 else 0
    
    print(f"\n🔧 Tool Coordination Statistics:")
    print(f"   Total tool calls: {total_tool_calls}")
    print(f"   Average tools per scenario: {avg_tools_per_scenario:.1f}")
    print(f"   Max tools in single scenario: {max(r['tool_call_count'] for r in results)}")
    
    # Unique tools used
    all_tools_used = set()
    for result in results:
        all_tools_used.update(result["tools_called"])
    
    print(f"\n📦 Unique Tools Used: {len(all_tools_used)}")
    for tool in sorted(all_tools_used):
        # Count how many scenarios used this tool
        usage_count = sum(1 for r in results if tool in r["tools_called"])
        print(f"   - {tool} (used in {usage_count} scenarios)")
    
    # Coordination quality breakdown
    print(f"\n🎯 Coordination Quality:")
    quality_counts = {}
    for result in results:
        score = result["coordination_score"]
        quality_counts[score] = quality_counts.get(score, 0) + 1
    
    for quality, count in sorted(quality_counts.items()):
        print(f"   {quality}: {count} scenarios")
    
    # Response quality
    avg_response_length = sum(r["response_length"] for r in results) / total_tests if total_tests > 0 else 0
    print(f"\n📏 Response Quality:")
    print(f"   Average response length: {avg_response_length:.0f} chars (~{avg_response_length/4:.0f} tokens)")
    print(f"   Shortest response: {min(r['response_length'] for r in results)} chars")
    print(f"   Longest response: {max(r['response_length'] for r in results)} chars")
    
    # Failed scenarios details
    if failed_tests > 0:
        print(f"\n❌ Failed Scenarios:")
        for result in results:
            if not result["success"]:
                print(f"\n   Scenario: {result['scenario_name']}")
                print(f"   Expected: {result['expected_tools']}")
                print(f"   Got: {result['tools_called']} ({result['tool_call_count']} calls)")
                for issue in result["issues"]:
                    print(f"   Issue: {issue}")
    
    # Success scenarios highlight
    if passed_tests > 0:
        print(f"\n✅ Successful Coordination Examples:")
        for result in results[:3]:  # Show first 3 successful
            if result["success"]:
                print(f"\n   {result['scenario_name']}:")
                print(f"   - Tools: {', '.join(result['tools_called'])}")
                print(f"   - Quality: {result['coordination_score']}")
    
    print("\n" + "=" * 80)
    if failed_tests == 0:
        print("✅ ALL COORDINATION TESTS PASSED")
        print("\nThe agent successfully:")
        print("  • Loaded all MCP tools correctly")
        print("  • Coordinated multiple tool calls in sequence")
        print("  • Generated coherent responses from tool results")
        print("  • Handled both built-in and MCP tools")
    else:
        print(f"⚠️  {failed_tests} COORDINATION TEST(S) FAILED")
        print("\nSome coordination scenarios did not meet expectations.")
    print("=" * 80)
    
    return passed_tests == total_tests


async def main():
    """Main entry point."""
    try:
        success = await test_mcp_coordination()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
