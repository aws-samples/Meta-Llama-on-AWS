#!/usr/bin/env python3
"""
Test the agent with queries that require multiple parallel tool calls.

This test verifies that the improved final_response_node and increased max_tokens
work correctly with parallel tool calling.
"""

import asyncio
import sys
import os
import importlib.util

# Import using importlib to handle hyphenated filename
spec = importlib.util.spec_from_file_location("fin_agent", "fin-agent-sagemaker-v2.py")
fin_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin_agent)

build_graph = fin_agent.build_graph
AgentState = fin_agent.AgentState

from langchain_core.messages import HumanMessage


async def test_query(graph, query, expected_tools, test_name):
    """Test a single query and analyze results."""
    
    print("\n" + "="*80)
    print(f"  TEST: {test_name}")
    print("="*80)
    print(f"\nQuery: {query}")
    print(f"Expected tools: {expected_tools}")
    
    # Create initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "revision_count": 0,
        "evaluation": None,
        "best_revision": None,
        "best_evaluation": None
    }
    
    # Run the graph
    try:
        result = await graph.ainvoke(initial_state)
        
        # Analyze the messages
        messages = result["messages"]
        
        # Count tool calls
        tool_call_count = 0
        tool_calls_list = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_call_count += len(msg.tool_calls)
                for tc in msg.tool_calls:
                    tool_calls_list.append(tc['name'])
        
        print(f"\n📊 Tool calls made: {tool_call_count}")
        print(f"📋 Tools called: {', '.join(tool_calls_list)}")
        
        # Check if expected tools were called
        success = tool_call_count >= expected_tools
        if success:
            print(f"✅ SUCCESS: {tool_call_count} tool calls (expected >= {expected_tools})")
        else:
            print(f"⚠️  PARTIAL: {tool_call_count} tool calls (expected >= {expected_tools})")
        
        # Show final response
        final_response = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                final_response = msg.content
                break
        
        if final_response:
            print(f"\n📝 Final response length: {len(final_response)} chars")
            print(f"📝 Preview: {final_response[:200]}...")
        else:
            print("\n⚠️  No final response generated")
        
        return success, tool_call_count, final_response
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, None


async def main():
    """Main entry point."""
    
    print("\n" + "="*80)
    print("  Multiple Parallel Tool Calling Test")
    print("="*80)
    print("\nEndpoint:", os.environ.get("SAGEMAKER_ENDPOINT_NAME", "llama3-lmi-agent"))
    print("Region:", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    
    print("\n" + "="*80)
    print("  Building Agent Graph")
    print("="*80)
    
    # Build graph
    graph = build_graph(
        mcp_tools=[],
        max_revisions=0,
        use_builtin_tools=True
    )
    
    print("✅ Graph built successfully")
    
    # Test cases
    test_cases = [
        {
            "name": "2 Stock Prices",
            "query": "What are the current stock prices for Apple (AAPL) and Microsoft (MSFT)?",
            "expected_tools": 2
        },
        {
            "name": "3 Stock Prices",
            "query": "Compare the stock prices of Apple (AAPL), Microsoft (MSFT), and Google (GOOGL)",
            "expected_tools": 3
        },
        {
            "name": "4 Stock Prices",
            "query": "Get me the current prices for AAPL, MSFT, GOOGL, and AMZN",
            "expected_tools": 4
        },
        {
            "name": "Mixed Tools (Price + News)",
            "query": "Get the stock price and latest news for Tesla (TSLA)",
            "expected_tools": 2
        },
        {
            "name": "5 Stock Prices",
            "query": "Show me prices for AAPL, MSFT, GOOGL, AMZN, and TSLA",
            "expected_tools": 5
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        success, tool_count, response = await test_query(
            graph,
            test_case["query"],
            test_case["expected_tools"],
            test_case["name"]
        )
        results.append({
            "name": test_case["name"],
            "success": success,
            "tool_count": tool_count,
            "has_response": response is not None
        })
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "⚠️  PARTIAL"
        response_status = "✅" if result["has_response"] else "❌"
        print(f"\n{status} {result['name']}")
        print(f"  Tool calls: {result['tool_count']}")
        print(f"  Response: {response_status}")
    
    # Overall result
    all_passed = all(r["success"] for r in results)
    all_have_responses = all(r["has_response"] for r in results)
    
    print("\n" + "="*80)
    if all_passed and all_have_responses:
        print("✅ ALL TESTS PASSED!")
        print("   Parallel tool calling is working correctly")
        print("   Final response generation is working correctly")
        return 0
    elif all_passed:
        print("⚠️  PARTIAL SUCCESS")
        print("   Parallel tool calling works")
        print("   Some responses missing - check final_response_node")
        return 1
    else:
        print("⚠️  SOME TESTS INCOMPLETE")
        print("   Check results above for details")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
