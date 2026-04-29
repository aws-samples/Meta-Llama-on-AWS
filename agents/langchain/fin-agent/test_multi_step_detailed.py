#!/usr/bin/env python3
"""
Detailed multi-step test showing all messages between tools.

This test provides comprehensive visibility into:
1. Agent's tool call decisions
2. Tool execution results
3. Message flow through the workflow
4. Final response synthesis
"""

import asyncio
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from typing import Any, Optional
from pydantic import Field

# Import the agent's build_graph
import importlib.util
spec = importlib.util.spec_from_file_location("fin_agent", "fin-agent-sagemaker-v2.py")
fin_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin_agent)

build_graph = fin_agent.build_graph
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "llama3-lmi-agent")


def create_test_tools():
    """Create tools for multi-step testing using StructuredTool."""
    
    # Tool 1: Get company ticker
    def get_company_ticker(company_name: str) -> str:
        """Get the stock ticker symbol for a company name.
        
        Args:
            company_name: Company name
        """
        return f"AAPL (called with: company_name={company_name})"
    
    ticker_tool = StructuredTool.from_function(
        func=get_company_ticker,
        name="get_company_ticker",
        description="Get the stock ticker symbol for a company name"
    )
    
    # Tool 2: Get stock price
    def get_stock_price(ticker: str) -> str:
        """Get current stock price for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
        """
        return f"$178.50 (called with: ticker={ticker})"
    
    price_tool = StructuredTool.from_function(
        func=get_stock_price,
        name="get_stock_price",
        description="Get current stock price for a ticker symbol"
    )
    
    # Tool 3: Get company revenue
    def get_company_revenue(ticker: str) -> str:
        """Get annual revenue for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
        """
        return f"$394 billion (called with: ticker={ticker})"
    
    revenue_tool = StructuredTool.from_function(
        func=get_company_revenue,
        name="get_company_revenue",
        description="Get annual revenue for a ticker symbol"
    )
    
    return [ticker_tool, price_tool, revenue_tool]


def print_separator(title: str, char: str = "="):
    """Print a formatted separator."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def print_message_details(msg, index: int):
    """Print detailed information about a message."""
    msg_type = type(msg).__name__
    
    print(f"Message {index}: {msg_type}")
    print("-" * 80)
    
    if isinstance(msg, HumanMessage):
        print(f"Role: user")
        print(f"Content: {msg.content}")
    
    elif isinstance(msg, AIMessage):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Role: assistant (calling tools)")
            print(f"Tool calls: {len(msg.tool_calls)}")
            for i, tc in enumerate(msg.tool_calls, 1):
                print(f"  {i}. {tc['name']}({tc['args']})")
                print(f"     Call ID: {tc['id']}")
        else:
            print(f"Role: assistant (text response)")
            content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"Content: {content_preview}")
    
    elif isinstance(msg, ToolMessage):
        print(f"Role: tool")
        print(f"Tool call ID: {msg.tool_call_id if hasattr(msg, 'tool_call_id') else 'N/A'}")
        print(f"Content: {msg.content}")
    
    print()


async def test_multi_step_detailed():
    """Run detailed multi-step test with full message visibility."""
    
    print_separator("DETAILED MULTI-STEP TEST WITH MESSAGE FLOW")
    
    print(f"🚀 Using endpoint: {SAGEMAKER_ENDPOINT_NAME}\n")
    
    # Create test tools
    print("🔧 Creating test tools...")
    test_tools = create_test_tools()
    print(f"✅ Created {len(test_tools)} test tools:")
    for tool in test_tools:
        print(f"   - {tool.name}: {tool.description}")
    
    # Build graph
    print("\n🔧 Building agent graph...")
    graph = build_graph(
        mcp_tools=test_tools,
        max_revisions=1,
        shortcut_quality_score=5.0,
        use_builtin_tools=False  # Only use test tools
    )
    print("✅ Graph built successfully")
    
    # Test query
    query = "What is Apple's stock ticker, current price, and annual revenue?"
    
    print_separator(f"TEST QUERY: {query}")
    
    # Run the agent
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "revision_count": 0,
        "evaluation": None,
        "best_revision": None,
        "best_evaluation": None
    }
    
    print("🔄 Processing query through agent workflow...\n")
    
    result = await graph.ainvoke(initial_state)
    
    # Analyze and display all messages
    print_separator("COMPLETE MESSAGE FLOW", "=")
    
    messages = result["messages"]
    print(f"Total messages in conversation: {len(messages)}\n")
    
    for i, msg in enumerate(messages, 1):
        print_message_details(msg, i)
    
    # Extract workflow stages
    print_separator("WORKFLOW STAGE ANALYSIS", "=")
    
    stage = 1
    tool_calls_made = []
    tool_results_received = []
    final_responses = []
    
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Stage {stage}: AGENT DECISION - Tool Calling")
            for tc in msg.tool_calls:
                tool_calls_made.append(tc['name'])
                print(f"  → Calling: {tc['name']}({tc['args']})")
            stage += 1
            print()
        
        elif isinstance(msg, ToolMessage):
            print(f"Stage {stage}: TOOL EXECUTION - Result Received")
            tool_results_received.append(msg.content)
            print(f"  → Result: {msg.content}")
            stage += 1
            print()
        
        elif isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            print(f"Stage {stage}: FINAL RESPONSE - Text Generation")
            final_responses.append(msg.content)
            content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            print(f"  → Response: {content_preview}")
            stage += 1
            print()
    
    # Summary statistics
    print_separator("SUMMARY STATISTICS", "=")
    
    print(f"Total tool calls made: {len(tool_calls_made)}")
    print(f"Tools called: {', '.join(tool_calls_made)}")
    print(f"\nTotal tool results received: {len(tool_results_received)}")
    print(f"\nFinal responses generated: {len(final_responses)}")
    
    # Show tool usage
    print("\n📊 Tool usage breakdown:")
    tool_names_used = set(tool_calls_made)
    for tool in test_tools:
        if tool.name in tool_names_used:
            count = tool_calls_made.count(tool.name)
            print(f"   - {tool.name}: {count} call(s)")
    
    # Extract and display final response
    if final_responses:
        print_separator("FINAL RESPONSE TO USER", "=")
        print(final_responses[-1])
    
    # Verification
    print_separator("VERIFICATION", "=")
    
    success = True
    issues = []
    
    if len(tool_calls_made) < 2:
        success = False
        issues.append(f"Expected at least 2 tool calls, got {len(tool_calls_made)}")
    
    if len(tool_results_received) != len(tool_calls_made):
        success = False
        issues.append(f"Tool calls ({len(tool_calls_made)}) != Tool results ({len(tool_results_received)})")
    
    if not final_responses:
        success = False
        issues.append("No final response generated")
    
    if success:
        print("✅ TEST PASSED")
        print(f"   - {len(tool_calls_made)} tool calls executed successfully")
        print(f"   - {len(tool_results_received)} tool results received")
        print(f"   - Final response generated correctly")
    else:
        print("❌ TEST FAILED")
        for issue in issues:
            print(f"   - {issue}")
    
    print_separator("TEST COMPLETE", "=")
    
    return success


async def main():
    """Main entry point."""
    try:
        success = await test_multi_step_detailed()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
