"""
Multi-tool financial agent using LangGraph with SageMaker Llama 3 function calling.

This version uses the ChatSagemakerWithTools wrapper with LlamaFunctionCallingHandler
to enable proper function calling support on SageMaker endpoints.

Environment variables:
    AWS credentials - Set up for SageMaker access via ~/.aws/credentials or environment variables
    
    SAGEMAKER_ENDPOINT_NAME - Your SageMaker endpoint name (default: 'llama3-lmi-agent')
                              Deploy endpoint first using: python deployment/deploy_llama3_lmi.py
    
    TAVILY_API_KEY - Your Tavily API key from https://tavily.com
    
    ALPHAVANTAGE_API_KEY - Your AlphaVantage API key from https://www.alphavantage.co/support/#api-key
    
    EDGAR_IDENTITY - Your Edgar AI identity email (required for SEC filings access)

Prerequisites:
    1. Deploy SageMaker endpoint with HuggingFace token:
       - Get token from: https://huggingface.co/settings/tokens
       - Accept Llama 3.1 license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
       - Set token: export HF_TOKEN="your_token_here"
       - Deploy: python deploy_llama3_lmi.py
    
    2. Set required API keys (see environment variables above)
    
    3. Run agent: uv run python fin-agent-sagemaker-v2.py

Note: The endpoint name must follow AWS naming rules: alphanumeric and hyphens only.
"""

import asyncio
import json
import os
import traceback
from typing import Annotated, Any, Dict, List, TypedDict
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

import boto3
import yfinance as yf
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

# Import the new SageMaker function calling components
from src.sagemaker_with_tools import ChatSagemakerWithTools
from src.content_handler import LlamaFunctionCallingHandler

# --- Configuration ---
# Missing environment variable and API keys are checked in run_cli() and will raise errors.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY")

# Debug mode — set via --debug flag or DEBUG=1 env var
# When off, only essential output is shown (tool calls, responses)
# When on, shows token counts, payload details, type coercion, etc.
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# MCP Server Configuration
# AlphaVantage MCP uses stdio transport with uvx for reliable connection
# Edgar MCP uses stdio transport with python module
MCP_SERVERS = {}

# Add Edgar if identity is configured
if EDGAR_IDENTITY:
    MCP_SERVERS["edgar"] = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "edgar.ai"],
        "env": {"EDGAR_IDENTITY": EDGAR_IDENTITY},
    }

# Add AlphaVantage if API key is configured
# Uses local server via uvx for reliable stdio transport
# Provides 80+ tools: technical indicators, fundamentals, economic data
if ALPHAVANTAGE_API_KEY:
    MCP_SERVERS["alphavantage"] = {
        "transport": "stdio",
        "command": "uvx",
        "args": ["av-mcp", ALPHAVANTAGE_API_KEY],
    }

SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "llama3-lmi-agent")

from botocore.config import Config as BotoConfig
sagemaker_client = boto3.Session().client(
    'sagemaker-runtime',
    config=BotoConfig(read_timeout=120)  # 70B model needs more time to generate
)
aws_region_name = boto3.Session().region_name


def _detect_endpoint_context_window(endpoint_name, fallback=16384):
    """Auto-detect the endpoint's max context window from its model config.

    Queries SageMaker to find the OPTION_MAX_MODEL_LEN environment variable
    set on the deployed model container. This avoids requiring the user to
    manually keep MAX_CONTEXT_TOKENS in sync with the deployment profile.

    Args:
        endpoint_name: SageMaker endpoint name.
        fallback: Default context window if detection fails.

    Returns:
        int: The endpoint's max model length.
    """
    try:
        sm_client = boto3.client("sagemaker", region_name=aws_region_name)
        # Endpoint → EndpointConfig → Model → Environment
        ep = sm_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = ep["EndpointConfigName"]
        ep_config = sm_client.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = ep_config["ProductionVariants"][0]["ModelName"]
        model = sm_client.describe_model(ModelName=model_name)
        env = model["PrimaryContainer"].get("Environment", {})
        max_len = env.get("OPTION_MAX_MODEL_LEN")
        if max_len:
            detected = int(max_len)
            if DEBUG_MODE:
                print(f"  🔍 Auto-detected endpoint context window: {detected} tokens")
            return detected
    except Exception as e:
        if DEBUG_MODE:
            print(f"  ⚠️  Could not auto-detect context window: {e}")
    return fallback


# Auto-detect context window from the live endpoint, or fall back to env var / default.
# The env var override is still supported for offline testing or edge cases.
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "0")) or \
    _detect_endpoint_context_window(SAGEMAKER_ENDPOINT_NAME)

# --- Evaluator Prompt ---

EVALUATOR_PROMPT = """You are an expert evaluator for a financial research AI assistant. Your task is to objectively assess the quality of the assistant's response based on the user's query, the tool results retrieved, and the final response provided.

## Evaluation Dimensions

Score each dimension from 1-5 using the rubrics below:

### 1. FACTUAL ACCURACY (factual_accuracy)
Measures whether the response accurately reflects the data from tool results without hallucination.

- **5 (Excellent)**: All facts, numbers, prices, and statistics are perfectly accurate and directly traceable to tool results. No hallucinated information.
- **4 (Good)**: Nearly all information is accurate with minor imprecisions that don't affect the overall correctness.
- **3 (Adequate)**: Mostly accurate but contains some unsupported claims or minor factual errors.
- **2 (Poor)**: Multiple factual errors or significant information not grounded in tool results.
- **1 (Unacceptable)**: Largely inaccurate, contains hallucinated data, or contradicts tool results.

### 2. TOOL SELECTION APPROPRIATENESS (tool_selection)
Measures whether the agent chose the right tools for the user's query.

- **5 (Excellent)**: Perfect tool selection—used exactly the tools needed, no unnecessary calls, covered all aspects of the query.
- **4 (Good)**: Appropriate tools selected with perhaps one minor suboptimal choice.
- **3 (Adequate)**: Reasonable tool selection but missed an opportunity or made an unnecessary call.
- **2 (Poor)**: Significant mismatch—used wrong tools or missed important ones.
- **1 (Unacceptable)**: Completely wrong tool selection that fails to address the query.

Tool Reference:
- yahoo_news: For stock/company news (requires ticker symbol)
- yahoo_stock_price: For current stock prices and market data (requires ticker symbol)
- tavily_search: For general web searches, non-stock queries, or when no ticker is available

### 3. COMPLETENESS (completeness)
Measures whether the response fully addresses all aspects of the user's query.

- **5 (Excellent)**: Comprehensively addresses every aspect of the query with appropriate depth.
- **4 (Good)**: Addresses all main points with minor gaps in secondary details.
- **3 (Adequate)**: Covers the core question but misses some relevant aspects.
- **2 (Poor)**: Significant omissions—key parts of the query are not addressed.
- **1 (Unacceptable)**: Fails to address the main question or provides superficial response.

### 4. CLARITY & STRUCTURE (clarity_structure)
Measures how well-organized, readable, and professionally formatted the response is.

- **5 (Excellent)**: Exceptionally clear, well-structured with appropriate headings/bullets, professional financial writing.
- **4 (Good)**: Clear and organized with good formatting, easy to follow.
- **3 (Adequate)**: Understandable but could be better organized or formatted.
- **2 (Poor)**: Disorganized, hard to follow, or poorly formatted.
- **1 (Unacceptable)**: Confusing, incoherent, or unreadable.

### 5. RELEVANCE (relevance)
Measures whether the response stays focused on what the user asked.

- **5 (Excellent)**: Perfectly focused—every piece of information directly relates to the query.
- **4 (Good)**: Mostly relevant with minimal tangential information.
- **3 (Adequate)**: Relevant but includes some unnecessary information.
- **2 (Poor)**: Contains significant irrelevant content or goes off-topic.
- **1 (Unacceptable)**: Mostly irrelevant or fails to address the user's actual question.

### 6. ACTIONABILITY (actionability)
Measures whether the response provides useful, actionable insights beyond raw data.

- **5 (Excellent)**: Provides clear insights, context, implications, and/or suggested next steps where appropriate.
- **4 (Good)**: Offers good context and some actionable insights.
- **3 (Adequate)**: Presents data with basic context but limited actionable guidance.
- **2 (Poor)**: Raw data dump with little interpretation or context.
- **1 (Unacceptable)**: No useful insights, just restates data without any added value.

## Output Format

You MUST respond with valid JSON only, no other text. Use this exact structure:

{
  "scores": {
    "factual_accuracy": <1-5>,
    "tool_selection": <1-5>,
    "completeness": <1-5>,
    "clarity_structure": <1-5>,
    "relevance": <1-5>,
    "actionability": <1-5>
  },
  "reasoning": {
    "factual_accuracy": "<1-2 sentence explanation>",
    "tool_selection": "<1-2 sentence explanation>",
    "completeness": "<1-2 sentence explanation>",
    "clarity_structure": "<1-2 sentence explanation>",
    "relevance": "<1-2 sentence explanation>",
    "actionability": "<1-2 sentence explanation>"
  },
  "summary": "<2-3 sentence overall assessment of response quality>"
}

## Evaluation Guidelines

1. Be objective and consistent in your scoring
2. Consider the financial domain context—users expect accuracy with financial data
3. Evaluate based on what was possible given the tool results, not hypothetically
4. If tools returned errors, evaluate how well the agent handled those errors
5. A response can be excellent in some dimensions while poor in others—score independently"""


# --- Tools ---

@tool
def yahoo_news(ticker: str) -> str:
    """Get the latest news for a stock ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return f"No news found for {ticker}"

        result = f"Latest news for {ticker}:\n\n"
        for i, article in enumerate(news[:5], 1):
            content = article.get("content", {})
            title = content.get("title", "No title")
            summary = content.get("summary", "No summary available")
            provider = content.get("provider", {}).get("displayName", "Unknown")
            result += f"{i}. {title}\n   Source: {provider}\n   Summary: {summary}\n\n"
        return result
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching news for {ticker}: {str(e)}"


@tool
def yahoo_stock_price(ticker: str) -> str:
    """Get the current stock price and basic info for a ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        prev_close = info.get("previousClose", "N/A")
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap:,.0f}"

        day_high = info.get("dayHigh", "N/A")
        day_low = info.get("dayLow", "N/A")
        volume = info.get("volume", "N/A")
        if isinstance(volume, (int, float)):
            volume = f"{volume:,}"

        name = info.get("shortName", ticker)

        return f"""Stock Information for {name} ({ticker}):
- Current Price: ${price}
- Previous Close: ${prev_close}
- Day Range: ${day_low} - ${day_high}
- Volume: {volume}
- Market Cap: {market_cap}"""
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching stock price for {ticker}: {str(e)}"


@tool
def tavily_search(query: str) -> str:
    """Search the internet for general information using Tavily.

    Args:
        query: The search query
    """
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=5)

        results = response.get("results", [])
        if not results:
            return f"No results found for: {query}"

        output = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "")
            output += f"{i}. {title}\n   {content[:200]}...\n   URL: {url}\n\n"
        return output
    except Exception as e:
        traceback.print_exc()
        return f"Error searching: {str(e)}"


# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    revision_count: int
    evaluation: dict | None  # Stores the latest evaluation result
    best_revision: str | None  # Stores the best revision content so far
    best_evaluation: dict | None  # Stores the best evaluation result so far


# --- Graph Nodes ---

tools = [yahoo_news, yahoo_stock_price, tavily_search]


def create_agent_node(llm_with_tools, available_tools):
    """Create the main agent node that decides which tools to use."""
    
    # Build tool list
    tool_list = "\n".join([f"- {tool.name}: {tool.description[:80]}..." for tool in available_tools[:15]])

    system_prompt = f"""You are a financial research assistant. Use the available tools to answer queries.

Tools:
{tool_list}

Rules:
1. Call ONE tool at a time. Wait for the result before calling the next.
2. For stock prices, use yahoo_stock_price(ticker). Do NOT use edgar_compare for prices.
3. edgar_compare only accepts: revenue, net_income, gross_profit, operating_income, assets, liabilities, equity, margins, growth. Do NOT pass cash, price, or any other metric.
4. If a tool returns incomplete data, call alternative tools for the missing items.
5. Always use proper JSON types in arguments: arrays [], booleans true/false, integers without quotes.
6. After collecting all data, synthesize a clear response."""

    # Pre-calculate tool schema overhead (this is constant per request)
    _tool_schema_tokens = 0
    for t in available_tools:
        _tool_schema_tokens += len(t.name) // 4 + len(getattr(t, 'description', '') or '') // 4
        if hasattr(t, 'args_schema'):
            schema = t.args_schema
            if isinstance(schema, dict):
                _tool_schema_tokens += len(json.dumps(schema)) // 4
            elif hasattr(schema, 'model_json_schema'):
                _tool_schema_tokens += len(json.dumps(schema.model_json_schema())) // 4
            elif hasattr(schema, 'schema'):
                _tool_schema_tokens += len(json.dumps(schema.schema())) // 4
    # Add overhead for JSON structure wrapping each tool
    _tool_schema_tokens += len(available_tools) * 20

    def _estimate_tokens(msgs):
        """Rough token estimate (~4 chars per token) with safety margin.
        
        The len/4 heuristic underestimates by ~10-15% vs the actual Llama
        tokenizer, especially for JSON-heavy tool schemas and structured
        content. We apply a 1.15x multiplier to compensate.
        """
        total = 0
        for m in msgs:
            total += len(str(m.content) if hasattr(m, 'content') else str(m)) // 4
            if hasattr(m, 'tool_calls') and m.tool_calls:
                total += len(json.dumps(m.tool_calls)) // 4
        return int(total * 1.15)  # 15% safety margin for tokenizer mismatch

    def _trim_messages(msgs, max_ctx=None, max_completion=512):
        """Drop oldest tool call/result pairs when approaching context limit.

        Preserves: system prompt (idx 0), user message (idx 1), and the
        most recent 4 messages so the model has enough recent context.
        
        Accounts for tool schema overhead that gets injected into every request.
        """
        if max_ctx is None:
            max_ctx = MAX_CONTEXT_TOKENS
        
        # Budget = context limit - completion tokens - tool schema overhead
        # Apply safety margin to tool schema too
        tool_overhead = int(_tool_schema_tokens * 1.15)
        budget = max_ctx - max_completion - tool_overhead
        
        if DEBUG_MODE:
            print(f"  📐 Token budget: {max_ctx} - {max_completion} completion - {tool_overhead} tools (w/ margin) = {budget}")
        
        # First pass: truncate long tool results (ToolMessage content)
        trimmed = []
        for m in msgs:
            if isinstance(m, ToolMessage) and m.content and len(str(m.content)) > 2500:
                # Truncate tool results to ~625 tokens to save space
                truncated_content = str(m.content)[:2500] + "\n...[truncated]"
                trimmed.append(ToolMessage(content=truncated_content, tool_call_id=m.tool_call_id))
            else:
                trimmed.append(m)
        
        # Second pass: drop oldest tool call/result pairs if still over budget
        head, tail = 2, 4  # keep first 2 (system + user) and last 4
        while _estimate_tokens(trimmed) > budget and len(trimmed) > head + tail:
            removed = trimmed.pop(head)
            if DEBUG_MODE:
                print(f"  🗑️  Trimmed {removed.__class__.__name__} to fit context window")
        
        if DEBUG_MODE:
            print(f"  📏 Context after trim: ~{_estimate_tokens(trimmed)} tokens (budget: {budget})")
        return trimmed

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        messages = _trim_messages(messages)
        response = llm_with_tools.invoke(messages)
        
        # Debug: Print what the model returned
        if DEBUG_MODE:
            print(f"\n[DEBUG] Agent response type: {type(response)}")
            print(f"[DEBUG] Has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                print(f"[DEBUG] Tool calls: {response.tool_calls}")
            if hasattr(response, 'content'):
                print(f"[DEBUG] Content preview: {str(response.content)[:200]}...")
        
        # CRITICAL: Validate and fix tool calls BEFORE returning
        # This prevents parallel calls from reaching SageMaker
        if hasattr(response, 'tool_calls') and response.tool_calls:
            fixed_calls = []
            needs_fixing = False
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                # Check for invalid edgar_compare with price metrics
                if tool_name == "edgar_compare":
                    metrics = tool_args.get("metrics", [])
                    identifiers = tool_args.get("identifiers", [])
                    
                    # Check if any price-related metric is in the list
                    price_metrics = {"price", "stock_price", "current_price", "market_price", "share_price"}
                    has_price = any(m.lower() in price_metrics for m in metrics)
                    
                    if has_price:
                        needs_fixing = True
                        if DEBUG_MODE:
                            print(f"\n⚠️  VALIDATION: Detected invalid edgar_compare call with price metrics")
                            print(f"   Metrics: {metrics}")
                            print(f"   Identifiers: {identifiers}")
                            print(f"   🔧 Converting to yahoo_stock_price calls...")
                        
                        # Create yahoo_stock_price calls for each identifier
                        for identifier in identifiers:
                            fixed_call = {
                                "name": "yahoo_stock_price",
                                "args": {"ticker": identifier},
                                "id": f"{tool_call.get('id', 'fixed')}-{identifier}",
                                "type": "tool_call"
                            }
                            fixed_calls.append(fixed_call)
                            if DEBUG_MODE:
                                print(f"   ✅ Created: yahoo_stock_price(ticker='{identifier}')")
                        
                        continue  # Skip the invalid edgar_compare call
                
                # Keep valid tool calls as-is
                fixed_calls.append(tool_call)
            
            # CRITICAL: Ensure only ONE tool call (no parallel calls)
            if len(fixed_calls) > 1:
                needs_fixing = True
                if DEBUG_MODE:
                    print(f"\n⚠️  VALIDATION: Detected {len(fixed_calls)} parallel tool calls")
                    print(f"   SageMaker endpoint only supports ONE tool call at a time")
                    print(f"   🔧 Keeping only the FIRST tool call: {fixed_calls[0]['name']}")
                    print(f"   ℹ️  Agent will call remaining tools in subsequent turns")
                fixed_calls = [fixed_calls[0]]  # Keep only the first call
            
            if needs_fixing:
                # Create a new AIMessage with fixed tool calls
                response = AIMessage(
                    content=response.content or "",
                    tool_calls=fixed_calls
                )
                if DEBUG_MODE:
                    print(f"   ✅ Tool calls validated and fixed\n")
        
        return {"messages": [response]}

    return agent


def create_final_response_node(llm_no_tools):
    """Create the final response node that generates text without tool calling.
    
    With parallel tool calling enabled, we can now use a more complete approach.
    """

    def final_response(state: AgentState) -> dict:
        if DEBUG_MODE:
            print("\n" + "="*80)
            print("🔍 FINAL_RESPONSE NODE - GENERATING RESPONSE")
            print("="*80)

        messages = state["messages"]
        
        # Extract user query
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # Extract tool results with reasonable truncation (500 chars per tool)
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, list):
                    content = str(content)
                # Truncate to 3000 chars to preserve financial data tables
                truncated = content[:3000]
                if len(content) > 3000:
                    truncated += "..."
                tool_results.append(truncated)
        
        # Build a clear prompt for final response
        system_msg = (
            "You are a helpful financial assistant. Synthesize the tool results below into a clear, "
            "comprehensive response. Present actual numbers and data from the results. "
            "Do NOT suggest the user visit other websites — answer directly with the data you have."
        )
        
        # Include user query and tool results
        context_parts = [f"User query: {user_query}"]
        if tool_results:
            context_parts.append(f"\nTool results:\n" + "\n---\n".join(tool_results))
        
        combined_context = "\n".join(context_parts)
        
        if DEBUG_MODE:
            print(f"📊 Context length: {len(combined_context)} chars (~{len(combined_context)//4} tokens)")
            print(f"📊 Tool results included: {len(tool_results)}")
            print("="*80)

        # Send with system message for better responses
        final_messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=combined_context)
        ]

        # Invoke with reasonable token limit
        response = llm_no_tools.invoke(final_messages)

        print("✅ Response generated")
        return {"messages": [response]}

    return final_response


def create_reviser_node(llm):
    """Create the reviser node that improves the response."""

    system_prompt = """You are a critical reviewer and response improver. Your job is to:

1. Review the previous response for accuracy, completeness, and clarity
2. Identify any gaps, errors, or areas that could be improved
3. Provide an improved, more comprehensive response

If the response is already excellent and complete, you can confirm it's good.
Always maintain factual accuracy based on the tool results provided.

Respond with your improved/revised response directly."""

    def reviser(state: AgentState) -> dict:
        messages = state["messages"]

        # Find the last AI response that's not a tool call
        last_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                last_response = msg.content
                break

        if not last_response:
            return {"messages": [], "revision_count": state.get("revision_count", 0)}

        revision_prompt = f"""Please review and improve this response if needed:

{last_response}

Provide your revised response:"""

        revision_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=revision_prompt)
        ]

        response = llm.invoke(revision_messages)
        new_count = state.get("revision_count", 0) + 1

        return {
            "messages": [AIMessage(content=f"[Revision {new_count}] {response.content}")],
            "revision_count": new_count
        }

    return reviser


def validate_and_fix_tool_calls(state: AgentState) -> dict:
    """
    Validate tool calls and fix common errors BEFORE execution.
    
    This is a safety layer that catches invalid tool calls that the model
    makes despite system prompt warnings. Specifically:
    - Prevents edgar_compare from being called with 'price' metrics
    - Converts invalid edgar_compare price calls to yahoo_stock_price calls
    - Ensures only ONE tool call at a time (no parallel calls)
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        return {"messages": []}  # No tool calls to validate
    
    fixed_calls = []
    needs_fixing = False
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        # Check for invalid edgar_compare with price metrics
        if tool_name == "edgar_compare":
            metrics = tool_args.get("metrics", [])
            identifiers = tool_args.get("identifiers", [])
            
            # Check if any price-related metric is in the list
            price_metrics = {"price", "stock_price", "current_price", "market_price", "share_price"}
            has_price = any(m.lower() in price_metrics for m in metrics)
            
            if has_price:
                needs_fixing = True
                if DEBUG_MODE:
                    print(f"\n⚠️  VALIDATION: Detected invalid edgar_compare call with price metrics")
                    print(f"   Metrics: {metrics}")
                    print(f"   Identifiers: {identifiers}")
                    print(f"   🔧 Converting to yahoo_stock_price calls...")
                
                # Create yahoo_stock_price calls for each identifier
                for identifier in identifiers:
                    fixed_call = {
                        "name": "yahoo_stock_price",
                        "args": {"ticker": identifier},
                        "id": f"{tool_call.get('id', 'fixed')}-{identifier}",
                        "type": "tool_call"
                    }
                    fixed_calls.append(fixed_call)
                    if DEBUG_MODE:
                        print(f"   ✅ Created: yahoo_stock_price(ticker='{identifier}')")
                
                continue  # Skip the invalid edgar_compare call
        
        # Keep valid tool calls as-is
        fixed_calls.append(tool_call)
    
    # CRITICAL: Ensure only ONE tool call (no parallel calls)
    if len(fixed_calls) > 1:
        needs_fixing = True
        if DEBUG_MODE:
            print(f"\n⚠️  VALIDATION: Detected {len(fixed_calls)} parallel tool calls")
            print(f"   SageMaker endpoint only supports ONE tool call at a time")
            print(f"   🔧 Keeping only the FIRST tool call: {fixed_calls[0]['name']}")
            print(f"   ℹ️  Agent will call remaining tools in subsequent turns")
        fixed_calls = [fixed_calls[0]]  # Keep only the first call
    
    if needs_fixing:
        # Create a new AIMessage with fixed tool calls
        fixed_message = AIMessage(
            content=last_message.content or "",
            tool_calls=fixed_calls
        )
        
        # Replace the last message
        new_messages = messages[:-1] + [fixed_message]
        if DEBUG_MODE:
            print(f"   ✅ Tool calls validated and fixed\n")
        return {"messages": new_messages}
    
    # No fixes needed
    return {"messages": []}


def check_tool_errors(state: AgentState) -> dict:
    """Inspect the last tool result. If it's an error, inject a hint so the model
    tries an alternative tool instead of retrying the same one."""
    messages = state["messages"]
    if not messages:
        return {"messages": []}

    last = messages[-1]
    if not isinstance(last, ToolMessage):
        return {"messages": []}

    content = str(last.content) if last.content else ""
    error_indicators = [
        "Error:",
        "error",
        "Traceback",
        "Exception",
        "validation error",
        "failed",
        "timed out",
    ]
    is_error = any(ind.lower() in content.lower() for ind in error_indicators)

    if not is_error:
        return {"messages": []}

    # Find which tool failed by looking at the preceding AIMessage
    failed_tool = "the previous tool"
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            failed_tool = msg.tool_calls[-1].get("name", failed_tool)
            break

    # Count how many times this specific tool has failed
    fail_count = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            msg_content = str(msg.content).lower()
            if any(ind.lower() in msg_content for ind in error_indicators):
                # Check if the preceding AI message called the same tool
                for prev in reversed(messages[:i]):
                    if isinstance(prev, AIMessage) and prev.tool_calls:
                        if prev.tool_calls[-1].get("name") == failed_tool:
                            fail_count += 1
                        break

    if DEBUG_MODE:
        print(f"\n⚠️  Tool error detected: {failed_tool} (failed {fail_count}x)")

    hint = (
        f"The tool '{failed_tool}' returned an error. Do NOT retry it. "
        f"Use a different tool to get the information. "
        f"For financial data, try tavily_search or yahoo_stock_price/yahoo_news instead."
    )
    return {"messages": [SystemMessage(content=hint)]}


def should_continue(state: AgentState) -> str:
    """Determine next step after agent node.
    
    This function enables multi-turn tool calling while preventing infinite loops.
    It allows the agent to call tools multiple times as long as it's making progress
    (calling different tools or with different arguments).
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If last message has tool calls, execute tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        current_calls = [(tc['name'], str(tc['args'])) for tc in last_message.tool_calls]
        
        # Count how many times we've seen these exact calls
        call_count = 0
        for msg in messages[:-1]:  # Exclude last message
            if isinstance(msg, AIMessage) and msg.tool_calls:
                msg_calls = [(tc['name'], str(tc['args'])) for tc in msg.tool_calls]
                if msg_calls == current_calls:  # Exact match
                    call_count += 1
        
        # If we've seen these exact calls 2+ times, it's a loop
        if call_count >= 2:
            if DEBUG_MODE:
                print(f"[Loop detected: {call_count} repetitions of {[c[0] for c in current_calls]}]")
            return "final_response"
        
        # Execute tools (validation already happened in agent node)
        return "tools"
    
    # No tool calls, generate final response
    return "final_response"


def create_should_continue_revision(max_revisions: int = 3, shortcut_quality_score: float = 4.95):
    """Create the revision continuation checker with configurable max revisions."""

    def should_continue_revision(state: AgentState) -> str:
        """Determine if more revisions are needed based on evaluation score."""
        revision_count = state.get("revision_count", 0)
        evaluation = state.get("evaluation")

        # If we've hit max revisions, stop
        if revision_count >= max_revisions:
            return END

        # If evaluation score is good enough, stop
        if evaluation and evaluation.get("overall_score", 0) >= shortcut_quality_score:
            return END

        # Otherwise, continue revising
        return "reviser"

    return should_continue_revision


def create_evaluator_node(llm):
    """Create the evaluator node that scores the response."""

    def evaluator(state: AgentState) -> dict:
        messages = state["messages"]

        # Extract user query (first human message)
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = str(msg.content)
                break

        # Extract tool results
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results.append(msg.content)

        # Extract tools called
        tools_called = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tools_called.append(tool_call["name"])

        # Get the last AI response (the one to evaluate)
        agent_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_response = str(msg.content)
                break

        if not agent_response:
            return {"evaluation": None}

        # Run evaluation
        evaluation = evaluate_response(
            llm=llm,
            user_query=user_query,
            tool_results=tool_results,
            agent_response=agent_response,
            tools_called=tools_called
        )

        # Print evaluation score
        overall_score = evaluation.get("overall_score", 0.0)
        print(f"[Evaluator: score = {overall_score}/5.0]")

        return {"evaluation": evaluation}

    return evaluator


def create_select_best_revision_node():
    """Create the node that selects the best revision based on evaluation score."""

    def select_best_revision(state: AgentState) -> dict:
        messages = state["messages"]
        current_evaluation = state.get("evaluation")
        best_evaluation = state.get("best_evaluation")

        # Get the current revision content (last AI message without tool calls)
        current_revision = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                current_revision = msg.content
                break

        if not current_evaluation:
            return {}

        current_score = current_evaluation.get("overall_score", 0.0)
        best_score = best_evaluation.get("overall_score", 0.0) if best_evaluation else 0.0

        # Compare and select the best
        if current_score > best_score:
            print(f"[Select Best: new best score {current_score}/5.0 > previous {best_score}/5.0]")
            return {
                "best_revision": current_revision,
                "best_evaluation": current_evaluation
            }
        else:
            print(f"[Select Best: keeping previous best {best_score}/5.0 >= current {current_score}/5.0]")
            return {}  # Keep existing best

    return select_best_revision


# --- Evaluator Function ---

def evaluate_response(
    llm,
    user_query: str,
    tool_results: list[str],
    agent_response: str,
    tools_called: list[str],
) -> dict:
    """
    Evaluate the agent's response using LLM-as-a-judge.

    Args:
        llm: The LLM instance to use for evaluation
        user_query: The original user query
        tool_results: List of tool result strings
        agent_response: The final response from the agent
        tools_called: List of tool names that were called

    Returns:
        Dictionary with scores, reasoning, overall_score, and summary
    """
    # Format tool results for evaluation
    tool_results_text = "\n\n".join(
        f"[Tool Result {i+1}]:\n{result}"
        for i, result in enumerate(tool_results)
    ) if tool_results else "No tool results available."

    tools_called_text = ", ".join(tools_called) if tools_called else "No tools called"

    evaluation_request = f"""## User Query
{user_query}

## Tools Called
{tools_called_text}

## Tool Results
{tool_results_text}

## Agent Response
{agent_response}

Please evaluate this response according to the rubric and provide your assessment in JSON format."""

    messages = [
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=evaluation_request)
    ]

    response = llm.invoke(messages)

    # Parse the JSON response
    try:
        # Extract JSON from response (handle potential markdown code blocks)
        response_text = str(response.content).strip()

        if response_text.startswith("```"):
            # Remove markdown code block
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        evaluation = json.loads(response_text)

        # Calculate overall score (average of all dimensions, scaled to 1.0-5.0)
        scores = evaluation.get("scores", {})
        if scores:
            total = sum(scores.values())
            overall = total / len(scores)  # Already 1-5 scale
            evaluation["overall_score"] = round(overall, 2)
        else:
            evaluation["overall_score"] = 0.0

        return evaluation

    except json.JSONDecodeError as e:
        traceback.print_exc()
        return {
            "error": f"Failed to parse evaluation response: {e}",
            "raw_response": response.content,
            "scores": {},
            "overall_score": 0.0,
            "reasoning": {},
            "summary": "Evaluation parsing failed"
        }


# --- Build Graph ---

def build_graph(mcp_tools: list, max_revisions: int = 3, shortcut_quality_score: float = 4.95, graph_filename: str | None = None, use_builtin_tools: bool = True):
    """Build the LangGraph workflow.

    Args:
        mcp_tools: List of MCP tools to include alongside built-in tools
        max_revisions: Maximum number of revisions before stopping (default: 3)
        shortcut_quality_score: Stop early if evaluation score reaches this threshold (default: 4.95)
        graph_filename: Filename for saving the graph visualization (optional)
        use_builtin_tools: Whether to include built-in tools (yahoo_news, yahoo_stock_price, tavily_search). 
                          Set to False for testing to avoid tool overlap (default: True)
    """

    # Combine built-in tools with MCP tools based on use_builtin_tools flag
    if use_builtin_tools:
        all_tools = tools + (mcp_tools or [])
    else:
        all_tools = mcp_tools or []  # Only use MCP/test tools

    # Initialize LlamaFunctionCallingHandler
    content_handler = LlamaFunctionCallingHandler()

    # Create ChatSagemakerWithTools instance
    llm = ChatSagemakerWithTools(
        endpoint_name=SAGEMAKER_ENDPOINT_NAME,
        region_name=aws_region_name,
        content_handler=content_handler,
        client=sagemaker_client,
        model_kwargs={"temperature": 0.1, "max_tokens": 2048}  # Increased for agent workflows with tool calling
    )

    # Bind tools using bind_tools method
    llm_with_tools = llm.bind_tools(all_tools)
    
    # Create LLM WITHOUT tools for final response generation
    # This forces the model to generate text instead of calling tools again
    llm_no_tools = ChatSagemakerWithTools(
        endpoint_name=SAGEMAKER_ENDPOINT_NAME,
        region_name=aws_region_name,
        content_handler=LlamaFunctionCallingHandler(),  # New handler instance
        client=sagemaker_client,
        model_kwargs={"temperature": 0.1, "max_tokens": 2048}  # Increased for agent workflows with tool calling
    )
    # Don't bind tools to llm_no_tools - this forces text generation

    # Create nodes
    agent_node = create_agent_node(llm_with_tools, all_tools)
    tool_node = ToolNode(all_tools, handle_tool_errors=True)
    final_response_node = create_final_response_node(llm_no_tools)
    reviser_node = create_reviser_node(llm_no_tools)  # Use llm_no_tools for reviser too
    evaluator_node = create_evaluator_node(llm_no_tools)  # Use llm_no_tools for evaluator
    select_best_node = create_select_best_revision_node()
    should_continue_revision = create_should_continue_revision(max_revisions, shortcut_quality_score)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("check_tool_errors", check_tool_errors)
    workflow.add_node("final_response", final_response_node)
    workflow.add_node("reviser", reviser_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("select_best_revision", select_best_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "check_tool_errors")  # After tools, check for errors first
    workflow.add_edge("check_tool_errors", "agent")  # Then back to agent (with hint if error)
    workflow.add_edge("final_response", END)  # Skip evaluator/reviser - go directly to END
    # Disabled for now due to context length issues:
    # workflow.add_edge("final_response", "reviser")
    # workflow.add_edge("reviser", "evaluator")
    # workflow.add_edge("evaluator", "select_best_revision")
    # workflow.add_conditional_edges("select_best_revision", should_continue_revision)

    graph = workflow.compile()

    # Get PNG bytes from the graph
    if graph_filename:
        png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open(graph_filename, "wb") as f:
            f.write(png_bytes)

        print(f"Graph saved to {graph_filename}")
    return graph


# graph_filename is optional argument for saving the graph visualization. Make it str or None.
async def run_cli(evaluate_mode: bool = False, max_revisions: int = 0, shortcut_quality_score: float = 4.95, graph_filename: str | None = None):  
    """Run the interactive CLI loop.

    Args:
        evaluate_mode: If True, evaluate each response using LLM-as-a-judge
        max_revisions: Maximum number of revisions before stopping (default: 3)
        shortcut_quality_score: Stop early if evaluation score reaches this threshold (default: 4.95)
        graph_filename: Filename for saving the graph visualization
    """

    # Check required API keys - make AlphaVantage optional for testing
    missing_keys = []
    
    if not TAVILY_API_KEY:
        missing_keys.append("TAVILY_API_KEY")
    if not SAGEMAKER_ENDPOINT_NAME:
        missing_keys.append("SAGEMAKER_ENDPOINT_NAME")
    
    # Optional keys - warn but don't fail
    optional_missing = []
    if not ALPHAVANTAGE_API_KEY:
        optional_missing.append("ALPHAVANTAGE_API_KEY (AlphaVantage MCP tools will be unavailable)")
    if not EDGAR_IDENTITY:
        optional_missing.append("EDGAR_IDENTITY (Edgar SEC filing tools will be unavailable)")
    
    if missing_keys:
        raise ValueError(f"Required environment variables not set: {', '.join(missing_keys)}")
    
    if optional_missing:
        print("\n⚠️  Optional API keys not set:")
        for key in optional_missing:
            print(f"   - {key}")
        print("   Agent will run with limited tools.\n")

    print("=" * 60)
    print("Financial Research Agent with Reflection (SageMaker v2)")
    if evaluate_mode:
        print("📋 Evaluation mode: ON")
    print(f"🔄 Max revisions: {max_revisions}")
    print(f"⭐ Shortcut quality score: {shortcut_quality_score}")
    print("=" * 60)
    print("Available tools:")
    print("  - Yahoo News (stock news)")
    print("  - Yahoo Stock Price (current prices)")
    print("  - Tavily Search (general web search)")
    print("  - AlphaVantage MCP (financial data)")
    print("  - Edgar AI MCP (SEC filings)")
    print("\nType 'quit' or 'exit' to stop.")
    print("=" * 60)

    print("\n🔌 Connecting to MCP servers...")

    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    mcp_tools = await mcp_client.get_tools()
    print(f"✅ Connected! Loaded {len(mcp_tools)} MCP tools")

    graph = build_graph(mcp_tools=mcp_tools, max_revisions=max_revisions, shortcut_quality_score=shortcut_quality_score, graph_filename=graph_filename)

    while True:
        try:
            user_input = input("\n📝 Your query: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\n🔄 Processing...")

            # Run the graph
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "revision_count": 0,
                "evaluation": None,
                "best_revision": None,
                "best_evaluation": None
            }

            result = await graph.ainvoke(initial_state) # type: ignore

            # Extract tools called and tool results
            tools_called = []
            tool_results = []
            final_response = ""

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_called.append(tool_call["name"])
                elif isinstance(msg, ToolMessage):
                    tool_results.append(msg.content)
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_response = msg.content  # Keep updating to get the last one

            # Display tools called
            if tools_called:
                print("\n🔧 Tools called: " + ", ".join(tools_called))

            # Display revision count
            revision_count = result.get("revision_count", 0)
            print(f"🔄 Revisions: {revision_count}")

            # Extract and display the final response
            print("\n" + "=" * 60)
            print("📊 Response:")
            print("=" * 60)
            print(f"\n{final_response}")

            # Display evaluation from state (always computed by graph)
            evaluation = result.get("evaluation")
            if evaluate_mode and evaluation:
                print("\n" + "=" * 60)
                print("📋 Evaluation (LLM-as-a-Judge):")
                print("=" * 60)

                if "error" not in evaluation:
                    scores = evaluation.get("scores", {})
                    print("\nScores (1-5 scale):")
                    for dim, score in scores.items():
                        dim_display = dim.replace("_", " ").title()
                        print(f"  • {dim_display}: {score}/5")

                    print(f"\n⭐ Overall Score: {evaluation.get('overall_score', 'N/A')}/5.0")

                    print("\nReasoning:")
                    reasoning = evaluation.get("reasoning", {})
                    for dim, reason in reasoning.items():
                        dim_display = dim.replace("_", " ").title()
                        print(f"  • {dim_display}: {reason}")

                    print(f"\nSummary: {evaluation.get('summary', 'N/A')}")
                else:
                    print(f"\n❌ Evaluation error: {evaluation.get('error')}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"\n❌ Error: {str(e)}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Financial Research Agent with Reflection (SageMaker v2)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Enable LLM-as-a-judge evaluation for each response")
    parser.add_argument("--max-revisions", type=int, default=3,
                        help="Maximum number of revisions before stopping (default: 3)")
    parser.add_argument("--shortcut-quality-score", type=float, default=4.95,
                        help="Stop early if evaluation score reaches this threshold (default: 4.95)")
    parser.add_argument("--graph-filename", type=str, default=None,
                        help="Filename for saving the graph visualization (optional)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output (token counts, payload details, type coercion)")
    args = parser.parse_args()

    # Set global DEBUG_MODE from CLI flag (overrides env var)
    if args.debug:
        DEBUG_MODE = True
        os.environ["DEBUG"] = "1"  # Propagate to content_handler via env

    asyncio.run(run_cli(
        evaluate_mode=args.evaluate,
        max_revisions=args.max_revisions,
        shortcut_quality_score=args.shortcut_quality_score,
        graph_filename=args.graph_filename
    ))
