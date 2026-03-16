"""
Multi-tool financial agent using LangGraph.

Environment variables:
    Set up AWS credentials for SageMaker access, e.g., via ~/.aws/credentials or environment variables.
    TAVILY_API_KEY - Your Tavily API key
    ALPHAVANTAGE_API_KEY - Your AlphaVantage API key
    EDGAR_IDENTITY - Your Edgar AI identity email
"""

import asyncio
import json
import os
import traceback
from typing import Annotated, Any, Dict, List, TypedDict

import boto3
import yfinance as yf
from langchain_aws.chat_models.sagemaker_endpoint import (
    ChatModelContentHandler,
    ChatSagemakerEndpoint,
)
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

# --- Configuration ---
# Missing environment variable and API keys are checked in run_cli() and will raise errors.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL") or "https://api.llama.com/compat/v1/"

# MCP Server Configuration
MCP_SERVERS = {
    "alphavantage": {
        "transport": "streamable_http",
        "url": f"https://mcp.alphavantage.co/mcp?apikey=${ALPHAVANTAGE_API_KEY}",
    },
    "edgar": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "edgar.ai"],
        "env": {"EDGAR_IDENTITY": EDGAR_IDENTITY},
    },
}

SAGEMAKER_ENDPOINT_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
sagemaker_client = boto3.Session().client('sagemaker-runtime')
aws_region_name = boto3.Session().region_name

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

# --- SageMaker LLM Setup ---

class LlamaChatHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, messages: List[BaseMessage], model_kwargs: Dict[str, Any]) -> bytes:
        # Custom logic to format messages into Llama's specific prompt format
        # This typically involves adding special tokens like [INST], [/INST], <|begin_of_text|>, etc.
        
        # Example for Llama 3 format (simplified)
        prompt = "<|begin_of_text|>"
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"
            elif isinstance(message, AIMessage):
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"
            elif isinstance(message, SystemMessage):
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{message.content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" # Model expects this to continue

        # The payload structure expected by the SageMaker TGI container
        input_payload = {
            "inputs": prompt,
            "parameters": {**model_kwargs, "max_new_tokens": 512, "return_full_text": False}
        }
        return json.dumps(input_payload).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        # Logic to parse the response from the SageMaker endpoint
        response_json = json.loads(output.decode("utf-8"))
        # Extract the generated text from the response
        return response_json[0]['generated_text']
    

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


def create_agent_node(llm_with_tools):
    """Create the main agent node that decides which tools to use."""

    system_prompt = """You are a helpful financial research assistant with access to the following tools:

1. yahoo_news - Get latest news for a stock ticker
2. yahoo_stock_price - Get current stock price and info for a ticker
3. tavily_search - Search the internet for general information

Based on the user's query, decide which tool(s) to use. You can use multiple tools if needed.
For stock-related queries, prefer Yahoo Finance tools. For general questions, use Tavily search.

Provide comprehensive and accurate responses based on the tool results."""

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent


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


def should_continue(state: AgentState) -> str:
    """Determine next step after agent node."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "reviser"


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


def create_evaluator_node():
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
    user_query: str,
    tool_results: list[str],
    agent_response: str,
    tools_called: list[str],
) -> dict:
    """
    Evaluate the agent's response using LLM-as-a-judge.

    Args:
        user_query: The original user query
        tool_results: List of tool result strings
        agent_response: The final response from the agent
        tools_called: List of tool names that were called

    Returns:
        Dictionary with scores, reasoning, overall_score, and summary
    """
    llm = ChatSagemakerEndpoint(
        endpoint_name=SAGEMAKER_ENDPOINT_MODEL_NAME,
        region_name=aws_region_name,
        content_handler=LlamaChatHandler(),
        client=sagemaker_client,
        model_kwargs={"temperature": 0.1}
    )

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

def build_graph(mcp_tools: list, max_revisions: int = 3, shortcut_quality_score: float = 4.95, graph_filename: str | None = None):
    """Build the LangGraph workflow.

    Args:
        mcp_tools: List of MCP tools to include alongside built-in tools
        max_revisions: Maximum number of revisions before stopping (default: 3)
        shortcut_quality_score: Stop early if evaluation score reaches this threshold (default: 4.95)
        graph_filename: Filename for saving the graph visualization (optional)
    """

    # Combine built-in tools with MCP tools
    all_tools = tools + (mcp_tools or [])

    # Initialize LLM
    llm = ChatSagemakerEndpoint(
        endpoint_name=SAGEMAKER_ENDPOINT_MODEL_NAME,
        region_name=aws_region_name,
        content_handler=LlamaChatHandler(),
        client=sagemaker_client,
        model_kwargs={"temperature": 0.1}
    )

    llm_with_tools = llm.bind_tools(all_tools)

    # Create nodes
    agent_node = create_agent_node(llm_with_tools)
    tool_node = ToolNode(all_tools)
    reviser_node = create_reviser_node(llm)
    evaluator_node = create_evaluator_node()
    select_best_node = create_select_best_revision_node()
    should_continue_revision = create_should_continue_revision(max_revisions, shortcut_quality_score)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reviser", reviser_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("select_best_revision", select_best_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")  # After tools, go back to agent
    workflow.add_edge("reviser", "evaluator")  # After revision, evaluate
    workflow.add_edge("evaluator", "select_best_revision")  # After evaluation, select best
    workflow.add_conditional_edges("select_best_revision", should_continue_revision)  # Then decide

    graph = workflow.compile()

    # Get PNG bytes from the graph
    if graph_filename:
        png_bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open(graph_filename, "wb") as f:
            f.write(png_bytes)

        print(f"Graph saved to {graph_filename}")
    return graph


# graph_filename is optional argument for saving the graph visualization. Make it str or None.
async def run_cli(evaluate_mode: bool = False, max_revisions: int = 3, shortcut_quality_score: float = 4.95, graph_filename: str | None = None):  
    """Run the interactive CLI loop.

    Args:
        evaluate_mode: If True, evaluate each response using LLM-as-a-judge
        max_revisions: Maximum number of revisions before stopping (default: 3)
        shortcut_quality_score: Stop early if evaluation score reaches this threshold (default: 4.95)
        graph_filename: Filename for saving the graph visualization
    """

    # exit if any of the required API keys are missing
    if not LLAMA_API_KEY:
        raise ValueError("LLAMA_API_KEY environment variable not set")
    if not LLAMA_BASE_URL:
        raise ValueError("LLAMA_BASE_URL environment variable not set")
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set")
    if not EDGAR_IDENTITY:
        raise ValueError("EDGAR_IDENTITY environment variable not set")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    print("=" * 60)
    print("Financial Research Agent with Reflection")
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

    parser = argparse.ArgumentParser(description="Financial Research Agent with Reflection")
    parser.add_argument("--evaluate", action="store_true",
                        help="Enable LLM-as-a-judge evaluation for each response")
    parser.add_argument("--max-revisions", type=int, default=3,
                        help="Maximum number of revisions before stopping (default: 3)")
    parser.add_argument("--shortcut-quality-score", type=float, default=4.95,
                        help="Stop early if evaluation score reaches this threshold (default: 4.95)")
    parser.add_argument("--graph-filename", type=str, default=None,
                        help="Filename for saving the graph visualization (optional)")
    args = parser.parse_args()

    asyncio.run(run_cli(
        evaluate_mode=args.evaluate,
        max_revisions=args.max_revisions,
        shortcut_quality_score=args.shortcut_quality_score,
        graph_filename=args.graph_filename
    ))
