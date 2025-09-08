from fastmcp import FastMCP
from dotenv import load_dotenv
from finnhub import Client
import logging
import os

MCP_SERVER_NAME = "mcp-finnhub"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(MCP_SERVER_NAME)

load_dotenv()

deps = ["finnhub-python", "python-dotenv"]

finnhub_client = Client(api_key=os.getenv("FINNHUB_API_KEY"))

mcp = FastMCP(MCP_SERVER_NAME, dependencies=deps)


@mcp.tool(name="list_news", description="List all latest market news")
def list_news(category: str = "general", count: int = 10):
    logger.info(f"Fetching {category} news")
    news =  finnhub_client.general_news(category)
    return news[:count]


@mcp.tool(name="get_market_data", description="Get market data for a given stock")
def get_market_data(stock: str):
    logger.info(f"Fetching market data for {stock}")
    return finnhub_client.quote(stock)


@mcp.tool(
    name="get_basic_financials", description="Get basic financials for a given stock"
)
def get_basic_financials(stock: str, metric: str = "all"):
    logger.info(f"Fetching basic financials for {stock}")
    return finnhub_client.company_basic_financials(stock, metric)


@mcp.tool(
    name="get_recommendation_trends",
    description="Get recommendation trends for a given stock",
)
def get_recommendation_trends(stock: str):
    logger.info(f"Fetching recommendation trends for {stock}")
    return finnhub_client.recommendation_trends(stock)


if __name__ == "__main__":
    mcp.run()

