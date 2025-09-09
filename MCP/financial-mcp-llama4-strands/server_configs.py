from mcp import StdioServerParameters

SERVER_CONFIGS = [
    # Yahoo Finance MCP
    StdioServerParameters(
        command="python",
        args=["-m", "yahoo_finance_server"]
    ),
    # Finnhub MCP
    StdioServerParameters(
        command="python",
        args=["server.py"]
    )
]

