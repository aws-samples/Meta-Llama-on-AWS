# Financial Analysis MCP Application

A financial analysis application built with Strands "Agents as Tools" pattern, AWS Bedrock (llama4-scout-17b-instruct-v1:0), and two Model Context Protocol (MCP) servers: Yahoo Finance and Finnhub. Runs on AWS SageMaker notebook instances (tested on ml.t3.medium).

## 🏗️ Architecture

- **Strands Tools** - Direct tool calls for data collection and analysis
- **AWS Bedrock** - Llama 4 Scout 17B Instruct model for AI analysis
- **MCP Servers** - [Yahoo Finance MCP Server](https://github.com/AgentX-ai/yahoo-finance-server) + [Finnhub MCP Server](https://github.com/catherinedparnell/mcp-finnhub/tree/main)
- **S3 Storage** - Complete dataset storage and JSON downloads
- **Gradio Interface** - Dark-themed web application

## 🚀 Features

- **Current Financial Data** - Stock prices, market cap, P/E ratios from Yahoo Finance and Finnhub
- **AI-Powered Analysis** - 5-sentence financial summaries using Llama 4
- **Data Persistence** - Complete datasets stored in S3 with timestamps
- **JSON Downloads** - Full financial data available for download
- **Dual MCP Integration** - Yahoo Finance MCP Server + Finnhub MCP Server
- **Professional Interface** - Clean, dark-themed web UI

## 📋 Prerequisites

- **AWS SageMaker Notebook Instance** (tested on ml.t3.medium, default instance works fine)
- **Finnhub API key** - FREE at [finnhub.io](https://finnhub.io) (click "Get free API Key", takes 60 seconds to sign up)
- AWS credentials with Bedrock and S3 access

## 🔗 MCP Servers Used

1. **[Yahoo Finance MCP Server](https://github.com/AgentX-ai/yahoo-finance-server)** - Stock data from Yahoo Finance
2. **[Finnhub MCP Server](https://github.com/catherinedparnell/mcp-finnhub/tree/main)** - Financial data from [Finnhub.io](https://finnhub.io)

## 🛠️ Installation

### 1. Upload and Extract
```bash
cd ~/SageMaker/
unzip FinancialAnalysisMCP.zip
```

### 2. Environment Setup
```bash
conda init bash
exec bash
conda create -n python311 python=3.11 -y
conda activate python311
```

### 3. Install Dependencies
```bash
pip install numpy==1.26.4

pip install -r requirements.txt
```

### 4. **CRITICAL: Create .env File with Finnhub API Key**
**⚠️ REQUIRED:** Create a `.env` file in your project directory with your Finnhub API key:
```bash
echo "FINNHUB_API_KEY=your_actual_api_key_here" > .env
```

**Or manually create `.env` file containing:**
```
FINNHUB_API_KEY=your_actual_api_key_here
```

**Get your FREE API key at [finnhub.io](https://finnhub.io) - takes 60 seconds!**

### 5. Run Application
```bash
python gradio_app.py
```

## 📁 Project Structure

```
├── gradio_app.py                 # Main application with Strands tools
├── server.py                     # Finnhub MCP server
├── server_configs.py             # MCP server configurations
├── s3_utils.py                   # S3 storage utilities
├── finance_coordinator_agent.py  # AWS Bedrock/Llama 4 configuration
├── config.py                     # AWS and model configuration
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🔧 How It Works

### 1. Data Collection Tool (`@tool data_collector`)
- Connects to Yahoo Finance and Finnhub MCP servers
- Collects current financial data (price, market cap, P/E ratio)
- Uploads complete datasets to S3 with timestamp organization
- Returns key metrics for analysis

### 2. Financial Analysis Tool (`@tool financial_analyzer`)
- Creates Strands Agent with AWS Bedrock Llama 4 model
- Generates 5-sentence financial analysis based on collected data
- Provides investment insights and valuation assessment

### 3. Workflow
1. User enters company ticker (e.g., AAPL, TSLA)
2. `data_collector` tool gathers current financial data from both MCP servers
3. `financial_analyzer` tool generates AI-powered analysis
4. Results displayed with download options for complete datasets

## 🎯 Example Usage

1. Launch the application: `python gradio_app.py`
2. Open the provided Gradio URL
3. Enter a stock ticker (e.g., "AAPL")
4. Click "🚀 Start Analysis"
5. View the 5-sentence AI analysis
6. Download complete JSON datasets if needed

## 📊 Sample Output

```
Based on the provided metrics, Apple Inc. (AAPL) is currently trading at $175.43, 
indicating strong market performance. The company's market capitalization of $2.75 
trillion demonstrates its position as one of the world's most valuable companies. 
With a P/E ratio of 28.5, AAPL shows reasonable valuation relative to earnings. 
The company's consistent profitability and strong brand loyalty support its premium 
valuation. Overall, AAPL represents a stable large-cap investment with continued 
growth potential in the technology sector.
```

## 🔑 Key Technologies

- **[Strands](https://github.com/strands-ai/strands)** - AI agent framework with tools pattern
- **[AWS Bedrock](https://aws.amazon.com/bedrock/)** - Managed AI service running Llama 4
- **[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)** - Standardized AI-tool communication
- **[Gradio](https://gradio.app/)** - Web interface framework
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Fast MCP server implementation

## 🏛️ Architecture Pattern

This application demonstrates the **"Agents as Tools"** pattern:
- Tools perform specific tasks (data collection, analysis)
- Direct tool calls provide reliable execution
- Agents used as specialized tools (not autonomous orchestrators)
- Predictable workflow with error handling

Strands agent tools can help handle tasks more effectively and increase throughput in certain workflows compared to working without them. The effectiveness depends on your specific use case.

## 🔒 Security Notes

- Store API keys in `.env` file (not in code)
- Use AWS IAM roles for SageMaker permissions
- S3 data is organized by company and timestamp
- No sensitive data logged or exposed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### Common Issues

**Python Version - Must use Python 3.11:**
```bash
conda create -n python311 python=3.11 -y
conda activate python311
```

**Numpy Installation Error:**
```bash
pip install numpy==1.26.4  # Install compatible version first
```

**Conda Activation Error:**
```bash
conda init bash
exec bash
```

**Missing Yahoo Finance Server:**
```bash
pip install yahoo-finance-server==0.1.1
```

**Finnhub API Error:**
- Verify your API key in `.env` file
- Check API rate limits

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review AWS Bedrock and SageMaker documentation
- Verify all dependencies are installed correctly

---

**Built with ❤️ using Strands, AWS Bedrock,Meta Llama4 and MCP**
