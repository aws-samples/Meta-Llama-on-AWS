""" Financial Analysis MCP Gradio Application - Agents as Tools Pattern

This module provides a dark mode web interface for financial analysis
using Yahoo Finance, TAM analysis, and AI-powered summaries.
"""
from s3_utils import upload_json_to_s3, download_from_s3
import gradio as gr
import logging
import sys
import time
import json
from datetime import datetime
from finance_coordinator_agent import new_finance_coordinator_agent
from config import AWS_CONFIG
from strands.tools import tool
from strands import Agent

def create_combined_financial_document(company_name, s3_paths):
    """Combine raw data from multiple sources into one structured document"""
    combined_doc = {
        "company": company_name,
        "timestamp": datetime.now().isoformat(),
        "analysis_metadata": {
            "sources": ["yahoo_finance", "finnhub"],
            "tools_used": []
        },
        "raw_data": {
            "yahoo_finance": {},
            "finnhub": {}
        }
    }
    
    for s3_path in s3_paths:
        try:
            raw_data = json.loads(download_from_s3(s3_path))
            
            if "yahoo" in s3_path:
                combined_doc["raw_data"]["yahoo_finance"] = raw_data
                combined_doc["analysis_metadata"]["tools_used"].append("yahoo_finance")
            elif "finnhub" in s3_path:
                combined_doc["raw_data"]["finnhub"] = raw_data  
                combined_doc["analysis_metadata"]["tools_used"].append("finnhub")
                
        except Exception as e:
            logger.error(f"Error combining data from {s3_path}: {e}")
    
    return combined_doc

@tool
def data_collector(company: str) -> str:
    """Collect financial data from Yahoo Finance and Finnhub, then upload to S3"""
    try:
        from mcp import stdio_client
        from strands.tools.mcp import MCPClient
        from server_configs import SERVER_CONFIGS
        
        logger.info(f"🔍 Collecting data for {company}")
        
        mcp_clients = [
            MCPClient(lambda: stdio_client(SERVER_CONFIGS[0])),  # Yahoo Finance
            MCPClient(lambda: stdio_client(SERVER_CONFIGS[1]))   # Finnhub
        ]
        
        with mcp_clients[0], mcp_clients[1]:
            # Get Yahoo Finance data
            yahoo_data = mcp_clients[0].call_tool_sync(
                name="get-ticker-info", 
                arguments={"symbol": company},
                tool_use_id=f"yahoo-{int(time.time())}"
            )
            
            # Get Finnhub data
            finnhub_data = mcp_clients[1].call_tool_sync(
                name="get_basic_financials", 
                arguments={"stock": company},
                tool_use_id=f"finnhub-{int(time.time())}"
            )
            
            # Combine and upload to S3
            combined_data = {
                "company": company,
                "timestamp": datetime.now().isoformat(),
                "yahoo_finance": yahoo_data,
                "finnhub": finnhub_data
            }
            
            s3_path = upload_json_to_s3(combined_data, company, "financial_data")
            logger.info(f"📤 Data uploaded to: {s3_path}")
            
            # Parse the nested Yahoo Finance data
            try:
                yahoo_content = json.loads(yahoo_data['content'][0]['text'])
                price = yahoo_content.get("regularMarketPrice", "N/A")
                market_cap = yahoo_content.get("marketCap", "N/A")
                pe_ratio = yahoo_content.get("trailingPE", "N/A")
            except:
                price = market_cap = pe_ratio = "N/A"
            
            return f"Data collected for {company}. Price: ${price}, Market Cap: {market_cap}, P/E: {pe_ratio}. S3_PATH: {s3_path}"
            
    except Exception as e:
        logger.error(f"Error in data_collector: {e}")
        return f"Error collecting data for {company}: {str(e)}"

@tool  
def financial_analyzer(company: str, metrics: str) -> str:
    """Analyze financial metrics and provide summary"""
    try:
        from finance_coordinator_agent import bedrock_model
        
        logger.info(f"📊 Analyzing {company}")
        
        analyzer_agent = Agent(
            model=bedrock_model,
            
system_prompt="Provide 5-sentence financial analysis based on given metrics.",
            tools=[]
        )
        
        response = analyzer_agent(f"Analyze {company} with these metrics: {metrics}")
        return extract_text_from_response(response)
        
    except Exception as e:
        logger.error(f"Error in financial_analyzer: {e}")
        return f"Analysis unavailable for {company}. Metrics: {metrics}"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("financial-analysis-app")

# Global variables
agent = None

def initialize_agent():
    """Initialize the finance coordinator agent"""
    global agent
    try:
        agent = new_finance_coordinator_agent()
        logger.info("Finance coordinator agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

def extract_text_from_response(response):
    """Extract text from the complex response structure"""
    try:
        # Handle AgentResult object with direct content access
        if hasattr(response, 'message'):
            message = response.message
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                # Handle list content
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and 'text' in content[0]:
                        return content[0]['text']
                    else:
                        return str(content[0])
                # Handle string content
                elif isinstance(content, str):
                    return content
        
        # Fallback to other attributes
        for attr in ['text', 'content', 'output', 'result']:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if value and str(value).strip():
                    return str(value)
        
        # Last resort - string conversion
        return str(response)
    
    except Exception as e:
        logger.error(f"Error extracting text from response: {e}")
        return str(response)

def extract_s3_paths(text):
    """Extract S3 paths from response text"""
    import re
    logger.info(f"DEBUG extract_s3_paths: searching in text = {str(text)[:200]}...")
    s3_paths = re.findall(r'S3_PATH:\s*(s3://[^\s\n]+)', str(text))
    logger.info(f"DEBUG extract_s3_paths: found paths = {s3_paths}")
    return s3_paths

def process_financial_analysis(company_name, analysis_type="comprehensive"):
    """Process financial analysis using direct tool calls"""
    
    if not company_name.strip():
        return "Please enter a company name.", "[]"
    
    try:
        logger.info(f"🚀 Starting analysis for {company_name}")
        
        # Call tools directly
        data_result = data_collector(company_name)
        s3_paths = extract_s3_paths(data_result)
        
        analysis_result = financial_analyzer(company_name, data_result)
        
        final_result = f"""# {company_name} Financial Analysis

{analysis_result}

📊 Full financial datasets available for download below."""
        
        logger.info(f"✅ Analysis completed for {company_name}")
        return final_result, str(s3_paths)
        
    except Exception as e:
        logger.error(f"Error in process_financial_analysis: {e}")
        return f"Error processing analysis: {str(e)}", "[]"

def reset_conversation():
    """Reset the conversation"""
    global agent
    agent = None
    return "Analysis system reset. Ready for new analysis."

# Create Gradio interface
with gr.Blocks(
    css="""
    .container {
        max-width: 1200px;
        margin: auto;
    }
    """,
    title="Financial Analysis MCP - Agents as Tools"
) as demo:
    
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem">
            <h1>📊 Financial Analysis MCP - Agents as Tools</h1>
            <p>AI-powered financial analysis using specialized agent tools</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            with gr.Group():
                gr.Markdown("### Analysis Input")
                company_input = gr.Textbox(
                    label="Company Name or Ticker",
                    placeholder="Enter company name or stock ticker (e.g., AAPL, Microsoft)",
                    value=""
                )
                analysis_type = gr.Dropdown(
                    label="Analysis Type",
                    choices=["comprehensive", "financial_only", "market_analysis"],
                    value="comprehensive"
                )
                analyze_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")
            
            # Results section
            with gr.Group():
                gr.Markdown("### Analysis Results")
                results_output = gr.Textbox(
                    label="Financial Analysis Summary",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    container=True
                )
                
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        label="📥 Download Summary Report",
                        variant="secondary"
                    )
                    json_download_btn = gr.DownloadButton(
                        label="📊 Download JSON Data",
                        variant="secondary"
                    )
                    reset_btn = gr.Button("🔄 Reset", variant="secondary")
        
        with gr.Column(scale=1):
            # Status and configuration panel
            with gr.Group():
                gr.Markdown("### System Status")
                status_display = gr.Label(value="Ready", label="Status")
                s3_paths_state = gr.State(value="[]")
                
                gr.Markdown("### Architecture")
                gr.Markdown("""
                **Agents as Tools Pattern:**
                - 🔧 Data Collector Agent
                - 📊 Financial Analyzer Agent  
                - 🎯 Direct Tool Calls
                - ⚡ Specialized & Efficient
                
                *Each agent focuses on one task*
                """)
                
                gr.Markdown("### Configuration")
                with gr.Group():
                    gr.Textbox(
                        label="AWS Region",
                        value=AWS_CONFIG["region"],
                        interactive=False
                    )
                    gr.Textbox(
                        label="Model ID", 
                        value=AWS_CONFIG["model_id"],
                        interactive=False
                    )

    # Event handlers
    def analyze_with_status(company, analysis_type):
        if not company.strip():
            return "Please enter a company name or ticker.", "Error", "[]"

        try:
            result, s3_paths_str = process_financial_analysis(company, analysis_type)
            final_status = "Analysis Complete ✅" if not result.startswith("Error") else "Error ❌"
            logger.info(f"DEBUG analyze_with_status: returning s3_paths_str = {s3_paths_str}")
            return result, final_status, s3_paths_str
        except Exception as e:
            logger.error(f"Error in analyze_with_status: {e}")
            return f"Error: {str(e)}", "Error ❌", "[]"

    def prepare_download(results_text):
        if results_text and not results_text.startswith("Error"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(results_text)
            return filename
        return None

    def prepare_json_download(s3_paths_str):
        logger.info(f"DEBUG prepare_json_download: s3_paths_str = {s3_paths_str}")
        if s3_paths_str and s3_paths_str != "[]":
            try:
                import ast
                paths = ast.literal_eval(s3_paths_str)
                logger.info(f"DEBUG prepare_json_download: parsed paths = {paths}")

                if paths:
                    # Use the first available path
                    download_path = paths[0]
                    
                    raw_data = download_from_s3(download_path)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"financial_data_{timestamp}.json"
                    with open(filename, "w") as f:
                        f.write(raw_data)
                    return filename
            except Exception as e:
                logger.error(f"Error preparing JSON download: {e}")
        return None

    analyze_btn.click(
        fn=analyze_with_status,
        inputs=[company_input, analysis_type],
        outputs=[results_output, status_display, s3_paths_state]
    ).then(
        fn=prepare_download,
        inputs=[results_output],
        outputs=[download_btn]
    ).then(
        fn=prepare_json_download,
        inputs=[s3_paths_state],
        outputs=[json_download_btn]
    )
    
    company_input.submit(
        fn=analyze_with_status,
        inputs=[company_input, analysis_type],
        outputs=[results_output, status_display, s3_paths_state]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        inputs=None,
        outputs=[status_display]
    ).then(
        fn=lambda: ("", "Ready", "[]"),
        inputs=None,
        outputs=[results_output, status_display, s3_paths_state]
    )

    json_download_btn.click(
        fn=prepare_json_download,
        inputs=[s3_paths_state],
        outputs=[json_download_btn]
    )

    download_btn.click(
        fn=prepare_download,
        inputs=[results_output],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    try:
        logger.info("Starting Financial Analysis MCP application with Agents as Tools...")
        
        # Launch the Gradio app
        demo.queue()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete.")

