# Hybrid PDF Document Analyzer - Web App + MCP Server

Complete hybrid application with both web interface and MCP server capabilities.

## Files
- `hybrid_flask_mcp.py` - Hybrid Flask + MCP server
- `streamlit_app.py` - Streamlit frontend interface  
- `bedrock_call.py` - Bedrock API call handler
- `requirements.txt` - Python dependencies
- `.env` - Environment configuration (optional, uses IAM role)

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **AWS Setup Required:**
- EC2 instance with IAM role for Bedrock access
- No AWS credentials in .env file (uses IAM role)

### IAM Role Policy
Create an IAM role with this policy and attach to your EC2 instance:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Action": [
				"bedrock:InvokeModel",
				"bedrock:ListFoundationModels",
				"bedrock:GetFoundationModel",
				"bedrock:InvokeModelWithResponseStream",
				"bedrock:GetInferenceProfile",
				"bedrock:ListInferenceProfiles"
			],
			"Resource": "*",
			"Effect": "Allow"
		}
	]
}
```

### Security Group Configuration
Add these inbound rules to your security group:

- **TCP 22** - 0.0.0.0/0 (SSH access)
- **TCP 5000** - 0.0.0.0/0 (Flask MCP Server)
- **TCP 8501** - 0.0.0.0/0 (Streamlit Access)

### Instance Requirements
- **Instance Type:** t3.large (demo), use larger for production
- **Region:** us-west-2 (required for Bedrock access)

## Running Options

### Option 1: Full Hybrid (Web + MCP)
```bash
# Terminal 1 - Hybrid server (Flask + MCP)
python3 hybrid_flask_mcp.py

# Terminal 2 - Streamlit frontend
streamlit run streamlit_app.py --server.address=0.0.0.0
```

### Option 2: Web App Only
```bash
# Terminal 1 - Flask only
python3 hybrid_flask_mcp.py --flask-only

# Terminal 2 - Streamlit frontend
streamlit run streamlit_app.py --server.address=0.0.0.0
```

### Option 3: MCP Server Only
```bash
python3 hybrid_flask_mcp.py --mcp-only
```

## Access

- **Web Interface:** http://localhost:8501 (humans)
- **HTTP API:** http://localhost:5000 (web requests)
- **MCP Protocol:** stdin/stdout JSON-RPC (AI tools)

## Testing Web App Workflow

1. **Go to:** http://your-ec2-publicIP:8501
2. **Enter PDF URL:** `https://soarworks.samhsa.gov/sites/default/files/media/documents/2023-05/MSR%20Sample%20Active%20Substance%20Use%202023.pdf`
3. **Ask question:** "What is this document about?"
4. **Click:** "Analyze Document"

## Testing MCP Server

```bash
# Test tool discovery
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python3 hybrid_flask_mcp.py --mcp-only

# Test PDF analysis
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "analyze_pdf_url", "arguments": {"url": "https://example.com/doc.pdf", "question": "What is this about?"}}}' | python3 hybrid_flask_mcp.py --mcp-only
```

## Architecture

```
PDF URL → Streamlit → Flask HTTP → bedrock_call.py → AWS Bedrock → Response
PDF URL → AI Tool → MCP JSON-RPC → bedrock_call.py → AWS Bedrock → Response
```

Three interfaces, same powerful backend!