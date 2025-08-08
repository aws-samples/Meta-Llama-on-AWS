#!/usr/bin/env python3
"""
Hybrid server that runs both Flask web app and simple MCP server
No external MCP dependencies required
"""
import threading
import json
import sys
import subprocess
import os
from flask import Flask, request, jsonify
import pdfplumber
import requests
from io import BytesIO
import traceback

# Flask App (existing web interface)
flask_app = Flask(__name__)

# Shared functions
def extract_pdf_text(url: str):
    try:
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text, None
    except Exception as e:
        return None, f"Error processing URL: {str(e)}"

def call_bedrock(prompt: str):
    try:
        result = subprocess.run(['python3', 'bedrock_call.py', prompt], 
                              env=os.environ, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Bedrock call failed: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling Bedrock: {str(e)}"

# Flask Routes (Web App)
@flask_app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@flask_app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            source = data.get('source')
            question = data.get('question')
            is_url = data.get('is_url', False)
        else:
            # Form data (file upload)
            source = request.form.get('source')
            question = request.form.get('question')
            is_url = request.form.get('is_url', 'false').lower() == 'true'

        # Validate inputs based on request type
        if is_url:
            if not source or not question:
                return jsonify({'error': 'Missing source or question'}), 400
        else:
            if not question or 'file' not in request.files:
                return jsonify({'error': 'Missing file or question'}), 400

        if is_url:
            document_text, error = extract_pdf_text(source)
        else:
            # Handle file upload
            if 'file' in request.files:
                uploaded_file = request.files['file']
                if uploaded_file.filename.endswith('.pdf'):
                    try:
                        pdf_content = BytesIO(uploaded_file.read())
                        with pdfplumber.open(pdf_content) as pdf:
                            text = ''
                            for page in pdf.pages:
                                text += page.extract_text() or ''
                        document_text, error = text, None
                    except Exception as e:
                        document_text, error = None, f"Error processing uploaded file: {str(e)}"
                else:
                    return jsonify({'error': 'Only PDF files are supported'}), 400
            else:
                return jsonify({'error': 'No file uploaded'}), 400

        if error:
            return jsonify({'error': error}), 400

        if not document_text:
            return jsonify({'error': 'Failed to extract text from document'}), 400

        prompt = f"Document: {document_text}\n\nQuestion: {question}\n\nProvide a comprehensive answer organized in clear paragraphs with section headings. Include all relevant details from the document.\n\nAnswer:"
        analysis = call_bedrock(prompt)
        
        return jsonify({
            'analysis': analysis,
            'status': 'success'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# MCP Server Functions
def handle_list_tools():
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [{
                "name": "analyze_pdf_url",
                "description": "Analyze a PDF document from a URL using Llama 3.2 11B"
            }]
        }
    }

def handle_call_tool(params):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name == "analyze_pdf_url":
        url = arguments.get("url")
        question = arguments.get("question")
        
        if not url or not question:
            return {"jsonrpc": "2.0", "error": {"code": -1, "message": "Missing URL or question"}}
        
        document_text, error = extract_pdf_text(url)
        if error:
            return {"jsonrpc": "2.0", "error": {"code": -1, "message": error}}
        
        prompt = f"Document: {document_text}\n\nQuestion: {question}\n\nProvide a comprehensive answer organized in clear paragraphs with section headings. Include all relevant details from the document.\n\nAnswer:"
        analysis = call_bedrock(prompt)
        
        return {
            "jsonrpc": "2.0",
            "result": {"content": [{"type": "text", "text": analysis}]}
        }

def mcp_server():
    """MCP server loop running in background thread."""
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "tools/list":
                response = handle_list_tools()
            elif method == "tools/call":
                response = handle_call_tool(params)
            else:
                response = {"jsonrpc": "2.0", "error": {"code": -1, "message": f"Unknown method: {method}"}}
            
            response["id"] = request.get("id")
            print(json.dumps(response))
            sys.stdout.flush()
        except:
            pass

def run_flask():
    """Run Flask in background thread."""
    flask_app.run(host='0.0.0.0', port=5000, debug=False)

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--flask-only":
        # Run only Flask for web interface
        flask_app.run(host='0.0.0.0', port=5000, debug=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "--mcp-only":
        # Run only MCP server
        mcp_server()
    else:
        # Run hybrid (Flask in background, MCP in foreground)
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("Flask server started on port 5000", file=sys.stderr)
        mcp_server()

if __name__ == "__main__":
    main()