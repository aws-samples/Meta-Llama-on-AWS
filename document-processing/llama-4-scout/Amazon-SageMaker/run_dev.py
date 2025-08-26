#!/usr/bin/env python3
"""
Development server runner for the Document Processing API
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def setup_environment():
    """Setup environment variables"""

def main():
    print("🚀 Starting Document Processing API Development Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the backend directory")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    try:
        install_dependencies()
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Start the server
    print("\n🌟 Starting FastAPI server...")
    port = os.getenv("API_PORT", "8001")
    print(f"📍 API will be available at: http://localhost:{port}")
    print(f"📖 API docs at: http://localhost:{port}/docs")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        port = os.getenv("API_PORT", "8001")
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", port])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
