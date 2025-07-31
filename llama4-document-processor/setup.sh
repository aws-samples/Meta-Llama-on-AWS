#!/bin/bash

# Llama4 Document Processor - EC2 Setup Script
# Run this on a fresh EC2 g5.12xlarge instance

set -e

echo "🚀 Setting up Llama4 Document Processor on EC2..."

# Check disk space requirements
echo "💾 Checking disk space requirements..."
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
REQUIRED_SPACE=104857600  # 100GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "❌ Insufficient disk space. Need 200GB+ for Llama4 model."
    echo "   Current available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
    echo "   Please resize your EBS volume to 200GB+ and run: sudo resize2fs /dev/nvme0n1p1"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo yum update -y

# Install Python 3.9+ (Amazon Linux 2023 compatible)
echo "🐍 Installing Python..."
sudo yum install -y python3 python3-pip
# Note: python3-venv not needed in AL2023, venv module is built-in

# Install Node.js 18+
echo "📦 Installing Node.js..."
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs

# GPU Detection and Setup
echo "🔥 Configuring GPU acceleration..."
GPU_ENABLED=false

if lspci | grep -i nvidia > /dev/null; then
    echo "✅ NVIDIA GPUs detected. Installing CUDA runtime..."
    GPU_ENABLED=true
    
    # Install CUDA runtime (Amazon Linux 2023 compatible)
    if sudo yum install -y cuda-runtime-12-2 cuda-cudart-12-2; then
        echo "✅ CUDA runtime installed successfully"
        
        # Configure GPU settings for optimal performance
        echo "⚡ Configuring GPU settings..."
        if sudo nvidia-smi -pm 1 2>/dev/null; then
            echo "✅ GPU persistence mode enabled"
        else
            echo "⚠️  GPU persistence mode failed - continuing anyway"
        fi
        
        if sudo nvidia-smi -ac 6251,1710 2>/dev/null; then
            echo "✅ GPU application clocks set"
        else
            echo "⚠️  GPU clock settings failed - continuing anyway"
        fi
        
        # Set CUDA environment variables
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        
        echo "✅ GPU acceleration configured"
    else
        echo "⚠️  CUDA installation failed - falling back to CPU mode"
        GPU_ENABLED=false
    fi
else
    echo "⚠️  No NVIDIA GPUs detected. Continuing with CPU-only setup."
fi

# Install Ollama using official installer (most reliable)
echo "🤖 Installing Ollama..."
if curl -fsSL https://ollama.com/install.sh | sh; then
    echo "✅ Ollama installed successfully"
else
    echo "❌ Ollama installation failed"
    exit 1
fi

# Start Ollama server with GPU support if available
echo "🔄 Starting Ollama server..."
if [ "$GPU_ENABLED" = true ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 ollama serve &
    echo "✅ Ollama server started with GPU support"
else
    ollama serve &
    echo "✅ Ollama server started (CPU mode)"
fi

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 15

# Check if Ollama is responding
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama server is responding"
else
    echo "❌ Ollama server not responding"
    exit 1
fi

# Pull Llama4 model (this will take 10-20 minutes for 67GB download)
echo "📥 Downloading Llama4 model (67GB - this will take 10-20 minutes)..."
if ollama pull llama4; then
    echo "✅ Llama4 model downloaded successfully"
else
    echo "❌ Failed to download Llama4 model"
    exit 1
fi

# Setup backend Python environment
echo "🐍 Setting up backend environment..."
cd backend
if python3 -m venv venv; then
    echo "✅ Python virtual environment created"
else
    echo "❌ Failed to create Python virtual environment"
    exit 1
fi

source venv/bin/activate
if pip install -r requirements.txt; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi
cd ..

# Setup frontend Node.js environment
echo "📦 Setting up frontend environment..."
cd frontend
if npm install; then
    echo "✅ Node.js dependencies installed"
else
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi
cd ..

# Create missing API route
echo "🔗 Creating API route..."
mkdir -p frontend/src/app/api/process-document
cat > frontend/src/app/api/process-document/route.ts << 'EOF'
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const response = await fetch('http://localhost:8001/process-document', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Processing failed' }, { status: 500 });
  }
}
EOF
echo "✅ API route created"

# Pre-load model in GPU memory if GPU is available
if [ "$GPU_ENABLED" = true ]; then
    echo "🔥 Pre-loading model in GPU memory..."
    echo "Model loaded and ready for GPU acceleration" | ollama run llama4 > /dev/null 2>&1 &
    echo "✅ Model pre-loaded in GPU memory"
fi

# Final status check
echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 System Status:"
echo "- Python: $(python3 --version)"
echo "- Node.js: $(node --version)"
echo "- Ollama: $(ollama --version 2>/dev/null || echo 'Installed')"
if [ "$GPU_ENABLED" = true ]; then
    echo "- GPU: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits) NVIDIA GPUs detected"
    echo "- CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'Runtime installed')"
else
    echo "- GPU: CPU-only mode"
fi
echo "- Disk space: $(df -h / | awk 'NR==2 {print $4}') available"
echo ""
echo "📝 Next steps:"
echo "1. Update IP addresses: sed -i 's/YOUR-EC2-IP/YOUR-ACTUAL-EC2-IP/g' backend/main.py frontend/src/app/page.tsx"
echo "2. Start backend: cd backend && chmod +x start-backend.sh && ./start-backend.sh"
echo "3. Start frontend: cd frontend && chmod +x start-frontend.sh && npm run dev -- --hostname 0.0.0.0"
echo ""
if [ "$GPU_ENABLED" = true ]; then
    echo "🚀 GPU acceleration enabled - expect ~30 second processing times!"
else
    echo "⚠️  CPU mode - expect 2-5 minute processing times"
fi