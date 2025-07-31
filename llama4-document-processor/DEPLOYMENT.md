# Llama4 Document Processor - Deployment Guide

## Overview
This app transforms AWS documentation into production-ready Python code using Llama4 on AWS EC2.

## Prerequisites
- AWS EC2 g5.12xlarge instance (4x NVIDIA A10G GPUs, 192GB RAM)
- Ubuntu/Amazon Linux 2
- 100GB+ storage

## Quick Start

### 1. Launch EC2 Instance
```bash
# Launch g5.12xlarge in us-west-2
# Security Group: Allow ports 22, 3000, 8001
# Storage: 100GB gp3
```

### 2. Connect to EC2
```bash
ssh -i your-key.pem ec2-user@YOUR-EC2-IP
```

### 3. Install Dependencies
```bash
# Update system
sudo yum update -y

# Install Python 3.9+
sudo yum install -y python3 python3-pip

# Install Node.js 18+
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### 4. Deploy Backend
```bash
# Copy backend files to ~/backend/
cd ~/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Llama4 model (67GB download)
ollama pull llama4

# Start backend
python run_dev.py
```

### 5. Deploy Frontend
```bash
# Copy frontend files to ~/frontend/
cd ~/frontend

# Install dependencies
npm install

# Install additional packages
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer

# Start frontend
npm run dev
```

### 6. Access Application
- Frontend: http://YOUR-EC2-IP:3000
- Backend API: http://YOUR-EC2-IP:8001

## Configuration

### Backend (.env)
```
OLLAMA_MODEL=llama4
OLLAMA_BASE_URL=http://localhost:11434
```

### Frontend (next.config.js)
Update CORS origins to match your EC2 IP:
```javascript
allow_origins=["http://YOUR-EC2-IP:3000"]
```

## Usage
1. Upload AWS API documentation (PDF, DOCX, TXT, MD)
2. Wait ~2 minutes for Llama4 processing
3. Copy/download generated Python boto3 code

## Costs
- EC2 g5.12xlarge: ~$7/hour
- Storage: ~$10/month
- No API fees (self-hosted Llama4)

## Troubleshooting
- Ensure security groups allow ports 3000, 8001
- Check Ollama is running: `ollama list`
- Verify GPU access: `nvidia-smi`
- Check logs in terminal outputs

## Architecture
- **Frontend**: Next.js + React + Tailwind CSS
- **Backend**: FastAPI + LangGraph + Ollama
- **AI**: Llama4 (67GB) on 4x A10G GPUs
- **Infrastructure**: AWS EC2 g5.12xlarge