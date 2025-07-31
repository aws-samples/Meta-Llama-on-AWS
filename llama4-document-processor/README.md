# Llama4 Document Processor

Transform AWS documentation into production-ready Python code using Llama4 on AWS EC2.

## 🚀 What It Does

Upload AWS API documentation (PDF, DOCX, TXT, MD) → Llama4 analyzes it → Get complete Python boto3 code in ~30 seconds with GPU acceleration.

**Example**: Upload AWS S3 docs → Get a complete S3Client class with create_bucket, upload_object, download_object, error handling, and usage examples.

## 🏗️ Architecture

**Frontend Layer**
- Next.js 14 - React framework with App Router
- React 18 - UI components and state management
- Tailwind CSS - Styling and responsive design
- TypeScript - Type safety

**Backend Layer**
- FastAPI - Python web framework with async support
- LangGraph - Workflow orchestration for document processing
- Pydantic - Data validation and serialization
- Uvicorn - ASGI server

**AI/ML Layer**
- Llama4 (67GB) - Large language model for code generation
- Ollama - Local LLM serving with GPU acceleration
- CUDA 12.2 - GPU compute runtime
- 4x NVIDIA A10G GPUs - Hardware acceleration

**Document Processing**
- LangChain - Document loading and text splitting
- Python-multipart - File upload handling
- Custom parsers - PDF, DOCX, TXT, MD support

**Infrastructure**
- AWS EC2 g5.12xlarge - 4 GPUs, 48 vCPUs, 192GB RAM
- Amazon Linux 2023 - Operating system
- 200GB+ gp3 storage - Fast SSD storage

## ⚡ Quick Start

### 1. Launch EC2
```bash
# Launch g5.12xlarge instance in us-west-2
# Storage: 200GB+ gp3 (REQUIRED for 67GB Llama4 model)
```

**Security Group Inbound Rules:**
```
Type         Protocol    Port    Source      Description
SSH          TCP         22      0.0.0.0/0   SSH access
Custom TCP   TCP         3000    0.0.0.0/0   Frontend web interface
Custom TCP   TCP         8001    0.0.0.0/0   Backend API
```

### 2. Copy Files & Setup System
```bash
# Copy deployment package to EC2
scp -r -i your-key.pem /path/to/Llama4final ec2-user@YOUR-EC2-IP:~/

# SSH to EC2
ssh -i your-key.pem ec2-user@YOUR-EC2-IP

# Run automated setup (installs everything + GPU acceleration)
cd Llama4final
chmod +x setup.sh
./setup.sh
```

**Note**: The setup.sh script will:
- Check disk space requirements (200GB+)
- Install Python 3 and pip (Amazon Linux 2023 compatible)
- Install Node.js 18+
- Detect and configure GPU acceleration
- Install CUDA runtime (12.2) if GPUs present
- Install Ollama with GPU support
- Download Llama4 model (67GB - takes 10-20 minutes)
- Setup backend Python environment
- Setup frontend Node.js environment
- Create missing API routes
- Pre-load model in GPU memory

### 3. Update Configuration
```bash
# Update IP addresses in 2 files:
# backend/main.py: Line 20 - CORS origins
# frontend/src/app/page.tsx: Line 21 - API endpoint
sed -i 's/YOUR-EC2-IP/YOUR-ACTUAL-EC2-IP/g' backend/main.py frontend/src/app/page.tsx
```

### 4. Start Servers
```bash
# Terminal 1 - Backend
cd ~/Llama4final/backend
chmod +x start-backend.sh
./start-backend.sh

# Terminal 2 - Frontend  
cd ~/Llama4final/frontend
chmod +x start-frontend.sh
./start-frontend.sh
```

**Note**: Always use full paths (~/Llama4final/) to avoid "directory not found" errors.

### 5. Access App
- **Web Interface**: http://YOUR-EC2-IP:3000
- **API Docs**: http://YOUR-EC2-IP:8001/docs

## 📁 Project Structure

```
Llama4final/
├── backend/                 # Python FastAPI server
│   ├── .env                # Environment variables
│   ├── main.py             # FastAPI app with CORS
│   ├── document_processor.py # Core processing logic
│   ├── document_parsers.py # File format parsers
│   ├── langgraph_workflow.py # LangGraph workflow
│   ├── llm_providers.py    # LLM integration
│   ├── models.py           # Pydantic models
│   ├── simple_workflow.py  # Simplified workflow
│   ├── requirements.txt    # Python dependencies
│   ├── run_dev.py          # Development server
│   └── start-backend.sh    # Backend startup script
├── frontend/               # Next.js React app
│   ├── src/app/
│   │   ├── api/process-document/route.ts # API proxy route
│   │   ├── globals.css     # Global styles
│   │   ├── layout.tsx      # App layout
│   │   └── page.tsx        # Main upload interface
│   ├── src/components/     # React components
│   │   ├── CodeDisplay.tsx # Generated code display
│   │   ├── DocumentUpload.tsx # File upload
│   │   └── ProcessingStatus.tsx # Loading states
│   ├── src/lib/           # Utility functions
│   ├── src/types/         # TypeScript types
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.ts # Tailwind configuration
│   ├── tsconfig.json      # TypeScript configuration
│   └── start-frontend.sh  # Frontend startup script
├── sample_docs/           # Example AWS documentation
│   └── aws_s3_api_sample.md
├── setup.sh              # Automated setup script
├── DEPLOYMENT_NOTES.md   # Deployment lessons learned
├── DEPLOYMENT.md         # Detailed deployment guide
└── README.md             # This file
```

## 💰 Costs

- **EC2 g5.12xlarge**: ~$7/hour when running
- **Storage**: ~$20/month for 200GB gp3
- **Llama4**: Free (self-hosted, no API fees)
- **Total**: ~$7/hour + $20/month storage

## 🎯 Use Cases

- **DevOps Teams**: Generate AWS infrastructure code from documentation
- **Developers**: Quickly implement AWS service integrations
- **Cloud Architects**: Prototype AWS solutions rapidly
- **Learning**: Understand AWS APIs through generated examples

## 🔧 Troubleshooting

- **"Directory not found" errors**: Always use full paths like `cd ~/Llama4final/frontend`
- **Port Issues**: Ensure security group allows 3000, 8001
- **Frontend not accessible**: Use `npm run dev -- --hostname 0.0.0.0` instead of just `npm run dev`
- **Permission denied on scripts**: Run `chmod +x script-name.sh` before executing
- **Ollama Issues**: Check `ollama list` shows llama4 model
- **Setup script fails**: Check disk space (need 200GB+) and GPU detection
- **Python venv errors**: Amazon Linux 2023 has built-in venv support
- **GPU Issues**: Verify `nvidia-smi` shows 4x A10G GPUs
- **Memory Issues**: Ensure 192GB RAM for 67GB Llama4 model
- **Slow processing**: CPU mode takes 2-5 minutes, GPU mode takes ~30 seconds

## 📝 Example Output

**Input**: AWS S3 API documentation  
**Output**: Complete Python class with:
- S3 client initialization
- Bucket operations (create, delete)
- Object operations (upload, download, list)
- Error handling and retries
- Usage examples and documentation

## 🚀 Performance

- **With GPU acceleration**: ~30 seconds processing time
- **CPU-only mode**: 2-5 minutes processing time
- **Model loading**: First inference takes 1-2 minutes (GPU warmup)
- **Subsequent requests**: Lightning fast with GPU memory loaded

## 🚀 Demo Ready

Perfect for viral content:
*"Just deployed 67GB Llama4 on 4x GPUs, processing documents 12x faster! From 5 minutes to 30 seconds! 🤯 #Llama4 #AWS #AI"*

## 📚 More Info

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions and troubleshooting.
See [DEPLOYMENT_NOTES.md](DEPLOYMENT_NOTES.md) for lessons learned during development.