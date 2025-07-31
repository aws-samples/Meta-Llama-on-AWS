# Deployment Notes - Production Ready

## 🔧 Critical Fixes Applied

### 1. Storage Requirements
**Issue**: 100GB insufficient for 67GB Llama4 model
**Fix**: Require 200GB+ EBS volume
```bash
# Resize EBS volume to 200GB in AWS Console
sudo resize2fs /dev/nvme0n1p1
```

### 2. GPU Detection and Setup
**Issue**: setup.sh failed on CUDA installation
**Fix**: Detect GPUs and install appropriate CUDA runtime
```bash
if lspci | grep -i nvidia > /dev/null; then
    sudo yum install -y cuda-runtime-12-2 cuda-cudart-12-2
fi
```

### 3. GPU Performance Optimization
**Issue**: GPUs running at low performance
**Fix**: Enable persistence mode and application clocks
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 6251,1710
```

### 4. Environment Variables
**Issue**: Missing CUDA environment for Ollama
**Fix**: Set proper environment variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 5. Missing API Route
**Issue**: Frontend couldn't communicate with backend
**Fix**: Create API proxy route
```typescript
// frontend/src/app/api/process-document/route.ts
export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const response = await fetch('http://localhost:8001/process-document', {
    method: 'POST',
    body: formData,
  });
  return NextResponse.json(await response.json());
}
```

### 6. Error Handling
**Issue**: Setup script failed silently
**Fix**: Comprehensive error checking and status reporting

## 🚀 Performance Results

### Before Optimization (CPU):
- Processing time: 2-5 minutes
- Memory usage: 67GB RAM
- GPU utilization: 0%

### After Optimization (4x A10G GPUs):
- Processing time: ~30 seconds
- Memory usage: ~67GB distributed across 4 GPUs
- GPU utilization: High during inference
- First inference: 1-2 minutes (GPU warmup)
- Subsequent inferences: 10-30 seconds

## 🎯 Deployment Success Indicators

✅ **GPU Memory Usage**: nvidia-smi shows ~18GB per GPU when model loaded
✅ **Fast Responses**: ollama run llama4 "hello" responds in seconds, not minutes
✅ **Layer Offloading**: Logs show "4 GPUs detected" with CUDA support
✅ **End-to-End Working**: Web app processes documents in ~30 seconds
✅ **Disk Space**: 200GB available for model storage

## 🔍 Troubleshooting Commands

```bash
# Check GPU status
nvidia-smi

# Check Ollama model loading
ollama list
ollama run llama4 "test"

# Check CUDA libraries
ls -la /usr/local/cuda*/lib64/libcudart*

# Check processes
ps aux | grep ollama
ps aux | grep uvicorn

# Check disk space
df -h

# Check environment
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH
```

## 💡 Key Insights

- **CUDA runtime ≠ NVIDIA drivers** - Both needed for GPU acceleration
- **Fresh installs work better** - Avoid manual installations when possible
- **GPU settings matter** - Persistence mode and clocks affect performance
- **Environment variables critical** - CUDA paths must be set correctly
- **Disk space planning** - 67GB model needs 200GB+ total storage
- **First inference slow** - GPU warmup is normal, subsequent requests fast

## 📋 Production Deployment Checklist

1. **Launch g5.12xlarge EC2** with 200GB+ storage
2. **Copy Llama4final package**: `scp -r Llama4final ec2-user@IP:~/`
3. **Run setup**: `cd Llama4final && chmod +x setup.sh && ./setup.sh`
4. **Update IPs**: `sed -i 's/YOUR-EC2-IP/ACTUAL-IP/g' backend/main.py frontend/src/app/page.tsx`
5. **Start servers**: Backend on 8001, Frontend on 3000
6. **Verify**: Test at http://IP:3000
7. **Monitor**: Check GPU usage with nvidia-smi

This deployment package is now production-ready and battle-tested! 🎉