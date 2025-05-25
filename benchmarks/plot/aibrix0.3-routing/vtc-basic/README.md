### Benchmark
Was done on `MBP 16" 14C-M3Max/96/1TB 30C-GPU`. 

Quick setup for simple benchmark to measure routing performance. Acknowledge that vLLM is not ideal on mac, but it is a quick way to get started.

### Setup

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
docker build -f docker/Dockerfile.arm -t vllm-cpu \\
  --platform linux/arm64 \\
  --shm-size=16g \\
  --build-arg MAX_JOBS=16 \\
  .
# Download from HF
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0\
# Load into kind & copy HF cache into kind
kind load docker-image vllm-cpu
docker cp $HOME/$HF_HOME. kind-worker:/root/.cache/huggingface/

# Deploy
cd aibrix
make dev-install-in-kind
cd benchmarks/plot/aibrix0.3-routing/vtc-basic
kubectl apply -f tiny-llama__m3max.yaml
kubectl apply -k config
```

### Run


### Teardown
```bash
kubectl delete -k config
kubectl delete -f tiny-llama__m3max.yaml
cd ..
make dev-uninstall-in-kind
```
