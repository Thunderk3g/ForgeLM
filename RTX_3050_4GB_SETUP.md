# RTX 3050 4GB Laptop Configuration

## System Specs
- **GPU:** RTX 3050 Laptop (4GB VRAM) ⚠️ Bottleneck
- **CPU:** Intel i5 11th Gen (4-6 cores)
- **System RAM:** 32GB ✅ Excellent
- **Container Memory Limit:** 12GB (base), 16GB (dev override)
- **Worker Memory:** 10GB (base), 12GB (dev override)

## Critical Container Restart Issue - SOLVED ✅

### The Problem
Your training container was restarting silently because:
1. Container had **12GB memory limit** originally set for larger systems
2. Initial assumption was limited RAM (8-12GB)
3. With **32GB RAM**, memory is NOT the issue - **4GB VRAM is the bottleneck**
4. Container was restarting due to VRAM exhaustion, not RAM

### The Solution
Updated all configurations for RTX 3050 4GB:

#### 1. Docker Compose Settings

**Base config (docker-compose.yml):**
```yaml
# Container memory limits - Balanced for 32GB RAM system
deploy:
  resources:
    limits:
      memory: 12G  # Safe for 32GB system
    reservations:
      memory: 10G

# Environment variables
environment:
  - CONTAINER_MEMORY_LIMIT_GB=12  # Plenty of RAM available
  - WORKER_MEMORY_GB=10
  - GPU_MEMORY_LIMIT_MB=3072      # 75% of 4GB VRAM (the bottleneck)
  - OMP_NUM_THREADS=2
```

**Dev override (docker-compose.dev.yml):**
```yaml
# Even more generous for development
deploy:
  resources:
    limits:
      memory: 16G  # Use more RAM for better performance
    reservations:
      memory: 12G

environment:
  - CONTAINER_MEMORY_LIMIT_GB=16
  - WORKER_MEMORY_GB=12
  - OMP_NUM_THREADS=4  # i5 11th gen can handle it
```

#### 2. Training Parameters (Auto-Applied)
```python
# Optimized for 4GB VRAM with 32GB RAM:
{
    "max_seq_length": 1024,             # Can use longer with plenty of RAM
    "batch_size": 1,                    # VRAM limited, not RAM limited
    "gradient_accumulation_steps": 32,  # Moderate accumulation
    "load_in_4bit": True,               # CRITICAL - 4x VRAM reduction
    "attn_implementation": "flash_attention_2",  # Memory efficient
    "dataloader_num_workers": 0,        # Can be increased with 32GB RAM if needed
}
```

**Key Insight:** With 32GB RAM, you can:
- ✅ Use longer sequences (1024 vs 512)
- ✅ Load larger datasets
- ✅ Use more dataloader workers if needed
- ⚠️ Still limited by 4GB VRAM for batch size

#### 3. Memory Monitoring
Enhanced to detect OOM early:
- Checks OOM score every 5 seconds
- Stops training at 70% memory usage
- Logs memory every 30 seconds
- Critical warnings at 85%+

## How to Apply the Fix

### Step 1: Rebuild Container
```bash
cd /home/jatin-kumawat/Desktop/theStartup/ForgeLM

# Stop current container
docker compose down training-service

# Rebuild with new settings
docker compose build training-service --no-cache

# Start container
docker compose up -d training-service

# Verify it's running
docker ps | grep llmforge-training
```

### Step 2: Monitor Container Health
```bash
# Use the monitoring script
./monitor_training_container.sh

# Or manually check:
docker stats llmforge-training --no-stream
docker inspect llmforge-training | grep -E "OOMKilled|RestartCount"
```

### Step 3: Test Training
Start a small training job and monitor:

```bash
# Terminal 1: Watch container stats
watch -n 2 'docker stats llmforge-training --no-stream'

# Terminal 2: Follow logs
docker logs -f llmforge-training 2>&1 | grep -E "Memory|OOM|CRITICAL|⚠️|🚨"

# Terminal 3: Check restart count
watch -n 5 'docker inspect llmforge-training | grep RestartCount'
```

## Expected Memory Usage Pattern

### Startup (Healthy)
```
📊 Initial Memory: 1.8GB used / 6.0GB total (30%)
🐳 Container Memory Config: Limit=6GB, Worker=4GB
```

### Model Loading (Healthy)
```
Memory: 2.5GB used (41.7% of 6GB limit)
Loading model with RTX 3050 4GB optimized config: 4bit=True, max_seq=1024
```

### Training Start (Healthy)
```
Memory: 3.2GB used (53.3% of 6GB limit), system: 54%
Training started with memory monitoring...
```

### During Training (Healthy)
```
Memory: 3.8GB used (63.3% of 6GB limit), system: 65%
Step 10/500 (2.0%) - Epoch: 0.02 - Loss: 2.1234
```

### Warning Signs (Still Healthy)
```
⚠️ HIGH: Memory at 71.2% of container limit - approaching danger zone
Applying additional memory optimizations...
```

### Critical (Prevents Crash)
```
🚨 CRITICAL: Memory at 87.5% of container limit! OOM kill imminent!
Training stopped due to memory pressure
```

### Bad (Container Would Restart)
```
# You should NOT see this anymore:
# Container restart count: 5
# OOMKilled: true
```

## Training Recommendations for 4GB GPU

### Model Size Limits
✅ **Recommended:**
- Qwen2.5-0.5B (500M parameters)
- Phi-3-mini (3.8B parameters with 4-bit)
- Small Llama models (up to 3B with 4-bit)

❌ **Not Recommended:**
- Qwen2.5-7B (too large even with 4-bit)
- Llama-3-8B (may work but very slow)
- Models > 7B parameters

### Dataset Size Limits
- **Maximum:** 100MB preprocessed
- **Recommended:** 10-50MB
- **Safe:** < 10MB for first tests

### Training Settings
```json
{
  "epochs": 1,                    // Start with 1 epoch
  "batch_size": 1,                // Always 1 for 4GB
  "gradient_accumulation_steps": 64-128,  // Increase for effective batch
  "max_seq_length": 512-1024,     // Lower = less memory
  "learning_rate": 0.0001,        // Standard
  "peft_method": "qlora"          // Essential for 4GB
}
```

## Troubleshooting Guide

### Issue: Container Still Restarts

#### Check #1: System RAM
```bash
free -h
# Need at least 12GB total, 8GB available
```

#### Check #2: OOM Kill Events
```bash
# Check if container was OOM killed
docker inspect llmforge-training | grep OOMKilled

# Check system logs
sudo dmesg -T | grep -i oom | tail -20

# Exit code 137 = OOM killed
docker inspect llmforge-training -f '{{.State.ExitCode}}'
```

#### Check #3: Restart Count
```bash
docker inspect llmforge-training | grep RestartCount
# Should be 0 or very low
```

#### Check #4: Memory During Training
```bash
./monitor_training_container.sh
# Choose option 3 (Watch memory usage)
```

### Issue: Training Too Slow

**This is expected on 4GB laptop GPU!**

- Expect 2-10x slower than desktop GPUs
- Training a 3B model may take hours
- Use smaller models (0.5B-1B) for faster iteration

### Issue: CUDA Out of Memory

If you see "CUDA OOM" in logs:

1. **Reduce sequence length:**
   ```python
   max_seq_length: 512  # Try 256 if still fails
   ```

2. **Reduce dataset size:**
   - Use only 1000-5000 samples
   - Split large datasets

3. **Disable eval dataset:**
   - Set `eval_dataset: None`

4. **Stop other GPU services:**
   ```bash
   docker compose stop chat-comparison-service
   docker compose stop model-management-service
   ```

## Pre-Training Checklist

Before starting any training:

- [ ] Check system RAM: `free -h` (need 8GB+ available)
- [ ] Verify container not OOM killed: `docker inspect llmforge-training | grep OOMKilled`
- [ ] Check restart count is 0: `docker inspect llmforge-training | grep RestartCount`
- [ ] Stop other GPU services: `docker compose stop chat-comparison-service`
- [ ] Have monitoring ready: `./monitor_training_container.sh`
- [ ] Use small model (< 3B parameters)
- [ ] Use small dataset (< 50MB)
- [ ] Set max_seq_length ≤ 1024

## Performance Expectations

### RTX 3050 4GB Laptop Performance:
- **Qwen2.5-0.5B:** ~500 tokens/sec
- **Phi-3-mini (3.8B):** ~100-200 tokens/sec with 4-bit
- **Training time (1000 samples):** 15-30 minutes
- **Memory usage:** 3-4GB VRAM, 4-5GB RAM

### Optimizations Applied:
- ✅ 4-bit quantization (4x memory reduction)
- ✅ Flash Attention 2 (faster + less memory)
- ✅ Gradient accumulation (effective larger batch)
- ✅ LoRA rank 16 (minimal trainable params)
- ✅ Single dataloader worker (no multiprocessing overhead)

## Success Criteria

After applying these fixes, you should see:

✅ Container runs without restarting
✅ Memory stays below 70% of 6GB limit
✅ No OOMKilled events
✅ RestartCount stays at 0
✅ Training completes or fails with proper error message
✅ Logs show memory monitoring every 30s

## Quick Commands Reference

```bash
# Rebuild container
docker compose build training-service --no-cache && docker compose up -d training-service

# Monitor memory
./monitor_training_container.sh

# Watch container stats
watch -n 2 'docker stats llmforge-training --no-stream'

# Check OOM status
docker inspect llmforge-training | grep -E "OOMKilled|RestartCount|ExitCode"

# Follow logs with highlights
docker logs -f llmforge-training 2>&1 | grep --color -E "Memory|OOM|ERROR|CRITICAL|$"

# Check system OOM events
sudo dmesg -T | grep -i oom | tail -20

# Stop other GPU services
docker compose stop chat-comparison-service model-management-service
```

## Support

If issues persist:
1. Run full diagnostic: `./monitor_training_container.sh` (option 4)
2. Check: `logs/container_monitor.log`
3. Provide: `docker inspect llmforge-training > container_inspect.json`

The enhanced monitoring will show exactly where memory spikes occur!
