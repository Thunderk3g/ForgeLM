# Training Container Restart Troubleshooting Guide

## Problem: Training container restarts without error logs

The `llmforge-training` container restarts when training starts, but `docker logs` shows no errors. This is a **silent OOM (Out-Of-Memory) kill** by Docker.

## Root Cause Analysis

### The Issue
Your RTX 4060 has **8GB VRAM**, but the container was configured with:
- Container memory limit: **12GB** (in docker-compose.yml)
- Environment variable: `CONTAINER_MEMORY_LIMIT_GB=12`

When training starts:
1. Model loads into VRAM (4-6GB with 4-bit quantization)
2. Dataset loads into RAM (1-3GB)
3. Training tensors and gradients created (2-4GB)
4. **Total exceeds 12GB** → Docker OOM-kills the container
5. Container auto-restarts due to `restart: unless-stopped`

### Why No Error Logs?
- OOM kills happen at the **kernel level** (outside the container)
- The container process is forcefully terminated before it can log anything
- Docker just restarts it silently

## Fixes Applied

### 1. Reduced Container Memory Limits ✅
**File:** `docker-compose.yml`

```yaml
# Before:
limits:
  memory: 12G
environment:
  - CONTAINER_MEMORY_LIMIT_GB=12
  - WORKER_MEMORY_GB=10

# After:
limits:
  memory: 8G  # Match your available RAM
environment:
  - CONTAINER_MEMORY_LIMIT_GB=8
  - WORKER_MEMORY_GB=6
```

### 2. Enhanced Memory Monitoring ✅
**File:** `src/microservices/training_service/src/services/training_service_core.py`

Added aggressive memory monitoring:
- Checks OOM score every 5 seconds
- Logs memory usage every 30 seconds
- Stops training at 70% memory usage (before OOM kill at 85%+)
- Logs critical warnings when approaching limits

### 3. Better Error Messages ✅
**Files:**
- `src/microservices/training_service/src/services/training_service_core.py`
- `src/microservices/training_service/src/main.py`

Added validation for:
- None/empty model paths
- None/empty dataset paths
- Missing directories
- Clear error messages instead of "NoneType" errors

## How to Verify the Fix

### Step 1: Rebuild the training container
```bash
docker compose down training-service
docker compose build training-service --no-cache
docker compose up -d training-service
```

### Step 2: Monitor container status
Use the monitoring script:
```bash
./monitor_training_container.sh
# Choose option 4 (Full diagnostic)
```

### Step 3: Check for OOM events
```bash
# Check if container was OOM killed
docker inspect llmforge-training | grep OOMKilled

# Check system OOM logs (requires sudo)
sudo dmesg -T | grep -i oom | tail -10

# Check restart count
docker inspect llmforge-training | grep RestartCount
```

### Step 4: Monitor during training
```bash
# Terminal 1: Watch container stats
watch -n 1 'docker stats llmforge-training --no-stream'

# Terminal 2: Follow logs
docker logs -f llmforge-training 2>&1 | grep -E "Memory|OOM|CRITICAL|⚠️|🚨"
```

## Expected Behavior After Fix

### Before Training Starts:
```
📊 Initial Memory: 2.1GB used / 8.0GB total (26%)
🐳 Container Memory Config: Limit=8GB, Worker=6GB
```

### During Training:
```
Memory: 4.2GB used (52.5% of 8GB limit), system: 53%
Memory: 5.1GB used (63.8% of 8GB limit), system: 64%
```

### If Memory Gets Too High:
```
⚠️ HIGH: Memory at 71.2% of container limit - approaching danger zone
🚨 CRITICAL: Memory limit approaching - stopping training to prevent container restart
Training stopped due to memory pressure
```

### Container Should NOT:
- ❌ Restart during training
- ❌ Show high restart count
- ❌ Have OOMKilled: true

## Additional Troubleshooting

### If Container Still Restarts:

#### 1. Check System RAM
```bash
free -h
# Ensure you have at least 8GB available
```

#### 2. Reduce Memory Further
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 6G  # Even more conservative
environment:
  - CONTAINER_MEMORY_LIMIT_GB=6
  - WORKER_MEMORY_GB=4
```

#### 3. Use Smaller Batch Size
The emergency mode already sets:
- `batch_size: 1`
- `gradient_accumulation_steps: 128`
- `max_seq_length: 128`

But you can make it even smaller in `training_service_core.py` line 292.

#### 4. Check Docker Resource Limits
```bash
# Check Docker daemon settings
docker info | grep -i memory

# On Windows WSL2, check .wslconfig:
cat ~/.wslconfig
# Should have:
# [wsl2]
# memory=16GB  # Or your available RAM
```

#### 5. Disable Other Services
If running multiple GPU services:
```bash
# Stop chat-comparison during training
docker compose stop chat-comparison-service

# Restart after training
docker compose start chat-comparison-service
```

## Monitoring Commands Quick Reference

```bash
# Check container status
docker ps -a | grep llmforge-training

# Check OOM kill status
docker inspect llmforge-training | grep OOMKilled

# Check restart count
docker inspect llmforge-training | grep RestartCount

# Watch memory in real-time
docker stats llmforge-training

# View last 100 log lines
docker logs llmforge-training --tail 100

# Follow logs with highlighting
docker logs -f llmforge-training 2>&1 | grep --color -E "Memory|OOM|ERROR|CRITICAL|$"

# Check system OOM events
sudo dmesg -T | grep -i oom | tail -20

# Check container exit code (137 = OOM killed)
docker inspect llmforge-training -f '{{.State.ExitCode}}'
```

## Prevention: Pre-Training Checklist

Before starting training:

- [ ] Check available system RAM: `free -h`
- [ ] Verify container memory limit matches available RAM
- [ ] Stop unnecessary containers: `docker compose stop chat-comparison-service`
- [ ] Clear old models from memory: `docker compose restart training-service`
- [ ] Monitor initial memory: `docker stats llmforge-training --no-stream`
- [ ] Have monitoring ready: `./monitor_training_container.sh`

## Success Criteria

✅ Container runs for full training duration without restart
✅ Memory usage stays below 70% of container limit
✅ No OOMKilled events in `docker inspect`
✅ RestartCount remains at 0 or very low
✅ Training completes or fails with proper error message

## Need More Help?

If container still restarts after these fixes:

1. **Capture full diagnostic:**
   ```bash
   ./monitor_training_container.sh  # Option 4
   docker inspect llmforge-training > container_inspect.json
   sudo dmesg -T > kernel_logs.txt
   ```

2. **Check logs directory:**
   ```bash
   ls -lah logs/
   cat logs/container_monitor.log
   ```

3. **Review status files:**
   ```bash
   cat logs/*_status.json | jq .
   ```

The enhanced logging will now show exactly where the memory spike happens!
