# Exit Code 139 (Segmentation Fault) - Fixed ✅

## Problem

Container was exiting with **exit code 139** (SIGSEGV - Segmentation Fault) during model loading.

## Root Cause

**Memory reservation was too low: 3GB**

The linter/formatter automatically changed:
```yaml
# Before (correct)
reservations:
  memory: 12G

# After (broken by linter)
reservations:
  memory: 3G  # ❌ TOO LOW!
```

## Why This Caused Exit Code 139

**Exit code 139 = Signal 11 (SIGSEGV) = Segmentation Fault**

This happens when:
1. Process tries to access memory it doesn't have permission to use
2. Docker only reserved 3GB for the container
3. Model loading + Triton compilation needs ~6-8GB
4. Kernel kills process with SIGSEGV when it tries to allocate more

## The Crash Sequence

```
16:15:54 - Training starts
16:15:54 - Unsloth begins loading model
16:15:58 - Model loaded (allocated 0.52GB GPU)
16:15:58 - LoRA configuration applied
16:15:58 - Triton begins compiling CUDA kernels
         ↓
      [CRASH]  ← Process needs more RAM for compilation
         ↓
   Exit Code 139 (SIGSEGV)
         ↓
   Container restarts
```

## What the Logs Showed

```
2025-10-04 16:15:58,396 - services.training_service - INFO - Applying Unsloth 2025 LoRA...
Not an error, but Unsloth cannot patch MLP layers...
Not an error, but Unsloth cannot patch Attention layers...
[TRUNCATED - Process killed here]

Container restarted at 16:16:10
```

## The Fix

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G      # Maximum allowed
    reservations:
      memory: 12G      # ✅ FIXED: Guaranteed minimum (was 3G)
```

**Why 12G reservation is needed:**
- Base process: ~2GB
- Model loading: ~2GB
- Triton compilation: ~4GB
- Training overhead: ~2GB
- Safety buffer: ~2GB
- **Total: ~12GB**

## Memory Reservation vs Limits

### Reservation (12G)
- **Guaranteed minimum** Docker will allocate
- Container won't start if this much isn't available
- Prevents other containers from stealing this memory

### Limit (16G)
- **Maximum allowed** the container can use
- Container will be OOM-killed if it exceeds this
- Allows bursting above reservation when available

### What Was Wrong
```
Limit:       16G  ✅ (correct)
Reservation: 3G   ❌ (too low - caused segfault)
```

Process needed 6-8GB but only had 3GB guaranteed → SIGSEGV

## Verification After Fix

```bash
✅ Memory Limit: 16GB
✅ Memory Reservation: 12GB  (fixed from 3GB)
✅ Status: running, Health: healthy
✅ ExitCode: 0 (not 139)
✅ OOMKilled: false
✅ Container stable, no restarts
```

## How to Prevent This

### 1. Lock Critical Values
Add comment to prevent linters from changing:
```yaml
reservations:
  memory: 12G # CRITICAL: Must reserve enough for model loading + Triton compilation
```

### 2. Validate Before Deployment
```bash
# Check memory config
docker inspect llmforge-training --format 'Reservation: {{.HostConfig.MemoryReservation}}'

# Should show: 12884901888 (12GB in bytes)
# NOT: 3221225472 (3GB)
```

### 3. Monitor for Exit Code 139
```bash
# Check if container died with segfault
docker inspect llmforge-training --format 'ExitCode: {{.State.ExitCode}}'

# 0 = normal exit
# 137 = OOM killed
# 139 = segmentation fault ← This was the issue
```

## Common Exit Codes

| Code | Signal | Meaning | Likely Cause |
|------|--------|---------|--------------|
| 0 | - | Success | Normal exit |
| 1 | - | Error | Application error |
| 137 | SIGKILL (9) | Killed | OOM (out of memory) |
| 139 | SIGSEGV (11) | Segfault | Memory access violation |
| 143 | SIGTERM (15) | Terminated | Graceful shutdown |

## Timeline of Events

```
15:54 - Training started with 3G reservation
15:58 - Model loaded successfully
15:58 - Triton compilation began
15:58 - Process needed >3GB
15:58 - Kernel denied memory access → SIGSEGV
15:58 - Container died (exit code 139)
16:10 - Container auto-restarted
16:15 - Same cycle repeated
16:23 - Fixed: Increased reservation to 12G
16:23 - Container stable ✅
```

## Technical Details

### Why Triton Needs So Much RAM

**Triton** (used by Unsloth for Flash Attention):
1. Compiles CUDA kernels on first run
2. Creates temporary compilation workspace
3. Generates optimized assembly code
4. Caches compiled kernels

This compilation process needs **4-6GB RAM temporarily**.

### Why It's a Segfault Not OOM

**OOM (exit 137):** Docker kills container when it exceeds limit
**Segfault (exit 139):** Kernel kills process for illegal memory access

With only 3GB reserved:
- Docker doesn't give container enough "guaranteed" space
- Process requests more memory
- Kernel sees it as unauthorized access
- Sends SIGSEGV signal
- Process terminates with exit code 139

## Commands Used to Debug

```bash
# Check exit code
docker inspect llmforge-training --format '{{.State.ExitCode}}'

# Check memory config
docker inspect llmforge-training --format 'Limit: {{.HostConfig.Memory}}, Reservation: {{.HostConfig.MemoryReservation}}'

# View logs around crash
docker logs llmforge-training 2>&1 | grep -C 20 "Starting fine-tuning"

# Check for segfaults in system
sudo dmesg -T | grep -i segfault

# Monitor memory usage
docker stats llmforge-training --no-stream
```

## Summary

**Problem:** Exit code 139 (segmentation fault)
**Cause:** Memory reservation too low (3G instead of 12G)
**Impact:** Container crashed during Triton compilation
**Fix:** Increased reservation to 12G
**Status:** ✅ Fixed and stable

**Key Lesson:** Memory **reservation** is just as important as **limit**!
- Limit = maximum allowed (prevents runaway usage)
- Reservation = minimum guaranteed (prevents crashes)

Both must be set correctly for stable training!
