# Safe Transition to Dev Configuration

## Current Situation ✅
Your containers are already running with the updated base config:
- **Memory Limit:** 12GB (good for 32GB RAM)
- **Container Memory:** 12GB
- **Worker Memory:** 10GB
- **Restart Count:** 1 (from earlier issue, now stable)
- **OOM Killed:** false (healthy!)

## Option 1: Keep Current Setup (Recommended for Now)

**Current config is actually good!** With 32GB RAM, 12GB limit is safe and working.

**Do nothing** - your containers are stable and running well.

```bash
# Just monitor to confirm stability
newgrp docker << 'EOF'
docker stats llmforge-training --no-stream
EOF
```

## Option 2: Upgrade to Dev Config (More RAM for Training)

If you want to use the dev override (16GB limit for even better performance):

### Step-by-step Safe Transition:

#### 1. Check Current Status
```bash
newgrp docker << 'EOF'
docker compose ps
docker ps --format "table {{.Names}}\t{{.Status}}"
EOF
```

#### 2. Gracefully Stop Services
```bash
# This preserves your data and volumes
newgrp docker << 'EOF'
docker compose down
EOF
```

#### 3. Start with Dev Config
```bash
newgrp docker << 'EOF'
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
EOF
```

#### 4. Verify New Config
```bash
newgrp docker << 'EOF'
# Check memory limit
docker inspect llmforge-training --format 'Memory Limit: {{.HostConfig.Memory}}' | awk '{print $1, $2, $3/1024/1024/1024 "GB"}'

# Check environment
docker exec llmforge-training env | grep -E "CONTAINER_MEMORY_LIMIT_GB|WORKER_MEMORY_GB"

# Check health
docker inspect llmforge-training --format 'OOM Killed: {{.State.OOMKilled}}, Restarts: {{.RestartCount}}'
EOF
```

Expected output with dev config:
```
Memory Limit: 16GB
CONTAINER_MEMORY_LIMIT_GB=16
WORKER_MEMORY_GB=12
OOM Killed: false, Restarts: 0
```

## Option 3: Rebuild Only Training Service (Minimal Disruption)

If you only want to update training service without touching other services:

```bash
newgrp docker << 'EOF'
# Stop only training service
docker compose stop training-service

# Remove only training container
docker compose rm -f training-service

# Rebuild and start with dev config
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d training-service

# Verify
docker ps | grep llmforge-training
docker inspect llmforge-training --format 'Memory: {{.HostConfig.Memory}}'
EOF
```

## Comparison: Current vs Dev Config

| Setting | Current (Base) | Dev Override | Benefit |
|---------|---------------|--------------|---------|
| Memory Limit | 12GB | 16GB | +33% more RAM |
| Worker Memory | 10GB | 12GB | More training headroom |
| OMP Threads | 2 | 4 | Better CPU usage |
| CPU Reservation | - | 4 cores | Guaranteed resources |

## When to Use Each Config

### Use Current Setup (base only) When:
- ✅ Everything is working fine (it is!)
- ✅ You want stability over max performance
- ✅ Running multiple services simultaneously

### Use Dev Config When:
- ✅ Training large datasets (>100MB)
- ✅ Using longer sequences (2048+ tokens)
- ✅ Want maximum training performance
- ✅ Not running other heavy services

## My Recommendation

**For now: Keep current setup!** Here's why:

1. **It's already working** - 12GB is plenty for 32GB RAM
2. **Stable** - No OOM kills, restart count is from old config
3. **All services running** - No disruption needed

**Later: Upgrade to dev config when:**
- You start training larger models
- You need faster preprocessing
- You want to optimize performance

## Quick Health Check Commands

```bash
# Check if training is stable (run this now)
newgrp docker << 'EOF'
echo "=== Container Status ==="
docker ps | grep llmforge-training

echo -e "\n=== Memory Usage ==="
docker stats llmforge-training --no-stream

echo -e "\n=== Health Check ==="
docker inspect llmforge-training --format 'Status: {{.State.Status}}, OOM: {{.State.OOMKilled}}, Restarts: {{.RestartCount}}'

echo -e "\n=== Recent Logs ==="
docker logs llmforge-training --tail 10
EOF
```

## File Structure Summary

### What You Have Now:
1. ✅ `docker-compose.yml` - Base config (12GB, currently in use)
2. ✅ `docker-compose.dev.yml` - Dev override (16GB, ready to use)
3. ✅ `docker-compose.prod.yml` - Prod override (for future)

### How to Use:
```bash
# Current (what's running now)
newgrp docker << 'EOF'
docker compose up -d
EOF

# Dev override (optional upgrade)
newgrp docker << 'EOF'
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
EOF

# Production (future)
newgrp docker << 'EOF'
export CONTAINER_MEMORY_GB=32
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
EOF
```

## Decision Time

**I recommend:** Do a health check first, then decide:

```bash
# Run this now
newgrp docker << 'EOF'
./monitor_training_container.sh
EOF
```

If monitor shows:
- ✅ Memory < 80% → Keep current config
- ⚠️ Memory > 80% → Upgrade to dev config
- ❌ OOM or restarts → Definitely upgrade to dev config

**What would you like to do?**
1. Keep current config (safe, working)
2. Upgrade to dev config (more performance)
3. Just monitor for now (smart choice)
