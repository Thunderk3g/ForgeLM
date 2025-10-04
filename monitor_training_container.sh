#!/bin/bash
# Monitor training container for restarts and OOM kills
# Usage: ./monitor_training_container.sh

set -e

CONTAINER_NAME="llmforge-training"
LOG_FILE="logs/container_monitor.log"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Training Container Monitor ===${NC}"
echo "Monitoring container: $CONTAINER_NAME"
echo "Log file: $LOG_FILE"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check container status
check_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
        RESTARTS=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER_NAME" 2>/dev/null || echo "0")
        echo -e "${GREEN}Container Status: $STATUS${NC}"
        echo -e "${YELLOW}Restart Count: $RESTARTS${NC}"

        # Check OOM kills
        if docker inspect "$CONTAINER_NAME" | grep -q '"OOMKilled": true'; then
            echo -e "${RED}⚠️  CONTAINER WAS OOM KILLED!${NC}"
            log_msg "CRITICAL: Container was OOM killed!"
        fi

        # Get memory stats
        MEMORY_STATS=$(docker stats "$CONTAINER_NAME" --no-stream --format "MEM: {{.MemUsage}} ({{.MemPerc}})" 2>/dev/null || echo "N/A")
        echo -e "${BLUE}$MEMORY_STATS${NC}"

        return 0
    else
        echo -e "${RED}Container not found!${NC}"
        return 1
    fi
}

# Function to check for OOM in dmesg (requires sudo)
check_oom_dmesg() {
    if command -v dmesg &> /dev/null; then
        echo ""
        echo -e "${BLUE}Checking system logs for OOM events...${NC}"
        OOM_EVENTS=$(sudo dmesg -T 2>/dev/null | grep -i "oom" | tail -5 || echo "No OOM events found or permission denied")
        if [[ "$OOM_EVENTS" != "No OOM events found"* ]]; then
            echo -e "${RED}Recent OOM events:${NC}"
            echo "$OOM_EVENTS"
        else
            echo -e "${GREEN}$OOM_EVENTS${NC}"
        fi
    fi
}

# Function to monitor container logs in real-time
monitor_logs() {
    echo ""
    echo -e "${BLUE}Monitoring container logs (Ctrl+C to stop)...${NC}"
    echo ""
    docker logs -f "$CONTAINER_NAME" 2>&1 | while read -r line; do
        # Highlight important log lines
        if echo "$line" | grep -qi "oom\|out of memory\|memory.*error\|restart"; then
            echo -e "${RED}$line${NC}"
            log_msg "ERROR: $line"
        elif echo "$line" | grep -qi "warning\|warn"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qi "memory.*GB\|memory.*percent"; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Function to watch memory usage
watch_memory() {
    echo ""
    echo -e "${BLUE}Watching memory usage (Ctrl+C to stop)...${NC}"
    echo ""
    while true; do
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            STATS=$(docker stats "$CONTAINER_NAME" --no-stream --format "{{.Container}} | MEM: {{.MemUsage}} ({{.MemPerc}}) | CPU: {{.CPUPerc}}" 2>/dev/null || echo "Error reading stats")
            echo -e "${GREEN}$(date '+%H:%M:%S')${NC} - $STATS"
            sleep 2
        else
            echo -e "${RED}Container stopped or restarted!${NC}"
            log_msg "Container stopped or restarted during monitoring"
            break
        fi
    done
}

# Main menu
echo "Select monitoring option:"
echo "1. Check container status and OOM events"
echo "2. Monitor logs in real-time"
echo "3. Watch memory usage"
echo "4. Full diagnostic (all checks)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        check_container
        check_oom_dmesg
        ;;
    2)
        check_container
        monitor_logs
        ;;
    3)
        check_container
        watch_memory
        ;;
    4)
        log_msg "=== Starting full diagnostic ==="
        check_container
        check_oom_dmesg
        echo ""
        echo "Starting memory monitoring for 60 seconds..."
        timeout 60 bash -c "$(declare -f watch_memory); watch_memory" || true
        echo ""
        echo "Check $LOG_FILE for full diagnostic log"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
