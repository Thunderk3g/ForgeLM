#!/bin/bash
set -e

# Fix permissions for mounted volumes 
# This ensures the llmforge user can write to directories even if they're owned by root
# due to Docker volume mounting behavior

echo "Fixing permissions for mounted directories..."

# Fix ownership of critical directories if they exist and we have permission
if [ -w /app ]; then
    # Only fix if we can actually change permissions (running as root initially)
    if [ "$(id -u)" = "0" ]; then
        echo "Running as root, fixing ownership..."
        chown -R llmforge:llmforge /app/models/fine_tuned 2>/dev/null || echo "Could not fix /app/models/fine_tuned ownership (may not exist or permissions denied)"
        chown -R llmforge:llmforge /app/logs 2>/dev/null || echo "Could not fix /app/logs ownership"
        chown -R llmforge:llmforge /app/data 2>/dev/null || echo "Could not fix /app/data ownership"
        chown -R llmforge:llmforge /app/datasets 2>/dev/null || echo "Could not fix /app/datasets ownership"
        
        # Switch to llmforge user and run the actual command
        echo "Switching to llmforge user and starting application..."
        exec su llmforge -c "$*"
    else
        echo "Not running as root, proceeding as current user..."
    fi
else
    echo "Cannot write to /app, proceeding with current permissions..."
fi

# If we reach here, we're not running as root or can't fix permissions
# Just run the command as-is
echo "Starting application..."
exec "$@"