@echo off
REM Build the shared base image first (one-time setup)
echo Building shared base image with common dependencies...
docker-compose --profile build-base build llmforge-base

echo Shared base image 'local/llmforge-base:latest' is ready!
echo Now you can run: docker-compose up