#!/usr/bin/env pwsh
# Windows GPU Sharing Deployment Script for LLMForge
# Optimized for Windows Docker Desktop with RTX 4060 8GB

param(
    [string]$Action = "deploy",
    [switch]$SkipChecks = $false,
    [switch]$BuildOnly = $false,
    [switch]$Verbose = $false
)

# Set error handling
$ErrorActionPreference = "Stop"

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

function Write-Success { param([string]$Text) Write-ColorOutput $Text "Green" }
function Write-Warning { param([string]$Text) Write-ColorOutput $Text "Yellow" }
function Write-Error { param([string]$Text) Write-ColorOutput $Text "Red" }
function Write-Info { param([string]$Text) Write-ColorOutput $Text "Cyan" }

# Main functions
function Test-Prerequisites {
    Write-Info "=== Checking Prerequisites ==="
    
    # Check Docker Desktop
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    } catch {
        Write-Error "Docker Desktop is not installed or not running!"
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker compose version
        Write-Success "Docker Compose found: $composeVersion"
    } catch {
        Write-Error "Docker Compose is not available!"
        exit 1
    }
    
    # Check NVIDIA Docker support
    try {
        $gpuInfo = docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            Write-Success "NVIDIA GPU detected: $gpuInfo"
        } else {
            Write-Warning "GPU detection failed - continuing anyway"
        }
    } catch {
        Write-Warning "Could not verify GPU support - Docker may need GPU runtime configuration"
    }
    
    # Check available system memory
    $totalMemory = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    Write-Info "System RAM: ${totalMemory}GB"
    
    if ($totalMemory -lt 16) {
        Write-Warning "Recommended 16GB+ RAM for optimal performance. Current: ${totalMemory}GB"
    }
    
    # Check disk space
    $freeSpace = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Write-Info "Available disk space: ${freeSpace}GB"
    
    if ($freeSpace -lt 20) {
        Write-Warning "Low disk space detected. At least 20GB recommended for Docker images and models."
    }
    
    Write-Success "Prerequisites check completed!"
}

function Stop-ExistingContainers {
    Write-Info "=== Stopping existing containers ==="
    
    $containers = docker ps -aq --filter "name=llmforge"
    if ($containers) {
        Write-Info "Stopping LLMForge containers..."
        docker stop $containers 2>$null
        docker rm $containers 2>$null
        Write-Success "Existing containers stopped and removed"
    } else {
        Write-Info "No existing LLMForge containers found"
    }
}

function Build-Services {
    Write-Info "=== Building Services ==="
    
    # Build base images first for better caching
    Write-Info "Building shared base image..."
    docker compose build base
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build base image!"
        exit 1
    }
    
    Write-Info "Building ML base image..."
    docker compose build ml-base
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build ML base image!"
        exit 1
    }
    
    # Build GPU services
    Write-Info "Building GPU-enabled services..."
    docker compose build training-service chat-comparison-service model-management-service
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build GPU services!"
        exit 1
    }
    
    # Build remaining services
    Write-Info "Building remaining services..."
    docker compose build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build all services!"
        exit 1
    }
    
    Write-Success "All services built successfully!"
}

function Deploy-Services {
    Write-Info "=== Deploying Services ==="
    
    # Start infrastructure services first
    Write-Info "Starting infrastructure services..."
    docker compose up -d redis
    Start-Sleep 10
    
    # Start orchestrator
    Write-Info "Starting orchestrator..."
    docker compose up -d orchestrator
    Start-Sleep 15
    
    # Start API gateway
    Write-Info "Starting API gateway..."
    docker compose up -d api-gateway
    Start-Sleep 10
    
    # Start GPU services with staggered deployment to avoid memory spikes
    Write-Info "Starting training service (primary GPU user)..."
    docker compose up -d training-service
    Start-Sleep 20
    
    Write-Info "Starting chat comparison service..."
    docker compose up -d chat-comparison-service
    Start-Sleep 15
    
    Write-Info "Starting model management service..."
    docker compose up -d model-management-service
    Start-Sleep 15
    
    # Start remaining services
    Write-Info "Starting remaining services..."
    docker compose up -d data-generation-service visualization-service factory-service
    
    Write-Success "All services started!"
}

function Test-ServiceHealth {
    Write-Info "=== Testing Service Health ==="
    
    $services = @(
        @{Name="Redis"; Port=6379; Type="tcp"}
        @{Name="Orchestrator"; Port=8080; Path="/health"}
        @{Name="API Gateway"; Port=8000; Path="/health"}
        @{Name="Training Service"; Port=8001; Path="/health"}
        @{Name="Chat Comparison"; Port=8002; Path="/health"}
        @{Name="Data Generation"; Port=8003; Path="/health"}
        @{Name="Visualization"; Port=8004; Path="/health"}
        @{Name="Model Management"; Port=8005; Path="/health"}
        @{Name="Factory"; Port=8006; Path="/health"}
    )
    
    Write-Info "Waiting for services to initialize..."
    Start-Sleep 30
    
    foreach ($service in $services) {
        $maxRetries = 6
        $retryCount = 0
        $healthy = $false
        
        while ($retryCount -lt $maxRetries -and -not $healthy) {
            try {
                if ($service.Type -eq "tcp") {
                    # Test TCP connection for Redis
                    $connection = Test-NetConnection -ComputerName localhost -Port $service.Port -InformationLevel Quiet
                    if ($connection) {
                        Write-Success "$($service.Name) is healthy (Port $($service.Port))"
                        $healthy = $true
                    }
                } else {
                    # Test HTTP health endpoint
                    $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)$($service.Path)" -UseBasicParsing -TimeoutSec 10
                    if ($response.StatusCode -eq 200) {
                        Write-Success "$($service.Name) is healthy"
                        $healthy = $true
                    }
                }
            } catch {
                $retryCount++
                if ($retryCount -eq $maxRetries) {
                    Write-Warning "$($service.Name) health check failed after $maxRetries attempts"
                } else {
                    Write-Info "Waiting for $($service.Name)... (attempt $retryCount/$maxRetries)"
                    Start-Sleep 10
                }
            }
        }
    }
}

function Show-GPUUsage {
    Write-Info "=== GPU Usage Status ==="
    
    try {
        # Check GPU usage in containers
        $gpuContainers = docker ps --filter "name=llmforge" --format "table {{.Names}}\t{{.Status}}" | Where-Object { $_ -match "(training|chat-comparison|model-management)" }
        Write-Info "GPU-enabled containers:"
        Write-Output $gpuContainers
        
        # Try to get GPU memory usage
        $gpuStats = docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($gpuStats) {
            Write-Info "GPU Memory and Utilization: $gpuStats"
        }
    } catch {
        Write-Warning "Could not retrieve GPU usage information"
    }
}

function Show-ServiceUrls {
    Write-Info "=== Service URLs ==="
    Write-ColorOutput "API Gateway:        http://localhost:8000" "White"
    Write-ColorOutput "Training Service:   http://localhost:8001" "White"
    Write-ColorOutput "Chat Comparison:    http://localhost:8002" "White"
    Write-ColorOutput "Data Generation:    http://localhost:8003" "White"
    Write-ColorOutput "Visualization:      http://localhost:8004" "White"
    Write-ColorOutput "Model Management:   http://localhost:8005" "White"
    Write-ColorOutput "Factory:            http://localhost:8006" "White"
    Write-ColorOutput "Orchestrator:       http://localhost:8080" "White"
    Write-ColorOutput ""
    Write-ColorOutput "Redis:              localhost:6379" "Yellow"
    Write-ColorOutput ""
    Write-ColorOutput "Health endpoints available at /health for each HTTP service" "Gray"
}

function Show-Usage {
    Write-Info "Windows GPU Sharing Deployment Script for LLMForge"
    Write-Output ""
    Write-Output "Usage: .\deploy-windows.ps1 [options]"
    Write-Output ""
    Write-Output "Actions:"
    Write-Output "  deploy     - Full deployment (default)"
    Write-Output "  build      - Build images only"
    Write-Output "  start      - Start existing services"
    Write-Output "  stop       - Stop all services"
    Write-Output "  restart    - Restart all services"
    Write-Output "  status     - Show service status"
    Write-Output "  clean      - Stop and remove all containers"
    Write-Output "  logs       - Show service logs"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -SkipChecks   Skip prerequisite checks"
    Write-Output "  -BuildOnly    Build images but don't start services"
    Write-Output "  -Verbose      Enable verbose output"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output "  .\deploy-windows.ps1 deploy"
    Write-Output "  .\deploy-windows.ps1 build -Verbose"
    Write-Output "  .\deploy-windows.ps1 status"
}

# Main execution logic
Write-Info "LLMForge Windows GPU Sharing Deployment"
Write-Info "========================================"

switch ($Action.ToLower()) {
    "deploy" {
        if (-not $SkipChecks) {
            Test-Prerequisites
        }
        Stop-ExistingContainers
        Build-Services
        if (-not $BuildOnly) {
            Deploy-Services
            Test-ServiceHealth
            Show-GPUUsage
            Show-ServiceUrls
        }
    }
    
    "build" {
        Build-Services
    }
    
    "start" {
        Write-Info "Starting services..."
        docker compose up -d
        Test-ServiceHealth
        Show-ServiceUrls
    }
    
    "stop" {
        Write-Info "Stopping services..."
        docker compose stop
        Write-Success "Services stopped"
    }
    
    "restart" {
        Write-Info "Restarting services..."
        docker compose restart
        Test-ServiceHealth
        Show-ServiceUrls
    }
    
    "status" {
        Write-Info "Service Status:"
        docker compose ps
        Show-GPUUsage
    }
    
    "clean" {
        Write-Info "Cleaning up..."
        docker compose down -v --remove-orphans
        docker system prune -f
        Write-Success "Cleanup completed"
    }
    
    "logs" {
        Write-Info "Showing service logs..."
        docker compose logs -f --tail=50
    }
    
    "help" {
        Show-Usage
    }
    
    default {
        Write-Error "Unknown action: $Action"
        Show-Usage
        exit 1
    }
}

Write-Success "Script completed successfully!"