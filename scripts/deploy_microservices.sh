#!/bin/bash

# LLMForge Microservices Deployment Script
# This script handles the complete deployment of the LLMForge microservices architecture

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="llmforge"
CHART_NAME="llmforge"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"localhost:5000"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists kubectl; then
        missing_tools+=("kubectl")
    fi
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists helm; then
        missing_tools+=("helm")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install these tools before continuing."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info >/dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    local services=(
        "orchestrator"
        "api_gateway"
        "training_service"
        "chat_comparison_service"
        "data_generation_service"
        "visualization_service"
        "model_management_service"
    )
    
    for service in "${services[@]}"; do
        print_status "Building $service..."
        
        docker build -t "${DOCKER_REGISTRY}/llmforge-${service}:${IMAGE_TAG}" \
            -f "docker/${service}/Dockerfile" .
        
        if [ $? -eq 0 ]; then
            print_success "Built $service image"
        else
            print_error "Failed to build $service image"
            exit 1
        fi
    done
}

# Push Docker images
push_images() {
    print_status "Pushing Docker images..."
    
    local services=(
        "orchestrator"
        "api_gateway"
        "training_service"
        "chat_comparison_service"
        "data_generation_service"
        "visualization_service"
        "model_management_service"
    )
    
    for service in "${services[@]}"; do
        print_status "Pushing $service..."
        
        docker push "${DOCKER_REGISTRY}/llmforge-${service}:${IMAGE_TAG}"
        
        if [ $? -eq 0 ]; then
            print_success "Pushed $service image"
        else
            print_error "Failed to push $service image"
            exit 1
        fi
    done
}

# Create namespace and secrets
setup_namespace() {
    print_status "Setting up namespace and secrets..."
    
    # Create namespace
    kubectl apply -f kubernetes/namespace.yaml
    
    # Create secrets if they don't exist
    if ! kubectl get secret llmforge-secrets -n $NAMESPACE >/dev/null 2>&1; then
        print_status "Creating secrets..."
        
        # Prompt for required secrets
        echo -n "Enter HuggingFace token: "
        read -s HUGGINGFACE_TOKEN
        echo
        
        echo -n "Enter Grafana admin password: "
        read -s GRAFANA_PASSWORD
        echo
        
        kubectl create secret generic llmforge-secrets \
            --from-literal=huggingface-token="$HUGGINGFACE_TOKEN" \
            --from-literal=grafana-password="$GRAFANA_PASSWORD" \
            -n $NAMESPACE
        
        print_success "Secrets created"
    else
        print_status "Secrets already exist, skipping creation"
    fi
}

# Deploy infrastructure components
deploy_infrastructure() {
    print_status "Deploying infrastructure components..."
    
    # Deploy ConfigMaps
    kubectl apply -f kubernetes/configmaps.yaml
    
    # Deploy storage
    kubectl apply -f kubernetes/storage.yaml
    
    # Wait for PVCs to be bound
    print_status "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n $NAMESPACE --timeout=300s
    
    print_success "Infrastructure deployed"
}

# Deploy Redis
deploy_redis() {
    print_status "Deploying Redis..."
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.enabled=false \
        --set architecture=standalone \
        --set master.persistence.size=20Gi \
        --set master.resources.requests.memory=1Gi \
        --set master.resources.limits.memory=2Gi \
        --wait
    
    print_success "Redis deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    kubectl apply -f kubernetes/monitoring.yaml
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=available deployment/prometheus -n $NAMESPACE --timeout=300s
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=available deployment/grafana -n $NAMESPACE --timeout=300s
    
    print_success "Monitoring stack deployed"
}

# Deploy orchestrator
deploy_orchestrator() {
    print_status "Deploying orchestrator..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llmforge-orchestrator
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llmforge-orchestrator
  template:
    metadata:
      labels:
        app: llmforge-orchestrator
        app.kubernetes.io/part-of: llmforge-platform
    spec:
      containers:
      - name: orchestrator
        image: ${DOCKER_REGISTRY}/llmforge-orchestrator:${IMAGE_TAG}
        ports:
        - containerPort: 8080
        env:
        - name: SERVICE_HOST
          value: "0.0.0.0"
        - name: SERVICE_PORT
          value: "8080"
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: llmforge-orchestrator
  namespace: $NAMESPACE
spec:
  selector:
    app: llmforge-orchestrator
  ports:
  - port: 8080
    targetPort: 8080
EOF

    kubectl wait --for=condition=available deployment/llmforge-orchestrator -n $NAMESPACE --timeout=300s
    print_success "Orchestrator deployed"
}

# Deploy microservices
deploy_microservices() {
    print_status "Deploying microservices..."
    
    # Deploy training service
    kubectl apply -f kubernetes/training-service.yaml
    
    # Deploy other services
    local services=(
        "api-gateway"
        "chat-comparison"
        "data-generation"
        "visualization"
        "model-management"
    )
    
    for service in "${services[@]}"; do
        print_status "Deploying $service service..."
        
        if [ -f "kubernetes/${service}-service.yaml" ]; then
            kubectl apply -f "kubernetes/${service}-service.yaml"
        else
            print_warning "Kubernetes manifest for $service not found, skipping..."
        fi
    done
    
    # Wait for all deployments to be ready
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment --all -n $NAMESPACE --timeout=600s
    
    print_success "Microservices deployed"
}

# Setup ingress
setup_ingress() {
    print_status "Setting up ingress..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llmforge-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: llmforge.local
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: llmforge-api-gateway
            port:
              number: 8000
      - path: /grafana
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
      - path: /prometheus
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
EOF

    print_success "Ingress configured"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Check if all pods are running
    local failed_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running -o name 2>/dev/null | wc -l)
    
    if [ "$failed_pods" -gt 0 ]; then
        print_warning "Some pods are not running:"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
    else
        print_success "All pods are running"
    fi
    
    # Test API Gateway connectivity
    print_status "Testing API Gateway connectivity..."
    kubectl port-forward service/llmforge-api-gateway 8000:8000 -n $NAMESPACE &
    PF_PID=$!
    sleep 5
    
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API Gateway is responding"
    else
        print_warning "API Gateway health check failed"
    fi
    
    kill $PF_PID 2>/dev/null || true
}

# Print deployment status
print_status_summary() {
    print_status "Deployment Summary"
    echo "===================="
    
    kubectl get pods -n $NAMESPACE
    echo
    kubectl get services -n $NAMESPACE
    echo
    
    print_status "Access URLs (after port-forwarding):"
    echo "API Gateway:    http://localhost:8000"
    echo "Grafana:        http://localhost:3000 (admin/admin)"
    echo "Prometheus:     http://localhost:9090"
    echo
    
    print_status "Port-forward commands:"
    echo "kubectl port-forward service/llmforge-api-gateway 8000:8000 -n $NAMESPACE"
    echo "kubectl port-forward service/grafana 3000:3000 -n $NAMESPACE"
    echo "kubectl port-forward service/prometheus 9090:9090 -n $NAMESPACE"
}

# Rollback function
rollback() {
    print_warning "Rolling back deployment..."
    
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    print_success "Rollback completed"
}

# Main deployment function
main() {
    local action=${1:-"deploy"}
    
    case $action in
        "deploy")
            print_status "Starting LLMForge microservices deployment..."
            
            check_prerequisites
            build_images
            push_images
            setup_namespace
            deploy_infrastructure
            deploy_redis
            deploy_monitoring
            deploy_orchestrator
            deploy_microservices
            setup_ingress
            health_check
            print_status_summary
            
            print_success "LLMForge microservices deployed successfully!"
            ;;
        "rollback")
            rollback
            ;;
        "health")
            health_check
            ;;
        "status")
            print_status_summary
            ;;
        *)
            echo "Usage: $0 [deploy|rollback|health|status]"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy the complete microservices stack (default)"
            echo "  rollback - Remove the entire deployment"
            echo "  health   - Perform health check on deployed services"
            echo "  status   - Show deployment status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"