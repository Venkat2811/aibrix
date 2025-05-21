#!/bin/bash
set -eo pipefail

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print section header
print_header() {
  echo -e "\n${BOLD}${GREEN}==== $1 ====${NC}\n"
}

# Print info message
print_info() {
  echo -e "${YELLOW}$1${NC}"
}

# Print error message
print_error() {
  echo -e "${RED}$1${NC}"
}

# Wait for resource to be ready
wait_for_resource() {
  local resource_type="$1"
  local resource_name="$2"
  local namespace="${3:-default}"
  
  print_info "Waiting for $resource_type/$resource_name in namespace $namespace to be ready..."
  
  # Different resources need different wait conditions
  case "$resource_type" in
    deployment)
      kubectl -n "$namespace" wait --for=condition=Available --timeout=300s deployment/"$resource_name"
      ;;
    pod)
      kubectl -n "$namespace" wait --for=condition=Ready --timeout=300s pod/"$resource_name"
      ;;
    service)
      # For services, we just check if they exist since there's no standard "ready" condition
      until kubectl -n "$namespace" get service "$resource_name" &>/dev/null; do
        sleep 5
        print_info "Waiting for service $resource_name..."
      done
      print_info "Service $resource_name is available."
      ;;
    *)
      print_error "Unknown resource type: $resource_type"
      return 1
      ;;
  esac
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
  print_error "kubectl is not installed. Please install it first."
  exit 1
fi

# Check if helm is available (for monitoring setup)
if ! command -v helm &> /dev/null; then
  print_error "helm is not installed. Please install it first."
  exit 1
fi

# Start AIBrix cluster
print_header "Starting AIBrix Cluster"
print_info "This will use the test-e2e script to start an AIBrix cluster..."

# Save the current directory to return to it later
current_dir=$(pwd)
cd "$(git rev-parse --show-toplevel)" || exit 1

# Start AIBrix using the test-e2e script
INSTALL_AIBRIX=true make test-e2e &
AIBRIX_PID=$!

# Wait for AIBrix to be ready
print_info "Waiting for AIBrix to start (30 seconds)..."
sleep 30

# Check if AIBrix started successfully
if ! kubectl get namespace aibrix-system &>/dev/null; then
  print_error "AIBrix cluster failed to start. Check the logs for details."
  kill $AIBRIX_PID
  exit 1
fi

# Setup Prometheus monitoring stack
print_header "Setting up Monitoring"

# Create prometheus namespace if it doesn't exist
kubectl create namespace prometheus --dry-run=client -o yaml | kubectl apply -f -

# Install prometheus-operator
print_info "Installing kube-prometheus-stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack --namespace prometheus --wait

# Apply service monitors from observability directory
print_info "Applying service monitors for AIBrix components..."
kubectl apply -f observability/monitor/service_monitor_controller_manager.yaml
kubectl apply -f observability/monitor/envoy_metrics_service.yaml
kubectl apply -f observability/monitor/envoy_metrics_service.yaml
kubectl apply -f observability/monitor/service_monitor_vllm.yaml

# Deploy TinyLlama
print_header "Deploying TinyLlama"
kubectl apply -f benchmarks/scenarios/gateway/vtc-basic-routing/tinyllama-deployment.yaml

# Wait for TinyLlama to be ready
wait_for_resource deployment tinyllama aibrix-system
print_info "TinyLlama is now deployed and ready!"

# Get service IP/port for TinyLlama
TINYLLAMA_SERVICE_IP=$(kubectl -n aibrix-system get service tinyllama -o jsonpath='{.spec.clusterIP}')
TINYLLAMA_SERVICE_PORT=$(kubectl -n aibrix-system get service tinyllama -o jsonpath='{.spec.ports[0].port}')
TINYLLAMA_ENDPOINT="http://${TINYLLAMA_SERVICE_IP}:${TINYLLAMA_SERVICE_PORT}"

print_info "TinyLlama service is available at: ${TINYLLAMA_ENDPOINT}"

# Configure gateway endpoint
GATEWAY_ENDPOINT="http://localhost:8888"
print_info "Gateway endpoint is: ${GATEWAY_ENDPOINT}"

# Run the benchmark
print_header "Running VTC Benchmark with ShareGPT Data"
export api_endpoint="${GATEWAY_ENDPOINT}"

# Create directories for benchmark results if they don't exist
mkdir -p benchmarks/scenarios/gateway/vtc-basic-routing/dataset
mkdir -p benchmarks/scenarios/gateway/vtc-basic-routing/workload
mkdir -p benchmarks/scenarios/gateway/vtc-basic-routing/results

# Download ShareGPT dataset if needed
print_info "Downloading ShareGPT dataset if needed..."
python3 benchmarks/scenarios/gateway/vtc-basic-routing/download_sharegpt.py

# Run the benchmark
print_info "Running VTC-basic routing benchmark..."
cd "$current_dir" || exit 1
python3 benchmarks/scenarios/gateway/vtc-basic-routing/run_benchmark.py

print_info "Benchmark completed! Check the results in:"
print_info "benchmarks/scenarios/gateway/vtc-basic-routing/results/"

# Setup port-forwarding for Grafana
print_header "Setting up Grafana port-forwarding"
print_info "Starting port-forward for Grafana. Access it at http://localhost:3000"
print_info "Default credentials are admin/prom-operator"
kubectl port-forward -n prometheus svc/prometheus-grafana 3000:80 &
GRAFANA_PID=$!

# Tell the user how to import dashboards
print_info "To import the AIBrix dashboards in Grafana:"
print_info "1. Log in to Grafana at http://localhost:3000"
print_info "2. Navigate to Dashboards > Import"
print_info "3. Upload or paste the JSON from these files:"
print_info "   - observability/grafana/AIBrix_Control_Plane_Runtime_Dashboard.json"
print_info "   - observability/grafana/AIBrix_Envoy_Gateway_Dashboard.json"
print_info "   - observability/grafana/AIBrix_vLLM_Engine_Dashboard.json"

print_header "Setup Complete"
print_info "Press Ctrl+C when you're done to clean up."

# Wait for user input to terminate
wait $AIBRIX_PID

# Clean up
kill $GRAFANA_PID 2>/dev/null || true
print_info "All processes terminated." 