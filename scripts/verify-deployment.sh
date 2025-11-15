#!/bin/bash
# Deployment Verification Script
# Comprehensive health checks for staging and production deployments
#
# Usage:
#   ./scripts/verify-deployment.sh <environment>
#
# Examples:
#   ./scripts/verify-deployment.sh dev
#   ./scripts/verify-deployment.sh staging
#   ./scripts/verify-deployment.sh production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
NAMESPACE="geo-climate-${ENVIRONMENT}"

if [ "$ENVIRONMENT" = "production" ]; then
    NAMESPACE="geo-climate"
fi

API_HOST="api.${ENVIRONMENT}.geo-climate.example.com"
if [ "$ENVIRONMENT" = "production" ]; then
    API_HOST="api.geo-climate.example.com"
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Geo Climate Deployment Verification${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Namespace: ${NAMESPACE}${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to print info
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if kubectl is configured
check_kubectl() {
    info "Checking kubectl configuration..."
    if ! kubectl cluster-info &> /dev/null; then
        error "kubectl is not configured or cluster is unreachable"
        exit 1
    fi
    success "kubectl configured and cluster reachable"
}

# Check namespace exists
check_namespace() {
    info "Checking namespace ${NAMESPACE}..."
    if kubectl get namespace ${NAMESPACE} &> /dev/null; then
        success "Namespace ${NAMESPACE} exists"
    else
        error "Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
}

# Check all pods are running
check_pods() {
    info "Checking pod status..."

    # Get pod status
    POD_STATUS=$(kubectl get pods -n ${NAMESPACE} -o json)

    # Count pods by status
    TOTAL_PODS=$(echo $POD_STATUS | jq '.items | length')
    RUNNING_PODS=$(echo $POD_STATUS | jq '[.items[] | select(.status.phase=="Running")] | length')
    PENDING_PODS=$(echo $POD_STATUS | jq '[.items[] | select(.status.phase=="Pending")] | length')
    FAILED_PODS=$(echo $POD_STATUS | jq '[.items[] | select(.status.phase=="Failed")] | length')

    info "Total pods: ${TOTAL_PODS}, Running: ${RUNNING_PODS}, Pending: ${PENDING_PODS}, Failed: ${FAILED_PODS}"

    if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ] && [ "$TOTAL_PODS" -gt 0 ]; then
        success "All ${TOTAL_PODS} pods are running"
    else
        warning "Not all pods are running"
        kubectl get pods -n ${NAMESPACE}

        # Show problematic pods
        if [ "$PENDING_PODS" -gt 0 ] || [ "$FAILED_PODS" -gt 0 ]; then
            error "Found ${PENDING_PODS} pending and ${FAILED_PODS} failed pods"
            kubectl get pods -n ${NAMESPACE} | grep -v Running | grep -v Completed || true
        fi
    fi

    # Check for pods with restarts
    RESTARTING_PODS=$(kubectl get pods -n ${NAMESPACE} -o json | jq '[.items[] | select(.status.containerStatuses[]?.restartCount > 3)] | length')
    if [ "$RESTARTING_PODS" -gt 0 ]; then
        warning "${RESTARTING_PODS} pods have more than 3 restarts"
        kubectl get pods -n ${NAMESPACE} --sort-by='.status.containerStatuses[0].restartCount' | tail -n 5
    fi
}

# Check deployments
check_deployments() {
    info "Checking deployments..."

    DEPLOYMENTS=$(kubectl get deployments -n ${NAMESPACE} -o json)

    # Check each deployment
    for deployment in $(echo $DEPLOYMENTS | jq -r '.items[].metadata.name'); do
        DESIRED=$(kubectl get deployment ${deployment} -n ${NAMESPACE} -o jsonpath='{.spec.replicas}')
        READY=$(kubectl get deployment ${deployment} -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}')

        if [ "$DESIRED" = "$READY" ]; then
            success "Deployment ${deployment}: ${READY}/${DESIRED} ready"
        else
            error "Deployment ${deployment}: ${READY}/${DESIRED} ready"
        fi
    done
}

# Check StatefulSets
check_statefulsets() {
    info "Checking StatefulSets..."

    STATEFULSETS=$(kubectl get statefulsets -n ${NAMESPACE} -o json 2>/dev/null || echo '{"items":[]}')
    COUNT=$(echo $STATEFULSETS | jq '.items | length')

    if [ "$COUNT" -eq 0 ]; then
        info "No StatefulSets found in namespace"
        return
    fi

    # Check each StatefulSet
    for sts in $(echo $STATEFULSETS | jq -r '.items[].metadata.name'); do
        DESIRED=$(kubectl get statefulset ${sts} -n ${NAMESPACE} -o jsonpath='{.spec.replicas}')
        READY=$(kubectl get statefulset ${sts} -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}')

        if [ "$DESIRED" = "$READY" ]; then
            success "StatefulSet ${sts}: ${READY}/${DESIRED} ready"
        else
            error "StatefulSet ${sts}: ${READY}/${DESIRED} ready"
        fi
    done
}

# Check services
check_services() {
    info "Checking services..."

    SERVICES=$(kubectl get svc -n ${NAMESPACE} -o json)
    COUNT=$(echo $SERVICES | jq '.items | length')

    success "Found ${COUNT} services"

    # Check for LoadBalancer services without external IP
    PENDING_LB=$(echo $SERVICES | jq '[.items[] | select(.spec.type=="LoadBalancer" and (.status.loadBalancer.ingress == null))] | length')
    if [ "$PENDING_LB" -gt 0 ]; then
        warning "${PENDING_LB} LoadBalancer services are pending external IP"
    fi
}

# Check ingress
check_ingress() {
    info "Checking ingress..."

    INGRESSES=$(kubectl get ingress -n ${NAMESPACE} -o json 2>/dev/null || echo '{"items":[]}')
    COUNT=$(echo $INGRESSES | jq '.items | length')

    if [ "$COUNT" -eq 0 ]; then
        warning "No ingress resources found"
        return
    fi

    success "Found ${COUNT} ingress resources"

    # Check each ingress
    for ingress in $(echo $INGRESSES | jq -r '.items[].metadata.name'); do
        HOSTS=$(kubectl get ingress ${ingress} -n ${NAMESPACE} -o jsonpath='{.spec.rules[*].host}')
        ADDRESS=$(kubectl get ingress ${ingress} -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

        if [ -n "$ADDRESS" ]; then
            success "Ingress ${ingress} has address: ${ADDRESS}"
            info "  Hosts: ${HOSTS}"
        else
            warning "Ingress ${ingress} has no address yet"
        fi
    done
}

# Check PVCs
check_pvcs() {
    info "Checking PersistentVolumeClaims..."

    PVCS=$(kubectl get pvc -n ${NAMESPACE} -o json)
    TOTAL=$(echo $PVCS | jq '.items | length')
    BOUND=$(echo $PVCS | jq '[.items[] | select(.status.phase=="Bound")] | length')

    if [ "$TOTAL" -eq "$BOUND" ]; then
        success "All ${TOTAL} PVCs are bound"
    else
        warning "${BOUND}/${TOTAL} PVCs are bound"
        kubectl get pvc -n ${NAMESPACE} | grep -v Bound || true
    fi
}

# Check secrets
check_secrets() {
    info "Checking secrets..."

    REQUIRED_SECRETS=("postgres-secrets" "redis-secrets" "geo-climate-secrets")

    for secret in "${REQUIRED_SECRETS[@]}"; do
        if kubectl get secret ${secret} -n ${NAMESPACE} &> /dev/null; then
            success "Secret ${secret} exists"
        else
            error "Secret ${secret} is missing"
        fi
    done
}

# Check ConfigMaps
check_configmaps() {
    info "Checking ConfigMaps..."

    CONFIGMAPS=$(kubectl get configmap -n ${NAMESPACE} -o json)
    COUNT=$(echo $CONFIGMAPS | jq '.items | length')

    success "Found ${COUNT} ConfigMaps"
}

# Check HPA
check_hpa() {
    info "Checking HorizontalPodAutoscaler..."

    HPAS=$(kubectl get hpa -n ${NAMESPACE} -o json 2>/dev/null || echo '{"items":[]}')
    COUNT=$(echo $HPAS | jq '.items | length')

    if [ "$COUNT" -eq 0 ]; then
        info "No HPA resources found"
        return
    fi

    # Check each HPA
    for hpa in $(echo $HPAS | jq -r '.items[].metadata.name'); do
        CURRENT=$(kubectl get hpa ${hpa} -n ${NAMESPACE} -o jsonpath='{.status.currentReplicas}')
        DESIRED=$(kubectl get hpa ${hpa} -n ${NAMESPACE} -o jsonpath='{.status.desiredReplicas}')
        MIN=$(kubectl get hpa ${hpa} -n ${NAMESPACE} -o jsonpath='{.spec.minReplicas}')
        MAX=$(kubectl get hpa ${hpa} -n ${NAMESPACE} -o jsonpath='{.spec.maxReplicas}')

        success "HPA ${hpa}: ${CURRENT}/${DESIRED} replicas (min: ${MIN}, max: ${MAX})"
    done
}

# Test API health endpoint
check_api_health() {
    info "Checking API health endpoint..."

    # Try internal service first
    if kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n ${NAMESPACE} -- \
        curl -sf http://geo-climate-api/health &> /dev/null; then
        success "API health endpoint is accessible internally"
    else
        error "API health endpoint is not accessible internally"
    fi

    # Try external endpoint if ingress is configured
    if [ -n "$API_HOST" ]; then
        if curl -sf https://${API_HOST}/health &> /dev/null; then
            success "API health endpoint is accessible externally at https://${API_HOST}/health"
        else
            warning "API health endpoint is not accessible externally (might be expected in dev/staging)"
        fi
    fi
}

# Check API readiness
check_api_readiness() {
    info "Checking API readiness..."

    if kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n ${NAMESPACE} -- \
        curl -sf http://geo-climate-api/health/ready &> /dev/null; then
        success "API is ready"
    else
        error "API is not ready"
    fi
}

# Check database connectivity
check_database() {
    info "Checking database connectivity..."

    # Get first API pod
    API_POD=$(kubectl get pods -n ${NAMESPACE} -l app=geo-climate-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -z "$API_POD" ]; then
        warning "No API pods found, skipping database check"
        return
    fi

    # Test database connection
    if kubectl exec -n ${NAMESPACE} ${API_POD} -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ.get('DATABASE_URL', 'postgresql://geo_climate:password@postgres:5432/geo_climate_prod'))
    conn.close()
    print('OK')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" &> /dev/null; then
        success "Database connectivity is working"
    else
        error "Database connectivity failed"
    fi
}

# Check Redis connectivity
check_redis() {
    info "Checking Redis connectivity..."

    # Get first API pod
    API_POD=$(kubectl get pods -n ${NAMESPACE} -l app=geo-climate-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -z "$API_POD" ]; then
        warning "No API pods found, skipping Redis check"
        return
    fi

    # Test Redis connection
    if kubectl exec -n ${NAMESPACE} ${API_POD} -- python -c "
import redis
import os
try:
    r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379/0'))
    r.ping()
    print('OK')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" &> /dev/null; then
        success "Redis connectivity is working"
    else
        error "Redis connectivity failed"
    fi
}

# Check metrics endpoint
check_metrics() {
    info "Checking metrics endpoint..."

    if kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n ${NAMESPACE} -- \
        curl -sf http://geo-climate-api:9090/metrics &> /dev/null; then
        success "Metrics endpoint is accessible"
    else
        warning "Metrics endpoint is not accessible"
    fi
}

# Check resource usage
check_resource_usage() {
    info "Checking resource usage..."

    # CPU usage
    echo ""
    info "CPU Usage:"
    kubectl top pods -n ${NAMESPACE} --sort-by=cpu | head -n 6

    # Memory usage
    echo ""
    info "Memory Usage:"
    kubectl top pods -n ${NAMESPACE} --sort-by=memory | head -n 6

    # Node usage
    echo ""
    info "Node Usage:"
    kubectl top nodes
}

# Check recent events
check_events() {
    info "Checking recent events..."

    echo ""
    warning "Recent Warning Events:"
    kubectl get events -n ${NAMESPACE} --field-selector type=Warning --sort-by='.lastTimestamp' | tail -n 10

    echo ""
    info "Recent Normal Events:"
    kubectl get events -n ${NAMESPACE} --field-selector type=Normal --sort-by='.lastTimestamp' | tail -n 5
}

# Summary
print_summary() {
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Verification Summary${NC}"
    echo -e "${BLUE}=====================================${NC}"

    # Count checks
    TOTAL_CHECKS=20
    echo ""
    info "Completed ${TOTAL_CHECKS} verification checks"
    echo ""

    # Next steps
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review any warnings or errors above"
    echo "2. Check application logs: kubectl logs -n ${NAMESPACE} -l app=geo-climate-api"
    echo "3. Run integration tests: pytest tests/integration/"
    echo "4. Run load tests: locust -f tests/load/locustfile.py"
    echo "5. Monitor metrics: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
    echo ""
}

# Main execution
main() {
    check_kubectl
    check_namespace
    echo ""

    check_pods
    check_deployments
    check_statefulsets
    echo ""

    check_services
    check_ingress
    echo ""

    check_pvcs
    check_secrets
    check_configmaps
    echo ""

    check_hpa
    echo ""

    check_api_health
    check_api_readiness
    check_database
    check_redis
    check_metrics
    echo ""

    check_resource_usage
    echo ""

    check_events

    print_summary
}

# Run main
main
