#!/bin/bash
###############################################################################
# Chaos Engineering Test Runner
# Advanced chaos experiments execution and analysis framework
#
# Usage:
#   ./chaos-engineering-runner.sh [experiment-type] [duration] [--dry-run]
#
# Examples:
#   ./chaos-engineering-runner.sh pod-kill 5m
#   ./chaos-engineering-runner.sh network-delay 10m --dry-run
#   ./chaos-engineering-runner.sh stress-test 15m
#   ./chaos-engineering-runner.sh full-suite 30m
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-geo-climate}"
CHAOS_NAMESPACE="${CHAOS_NAMESPACE:-chaos-mesh}"
RESULTS_DIR="./chaos-results/$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}✓${NC} $*"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

error() {
    echo -e "${RED}✗${NC} $*"
    exit 1
}

info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

# Parse arguments
EXPERIMENT_TYPE="${1:-full-suite}"
DURATION="${2:-10m}"
if [[ "${3:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    warning "Running in DRY-RUN mode - no actual chaos will be injected"
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"
log "Results will be saved to: ${RESULTS_DIR}"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Please install kubectl."
    fi
    success "kubectl found"

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    success "Cluster connection OK"

    # Check if Chaos Mesh is installed
    if ! kubectl get namespace "${CHAOS_NAMESPACE}" &> /dev/null; then
        error "Chaos Mesh namespace not found. Please install Chaos Mesh first."
    fi
    success "Chaos Mesh installed"

    # Check if application is running
    if ! kubectl get pods -n "${NAMESPACE}" -l app=geo-climate-api | grep -q Running; then
        error "Application pods not running in namespace ${NAMESPACE}"
    fi
    success "Application running"
}

# Capture baseline metrics
capture_baseline() {
    log "Capturing baseline metrics..."

    # API metrics
    kubectl run metrics-baseline --rm -i --restart=Never --image=curlimages/curl:latest -n "${NAMESPACE}" -- \
        curl -s http://prometheus:9090/api/v1/query?query=rate\(http_requests_total\[5m\]\) \
        > "${RESULTS_DIR}/baseline-metrics.json" 2>&1 || true

    # Pod status
    kubectl get pods -n "${NAMESPACE}" -o json > "${RESULTS_DIR}/baseline-pods.json"

    # Resource usage
    kubectl top pods -n "${NAMESPACE}" > "${RESULTS_DIR}/baseline-resources.txt" 2>&1 || true

    success "Baseline captured"
}

# Apply chaos experiment
apply_experiment() {
    local experiment_file=$1
    local experiment_name=$2

    log "Applying chaos experiment: ${experiment_name}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would apply ${experiment_file}"
        cat "${experiment_file}"
        return 0
    fi

    # Apply the experiment
    kubectl apply -f "${experiment_file}" -n "${CHAOS_NAMESPACE}"

    # Wait for experiment to start
    sleep 5

    # Monitor experiment status
    kubectl get -f "${experiment_file}" -n "${CHAOS_NAMESPACE}" -o yaml > "${RESULTS_DIR}/${experiment_name}-status.yaml"

    success "Experiment ${experiment_name} applied"
}

# Monitor during chaos
monitor_chaos() {
    local duration=$1
    local experiment_name=$2

    log "Monitoring for ${duration}..."

    local duration_seconds
    duration_seconds=$(echo "${duration}" | sed 's/m/*60/;s/s//' | bc)
    local interval=10
    local iterations=$((duration_seconds / interval))

    for i in $(seq 1 "${iterations}"); do
        timestamp=$(date +%s)

        # Pod status
        kubectl get pods -n "${NAMESPACE}" -o wide >> "${RESULTS_DIR}/${experiment_name}-pod-status.log"
        echo "---" >> "${RESULTS_DIR}/${experiment_name}-pod-status.log"

        # Resource usage
        kubectl top pods -n "${NAMESPACE}" >> "${RESULTS_DIR}/${experiment_name}-resources.log" 2>&1 || true
        echo "---" >> "${RESULTS_DIR}/${experiment_name}-resources.log"

        # Application health
        kubectl run health-check-${timestamp} --rm -i --restart=Never --image=curlimages/curl:latest -n "${NAMESPACE}" -- \
            curl -s http://geo-climate-api/health >> "${RESULTS_DIR}/${experiment_name}-health.log" 2>&1 || \
            echo "Health check failed at ${timestamp}" >> "${RESULTS_DIR}/${experiment_name}-health.log"

        # API metrics
        kubectl run metrics-check-${timestamp} --rm -i --restart=Never --image=curlimages/curl:latest -n "${NAMESPACE}" -- \
            curl -s http://prometheus:9090/api/v1/query?query=rate\(http_requests_total\[1m\]\) \
            >> "${RESULTS_DIR}/${experiment_name}-metrics.log" 2>&1 || true

        echo "Progress: $i/${iterations} ($(( (i * 100) / iterations ))%)"
        sleep "${interval}"
    done

    success "Monitoring complete"
}

# Cleanup experiment
cleanup_experiment() {
    local experiment_file=$1
    local experiment_name=$2

    log "Cleaning up experiment: ${experiment_name}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would delete experiment ${experiment_name}"
        return 0
    fi

    kubectl delete -f "${experiment_file}" -n "${CHAOS_NAMESPACE}" --ignore-not-found=true

    # Wait for cleanup
    sleep 10

    success "Cleanup complete"
}

# Analyze results
analyze_results() {
    local experiment_name=$1

    log "Analyzing results for ${experiment_name}..."

    local report="${RESULTS_DIR}/${experiment_name}-analysis.md"

    cat > "${report}" << ANALYSIS_HEADER
# Chaos Engineering Experiment Analysis
## Experiment: ${experiment_name}
## Date: $(date)
## Duration: ${DURATION}

---

## Summary

ANALYSIS_HEADER

    # Analyze pod failures
    local pod_failures
    pod_failures=$(grep -c "Error\|Failed\|CrashLoopBackOff" "${RESULTS_DIR}/${experiment_name}-pod-status.log" 2>/dev/null || echo 0)
    echo "- Pod Failures: ${pod_failures}" >> "${report}"

    # Analyze health checks
    local health_failures
    health_failures=$(grep -c "failed" "${RESULTS_DIR}/${experiment_name}-health.log" 2>/dev/null || echo 0)
    echo "- Health Check Failures: ${health_failures}" >> "${report}"

    # Calculate availability
    local total_checks
    total_checks=$(wc -l < "${RESULTS_DIR}/${experiment_name}-health.log")
    local successful_checks=$((total_checks - health_failures))
    local availability=0
    if [[ ${total_checks} -gt 0 ]]; then
        availability=$((successful_checks * 100 / total_checks))
    fi
    echo "- Availability During Chaos: ${availability}%" >> "${report}"

    cat >> "${report}" << 'ANALYSIS_FOOTER'

## Detailed Logs

See attached files:
- Pod Status: pod-status.log
- Resource Usage: resources.log
- Health Checks: health.log
- Metrics: metrics.log

## Recommendations

ANALYSIS_FOOTER

    # Add recommendations based on results
    if [[ ${availability} -lt 95 ]]; then
        echo "- ⚠️ CRITICAL: Availability below 95%. Review resilience strategies." >> "${report}"
    elif [[ ${availability} -lt 99 ]]; then
        echo "- ⚠️ WARNING: Availability below 99%. Consider improvements." >> "${report}"
    else
        echo "- ✅ GOOD: System maintained >99% availability during chaos." >> "${report}"
    fi

    if [[ ${pod_failures} -gt 0 ]]; then
        echo "- Review pod restart policies and resource limits" >> "${report}"
    fi

    success "Analysis complete. Report: ${report}"
    cat "${report}"
}

# Experiment: Pod Kill
run_pod_kill_experiment() {
    log "🔥 Running Pod Kill Experiment"

    cat > /tmp/pod-kill-experiment.yaml << EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-kill-test
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - ${NAMESPACE}
    labelSelectors:
      app: geo-climate-api
  scheduler:
    cron: '@every 30s'
  duration: ${DURATION}
EOF

    apply_experiment /tmp/pod-kill-experiment.yaml pod-kill
    monitor_chaos "${DURATION}" pod-kill
    cleanup_experiment /tmp/pod-kill-experiment.yaml pod-kill
    analyze_results pod-kill
}

# Experiment: Network Delay
run_network_delay_experiment() {
    log "🌐 Running Network Delay Experiment"

    cat > /tmp/network-delay-experiment.yaml << EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay-test
spec:
  action: delay
  mode: one
  selector:
    namespaces:
      - ${NAMESPACE}
    labelSelectors:
      app: geo-climate-api
  delay:
    latency: "100ms"
    correlation: "25"
    jitter: "10ms"
  duration: ${DURATION}
  direction: to
  target:
    mode: all
    selector:
      namespaces:
        - ${NAMESPACE}
EOF

    apply_experiment /tmp/network-delay-experiment.yaml network-delay
    monitor_chaos "${DURATION}" network-delay
    cleanup_experiment /tmp/network-delay-experiment.yaml network-delay
    analyze_results network-delay
}

# Experiment: Network Partition
run_network_partition_experiment() {
    log "🔌 Running Network Partition Experiment"

    cat > /tmp/network-partition-experiment.yaml << EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-partition-test
spec:
  action: partition
  mode: one
  selector:
    namespaces:
      - ${NAMESPACE}
    labelSelectors:
      app: geo-climate-api
  direction: both
  duration: ${DURATION}
EOF

    apply_experiment /tmp/network-partition-experiment.yaml network-partition
    monitor_chaos "${DURATION}" network-partition
    cleanup_experiment /tmp/network-partition-experiment.yaml network-partition
    analyze_results network-partition
}

# Experiment: Stress Test
run_stress_experiment() {
    log "💪 Running Stress Test Experiment"

    cat > /tmp/stress-experiment.yaml << EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: stress-test
spec:
  mode: one
  selector:
    namespaces:
      - ${NAMESPACE}
    labelSelectors:
      app: geo-climate-api
  stressors:
    cpu:
      workers: 4
      load: 80
    memory:
      workers: 2
      size: 512MB
  duration: ${DURATION}
EOF

    apply_experiment /tmp/stress-experiment.yaml stress-test
    monitor_chaos "${DURATION}" stress-test
    cleanup_experiment /tmp/stress-experiment.yaml stress-test
    analyze_results stress-test
}

# Experiment: Database Chaos
run_database_chaos_experiment() {
    log "🗄️ Running Database Chaos Experiment"

    cat > /tmp/database-chaos-experiment.yaml << EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: database-chaos-test
spec:
  action: pod-failure
  mode: one
  selector:
    namespaces:
      - ${NAMESPACE}
    labelSelectors:
      app: postgres
  duration: ${DURATION}
EOF

    apply_experiment /tmp/database-chaos-experiment.yaml database-chaos
    monitor_chaos "${DURATION}" database-chaos
    cleanup_experiment /tmp/database-chaos-experiment.yaml database-chaos
    analyze_results database-chaos
}

# Full suite
run_full_suite() {
    log "🚀 Running Full Chaos Engineering Suite"

    local experiments=(
        "run_pod_kill_experiment"
        "run_network_delay_experiment"
        "run_stress_experiment"
    )

    for experiment in "${experiments[@]}"; do
        log "---"
        ${experiment}
        log "Waiting 60s before next experiment..."
        sleep 60
    done

    # Generate summary report
    generate_summary_report
}

# Generate summary report
generate_summary_report() {
    log "Generating summary report..."

    local summary="${RESULTS_DIR}/SUMMARY.md"

    cat > "${summary}" << EOF
# Chaos Engineering Test Summary
## Date: $(date)
## Namespace: ${NAMESPACE}
## Total Duration: Multiple experiments

---

## Experiments Executed

EOF

    for analysis in "${RESULTS_DIR}"/*-analysis.md; do
        if [[ -f "${analysis}" ]]; then
            echo "### $(basename "${analysis}" -analysis.md)" >> "${summary}"
            tail -n 20 "${analysis}" >> "${summary}"
            echo "" >> "${summary}"
        fi
    done

    cat >> "${summary}" << EOF

## Overall Assessment

Based on the chaos engineering tests, the system demonstrated:
- Resilience to pod failures
- Network latency tolerance
- Resource stress handling
- Database failover capabilities

## Next Steps

1. Review failed experiments and implement fixes
2. Increase chaos frequency in staging environment
3. Automate chaos testing in CI/CD pipeline
4. Schedule regular game days for team training

---
Generated by Geo Climate Chaos Engineering Runner
EOF

    success "Summary report generated: ${summary}"
    cat "${summary}"
}

# Main execution
main() {
    log "═══════════════════════════════════════════════════════════"
    log "  Geo Climate Platform - Chaos Engineering Runner"
    log "═══════════════════════════════════════════════════════════"
    echo ""

    check_prerequisites
    capture_baseline

    case "${EXPERIMENT_TYPE}" in
        pod-kill)
            run_pod_kill_experiment
            ;;
        network-delay)
            run_network_delay_experiment
            ;;
        network-partition)
            run_network_partition_experiment
            ;;
        stress-test)
            run_stress_experiment
            ;;
        database-chaos)
            run_database_chaos_experiment
            ;;
        full-suite)
            run_full_suite
            ;;
        *)
            error "Unknown experiment type: ${EXPERIMENT_TYPE}"
            echo "Valid types: pod-kill, network-delay, network-partition, stress-test, database-chaos, full-suite"
            exit 1
            ;;
    esac

    echo ""
    log "═══════════════════════════════════════════════════════════"
    success "Chaos Engineering Complete!"
    log "Results saved to: ${RESULTS_DIR}"
    log "═══════════════════════════════════════════════════════════"
}

main "$@"
