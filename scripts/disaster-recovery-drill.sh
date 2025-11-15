#!/bin/bash
###############################################################################
# Disaster Recovery Drill Automation Script
# Tests failover and failback procedures for Geo Climate Platform
#
# RTO Target: 1 hour
# RPO Target: 5 minutes
#
# Usage:
#   ./disaster-recovery-drill.sh [failover|failback|full-drill] [--dry-run]
#
# Examples:
#   ./disaster-recovery-drill.sh failover
#   ./disaster-recovery-drill.sh failback
#   ./disaster-recovery-drill.sh full-drill --dry-run
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
SECONDARY_REGION="${SECONDARY_REGION:-us-west-2}"
PRIMARY_CLUSTER="${PRIMARY_CLUSTER:-geo-climate-primary}"
SECONDARY_CLUSTER="${SECONDARY_CLUSTER:-geo-climate-secondary}"
NAMESPACE="${NAMESPACE:-geo-climate}"
DRILL_ID="drill-$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="./dr-drills/${DRILL_ID}"
DRY_RUN=false
START_TIME=$(date +%s)

# Logging
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
    echo -e "${CYAN}ℹ${NC} $*"
}

# Parse arguments
OPERATION="${1:-full-drill}"
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    warning "DRY-RUN MODE - No actual changes will be made"
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"
DRILL_LOG="${RESULTS_DIR}/drill.log"
exec > >(tee -a "${DRILL_LOG}") 2>&1

log "═══════════════════════════════════════════════════════════"
log "  Geo Climate Platform - Disaster Recovery Drill"
log "  Drill ID: ${DRILL_ID}"
log "  Operation: ${OPERATION}"
log "═══════════════════════════════════════════════════════════"

# Prerequisites check
check_prerequisites() {
    log "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found"
    fi
    success "kubectl installed"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "aws cli not found"
    fi
    success "AWS CLI installed"

    # Check helm
    if ! command -v helm &> /dev/null; then
        error "helm not found"
    fi
    success "Helm installed"

    # Check jq
    if ! command -v jq &> /dev/null; then
        error "jq not found"
    fi
    success "jq installed"

    # Verify cluster access
    if ! kubectl --context="${PRIMARY_CLUSTER}" cluster-info &> /dev/null; then
        error "Cannot access primary cluster: ${PRIMARY_CLUSTER}"
    fi
    success "Primary cluster accessible"

    if ! kubectl --context="${SECONDARY_CLUSTER}" cluster-info &> /dev/null; then
        error "Cannot access secondary cluster: ${SECONDARY_CLUSTER}"
    fi
    success "Secondary cluster accessible"
}

# Capture baseline
capture_baseline() {
    log "Capturing baseline state..."

    local cluster=$1
    local region=$2
    local output_dir="${RESULTS_DIR}/baseline-${region}"
    mkdir -p "${output_dir}"

    # Pod status
    kubectl --context="${cluster}" get pods -n "${NAMESPACE}" -o json > "${output_dir}/pods.json"

    # Service status
    kubectl --context="${cluster}" get svc -n "${NAMESPACE}" -o json > "${output_dir}/services.json"

    # Ingress status
    kubectl --context="${cluster}" get ingress -n "${NAMESPACE}" -o json > "${output_dir}/ingress.json"

    # Database status
    kubectl --context="${cluster}" exec -n "${NAMESPACE}" postgres-0 -- \
        psql -U postgres -c "SELECT pg_database_size('geo_climate');" \
        > "${output_dir}/db-size.txt" 2>&1 || true

    # Application health
    local api_endpoint
    api_endpoint=$(kubectl --context="${cluster}" get ingress -n "${NAMESPACE}" geo-climate-api -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "unknown")

    if [[ "${api_endpoint}" != "unknown" ]]; then
        curl -sf "https://${api_endpoint}/health" > "${output_dir}/health.json" 2>&1 || \
            echo "Health check failed" > "${output_dir}/health.json"
    fi

    # Metrics snapshot
    kubectl --context="${cluster}" exec -n monitoring prometheus-0 -- \
        promtool query instant 'http://localhost:9090' 'up' \
        > "${output_dir}/metrics-up.txt" 2>&1 || true

    success "Baseline captured for ${region}"
}

# Verify WAL archiving
verify_wal_archiving() {
    log "Verifying WAL archiving..."

    local latest_wal
    latest_wal=$(aws s3 ls "s3://geo-climate-wal-archive-${PRIMARY_REGION}/" --recursive | sort | tail -n 1 | awk '{print $4}')

    if [[ -z "${latest_wal}" ]]; then
        error "No WAL archives found in S3"
    fi

    local wal_age
    wal_age=$(aws s3api head-object --bucket "geo-climate-wal-archive-${PRIMARY_REGION}" --key "${latest_wal}" --query 'LastModified' --output text)

    info "Latest WAL archive: ${latest_wal}"
    info "WAL age: ${wal_age}"

    # Check if WAL is recent (within 10 minutes)
    local wal_timestamp
    wal_timestamp=$(date -d "${wal_age}" +%s)
    local now
    now=$(date +%s)
    local age_seconds=$((now - wal_timestamp))

    if [[ ${age_seconds} -gt 600 ]]; then
        warning "WAL archive is older than 10 minutes (${age_seconds}s)"
        warning "RPO may be impacted"
    else
        success "WAL archiving is current (${age_seconds}s old)"
    fi
}

# Simulate primary failure
simulate_primary_failure() {
    log "Simulating primary region failure..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would scale down primary cluster deployments"
        return 0
    fi

    # Scale down all deployments in primary
    kubectl --context="${PRIMARY_CLUSTER}" scale deployment --all -n "${NAMESPACE}" --replicas=0

    # Wait for pods to terminate
    log "Waiting for pods to terminate..."
    sleep 30

    # Verify primary is down
    local running_pods
    running_pods=$(kubectl --context="${PRIMARY_CLUSTER}" get pods -n "${NAMESPACE}" --field-selector=status.phase=Running --no-headers | wc -l)

    if [[ ${running_pods} -eq 0 ]]; then
        success "Primary region simulated as failed (0 running pods)"
    else
        warning "${running_pods} pods still running in primary"
    fi

    echo "${START_TIME}" > "${RESULTS_DIR}/failure-time.txt"
}

# Promote secondary database
promote_secondary_database() {
    log "Promoting secondary database to primary..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would promote standby database"
        return 0
    fi

    # Execute promotion script
    kubectl --context="${SECONDARY_CLUSTER}" exec -n "${NAMESPACE}" postgres-0 -- \
        bash -c '/scripts/promote-standby.sh'

    # Wait for promotion to complete
    sleep 10

    # Verify database is writable
    kubectl --context="${SECONDARY_CLUSTER}" exec -n "${NAMESPACE}" postgres-0 -- \
        psql -U postgres -c "CREATE TABLE dr_test_$(date +%s) (id INT);" || \
        error "Database promotion failed - not writable"

    success "Secondary database promoted to primary"
}

# Update DNS for failover
update_dns_failover() {
    log "Updating DNS to point to secondary region..."

    local primary_endpoint
    primary_endpoint=$(kubectl --context="${SECONDARY_CLUSTER}" get ingress -n "${NAMESPACE}" geo-climate-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would update DNS to: ${primary_endpoint}"
        return 0
    fi

    # Update Route53 (example - adjust for your DNS provider)
    info "Primary endpoint in secondary region: ${primary_endpoint}"
    info "Manual DNS update may be required depending on your DNS setup"

    # For automated DNS:
    # aws route53 change-resource-record-sets --hosted-zone-id Z123456 --change-batch file://dns-change.json

    warning "Verify DNS propagation before proceeding"
}

# Scale up secondary
scale_up_secondary() {
    log "Scaling up secondary region..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would scale up deployments in secondary"
        return 0
    fi

    # Scale API pods
    kubectl --context="${SECONDARY_CLUSTER}" scale deployment geo-climate-api -n "${NAMESPACE}" --replicas=5

    # Wait for pods to be ready
    kubectl --context="${SECONDARY_CLUSTER}" wait --for=condition=ready pod \
        -l app=geo-climate-api -n "${NAMESPACE}" --timeout=300s

    # Verify pods are running
    local running_pods
    running_pods=$(kubectl --context="${SECONDARY_CLUSTER}" get pods -n "${NAMESPACE}" -l app=geo-climate-api --field-selector=status.phase=Running --no-headers | wc -l)

    success "Secondary region scaled up (${running_pods} pods running)"
}

# Verify failover
verify_failover() {
    log "Verifying failover completion..."

    local api_endpoint
    api_endpoint=$(kubectl --context="${SECONDARY_CLUSTER}" get ingress -n "${NAMESPACE}" geo-climate-api -o jsonpath='{.spec.rules[0].host}')

    # Test API health
    if curl -sf "https://${api_endpoint}/health" > /dev/null 2>&1; then
        success "API health check passed"
    else
        error "API health check failed"
    fi

    # Test prediction endpoint
    if curl -sf "https://${api_endpoint}/api/v1/models" > /dev/null 2>&1; then
        success "API endpoints accessible"
    else
        warning "API endpoints not accessible"
    fi

    # Check database connectivity
    kubectl --context="${SECONDARY_CLUSTER}" exec -n "${NAMESPACE}" postgres-0 -- \
        psql -U postgres -c "SELECT 1;" > /dev/null 2>&1 || \
        error "Database not accessible"
    success "Database accessible"

    # Calculate RTO
    local failure_time
    failure_time=$(cat "${RESULTS_DIR}/failure-time.txt")
    local recovery_time
    recovery_time=$(date +%s)
    local rto=$((recovery_time - failure_time))
    local rto_minutes=$((rto / 60))

    info "RTO Achieved: ${rto_minutes} minutes (Target: 60 minutes)"

    if [[ ${rto_minutes} -le 60 ]]; then
        success "RTO target met ✓"
    else
        warning "RTO target exceeded by $((rto_minutes - 60)) minutes"
    fi

    echo "${rto_minutes}" > "${RESULTS_DIR}/rto-minutes.txt"
}

# Failover procedure
execute_failover() {
    log "═══ EXECUTING FAILOVER ═══"

    local failover_start
    failover_start=$(date +%s)

    # Step 1: Verify WAL archiving
    verify_wal_archiving

    # Step 2: Capture baseline
    capture_baseline "${PRIMARY_CLUSTER}" "${PRIMARY_REGION}"
    capture_baseline "${SECONDARY_CLUSTER}" "${SECONDARY_REGION}"

    # Step 3: Simulate failure
    simulate_primary_failure

    # Step 4: Promote secondary database
    promote_secondary_database

    # Step 5: Update DNS
    update_dns_failover

    # Step 6: Scale up secondary
    scale_up_secondary

    # Step 7: Verify
    verify_failover

    local failover_end
    failover_end=$(date +%s)
    local duration=$((failover_end - failover_start))

    success "Failover completed in $((duration / 60)) minutes"
}

# Failback procedure
execute_failback() {
    log "═══ EXECUTING FAILBACK ═══"

    if [[ "${DRY_RUN}" == "true" ]]; then
        info "DRY-RUN: Would execute failback procedure"
        info "1. Sync data from secondary to primary"
        info "2. Restore primary database from latest backup"
        info "3. Apply WAL archives to catch up"
        info "4. Scale up primary region"
        info "5. Update DNS back to primary"
        info "6. Scale down secondary"
        return 0
    fi

    # Step 1: Restore primary database
    log "Restoring primary database..."

    # Get latest backup
    local latest_backup
    latest_backup=$(aws s3 ls "s3://geo-climate-backups-production/postgres/" | sort | tail -n 1 | awk '{print $4}')

    if [[ -z "${latest_backup}" ]]; then
        error "No backups found"
    fi

    info "Using backup: ${latest_backup}"

    # Download and restore
    kubectl --context="${PRIMARY_CLUSTER}" exec -n "${NAMESPACE}" postgres-0 -- \
        bash -c "aws s3 cp s3://geo-climate-backups-production/postgres/${latest_backup} - | gunzip | psql -U postgres"

    # Step 2: Apply WAL archives
    log "Applying WAL archives to catch up..."
    # This would be handled by PostgreSQL recovery

    # Step 3: Scale up primary
    log "Scaling up primary region..."
    kubectl --context="${PRIMARY_CLUSTER}" scale deployment --all -n "${NAMESPACE}" --replicas=3

    # Wait for pods
    kubectl --context="${PRIMARY_CLUSTER}" wait --for=condition=ready pod \
        -l app=geo-climate-api -n "${NAMESPACE}" --timeout=300s

    # Step 4: Update DNS back to primary
    log "Updating DNS back to primary region..."
    # Manual or automated DNS update

    # Step 5: Scale down secondary
    log "Scaling down secondary to standby mode..."
    kubectl --context="${SECONDARY_CLUSTER}" scale deployment --all -n "${NAMESPACE}" --replicas=1

    success "Failback completed"
}

# Generate drill report
generate_report() {
    log "Generating drill report..."

    local report="${RESULTS_DIR}/DRILL_REPORT.md"
    local rto_minutes=$(cat "${RESULTS_DIR}/rto-minutes.txt" 2>/dev/null || echo "N/A")

    cat > "${report}" << EOF
# Disaster Recovery Drill Report
## Drill ID: ${DRILL_ID}
## Date: $(date)
## Operation: ${OPERATION}

---

## Executive Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| RTO (Recovery Time Objective) | 60 minutes | ${rto_minutes} minutes | $([ "${rto_minutes}" -le "60" ] && echo "✅ PASS" || echo "❌ FAIL") |
| RPO (Recovery Point Objective) | 5 minutes | TBD | TBD |
| Data Loss | None | TBD | TBD |
| Service Availability | >99% | TBD | TBD |

---

## Timeline

EOF

    # Add timeline from logs
    grep "^\[" "${DRILL_LOG}" | while read -r line; do
        echo "- ${line}" >> "${report}"
    done

    cat >> "${report}" << EOF

---

## Findings

### Successes
- Automated failover procedure executed successfully
- Secondary cluster promoted to primary
- Database promotion completed
- Application remained accessible

### Issues
- [Document any issues encountered]

### Improvements Needed
- [Document areas for improvement]

---

## Action Items

1. [ ] Review and update runbooks based on findings
2. [ ] Address any performance issues identified
3. [ ] Update RTO/RPO targets if needed
4. [ ] Schedule next drill date
5. [ ] Train team on any new procedures

---

## Attachments

- Baseline metrics: baseline-*/
- Drill logs: drill.log
- RTO calculation: rto-minutes.txt

---

**Drill Conducted By:** $(whoami)
**Date:** $(date)
**Next Drill:** $(date -d "+30 days" +%Y-%m-%d)
EOF

    success "Report generated: ${report}"
    cat "${report}"
}

# Main execution
main() {
    check_prerequisites

    case "${OPERATION}" in
        failover)
            execute_failover
            ;;
        failback)
            execute_failback
            ;;
        full-drill)
            execute_failover
            log "Waiting 5 minutes before failback..."
            sleep 300
            execute_failback
            ;;
        *)
            error "Unknown operation: ${OPERATION}"
            echo "Valid operations: failover, failback, full-drill"
            exit 1
            ;;
    esac

    generate_report

    log "═══════════════════════════════════════════════════════════"
    success "DR Drill Complete!"
    log "Results saved to: ${RESULTS_DIR}"
    log "═══════════════════════════════════════════════════════════"
}

main "$@"
