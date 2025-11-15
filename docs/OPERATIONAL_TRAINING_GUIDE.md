# 🎓 Operational Training Guide
## Geo Climate Platform - Production Operations

**Target Audience:** DevOps Engineers, SREs, On-Call Engineers
**Duration:** 8-16 hours (self-paced)
**Prerequisites:** Kubernetes basics, monitoring fundamentals
**Version:** 1.0.0

---

## 📋 Table of Contents

1. [Training Overview](#training-overview)
2. [Module 1: Platform Architecture](#module-1-platform-architecture)
3. [Module 2: Monitoring & Observability](#module-2-monitoring--observability)
4. [Module 3: Incident Response](#module-3-incident-response)
5. [Module 4: Runbook Execution](#module-4-runbook-execution)
6. [Module 5: Disaster Recovery](#module-5-disaster-recovery)
7. [Module 6: Chaos Engineering](#module-6-chaos-engineering)
8. [Module 7: Performance Optimization](#module-7-performance-optimization)
9. [Hands-On Labs](#hands-on-labs)
10. [Certification](#certification)

---

## Training Overview

### Learning Objectives

By the end of this training, you will be able to:

✅ Understand the complete platform architecture
✅ Navigate Grafana dashboards and Prometheus metrics
✅ Respond to production incidents following runbooks
✅ Execute disaster recovery procedures
✅ Perform chaos engineering experiments
✅ Optimize system performance based on metrics
✅ Participate effectively in on-call rotation

### Training Structure

- **Theory**: 30% - Understand concepts and architecture
- **Hands-On**: 50% - Practice real scenarios
- **Assessment**: 20% - Validate knowledge

---

## Module 1: Platform Architecture

### Duration: 2 hours

### 1.1 System Components

#### **API Layer**
```
┌─────────────────────────────────────────┐
│          Load Balancer (NGINX)          │
│         SSL/TLS Termination             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      API Pods (3+ replicas)             │
│  - FastAPI Application                  │
│  - OAuth2/JWT Authentication            │
│  - Rate Limiting (Redis)                │
│  - Circuit Breakers                     │
└────────────────┬────────────────────────┘
```

**Key Files:**
- `k8s/production/04-api-deployment.yaml`
- `source/api/main.py`
- `source/api/auth.py`

**Practice:**
```bash
# Scale API pods
kubectl scale deployment geo-climate-api -n geo-climate --replicas=5

# View pod logs
kubectl logs -f deployment/geo-climate-api -n geo-climate

# Check pod health
kubectl get pods -n geo-climate -l app=geo-climate-api -o wide
```

#### **Data Layer**
```
┌─────────────────────────────────────────┐
│     PostgreSQL StatefulSet              │
│  - Primary + Standby replicas           │
│  - WAL archiving to S3                  │
│  - Point-in-time recovery               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│     Redis Cluster                       │
│  - Rate limiting data                   │
│  - Session storage                      │
│  - Cache layer                          │
└─────────────────────────────────────────┘
```

**Key Files:**
- `k8s/production/08-postgres.yaml`
- `k8s/production/09-redis.yaml`
- `k8s/dr/cross-region-replication.yaml`

**Practice:**
```bash
# Check database status
kubectl exec -it postgres-0 -n geo-climate -- psql -U postgres -c "\l"

# Check Redis cluster info
kubectl exec -it redis-0 -n geo-climate -- redis-cli cluster info

# View database connections
kubectl exec -it postgres-0 -n geo-climate -- psql -U postgres -c "SELECT * FROM pg_stat_activity;"
```

#### **Observability Stack**
```
┌─────────────────────────────────────────┐
│          Prometheus                     │
│  - Metrics collection (15s interval)    │
│  - 30-day retention                     │
│  - Alert evaluation                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          Grafana                        │
│  - Visualization dashboards             │
│  - Alert management                     │
│  - Data source integration              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          Loki Stack                     │
│  - Log aggregation                      │
│  - 90-day retention                     │
│  - LogQL queries                        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          Jaeger                         │
│  - Distributed tracing                  │
│  - Request flow visualization           │
│  - Performance analysis                 │
└─────────────────────────────────────────┘
```

**Key Files:**
- `monitoring/prometheus/prometheus.yml`
- `k8s/monitoring/grafana-production.yaml`
- `k8s/logging/loki-stack-production.yaml`
- `k8s/tracing/jaeger-distributed-tracing.yaml`

**Practice:**
```bash
# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Visit: http://localhost:9090

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80
# Visit: http://localhost:3000

# Access Jaeger
kubectl port-forward -n observability svc/jaeger-query 16686:16686
# Visit: http://localhost:16686
```

### 1.2 Data Flow

**Prediction Request Flow:**
```
1. Client → NGINX Ingress (SSL/TLS)
2. NGINX → API Pod (Load balanced)
3. API → Rate Limiter (Redis check)
4. API → Authentication (JWT validation)
5. API → Model Service (Prediction)
6. Model → Database (Feature data if needed)
7. Model → Response
8. API → Client (JSON response)

Monitoring:
- Prometheus: Metrics at each step
- Loki: Logs from all components
- Jaeger: Distributed trace
```

### 1.3 Architecture Quiz

**Question 1:** How many API pod replicas should run in production minimum?
**Answer:** 3 (for high availability across failure domains)

**Question 2:** What happens if Redis goes down?
**Answer:** Rate limiting fails open (allows requests), but system continues operating. Cache misses increase load on database.

**Question 3:** Where are application logs stored?
**Answer:** Loki (S3/GCS backend), 90-day retention

---

## Module 2: Monitoring & Observability

### Duration: 3 hours

### 2.1 Grafana Dashboards

#### **System Overview Dashboard**

**Location:** Grafana → Dashboards → Geo Climate → System Overview

**Key Panels:**
- Request Rate (RPS)
- Response Time (P50, P95, P99)
- Error Rate (5xx, 4xx)
- Active Connections
- Pod Status

**What to look for:**
- ✅ RPS: Steady during business hours, drops at night
- ✅ P95 < 100ms
- ✅ Error rate < 0.1%
- ⚠️ Sudden RPS spike = potential attack or viral event
- ⚠️ P95 > 500ms = performance degradation
- 🚨 Error rate > 1% = incident

**Practice Exercise:**
```
1. Open System Overview dashboard
2. Set time range to "Last 24 hours"
3. Identify the peak traffic time
4. Check if any alerts fired during peak
5. Document your findings
```

#### **API Performance Dashboard**

**Location:** Grafana → Dashboards → Application → API Performance

**Key Panels:**
- Endpoint latency breakdown
- Top 10 slowest endpoints
- Rate limiting metrics
- Connection pool saturation

**Investigation Workflow:**
```bash
# If you see high latency:
1. Check "Top 10 Slowest Endpoints" panel
2. Identify the problematic endpoint
3. Check Jaeger traces for that endpoint
4. Look at database query times
5. Check for N+1 queries or missing indexes

# If you see rate limiting:
1. Check which users are throttled
2. Verify if it's legitimate traffic
3. Consider scaling up limits for valid users
4. Investigate if it's an attack pattern
```

#### **ML Model Performance Dashboard**

**Location:** Grafana → Dashboards → ML Models → Model Performance

**Key Metrics:**
- Prediction latency (P50, P95, P99)
- Model error rate
- Prediction confidence distribution
- Feature drift detection
- Cache hit rate

**Model Health Indicators:**
- ✅ P95 latency < 50ms
- ✅ Error rate < 1%
- ✅ Avg confidence > 85%
- ✅ Cache hit rate > 90%
- ⚠️ Confidence dropping = potential model drift
- ⚠️ Error rate increasing = data quality issues
- 🚨 Latency spike = model loading issues

### 2.2 Prometheus Queries

**Essential Queries:**

```promql
# API Request Rate
sum(rate(http_requests_total{job="geo-climate-api"}[5m]))

# P95 Latency
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)

# Error Rate %
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# Pod CPU Usage
sum(rate(container_cpu_usage_seconds_total{pod=~"geo-climate-api.*"}[5m]))
by (pod)

# Database Connection Pool Utilization
database_connections_active / database_connections_max * 100

# Model Prediction Success Rate
sum(rate(model_predictions_total{status="success"}[5m])) /
sum(rate(model_predictions_total[5m])) * 100
```

**Practice Exercise:**
```
1. Open Prometheus UI
2. Execute each query above
3. Understand what each query measures
4. Create a custom query for:
   - Number of active users
   - API quota usage by tier
   - Redis memory usage
```

### 2.3 Log Analysis with Loki

**LogQL Queries:**

```logql
# All errors in last hour
{namespace="geo-climate"} |= "ERROR" | json

# Failed predictions
{app="geo-climate-api"} | json | prediction_status="error"

# Slow queries (>1s)
{app="geo-climate-api"} | json | duration > 1000

# Rate limiting events
{app="geo-climate-api"} |= "rate limit exceeded"

# Database connection errors
{app="postgres"} |= "connection refused"

# Authentication failures
{app="geo-climate-api"} | json | event="auth_failed"
```

**Investigation Workflow:**
```
1. Start with broad query (all errors)
2. Filter by time range (incident window)
3. Add specific filters (pod, endpoint, user)
4. Correlate with metrics (Prometheus)
5. Find distributed trace (Jaeger)
6. Identify root cause
```

### 2.4 Distributed Tracing

**Using Jaeger:**

```
1. Open Jaeger UI
2. Select service: geo-climate-api
3. Filter by operation (endpoint)
4. Set time range
5. Click "Find Traces"
6. Select slow trace (>100ms)
7. Analyze span breakdown:
   - Authentication: should be <5ms
   - Database query: should be <20ms
   - Model inference: should be <30ms
   - Total: should be <100ms
```

**Common Issues:**
- Long authentication span = JWT validation slow
- Long database span = missing index or N+1 queries
- Long model span = model not cached, loading from disk
- Long external API span = network issues or 3rd party slowness

---

## Module 3: Incident Response

### Duration: 2 hours

### 3.1 Incident Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| **P1 (Critical)** | Complete service outage | 5 minutes | API down, database unavailable |
| **P2 (High)** | Major feature broken | 30 minutes | Predictions failing, auth broken |
| **P3 (Medium)** | Minor feature broken | 2 hours | Specific endpoint slow |
| **P4 (Low)** | Cosmetic issues | Next business day | Dashboard formatting |

### 3.2 Incident Response Workflow

```
┌─────────────────────────────────────────┐
│  1. DETECT                              │
│  - Alert fires                          │
│  - User report                          │
│  - Monitoring shows issue               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. ASSESS                              │
│  - Determine severity (P1-P4)           │
│  - Check impact (users, revenue)        │
│  - Review dashboards                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. RESPOND                             │
│  - Page on-call engineer (P1/P2)        │
│  - Create incident in PagerDuty         │
│  - Start war room (Slack/Zoom)          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. INVESTIGATE                         │
│  - Follow runbook                       │
│  - Check logs (Loki)                    │
│  - Check metrics (Grafana)              │
│  - Check traces (Jaeger)                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  5. MITIGATE                            │
│  - Apply fix or workaround              │
│  - Test fix in staging first            │
│  - Deploy to production                 │
│  - Verify resolution                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  6. COMMUNICATE                         │
│  - Update status page                   │
│  - Notify stakeholders                  │
│  - Document actions taken               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  7. POST-MORTEM                         │
│  - Root cause analysis                  │
│  - Action items                         │
│  - Process improvements                 │
└─────────────────────────────────────────┘
```

### 3.3 Common Incidents & Runbooks

#### **Incident: API High Error Rate**

**Symptoms:**
- Alert: "High Error Rate" (Grafana/PagerDuty)
- Dashboard shows >1% 5xx errors
- User reports of failures

**Runbook:**
```bash
# 1. Check current error rate
kubectl logs -n geo-climate deployment/geo-climate-api --tail=100 | grep ERROR

# 2. Identify failing endpoint
# Open Grafana → API Performance → Error Breakdown by Endpoint

# 3. Check pod health
kubectl get pods -n geo-climate -l app=geo-climate-api

# 4. Check recent deployments
kubectl rollout history deployment/geo-climate-api -n geo-climate

# 5. If recent deployment caused it, rollback
kubectl rollout undo deployment/geo-climate-api -n geo-climate

# 6. If database issue, check connections
kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# 7. If rate limiting issue, check Redis
kubectl exec -it redis-0 -n geo-climate -- redis-cli INFO

# 8. Scale up if traffic spike
kubectl scale deployment geo-climate-api -n geo-climate --replicas=10

# 9. Monitor recovery
watch kubectl get pods -n geo-climate
```

**Post-Incident:**
- Document root cause
- Create preventive measures
- Update runbook if needed

#### **Incident: Database Connection Pool Exhausted**

**Symptoms:**
- Alert: "Database Connection Pool Exhausted"
- API returning 503 errors
- Slow query times

**Runbook:**
```bash
# 1. Check connection pool status
kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U postgres -c "
    SELECT count(*), state
    FROM pg_stat_activity
    GROUP BY state;
  "

# 2. Identify long-running queries
kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U postgres -c "
    SELECT pid, now() - query_start AS duration, query
    FROM pg_stat_activity
    WHERE state = 'active'
    ORDER BY duration DESC;
  "

# 3. Kill problematic queries (if necessary)
kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U postgres -c "SELECT pg_terminate_backend(<PID>);"

# 4. Scale API pods (increases total connections)
kubectl scale deployment geo-climate-api -n geo-climate --replicas=5

# 5. Restart API pods to reset connections
kubectl rollout restart deployment/geo-climate-api -n geo-climate

# 6. Monitor recovery
watch "kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U postgres -c 'SELECT count(*) FROM pg_stat_activity;'"
```

### 3.4 Hands-On Simulation

**Exercise: Simulated Incident Response**

**Scenario:** You receive a PagerDuty alert at 2 AM: "API High Latency - P95 > 1s"

**Your Task:**
1. Login to Grafana
2. Identify which endpoint is slow
3. Check Jaeger traces
4. Find the root cause
5. Document mitigation steps
6. Write incident report

**Expected Actions:**
- Check API Performance Dashboard
- Run Prometheus queries
- Analyze Loki logs
- Review Jaeger traces
- Scale resources if needed
- Create post-mortem

---

## Module 4: Runbook Execution

### Duration: 2 hours

### Common Operational Tasks

#### **Task: Deploy New Version**

```bash
# 1. Build and push image
docker build -t ghcr.io/dogaaydinn/geo_sentiment_climate:v1.2.3 .
docker push ghcr.io/dogaaydinn/geo_sentiment_climate:v1.2.3

# 2. Update Helm values
vim helm/geo-climate/values-prod.yaml
# Change image.tag to v1.2.3

# 3. Deploy to staging first
helm upgrade geo-climate-staging ./helm/geo-climate \
  --values helm/geo-climate/values-staging.yaml \
  --namespace geo-climate-staging

# 4. Run smoke tests
./scripts/verify-deployment.sh staging

# 5. Deploy to production (blue-green)
helm upgrade geo-climate ./helm/geo-climate \
  --values helm/geo-climate/values-prod.yaml \
  --namespace geo-climate

# 6. Monitor deployment
kubectl rollout status deployment/geo-climate-api -n geo-climate
watch kubectl get pods -n geo-climate

# 7. Verify
./scripts/verify-deployment.sh production

# 8. If issues, rollback
kubectl rollout undo deployment/geo-climate-api -n geo-climate
```

#### **Task: Scale for Traffic Spike**

```bash
# 1. Check current load
kubectl top pods -n geo-climate

# 2. Check current replicas
kubectl get hpa -n geo-climate

# 3. Manual scale (if HPA not fast enough)
kubectl scale deployment geo-climate-api -n geo-climate --replicas=20

# 4. Monitor scaling
watch kubectl get pods -n geo-climate

# 5. Check if scaling helped
# Open Grafana → API Performance

# 6. After traffic normalizes, scale down
kubectl scale deployment geo-climate-api -n geo-climate --replicas=3
```

#### **Task: Rotate Secrets**

```bash
# 1. Generate new secrets
export NEW_JWT_SECRET=$(openssl rand -base64 32)

# 2. Update Kubernetes secret
kubectl create secret generic geo-climate-secrets \
  --from-literal=jwt-secret=${NEW_JWT_SECRET} \
  --namespace geo-climate \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Rolling restart API pods
kubectl rollout restart deployment/geo-climate-api -n geo-climate

# 4. Monitor restart
kubectl rollout status deployment/geo-climate-api -n geo-climate

# 5. Verify new secret is used
kubectl logs deployment/geo-climate-api -n geo-climate | grep "JWT initialized"
```

---

## Module 5: Disaster Recovery

### Duration: 2 hours

### 5.1 DR Strategy

**RTO (Recovery Time Objective):** 1 hour
**RPO (Recovery Point Objective):** 5 minutes

### 5.2 DR Drill Procedure

**See:** `scripts/disaster-recovery-drill.sh` (created below)

**Monthly DR Drill Checklist:**
- [ ] Announce drill to team
- [ ] Capture baseline metrics
- [ ] Simulate primary region failure
- [ ] Execute failover to secondary region
- [ ] Verify all services operational
- [ ] Test data integrity
- [ ] Simulate failback to primary
- [ ] Verify full recovery
- [ ] Document lessons learned
- [ ] Update runbooks

### 5.3 Backup Verification

```bash
# Verify backups exist
aws s3 ls s3://geo-climate-backups-production/postgres/

# Test backup restore (staging environment)
./scripts/restore-backup.sh geo_climate_20240115_120000.sql.gz staging

# Verify data integrity
psql -h staging-db -U postgres -d geo_climate -c "
  SELECT count(*) FROM predictions;
  SELECT max(created_at) FROM predictions;
"
```

---

## Module 6: Chaos Engineering

### Duration: 2 hours

### 6.1 Chaos Principles

**Why Chaos Engineering?**
- Proactively find weaknesses
- Build confidence in system resilience
- Improve incident response skills
- Validate DR procedures

### 6.2 Running Chaos Experiments

**See:** `scripts/chaos-engineering-runner.sh`

**Safe Chaos Practices:**
- Always run in non-production first
- Start small (single pod)
- Gradually increase blast radius
- Monitor closely during experiment
- Have rollback plan ready
- Document all findings

**Example Experiment:**
```bash
# Run pod-kill experiment (5 minutes)
./scripts/chaos-engineering-runner.sh pod-kill 5m

# Expected behavior:
# - Kubernetes restarts killed pods
# - Load balancer routes around failed pods
# - API maintains >99% availability
# - No user-visible impact
```

---

## Module 7: Performance Optimization

### Duration: 1 hour

### 7.1 Performance Tuning Workflow

```
1. Identify bottleneck (Grafana dashboards)
2. Measure baseline (Prometheus metrics)
3. Hypothesize fix
4. Test in staging
5. Measure improvement
6. Deploy to production
7. Monitor impact
8. Document changes
```

### 7.2 Common Optimizations

**Database Query Optimization:**
```sql
-- Add index for common queries
CREATE INDEX CONCURRENTLY idx_predictions_user_created
ON predictions(user_id, created_at DESC);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM predictions
WHERE user_id = 123 ORDER BY created_at DESC LIMIT 10;
```

**Redis Cache Tuning:**
```bash
# Increase memory limit
kubectl set resources statefulset redis -n geo-climate \
  --limits=memory=4Gi \
  --requests=memory=2Gi

# Check cache hit rate
redis-cli INFO stats | grep keyspace_hits
```

**API Pod Resources:**
```yaml
resources:
  requests:
    memory: "1Gi"  # Increase from 512Mi
    cpu: "500m"    # Increase from 250m
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

---

## Hands-On Labs

### Lab 1: Incident Response Simulation (30 min)

**Scenario:** High latency alert

**Tasks:**
1. Investigate using Grafana
2. Check logs with Loki
3. Analyze traces with Jaeger
4. Identify root cause
5. Apply mitigation
6. Write incident report

### Lab 2: Deploy New Version (30 min)

**Tasks:**
1. Build Docker image
2. Deploy to staging
3. Run smoke tests
4. Deploy to production (blue-green)
5. Monitor rollout
6. Verify success

### Lab 3: DR Failover Drill (45 min)

**Tasks:**
1. Simulate primary region failure
2. Execute failover procedure
3. Verify secondary region operational
4. Test application functionality
5. Failback to primary
6. Document findings

### Lab 4: Chaos Engineering (30 min)

**Tasks:**
1. Run pod-kill experiment
2. Monitor system behavior
3. Analyze resilience
4. Document improvements needed

---

## Certification

### Final Assessment

**Written Exam (60 minutes):**
- 50 multiple choice questions
- Architecture knowledge
- Incident response procedures
- Monitoring & observability
- DR concepts

**Passing Score:** 80%

**Practical Exam (90 minutes):**
- Respond to 2 simulated incidents
- Execute deployment
- Perform DR drill
- Run chaos experiment

**Passing Criteria:**
- Complete all tasks within time limit
- Follow runbooks correctly
- Demonstrate troubleshooting skills
- Document actions clearly

### Certification Levels

**Level 1: Operator**
- Can monitor dashboards
- Can execute runbooks
- Can respond to incidents with guidance

**Level 2: Engineer**
- Can troubleshoot independently
- Can modify runbooks
- Can perform DR drills

**Level 3: Expert**
- Can architect solutions
- Can write new runbooks
- Can lead incident response
- Can train others

---

## Additional Resources

### Documentation
- [Operational Runbooks](OPERATIONAL_RUNBOOKS.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [API Documentation](https://api.geo-climate.example.com/docs)

### Tools
- [Grafana Dashboards](https://grafana.geo-climate.example.com)
- [Prometheus](https://prometheus.geo-climate.example.com)
- [Jaeger Tracing](https://jaeger.geo-climate.example.com)

### Support
- Slack: #geo-climate-ops
- Email: ops@geo-climate.example.com
- On-call: PagerDuty escalation policy

---

**Training Version:** 1.0.0
**Last Updated:** November 2025
**Maintained By:** Platform Engineering Team
