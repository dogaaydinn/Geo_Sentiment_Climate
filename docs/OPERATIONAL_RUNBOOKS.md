# Operational Runbooks

Complete operational procedures for the Geo Climate Platform.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Incident Response](#incident-response)
3. [Common Issues](#common-issues)
4. [Scaling Procedures](#scaling-procedures)
5. [Deployment Procedures](#deployment-procedures)
6. [Backup and Recovery](#backup-and-recovery)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Security Incidents](#security-incidents)

---

## Daily Operations

### Morning Health Check

**Frequency**: Every business day at 9 AM
**Duration**: 10-15 minutes
**Owner**: Platform Team

#### Checklist

- [ ] Check Slack #alerts channel for overnight alerts
- [ ] Review Grafana dashboards for anomalies
- [ ] Check pod status: `kubectl get pods -n geo-climate`
- [ ] Review error logs from last 24h
- [ ] Verify backup completion
- [ ] Check certificate expiration (should be > 30 days)
- [ ] Review cost dashboard for anomalies
- [ ] Check deployment pipeline status

#### Commands

```bash
# Quick health check
./scripts/verify-deployment.sh production

# Check recent errors
kubectl logs -n geo-climate -l app=geo-climate-api --since=24h | grep ERROR | tail -n 20

# Verify backups
kubectl get jobs -n geo-climate | grep backup

# Check certificate expiration
kubectl get certificate -n geo-climate -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.notAfter}{"\n"}{end}'
```

### Weekly Tasks

**Every Monday at 10 AM**

- [ ] Review capacity planning metrics
- [ ] Check for security updates
- [ ] Review and clean up old resources
- [ ] Update runbooks if needed
- [ ] Review incident log from previous week
- [ ] Test disaster recovery plan (monthly)

### Monthly Tasks

**First Monday of each month**

- [ ] Full disaster recovery drill
- [ ] Security audit
- [ ] Cost optimization review
- [ ] Update dependencies
- [ ] Review and rotate API keys
- [ ] Review access logs for anomalies
- [ ] Capacity planning review

---

## Incident Response

### Severity Levels

#### P0 - Critical (Production Down)
- **Response Time**: Immediate
- **Resolution Time**: 1 hour
- **Examples**: API completely unavailable, data loss, security breach
- **Actions**:
  1. Page on-call engineer immediately
  2. Create incident channel (#incident-YYYY-MM-DD)
  3. Notify stakeholders
  4. Start incident log
  5. Invoke disaster recovery if needed

#### P1 - High (Major Degradation)
- **Response Time**: 15 minutes
- **Resolution Time**: 4 hours
- **Examples**: High latency, elevated error rates, partial outage
- **Actions**:
  1. Notify on-call engineer
  2. Create incident channel
  3. Investigate root cause
  4. Implement temporary fix
  5. Plan permanent fix

#### P2 - Medium (Minor Degradation)
- **Response Time**: 1 hour
- **Resolution Time**: 24 hours
- **Examples**: Non-critical feature unavailable, minor performance issues
- **Actions**:
  1. Create Jira ticket
  2. Investigate during business hours
  3. Plan fix in next sprint

#### P3 - Low (Cosmetic)
- **Response Time**: Best effort
- **Resolution Time**: Next release
- **Examples**: UI glitches, minor logging issues
- **Actions**:
  1. Create Jira ticket
  2. Prioritize in backlog

### Incident Response Process

#### 1. Detection (0-5 minutes)

```bash
# Triggered by:
# - Alert in #alerts channel
# - User report
# - Automated monitoring

# Immediate actions:
# 1. Acknowledge alert
# 2. Check dashboard: https://grafana.geo-climate.example.com
# 3. Verify issue
```

#### 2. Triage (5-15 minutes)

```bash
# Assess severity
# Check scope of impact
kubectl get pods -n geo-climate
kubectl get events -n geo-climate --sort-by='.lastTimestamp' | tail -n 20

# Quick diagnosis
kubectl logs -n geo-climate -l app=geo-climate-api --tail=100

# Check metrics
# - Error rate in Prometheus
# - Latency P95/P99
# - Resource utilization
```

#### 3. Communication (Throughout)

**Slack Template**:
```
🚨 INCIDENT ALERT - P[0/1/2]
Title: [Brief description]
Impact: [What's affected]
Status: Investigating
Time: [HH:MM UTC]
Incident Commander: @[name]
Incident Channel: #incident-YYYY-MM-DD
Next Update: [In 30 min]
```

**Status Updates** (every 30 minutes):
```
📊 INCIDENT UPDATE
Status: [Investigating/Identified/Fixing/Resolved]
Current Actions: [What we're doing]
Impact: [Updated impact assessment]
ETA: [Expected resolution time]
Next Update: [In 30 min]
```

#### 4. Mitigation (Parallel to Investigation)

```bash
# Quick fixes to restore service

# 1. Scale up if resource issue
kubectl scale deployment geo-climate-api --replicas=10 -n geo-climate

# 2. Restart pods if stuck
kubectl rollout restart deployment/geo-climate-api -n geo-climate

# 3. Rollback if bad deployment
helm rollback geo-climate -n geo-climate

# 4. Isolate problematic component
kubectl cordon <node-name>  # Prevent scheduling on bad node
kubectl drain <node-name>   # Move pods off bad node

# 5. Enable maintenance mode if needed
kubectl scale deployment geo-climate-api --replicas=1 -n geo-climate
```

#### 5. Investigation

```bash
# Gather evidence
# 1. Collect logs
kubectl logs -n geo-climate deployment/geo-climate-api --since=1h > incident-logs.txt

# 2. Check events
kubectl get events -n geo-climate --sort-by='.lastTimestamp' > incident-events.txt

# 3. Check metrics
# Save screenshots from Grafana

# 4. Check database
kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
SELECT pid, usename, application_name, state, query_start, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;
"

# 5. Check Redis
kubectl exec -it redis-0 -n geo-climate -- redis-cli INFO stats
kubectl exec -it redis-0 -n geo-climate -- redis-cli SLOWLOG GET 10
```

#### 6. Resolution

```bash
# Implement fix
# 1. Deploy fix via CI/CD or manual patch
# 2. Verify fix with monitoring
# 3. Gradually restore traffic (if canary)
# 4. Monitor for 30 minutes
# 5. Declare resolved if stable
```

#### 7. Post-Incident Review (Within 48 hours)

**Template**: `docs/incidents/YYYY-MM-DD-incident-name.md`

```markdown
# Incident: [Title]

**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: P[0/1/2]
**Incident Commander**: [Name]

## Summary
[Brief description of what happened]

## Impact
- Users affected: [Number/percentage]
- Revenue impact: $[amount]
- Services affected: [List]
- Duration: [HH:MM]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| 14:30 | Issue detected via alert |
| 14:35 | Incident declared, IC assigned |
| 14:45 | Root cause identified |
| 15:00 | Fix deployed |
| 15:30 | Service restored |
| 16:00 | Incident resolved |

## Root Cause
[Technical description of what went wrong]

## Resolution
[What was done to fix it]

## Action Items
- [ ] [Action 1] - Owner: [Name] - Due: [Date]
- [ ] [Action 2] - Owner: [Name] - Due: [Date]

## Lessons Learned
[What we learned and will do differently]

## Prevention
[How we'll prevent this in the future]
```

---

## Common Issues

### High API Latency

**Symptoms**:
- P95 latency > 1 second
- Alert: `HighAPILatency` firing
- Slow response times from API

**Investigation**:

```bash
# 1. Check pod CPU/memory
kubectl top pods -n geo-climate --sort-by=memory

# 2. Check database performance
kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# 3. Check Redis hit rate
kubectl exec -it redis-0 -n geo-climate -- redis-cli INFO stats | grep hit

# 4. Check external API latency
kubectl logs -n geo-climate -l app=geo-climate-api --tail=100 | grep "external_api"
```

**Common Causes & Solutions**:

1. **Database slow queries**
   ```bash
   # Add indexes
   kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
   CREATE INDEX CONCURRENTLY idx_name ON table_name (column_name);
   "

   # Analyze tables
   kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "VACUUM ANALYZE;"
   ```

2. **Insufficient resources**
   ```bash
   # Scale horizontally
   kubectl patch hpa geo-climate-api -n geo-climate -p '{"spec":{"minReplicas":5}}'

   # Or scale vertically (update deployment)
   kubectl set resources deployment geo-climate-api -n geo-climate \
     --limits=cpu=4000m,memory=8Gi \
     --requests=cpu=1000m,memory=2Gi
   ```

3. **Cache misses**
   ```bash
   # Warm up cache
   kubectl exec -it redis-0 -n geo-climate -- redis-cli FLUSHALL
   # Then trigger cache warming endpoint
   curl -X POST https://api.geo-climate.example.com/admin/cache/warm
   ```

4. **External API slowness**
   ```bash
   # Increase timeout or implement circuit breaker
   # Check circuit breaker status in metrics
   ```

### High Error Rate

**Symptoms**:
- Error rate > 1%
- Alert: `HighErrorRate` firing
- 5xx errors in logs

**Investigation**:

```bash
# 1. Check error distribution
kubectl logs -n geo-climate -l app=geo-climate-api --since=1h | \
  grep -E "ERROR|5\d{2}" | \
  awk '{print $NF}' | \
  sort | uniq -c | sort -rn | head -10

# 2. Check specific errors
kubectl logs -n geo-climate -l app=geo-climate-api --tail=50 | grep ERROR

# 3. Check database errors
kubectl logs -n geo-climate postgres-0 | grep ERROR | tail -20

# 4. Check Redis errors
kubectl logs -n geo-climate redis-0 | grep ERROR | tail -20
```

**Common Causes**:

1. **Database connection pool exhausted**
   ```bash
   # Check active connections
   kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
   SELECT count(*) FROM pg_stat_activity WHERE state != 'idle';
   "

   # Increase pool size in deployment
   # Or scale PgBouncer
   kubectl scale deployment pgbouncer --replicas=4 -n geo-climate
   ```

2. **Dependency failure**
   ```bash
   # Check circuit breaker status
   # Check external service health
   curl -v https://external-api.example.com/health

   # Implement fallback or retry logic
   ```

3. **Memory leaks**
   ```bash
   # Check memory usage trend
   kubectl top pods -n geo-climate -l app=geo-climate-api

   # Restart pods if memory increasing
   kubectl rollout restart deployment/geo-climate-api -n geo-climate
   ```

### Database Connection Issues

**Symptoms**:
- Can't connect to database
- Connection timeout errors
- "Too many connections" errors

**Investigation**:

```bash
# 1. Check database pod
kubectl get pods -n geo-climate -l app=postgres

# 2. Check database logs
kubectl logs -n geo-climate postgres-0 --tail=100

# 3. Check connections
kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
SELECT count(*), state
FROM pg_stat_activity
GROUP BY state;
"

# 4. Check network policy
kubectl get networkpolicy -n geo-climate
kubectl describe networkpolicy geo-climate-network-policy -n geo-climate
```

**Solutions**:

```bash
# 1. Kill idle connections
kubectl exec -it postgres-0 -n geo-climate -- psql -U geo_climate -d geo_climate_prod -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < current_timestamp - INTERVAL '15 minutes';
"

# 2. Increase max_connections
kubectl exec -it postgres-0 -n geo-climate -- psql -U postgres -c "
ALTER SYSTEM SET max_connections = 300;
"
# Restart postgres
kubectl delete pod postgres-0 -n geo-climate

# 3. Check firewall/network policy
kubectl exec -it <api-pod> -n geo-climate -- nc -zv postgres 5432
```

### Certificate Issues

**Symptoms**:
- HTTPS not working
- Certificate expired warning
- cert-manager errors

**Investigation**:

```bash
# 1. Check certificate status
kubectl get certificate -n geo-climate
kubectl describe certificate geo-climate-tls-secret -n geo-climate

# 2. Check certificate request
kubectl get certificaterequest -n geo-climate
kubectl describe certificaterequest <name> -n geo-climate

# 3. Check challenge
kubectl get challenge -n geo-climate
kubectl describe challenge <name> -n geo-climate

# 4. Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager --tail=100
```

**Solutions**:

```bash
# 1. Delete and recreate certificate
kubectl delete certificate geo-climate-tls-secret -n geo-climate
kubectl apply -f k8s/ingress/dns-ssl-setup.yaml

# 2. Check DNS propagation
dig api.geo-climate.example.com

# 3. Check ingress annotation
kubectl get ingress -n geo-climate -o yaml | grep cert-manager

# 4. Test HTTP-01 challenge
curl -v http://api.geo-climate.example.com/.well-known/acme-challenge/test
```

### Pod Crash Loop

**Symptoms**:
- Pods restarting continuously
- CrashLoopBackOff status
- High restart count

**Investigation**:

```bash
# 1. Check pod status
kubectl get pods -n geo-climate -l app=geo-climate-api

# 2. Check logs
kubectl logs -n geo-climate <pod-name> --previous

# 3. Describe pod for events
kubectl describe pod <pod-name> -n geo-climate

# 4. Check resource limits
kubectl get pod <pod-name> -n geo-climate -o jsonpath='{.spec.containers[0].resources}'
```

**Common Causes**:

1. **OOM (Out of Memory)**
   ```bash
   # Check if OOMKilled
   kubectl get pods -n geo-climate -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}' | grep OOM

   # Increase memory limits
   kubectl set resources deployment geo-climate-api -n geo-climate \
     --limits=memory=8Gi \
     --requests=memory=2Gi
   ```

2. **Application error**
   ```bash
   # Check startup logs
   kubectl logs <pod-name> -n geo-climate --previous

   # Fix application code and redeploy
   ```

3. **Liveness probe failure**
   ```bash
   # Check liveness probe
   kubectl get pod <pod-name> -n geo-climate -o jsonpath='{.spec.containers[0].livenessProbe}'

   # Adjust probe timing
   kubectl patch deployment geo-climate-api -n geo-climate -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "api",
             "livenessProbe": {
               "initialDelaySeconds": 60,
               "timeoutSeconds": 10
             }
           }]
         }
       }
     }
   }'
   ```

---

## Scaling Procedures

### Horizontal Scaling (Pods)

**When to scale**:
- CPU usage > 70% sustained
- Memory usage > 80% sustained
- Response time degradation
- Planned traffic increase

**Manual scaling**:

```bash
# Scale up
kubectl scale deployment geo-climate-api --replicas=10 -n geo-climate

# Verify
kubectl get pods -n geo-climate -l app=geo-climate-api -w

# Update HPA minimum
kubectl patch hpa geo-climate-api -n geo-climate -p '{"spec":{"minReplicas":10}}'
```

**Automatic scaling** (already configured):

```yaml
# Configured in deployment
spec:
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Vertical Scaling (Resources)

**When to scale**:
- Consistent resource exhaustion
- Can't scale horizontally further
- Memory leaks suspected

**Procedure**:

```bash
# 1. Update resource requests/limits
kubectl set resources deployment geo-climate-api -n geo-climate \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=1000m,memory=2Gi

# 2. Monitor rollout
kubectl rollout status deployment/geo-climate-api -n geo-climate

# 3. Verify pods are stable
kubectl get pods -n geo-climate -l app=geo-climate-api -w

# 4. Monitor metrics for 1 hour
```

### Database Scaling

**Read replicas**:

```bash
# Already configured in values-prod.yaml
# To increase:
helm upgrade geo-climate ./helm/geo-climate \
  --namespace geo-climate \
  --values ./helm/geo-climate/values-prod.yaml \
  --set postgresql.readReplicas.replicaCount=3
```

**Vertical scaling (storage)**:

```bash
# Expand PVC
kubectl patch pvc postgres-data-postgres-0 -n geo-climate -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# Verify expansion
kubectl get pvc -n geo-climate postgres-data-postgres-0 -w
```

---

## Deployment Procedures

### Standard Deployment

See `.github/workflows/ci-cd.yaml` for automated deployment.

**Manual deployment**:

```bash
# 1. Pre-deployment checks
./scripts/verify-deployment.sh staging

# 2. Deploy to staging
helm upgrade geo-climate ./helm/geo-climate \
  --namespace geo-climate-staging \
  --values ./helm/geo-climate/values-staging.yaml \
  --set api.image.tag=v2.1.0

# 3. Verify staging
./scripts/verify-deployment.sh staging
./tests/load/load-test-runner.sh staging smoke

# 4. Deploy to production (canary)
# See Canary Deployment section below
```

### Canary Deployment

**Gradual rollout to minimize risk**:

```bash
# 1. Deploy canary (10% traffic)
kubectl set image deployment/geo-climate-api \
  api=ghcr.io/dogaaydinn/geo_sentiment_climate:v2.1.0 \
  -n geo-climate

kubectl patch deployment geo-climate-api -n geo-climate -p '{"spec":{"replicas":1}}'

# 2. Monitor for 15 minutes
# Check error rate, latency, logs

# 3. If successful, scale to 50%
kubectl scale deployment geo-climate-api --replicas=3 -n geo-climate

# 4. Monitor for 15 minutes

# 5. If successful, complete rollout
helm upgrade geo-climate ./helm/geo-climate \
  --namespace geo-climate \
  --values ./helm/geo-climate/values-prod.yaml \
  --set api.image.tag=v2.1.0
```

### Rollback Procedures

**Quick rollback**:

```bash
# Helm rollback (recommended)
helm rollback geo-climate -n geo-climate

# Or kubectl rollback
kubectl rollout undo deployment/geo-climate-api -n geo-climate

# Rollback to specific revision
helm rollback geo-climate 3 -n geo-climate

# Verify
kubectl rollout status deployment/geo-climate-api -n geo-climate
```

**Database migration rollback**:

```bash
# Check migration history
kubectl exec -it deployment/geo-climate-api -n geo-climate -- alembic history

# Rollback one version
kubectl exec -it deployment/geo-climate-api -n geo-climate -- alembic downgrade -1

# Rollback to specific version
kubectl exec -it deployment/geo-climate-api -n geo-climate -- alembic downgrade <revision>
```

---

## Backup and Recovery

See `docs/DEPLOYMENT_GUIDE.md#disaster-recovery` for complete procedures.

### Quick Backup

```bash
# Database backup
kubectl create job --from=cronjob/postgres-backup manual-backup-$(date +%s) -n geo-climate

# Verify backup
kubectl logs job/manual-backup-<timestamp> -n geo-climate
```

### Quick Restore

```bash
# 1. Scale down API
kubectl scale deployment geo-climate-api --replicas=0 -n geo-climate

# 2. Restore database
gunzip -c backup-20250114-120000.sql.gz | \
  kubectl exec -i postgres-0 -n geo-climate -- \
  psql -U geo_climate -d geo_climate_prod

# 3. Scale up API
kubectl scale deployment geo-climate-api --replicas=3 -n geo-climate

# 4. Verify
./scripts/verify-deployment.sh production
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

**Golden Signals**:
1. **Latency**: P50, P95, P99 response times
2. **Traffic**: Requests per second
3. **Errors**: Error rate (%)
4. **Saturation**: CPU, memory, disk usage

**Access dashboards**:
- Grafana: `https://grafana.geo-climate.example.com`
- Prometheus: `https://prometheus.geo-climate.example.com`

### Alert Configuration

See `monitoring/prometheus/alerts/geo_climate_alerts.yml`

**Key alerts**:
- HighAPILatency (P95 > 1s for 10m)
- HighErrorRate (> 1% for 5m)
- PodCrashLooping (restarts > 3 in 5m)
- HighCPUUsage (> 80% for 15m)
- DatabaseUnavailable (can't connect for 1m)

---

## Security Incidents

### Suspected Breach

**Immediate actions**:

```bash
# 1. Isolate affected components
kubectl scale deployment geo-climate-api --replicas=1 -n geo-climate
kubectl cordon <affected-node>

# 2. Collect evidence
kubectl logs -n geo-climate --all-containers=true --since=24h > security-logs-$(date +%s).txt
kubectl get events -n geo-climate --sort-by='.lastTimestamp' > security-events-$(date +%s).txt

# 3. Rotate all secrets
# See k8s/secrets/external-secrets-operator.yaml

# 4. Review access logs
kubectl logs -n geo-climate -l app=geo-climate-api | grep -E "admin|auth|login" > access-audit.txt

# 5. Contact security team
# security@example.com
```

### Suspected DDoS

```bash
# 1. Enable aggressive rate limiting
kubectl patch configmap geo-climate-config -n geo-climate -p '{"data":{"RATE_LIMIT":"10"}}'

# 2. Check request sources
kubectl logs -n geo-climate -l app=nginx-ingress | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | sort | uniq -c | sort -rn | head -20

# 3. Block malicious IPs in firewall/WAF
# Cloud provider specific

# 4. Enable CloudFlare DDoS protection
# If using CloudFlare
```

---

**Last Updated**: January 2025
**Maintained by**: Platform Team
**Review Cycle**: Monthly

For questions or updates to this runbook, contact platform-team@example.com
