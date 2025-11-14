# Geo Climate Platform - Production Deployment Guide

Complete guide for deploying the Geo Climate Air Quality Prediction Platform to production Kubernetes clusters.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Local Development](#local-development)
4. [Staging Deployment](#staging-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Disaster Recovery](#disaster-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)
10. [Operational Runbooks](#operational-runbooks)

---

## Prerequisites

### Required Tools

```bash
# Kubernetes CLI
kubectl version --client  # >= 1.24

# Helm package manager
helm version  # >= 3.12

# Docker for building images
docker version  # >= 24.0

# Optional but recommended
# - kubectx/kubens: Easy context/namespace switching
# - k9s: Terminal UI for Kubernetes
# - stern: Multi-pod log tailing
```

### Cloud Provider Requirements

#### AWS EKS
```bash
# AWS CLI
aws --version  # >= 2.13

# eksctl
eksctl version  # >= 0.150

# Create EKS cluster
eksctl create cluster \
  --name geo-climate-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 20 \
  --managed
```

#### GCP GKE
```bash
# gcloud CLI
gcloud version  # >= 450.0

# Create GKE cluster
gcloud container clusters create geo-climate-prod \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 20
```

#### Azure AKS
```bash
# Azure CLI
az version  # >= 2.50

# Create AKS cluster
az aks create \
  --resource-group geo-climate-rg \
  --name geo-climate-prod \
  --node-count 3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20 \
  --node-vm-size Standard_D4s_v3
```

### Access Requirements

- Kubernetes cluster admin access
- Container registry access (GitHub Container Registry)
- DNS management for domain configuration
- SSL certificate management (Let's Encrypt or cloud provider)

---

## Infrastructure Setup

### 1. Namespace and RBAC

Create the namespace and set up RBAC:

```bash
# Create namespace
kubectl apply -f k8s/production/01-namespace.yaml

# Verify namespace creation
kubectl get namespace geo-climate

# Create RBAC (ServiceAccounts, Roles, RoleBindings)
kubectl apply -f k8s/production/02-rbac.yaml

# Verify RBAC
kubectl get serviceaccounts -n geo-climate
kubectl get roles -n geo-climate
kubectl get rolebindings -n geo-climate
```

### 2. Storage Classes

Configure storage classes for your cloud provider:

```bash
# Apply storage configuration
kubectl apply -f k8s/production/07-storage.yaml

# Verify storage classes
kubectl get storageclass

# Verify PVCs
kubectl get pvc -n geo-climate
```

### 3. Secrets Management

**CRITICAL**: Never commit actual secrets to version control!

```bash
# Create database secrets
kubectl create secret generic postgres-secrets \
  --from-literal=POSTGRES_USER=geo_climate \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_DB=geo_climate_prod \
  --from-literal=DATABASE_URL="postgresql://geo_climate:$(openssl rand -base64 32)@postgres:5432/geo_climate_prod" \
  --namespace=geo-climate

# Create Redis secrets
kubectl create secret generic redis-secrets \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=REDIS_URL="redis://:$(openssl rand -base64 32)@redis:6379/0" \
  --namespace=geo-climate

# Create application secrets
kubectl create secret generic geo-climate-secrets \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=JWT_REFRESH_SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=ENCRYPTION_KEY=$(openssl rand -base64 32) \
  --namespace=geo-climate

# Verify secrets
kubectl get secrets -n geo-climate
```

**Best Practice**: Use external secret management:

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  --namespace external-secrets-system \
  --create-namespace

# Configure with AWS Secrets Manager, HashiCorp Vault, etc.
```

### 4. ConfigMaps

```bash
# Create application ConfigMap
kubectl apply -f k8s/production/03-configmap.yaml

# Verify ConfigMap
kubectl get configmap geo-climate-config -n geo-climate -o yaml
```

---

## Local Development

### Quick Start with Docker Compose

```bash
# Start all services locally
docker-compose up -d

# View logs
docker-compose logs -f api

# Run migrations
docker-compose exec api alembic upgrade head

# Create admin user
docker-compose exec api python -c "
from source.api.database import SessionLocal, init_db, create_admin_user
init_db()
db = SessionLocal()
create_admin_user('admin', 'admin@example.com', 'admin123', db)
"

# Access API
curl http://localhost:8000/health
```

### Local Kubernetes with Minikube/Kind

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Or use Kind
kind create cluster --name geo-climate-dev

# Deploy to local cluster
helm install geo-climate ./helm/geo-climate \
  --namespace geo-climate-dev \
  --create-namespace \
  --values ./helm/geo-climate/values-dev.yaml

# Port forward to access API
kubectl port-forward -n geo-climate-dev svc/geo-climate-api 8000:80

# Access at http://localhost:8000
```

---

## Staging Deployment

Staging should mirror production as closely as possible.

### 1. Prepare Helm Dependencies

```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Update chart dependencies
cd helm/geo-climate
helm dependency update
```

### 2. Deploy to Staging

```bash
# Deploy with staging values
helm install geo-climate ./helm/geo-climate \
  --namespace geo-climate-staging \
  --create-namespace \
  --values ./helm/geo-climate/values-staging.yaml \
  --set postgresql.auth.password=$DB_PASSWORD_STAGING \
  --set redis.auth.password=$REDIS_PASSWORD_STAGING \
  --set secrets.jwtSecretKey=$JWT_SECRET_STAGING \
  --wait \
  --timeout 15m

# Monitor deployment
kubectl get pods -n geo-climate-staging -w

# Check deployment status
helm status geo-climate -n geo-climate-staging
```

### 3. Verify Staging Deployment

```bash
# Check all pods are running
kubectl get pods -n geo-climate-staging

# Check services
kubectl get svc -n geo-climate-staging

# Check ingress
kubectl get ingress -n geo-climate-staging

# Run smoke tests
kubectl run smoke-tests \
  --image=curlimages/curl:latest \
  --rm -i --restart=Never \
  -n geo-climate-staging \
  -- curl -f http://geo-climate-api/health
```

### 4. Run Integration Tests

```bash
# Port forward to run tests
kubectl port-forward -n geo-climate-staging svc/geo-climate-api 8000:80 &

# Run integration tests
pytest tests/integration/ -v

# Run load tests
locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m --host http://localhost:8000
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] All tests passing in staging
- [ ] Load tests completed successfully
- [ ] Database backup verified
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] On-call team notified
- [ ] Change management ticket approved
- [ ] SSL certificates valid
- [ ] DNS records configured

### 1. Create Production Namespace

```bash
kubectl apply -f k8s/production/01-namespace.yaml
kubectl apply -f k8s/production/02-rbac.yaml
```

### 2. Configure Secrets

```bash
# Use strong, randomly generated passwords
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -hex 64)

# Create secrets
kubectl create secret generic postgres-secrets \
  --from-literal=POSTGRES_PASSWORD=$DB_PASSWORD \
  --namespace=geo-climate

kubectl create secret generic redis-secrets \
  --from-literal=REDIS_PASSWORD=$REDIS_PASSWORD \
  --namespace=geo-climate

kubectl create secret generic geo-climate-secrets \
  --from-literal=JWT_SECRET_KEY=$JWT_SECRET \
  --namespace=geo-climate

# Store passwords in secure location (1Password, Vault, etc.)
```

### 3. Deploy Infrastructure Components

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/production/08-postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod/postgres-0 -n geo-climate --timeout=300s

# Deploy Redis
kubectl apply -f k8s/production/09-redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod/redis-0 -n geo-climate --timeout=300s
```

### 4. Deploy Application

```bash
# Deploy ConfigMap
kubectl apply -f k8s/production/03-configmap.yaml

# Deploy API
kubectl apply -f k8s/production/04-api-deployment.yaml

# Deploy Services and Ingress
kubectl apply -f k8s/production/05-services-ingress.yaml

# Monitor rollout
kubectl rollout status deployment/geo-climate-api -n geo-climate
```

### 5. Alternative: Helm Deployment (Recommended)

```bash
# Deploy with Helm (recommended)
helm install geo-climate ./helm/geo-climate \
  --namespace geo-climate \
  --create-namespace \
  --values ./helm/geo-climate/values-prod.yaml \
  --set api.image.tag=v2.0.0 \
  --set postgresql.auth.password=$DB_PASSWORD \
  --set redis.auth.password=$REDIS_PASSWORD \
  --set secrets.jwtSecretKey=$JWT_SECRET \
  --wait \
  --timeout 20m

# Monitor deployment
watch kubectl get pods -n geo-climate
```

### 6. Post-Deployment Verification

```bash
# Check all pods are running
kubectl get pods -n geo-climate

# Check HPA status
kubectl get hpa -n geo-climate

# Check PVC status
kubectl get pvc -n geo-climate

# Test health endpoint
kubectl run curl --image=curlimages/curl:latest --rm -i --restart=Never -n geo-climate -- \
  curl -f http://geo-climate-api/health

# Check metrics endpoint
kubectl run curl --image=curlimages/curl:latest --rm -i --restart=Never -n geo-climate -- \
  curl -f http://geo-climate-api/metrics

# Test external access
curl https://api.geo-climate.example.com/health
```

### 7. Database Initialization

```bash
# Run migrations
kubectl exec -it deployment/geo-climate-api -n geo-climate -- \
  alembic upgrade head

# Create initial admin user
kubectl exec -it deployment/geo-climate-api -n geo-climate -- \
  python -c "
from source.api.database import SessionLocal, create_admin_user
db = SessionLocal()
create_admin_user('admin', 'admin@geo-climate.com', '$(openssl rand -base64 24)', db)
print('Admin user created successfully!')
"
```

---

## Monitoring and Observability

### Prometheus

```bash
# Port forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090

# Access Prometheus UI at http://localhost:9090

# Key metrics to monitor:
# - http_request_duration_seconds
# - http_requests_total
# - system_cpu_usage_percent
# - database_connections_active
```

### Grafana

```bash
# Port forward to Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Access Grafana at http://localhost:3000
# Default credentials: admin / admin (change immediately!)

# Import dashboard
# - Dashboard ID: geo_climate_overview.json
```

### Logs

```bash
# View API logs
kubectl logs -f deployment/geo-climate-api -n geo-climate

# View logs from all API pods
kubectl logs -f -l app=geo-climate-api -n geo-climate

# Use stern for multi-pod logging
stern geo-climate-api -n geo-climate

# Filter logs
kubectl logs deployment/geo-climate-api -n geo-climate | grep ERROR
```

### Alerts

Key alerts to configure:

1. **High API Latency**: P95 > 1s for 10 minutes
2. **High Error Rate**: > 1% errors for 5 minutes
3. **Pod Crash Loop**: Restarts > 3 in 5 minutes
4. **High CPU Usage**: > 80% for 15 minutes
5. **High Memory Usage**: > 85% for 10 minutes
6. **Database Unavailable**: Cannot connect for 1 minute
7. **Low Disk Space**: < 15% free space

---

## Disaster Recovery

### Database Backup and Restore

#### Automated Backups

Backups run daily via CronJob:

```bash
# Check backup CronJob
kubectl get cronjob postgres-backup -n geo-climate

# Trigger manual backup
kubectl create job --from=cronjob/postgres-backup manual-backup-$(date +%s) -n geo-climate

# List backups
kubectl exec -it postgres-0 -n geo-climate -- ls -lh /backups/
```

#### Manual Backup

```bash
# Create manual backup
kubectl exec -it postgres-0 -n geo-climate -- \
  pg_dump -U geo_climate -d geo_climate_prod | \
  gzip > backup-$(date +%Y%m%d-%H%M%S).sql.gz

# Upload to S3 (AWS)
aws s3 cp backup-*.sql.gz s3://geo-climate-backups/
```

#### Restore from Backup

```bash
# Stop API to prevent writes
kubectl scale deployment geo-climate-api --replicas=0 -n geo-climate

# Restore database
gunzip -c backup-20250114-120000.sql.gz | \
  kubectl exec -i postgres-0 -n geo-climate -- \
  psql -U geo_climate -d geo_climate_prod

# Restart API
kubectl scale deployment geo-climate-api --replicas=3 -n geo-climate
```

### Cluster Disaster Recovery

#### Backup Kubernetes Resources

```bash
# Install Velero for cluster backups
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket geo-climate-velero-backups \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1

# Create backup schedule
velero schedule create daily-backup \
  --schedule="0 2 * * *" \
  --include-namespaces geo-climate

# Manual backup
velero backup create manual-backup --include-namespaces geo-climate

# Restore from backup
velero restore create --from-backup manual-backup
```

---

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n geo-climate

# Describe pod for events
kubectl describe pod <pod-name> -n geo-climate

# Check logs
kubectl logs <pod-name> -n geo-climate

# Check previous logs if crashed
kubectl logs <pod-name> --previous -n geo-climate

# Common causes:
# - Image pull errors: Check image repository and credentials
# - Resource constraints: Check node resources
# - ConfigMap/Secret missing: Verify all secrets exist
# - Init container failures: Check init container logs
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl run pg-test --image=postgres:15-alpine --rm -it --restart=Never -n geo-climate -- \
  psql -h postgres -U geo_climate -d geo_climate_prod -c "SELECT 1"

# Check database pod
kubectl get pod postgres-0 -n geo-climate
kubectl logs postgres-0 -n geo-climate

# Check service
kubectl get svc postgres -n geo-climate
kubectl describe svc postgres -n geo-climate

# Verify secret
kubectl get secret postgres-secrets -n geo-climate -o yaml
```

#### High Latency

```bash
# Check metrics
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090

# Query in Prometheus:
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Check HPA status
kubectl get hpa -n geo-climate

# Check pod resources
kubectl top pods -n geo-climate

# Potential solutions:
# - Scale up: kubectl scale deployment geo-climate-api --replicas=10
# - Increase resources: Update deployment resource limits
# - Check database performance: Slow queries, connection pool
# - Check Redis: Connection issues, memory limits
```

#### Certificate Issues

```bash
# Check certificate
kubectl get certificate -n geo-climate
kubectl describe certificate geo-climate-tls -n geo-climate

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manually renew certificate
kubectl delete certificate geo-climate-tls -n geo-climate
kubectl apply -f k8s/production/05-services-ingress.yaml
```

### Debug Commands

```bash
# Get full cluster status
kubectl get all -n geo-climate

# Run diagnostic container
kubectl run debug --image=nicolaka/netshoot --rm -it --restart=Never -n geo-climate -- /bin/bash

# Check DNS resolution
kubectl run dns-test --image=busybox --rm -it --restart=Never -n geo-climate -- nslookup postgres

# Check network connectivity
kubectl run net-test --image=nicolaka/netshoot --rm -it --restart=Never -n geo-climate -- \
  curl -v http://geo-climate-api/health
```

---

## Security Best Practices

### 1. Network Security

```bash
# Apply NetworkPolicy
kubectl apply -f k8s/production/02-rbac.yaml

# Verify NetworkPolicy
kubectl get networkpolicy -n geo-climate
kubectl describe networkpolicy geo-climate-network-policy -n geo-climate
```

### 2. Pod Security

```bash
# Enable Pod Security Standards
kubectl label namespace geo-climate \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### 3. Secret Encryption

Ensure secrets are encrypted at rest in etcd:

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-secret>
      - identity: {}
```

### 4. RBAC Audit

```bash
# Review RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:geo-climate:geo-climate-api

# Audit who can access secrets
kubectl auth can-i get secrets --namespace=geo-climate --as=<user>
```

### 5. Image Security

```bash
# Scan images with Trivy
trivy image ghcr.io/dogaaydinn/geo_sentiment_climate:latest

# Use only signed images
kubectl apply -f - <<EOF
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: geo-climate-policy
spec:
  images:
    - glob: "ghcr.io/dogaaydinn/geo_sentiment_climate:*"
  authorities:
    - keyless:
        url: https://fulcio.sigstore.dev
EOF
```

---

## Operational Runbooks

### Scaling

```bash
# Manual scaling
kubectl scale deployment geo-climate-api --replicas=10 -n geo-climate

# Update HPA min/max
kubectl patch hpa geo-climate-api -n geo-climate -p '{"spec":{"minReplicas":5,"maxReplicas":30}}'

# Disable autoscaling temporarily
kubectl delete hpa geo-climate-api -n geo-climate
```

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/geo-climate-api \
  api=ghcr.io/dogaaydinn/geo_sentiment_climate:v2.1.0 \
  -n geo-climate

# Monitor rollout
kubectl rollout status deployment/geo-climate-api -n geo-climate

# Check rollout history
kubectl rollout history deployment/geo-climate-api -n geo-climate

# Rollback if needed
kubectl rollout undo deployment/geo-climate-api -n geo-climate

# Rollback to specific revision
kubectl rollout undo deployment/geo-climate-api --to-revision=3 -n geo-climate
```

### Maintenance Mode

```bash
# Scale down to single replica
kubectl scale deployment geo-climate-api --replicas=1 -n geo-climate

# Or use PodDisruptionBudget to prevent evictions
kubectl apply -f - <<EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: geo-climate-api-pdb-maintenance
  namespace: geo-climate
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: geo-climate-api
EOF

# Return to normal after maintenance
kubectl delete pdb geo-climate-api-pdb-maintenance -n geo-climate
kubectl scale deployment geo-climate-api --replicas=3 -n geo-climate
```

### Emergency Procedures

#### Complete System Failure

```bash
# 1. Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces

# 2. Restore from Velero backup
velero restore create emergency-restore --from-backup daily-backup-20250114

# 3. Verify restoration
kubectl get all -n geo-climate

# 4. Test endpoints
curl https://api.geo-climate.example.com/health
```

#### Database Corruption

```bash
# 1. Stop all writes
kubectl scale deployment geo-climate-api --replicas=0 -n geo-climate

# 2. Restore from latest backup
# See "Database Backup and Restore" section

# 3. Verify data integrity
kubectl exec -it postgres-0 -n geo-climate -- \
  psql -U geo_climate -d geo_climate_prod -c "ANALYZE VERBOSE;"

# 4. Restart API
kubectl scale deployment geo-climate-api --replicas=3 -n geo-climate
```

---

## CI/CD Integration

The platform uses GitHub Actions for automated deployment. See `.github/workflows/ci-cd.yaml`.

### Required Secrets

Configure these secrets in GitHub repository settings:

```
# Kubernetes access
KUBECONFIG_DEV
KUBECONFIG_STAGING
KUBECONFIG_PROD

# Database passwords
DB_PASSWORD_DEV
DB_PASSWORD_STAGING
DB_PASSWORD_PROD

# Redis passwords
REDIS_PASSWORD_STAGING
REDIS_PASSWORD_PROD

# Application secrets
JWT_SECRET_DEV
JWT_SECRET_STAGING
JWT_SECRET_PROD

# Notifications
SLACK_WEBHOOK
```

### Manual Deployment Trigger

```bash
# Trigger deployment via workflow dispatch
gh workflow run ci-cd.yaml \
  --ref main \
  -f environment=production \
  -f version=v2.0.0
```

---

## Support and Contacts

### Documentation

- API Documentation: https://api.geo-climate.example.com/docs
- Grafana Dashboards: https://grafana.geo-climate.example.com
- Prometheus: https://prometheus.geo-climate.example.com

### Contacts

- Platform Team: platform-team@example.com
- On-Call: oncall@example.com
- Security: security@example.com

### Emergency Escalation

1. Check Runbook (this document)
2. Contact On-Call Engineer
3. Escalate to Platform Team Lead
4. Emergency: Contact CTO

---

**Last Updated**: January 14, 2025
**Version**: 2.0.0
**Maintained by**: Platform Team
