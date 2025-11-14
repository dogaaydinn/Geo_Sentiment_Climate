# Quick Deployment Guide

## 🚀 Deploy to Production in 5 Steps

### Prerequisites
- AWS account with CLI configured
- kubectl installed
- Terraform installed
- Docker installed

### Step 1: Configure Secrets (5 minutes)

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update infrastructure/kubernetes/secrets.yaml with:
# - SECRET_KEY (generated above)
# - POSTGRES_PASSWORD (secure password)
# - Any AWS/cloud credentials
```

### Step 2: Deploy AWS Infrastructure (15 minutes)

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Deploy (creates EKS, RDS, Redis, S3)
terraform apply -auto-approve

# Get cluster credentials
aws eks update-kubeconfig --name geo-climate-prod --region us-east-1
```

### Step 3: Deploy to Kubernetes (10 minutes)

```bash
# Option A: Deploy with Helm (recommended)
helm install geo-climate infrastructure/kubernetes/helm/ \
  --namespace geo-climate \
  --create-namespace

# Option B: Deploy with kubectl
kubectl apply -f infrastructure/kubernetes/

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=geo-climate-api -n geo-climate --timeout=300s
```

### Step 4: Initialize Database (5 minutes)

```bash
# Run migrations
kubectl exec -it deployment/geo-climate-api -n geo-climate -- alembic upgrade head

# Create initial admin user (optional)
kubectl exec -it deployment/geo-climate-api -n geo-climate -- python -m source.scripts.create_admin
```

### Step 5: Verify Deployment (5 minutes)

```bash
# Check all resources
kubectl get all -n geo-climate

# Get ingress URL
kubectl get ingress -n geo-climate

# Test API
INGRESS_URL=$(kubectl get ingress geo-climate-ingress -n geo-climate -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl https://$INGRESS_URL/health

# Check metrics
kubectl port-forward svc/prometheus -n geo-climate 9090:9090 &
open http://localhost:9090

# Check dashboards
kubectl port-forward svc/grafana -n geo-climate 3000:3000 &
open http://localhost:3000
```

## 🎯 Post-Deployment

### Configure DNS
```bash
# Get load balancer hostname
kubectl get ingress -n geo-climate

# Create CNAME record:
# api.your-domain.com -> [load-balancer-hostname]
```

### Setup SSL/TLS
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply Let's Encrypt issuer (already in manifests)
kubectl apply -f infrastructure/kubernetes/cert-issuer.yaml
```

### Enable Auto-scaling
```bash
# Install metrics server (if not present)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# HPA is already configured in hpa.yaml
kubectl get hpa -n geo-climate
```

## 📊 Monitoring

- **Grafana**: http://grafana.your-domain.com (default: admin/admin_change_me)
- **Prometheus**: http://prometheus.your-domain.com
- **API Docs**: https://api.your-domain.com/docs

## 🔒 Security Checklist

- [ ] Update all default passwords in secrets.yaml
- [ ] Configure CORS allowed origins in configmap.yaml
- [ ] Enable SSL/TLS with valid certificates
- [ ] Configure rate limiting tiers
- [ ] Setup backup schedules for RDS
- [ ] Enable CloudWatch logging
- [ ] Configure Prometheus alerting
- [ ] Setup incident response procedures

## 💰 Cost Optimization

```bash
# View current costs
terraform cost

# Scale down non-production
kubectl scale deployment geo-climate-api --replicas=1 -n geo-climate-dev

# Use spot instances for ML workloads (already configured in Terraform)
```

## 🆘 Troubleshooting

### Pods not starting
```bash
kubectl describe pod <pod-name> -n geo-climate
kubectl logs <pod-name> -n geo-climate
```

### Database connection issues
```bash
# Check database connectivity
kubectl exec -it deployment/geo-climate-api -n geo-climate -- python -c "from source.database.base import check_db_connection; print(check_db_connection())"
```

### Redis connection issues
```bash
# Check Redis
kubectl exec -it deployment/redis -n geo-climate -- redis-cli ping
```

## 📞 Support

- Documentation: https://api.your-domain.com/docs
- Issues: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
- Email: dogaa882@gmail.com

---

**Total Deployment Time: ~40 minutes**
**Difficulty: Intermediate**
**Cost: $350-700/month (scales with usage)**
