# Geo Climate Helm Chart

Helm chart for deploying Geo Sentiment Climate air quality prediction platform on Kubernetes.

## Prerequisites

- Kubernetes 1.20+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure
- Ingress controller (nginx recommended)

## Installation

### Add the repository

```bash
helm repo add geo-climate https://charts.geo-climate.example.com
helm repo update
```

### Install the chart

```bash
# Install with default values
helm install geo-climate geo-climate/geo-climate -n geo-climate --create-namespace

# Install with custom values
helm install geo-climate geo-climate/geo-climate -n geo-climate --create-namespace -f values-production.yaml
```

### Upgrade

```bash
helm upgrade geo-climate geo-climate/geo-climate -n geo-climate -f values-production.yaml
```

### Uninstall

```bash
helm uninstall geo-climate -n geo-climate
```

## Configuration

The following table lists the configurable parameters and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.replicaCount` | Number of API replicas | `3` |
| `api.image.repository` | API image repository | `ghcr.io/dogaaydinn/geo_sentiment_climate` |
| `api.image.tag` | API image tag | `latest` |
| `api.autoscaling.enabled` | Enable HPA | `true` |
| `api.autoscaling.minReplicas` | Minimum replicas | `3` |
| `api.autoscaling.maxReplicas` | Maximum replicas | `20` |
| `postgresql.enabled` | Enable PostgreSQL | `true` |
| `redis.enabled` | Enable Redis | `true` |
| `ingress.enabled` | Enable ingress | `true` |
| `monitoring.prometheus.enabled` | Enable Prometheus metrics | `true` |

## Examples

### Production deployment

```yaml
# values-production.yaml
api:
  replicaCount: 5
  image:
    tag: "2.0.0"

  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "8Gi"
      cpu: "4000m"

  autoscaling:
    minReplicas: 5
    maxReplicas: 50

postgresql:
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"

ingress:
  hosts:
    - host: api.geo-climate.com
      paths:
        - path: /
          pathType: Prefix
```

### Development deployment

```yaml
# values-dev.yaml
api:
  replicaCount: 1

  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

  autoscaling:
    enabled: false

postgresql:
  primary:
    persistence:
      size: 10Gi

ingress:
  hosts:
    - host: api.dev.geo-climate.com
      paths:
        - path: /
          pathType: Prefix
```

## Monitoring

The chart includes Prometheus metrics exposition on `/metrics` endpoint.

Access metrics:
```bash
kubectl port-forward -n geo-climate svc/geo-climate-api-service 9090:9090
curl http://localhost:9090/metrics
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n geo-climate
kubectl describe pod <pod-name> -n geo-climate
kubectl logs <pod-name> -n geo-climate
```

### Check services
```bash
kubectl get svc -n geo-climate
kubectl get ingress -n geo-climate
```

### Check HPA
```bash
kubectl get hpa -n geo-climate
kubectl describe hpa geo-climate-api-hpa -n geo-climate
```

## License

Apache 2.0
