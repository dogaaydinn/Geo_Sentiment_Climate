# Load Testing Guide

This directory contains load testing scripts using [Locust](https://locust.io/).

## Prerequisites

```bash
pip install locust
```

## Running Load Tests

### Quick Start (Web UI)

```bash
# Start Locust with web interface
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Then open http://localhost:8089 in your browser
# Enter number of users and spawn rate
```

### Headless Mode (CI/CD)

#### Basic Load Test (10 users, 2 minutes)
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 10 \
    --spawn-rate 2 \
    --run-time 2m \
    --headless
```

#### Medium Load Test (50 users, 5 minutes)
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --html=reports/load_test_50users.html
```

#### High Load Test (100 users, 5 minutes)
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --html=reports/load_test_100users.html \
    --csv=reports/load_test_100users
```

#### Stress Test (200 users, 10 minutes)
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 200 \
    --spawn-rate 20 \
    --run-time 10m \
    --headless \
    --html=reports/stress_test_200users.html \
    --csv=reports/stress_test_200users
```

### Using Specific User Classes

#### Realistic Users Only
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 30 \
    --spawn-rate 3 \
    --run-time 5m \
    --headless \
    RealisticUser
```

#### Stress Test with Aggressive Users
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 20 \
    --run-time 3m \
    --headless \
    APIStressUser
```

## User Classes

### GeoClimateUser (Default)
- Simulates normal API usage
- Wait time: 1-3 seconds between requests
- Tasks:
  - Health check (50% of requests)
  - List models (25%)
  - Get metrics (15%)
  - Make prediction (10%)
  - Check API docs (5%)

### APIStressUser
- Simulates aggressive/stress testing
- Wait time: 0.1-0.5 seconds (very fast)
- Tasks:
  - Rapid health checks
  - Rapid predictions

### RealisticUser
- Simulates realistic user behavior
- Wait time: 2-8 seconds
- Tasks:
  - Browse documentation and explore
  - Check available models
  - Make predictions
  - Check system metrics

## Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Avg Response Time | < 100ms | < 200ms | > 500ms |
| 95th Percentile | < 200ms | < 500ms | > 1000ms |
| Requests/sec | > 100 | > 50 | < 20 |
| Error Rate | < 1% | < 5% | > 10% |

## Interpreting Results

### Key Metrics

- **RPS (Requests Per Second)**: Higher is better. Indicates system throughput.
- **Response Time**: Lower is better. Time to complete a request.
- **Failure Rate**: Should be near 0%. High failure rate indicates issues.
- **User Count**: Maximum concurrent users system can handle.

### Percentiles

- **P50 (Median)**: Half of requests are faster than this
- **P95**: 95% of requests are faster than this (important for user experience)
- **P99**: 99% of requests are faster than this (catches outliers)

### Warning Signs

⚠️ **High Response Time**: > 500ms average indicates performance issues
⚠️ **High Failure Rate**: > 5% indicates stability issues
⚠️ **Low RPS**: < 50 RPS indicates throughput bottleneck
⚠️ **High P95/P99**: Large gap between avg and P95/P99 indicates inconsistency

## Example Output

```
==========================================
🏁 LOAD TEST COMPLETE
==========================================

📊 RESULTS SUMMARY:
  Total requests: 5,234
  Total failures: 23
  Failure rate: 0.44%
  Average response time: 87.23ms
  Min response time: 12.45ms
  Max response time: 1,234.56ms
  Requests per second: 87.23

📈 PERCENTILES:
  50th percentile: 45.12ms
  75th percentile: 89.34ms
  90th percentile: 156.78ms
  95th percentile: 234.56ms
  99th percentile: 567.89ms
==========================================
```

## Advanced Usage

### Distributed Load Testing

Run on multiple machines for higher load:

```bash
# Master node
locust -f tests/load/locustfile.py --master --host=http://localhost:8000

# Worker nodes (run on separate machines)
locust -f tests/load/locustfile.py --worker --master-host=<master-ip>
```

### Custom Configuration

Create a `locust.conf` file:

```ini
host = http://localhost:8000
users = 100
spawn-rate = 10
run-time = 5m
headless = true
```

Then run:
```bash
locust -f tests/load/locustfile.py --config=locust.conf
```

## Troubleshooting

### Connection Errors
- Ensure API is running: `curl http://localhost:8000/health`
- Check firewall settings
- Verify host URL is correct

### High Failure Rates
- Check API logs for errors
- Reduce number of users/spawn rate
- Increase wait times between requests

### Low RPS
- Check system resources (CPU, memory)
- Profile the API for bottlenecks
- Consider horizontal scaling

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run Load Tests
  run: |
    docker-compose up -d
    sleep 10
    locust -f tests/load/locustfile.py \
      --host=http://localhost:8000 \
      --users 50 \
      --spawn-rate 5 \
      --run-time 2m \
      --headless \
      --html=reports/load_test.html
    docker-compose down
```

## References

- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Best Practices](https://locust.io/docs/latest/writing-a-locustfile.html)
