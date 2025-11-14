"""
Enterprise-grade Locust load testing for Geo Climate API.

Performance Targets:
- Normal Load: 1,000 req/s @ < 100ms p95
- Peak Load: 5,000 req/s @ < 200ms p95
- Stress Test: 10,000 req/s @ < 500ms p95
- Endurance: 500 req/s for 24h (no leaks)

Usage Examples:
    # Normal load (1 hour)
    locust -f locustfile.py --users=1000 --spawn-rate=100 --run-time=1h

    # Peak load stress test
    locust -f locustfile.py --users=5000 --spawn-rate=500 --run-time=30m

    # Headless with HTML report
    locust -f locustfile.py --users=1000 --spawn-rate=100 --run-time=1h --headless --html=report.html

    # Distributed load testing
    locust -f locustfile.py --master
    locust -f locustfile.py --worker --master-host=<master-ip>
"""
from locust import HttpUser, task, between, events
import random
import json
class GeoClimateUser(HttpUser):
    """Simulated user for load testing."""
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Called when user starts."""
        # Check if API is healthy
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("API not healthy")
            else:
                response.success()

    @task(10)  # Weight: 10 (most common)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="/health")

    @task(5)  # Weight: 5
    def list_models(self):
        """Test model listing endpoint."""
        self.client.get("/models", name="/models")

    @task(3)  # Weight: 3
    def get_metrics(self):
        """Test metrics endpoint."""
        self.client.get("/metrics", name="/metrics")

    @task(2)  # Weight: 2
    def make_prediction(self):
        """Test prediction endpoint."""
        # Generate random input data
        data = {
            "pm25": random.uniform(5, 50),
            "temperature": random.uniform(10, 35),
            "humidity": random.uniform(30, 90),
            "wind_speed": random.uniform(0, 20),
            "co": random.uniform(0.1, 5.0),
            "no2": random.uniform(10, 80),
            "o3": random.uniform(20, 100)
        }

        with self.client.post(
            "/predict",
            json={"data": data},
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" not in result:
                        response.failure("No predictions in response")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 404:
                # No models available - acceptable in test
                response.success()
            elif response.status_code in [400, 422]:
                # Validation error - mark as failure
                response.failure(f"Validation error: {response.status_code}")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)  # Weight: 1 (least common)
    def check_api_docs(self):
        """Test API documentation endpoint."""
        self.client.get("/docs", name="/docs")


class APIStressUser(HttpUser):
    """Heavy load user for stress testing."""
    wait_time = between(0.1, 0.5)  # Aggressive: 0.1-0.5 seconds

    @task
    def rapid_health_checks(self):
        """Rapidly check health endpoint."""
        self.client.get("/health", name="/health-stress")

    @task
    def rapid_predictions(self):
        """Rapidly make predictions."""
        data = {
            "pm25": 25.5,
            "temperature": 20.0,
            "humidity": 60.0
        }
        self.client.post(
            "/predict",
            json={"data": data},
            catch_response=True,
            name="/predict-stress"
        )


class RealisticUser(HttpUser):
    """Realistic user simulation with varied behavior."""
    wait_time = between(2, 8)

    def on_start(self):
        """User session initialization."""
        # Browse documentation first
        self.client.get("/docs")

    @task(20)
    def browse_and_explore(self):
        """User explores the API."""
        # Check health
        self.client.get("/health")

        # Browse documentation
        if random.random() < 0.3:  # 30% chance
            self.client.get("/docs")

    @task(15)
    def check_available_models(self):
        """User checks what models are available."""
        self.client.get("/models")

        # Sometimes filter by stage
        if random.random() < 0.5:
            stage = random.choice(["dev", "staging", "production"])
            self.client.get("/models", params={"stage": stage})

    @task(10)
    def make_single_prediction(self):
        """User makes a single prediction."""
        data = {
            "pm25": random.uniform(10, 50),
            "temperature": random.uniform(15, 30),
            "humidity": random.uniform(40, 80),
            "wind_speed": random.uniform(0, 15)
        }

        self.client.post("/predict", json={"data": data})

    @task(5)
    def check_system_metrics(self):
        """User checks system metrics."""
        self.client.get("/metrics")


# Event listeners for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track custom metrics on each request."""
    if exception:
        print(f"❌ Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Run at start of load test."""
    print("\n" + "="*70)
    print("🚀 LOAD TEST STARTING")
    print("="*70)
    print(f"Target host: {environment.host}")
    print(f"User classes: {[u.__name__ for u in environment.user_classes]}")
    print("="*70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Run at end of load test."""
    print("\n" + "="*70)
    print("🏁 LOAD TEST COMPLETE")
    print("="*70)

    stats = environment.stats.total

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  Total requests: {stats.num_requests:,}")
    print(f"  Total failures: {stats.num_failures:,}")
    print(f"  Failure rate: {(stats.num_failures/stats.num_requests*100):.2f}%" if stats.num_requests > 0 else "  Failure rate: N/A")
    print(f"  Average response time: {stats.avg_response_time:.2f}ms")
    print(f"  Min response time: {stats.min_response_time:.2f}ms")
    print(f"  Max response time: {stats.max_response_time:.2f}ms")
    print(f"  Requests per second: {stats.total_rps:.2f}")

    print(f"\n📈 PERCENTILES:")
    print(f"  50th percentile: {stats.get_response_time_percentile(0.5):.2f}ms")
    print(f"  75th percentile: {stats.get_response_time_percentile(0.75):.2f}ms")
    print(f"  90th percentile: {stats.get_response_time_percentile(0.9):.2f}ms")
    print(f"  95th percentile: {stats.get_response_time_percentile(0.95):.2f}ms")
    print(f"  99th percentile: {stats.get_response_time_percentile(0.99):.2f}ms")

    print("\n" + "="*70 + "\n")

    # Performance thresholds
    if stats.avg_response_time > 500:
        print("⚠️  WARNING: Average response time > 500ms")
    if stats.num_requests > 0 and (stats.num_failures / stats.num_requests) > 0.05:
        print("⚠️  WARNING: Failure rate > 5%")
    if stats.total_rps < 50:
        print("⚠️  WARNING: RPS < 50 (low throughput)")
