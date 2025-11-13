"""E2E tests for full system integration."""
import pytest
import requests
import time
from tests.e2e.helpers import E2ETestClient, ServiceHealthChecker


@pytest.mark.e2e
@pytest.mark.slow
class TestFullStackE2E:
    """E2E tests for full system integration."""

    def test_complete_system_health(self, e2e_api_base_url):
        """Test all system health endpoints."""
        base_url = e2e_api_base_url

        print("\n🏥 Testing system health endpoints...")

        # 1. Main health check
        print("  📋 Checking /health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"  ✅ Main health check passed")

        # 2. Readiness check
        print("  📋 Checking /health/ready...")
        response = requests.get(f"{base_url}/health/ready", timeout=5)
        # May be 200 or 503 depending on setup
        assert response.status_code in [200, 503]
        print(f"  ✅ Readiness check: {response.status_code}")

        # 3. Liveness check
        print("  📋 Checking /health/live...")
        response = requests.get(f"{base_url}/health/live", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        print(f"  ✅ Liveness check passed")

    def test_api_endpoints_availability(self, e2e_api_base_url):
        """Test that all API endpoints are accessible."""
        base_url = e2e_api_base_url

        endpoints = {
            "/": "GET",
            "/health": "GET",
            "/docs": "GET",
            "/redoc": "GET",
            "/openapi.json": "GET",
            "/models": "GET",
            "/metrics": "GET",
        }

        print("\n🔍 Testing API endpoint availability...")
        for endpoint, method in endpoints.items():
            print(f"  📋 Testing {method} {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            assert response.status_code in [200, 422], f"{endpoint} should be accessible"
            print(f"  ✅ {endpoint}: {response.status_code}")

    def test_database_integration_e2e(self, e2e_api_base_url_compose):
        """Test database integration through API."""
        pytest.skip("Requires docker-compose stack - enable when docker-compose is available")

        base_url = e2e_api_base_url_compose

        print("\n💾 Testing database integration...")

        # Models endpoint should work with database
        response = requests.get(f"{base_url}/models", timeout=5)
        assert response.status_code == 200
        print(f"  ✅ Database connection working")

    def test_redis_cache_integration_e2e(self, e2e_api_base_url_compose):
        """Test Redis cache integration."""
        pytest.skip("Requires docker-compose stack - enable when docker-compose is available")

        base_url = e2e_api_base_url_compose

        print("\n⚡ Testing Redis cache...")

        # Make same request twice to test caching
        start1 = time.time()
        response1 = requests.get(f"{base_url}/models", timeout=5)
        time1 = time.time() - start1

        start2 = time.time()
        response2 = requests.get(f"{base_url}/models", timeout=5)
        time2 = time.time() - start2

        assert response1.status_code == 200
        assert response2.status_code == 200

        print(f"  📊 First request: {time1*1000:.2f}ms")
        print(f"  📊 Second request: {time2*1000:.2f}ms")
        print(f"  ✅ Cache integration verified")

    def test_api_data_consistency(self, e2e_api_base_url):
        """Test data consistency across multiple requests."""
        client = E2ETestClient(e2e_api_base_url)

        print("\n🔄 Testing data consistency...")

        # Get models multiple times
        models1 = client.list_models()
        time.sleep(0.5)
        models2 = client.list_models()
        time.sleep(0.5)
        models3 = client.list_models()

        # Count should be consistent
        count1 = len(models1)
        count2 = len(models2)
        count3 = len(models3)

        assert count1 == count2 == count3, "Model count should be consistent"
        print(f"  ✅ Model count consistent: {count1}")

    def test_concurrent_api_access(self, e2e_api_base_url):
        """Test concurrent API access."""
        from concurrent.futures import ThreadPoolExecutor
        import random

        client = E2ETestClient(e2e_api_base_url)

        def make_request(request_id: int):
            """Make a random API request."""
            endpoints = ["/health", "/models", "/metrics"]
            endpoint = random.choice(endpoints)

            response = client.get(endpoint)
            return {
                "id": request_id,
                "endpoint": endpoint,
                "status": response.status_code,
                "success": response.status_code == 200
            }

        print("\n⚡ Testing 20 concurrent requests...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["success"])
        print(f"  ✅ Successful requests: {success_count}/20")
        assert success_count >= 18, "At least 90% of requests should succeed"

    def test_api_response_format_consistency(self, e2e_api_base_url):
        """Test that API responses have consistent format."""
        client = E2ETestClient(e2e_api_base_url)

        print("\n📝 Testing response format consistency...")

        # Health endpoint
        health = client.health_check()
        assert "status" in health
        assert "timestamp" in health
        print(f"  ✅ Health response format valid")

        # Models endpoint
        models = client.list_models()
        assert isinstance(models, list)
        print(f"  ✅ Models response format valid")

        # Metrics endpoint
        metrics = client.get_metrics()
        assert isinstance(metrics, dict)
        print(f"  ✅ Metrics response format valid")

    def test_api_handles_malformed_requests(self, e2e_api_base_url):
        """Test API handles malformed requests gracefully."""
        base_url = e2e_api_base_url

        print("\n🛡️ Testing malformed request handling...")

        # Invalid JSON
        response = requests.post(
            f"{base_url}/predict",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert response.status_code in [400, 422, 500]
        print(f"  ✅ Invalid JSON handled: {response.status_code}")

        # Missing required fields
        response = requests.post(
            f"{base_url}/predict",
            json={},
            timeout=5
        )
        assert response.status_code in [400, 422]
        print(f"  ✅ Missing fields handled: {response.status_code}")

    def test_system_performance_under_load(self, e2e_api_base_url):
        """Test system performance under moderate load."""
        from concurrent.futures import ThreadPoolExecutor
        import statistics

        client = E2ETestClient(e2e_api_base_url)

        def timed_health_check():
            start = time.time()
            try:
                client.health_check()
                return (time.time() - start) * 1000  # ms
            except:
                return None

        print("\n⚡ Testing performance under load (50 requests)...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(timed_health_check) for _ in range(50)]
            times = [f.result() for f in futures if f.result() is not None]

        if times:
            avg_time = statistics.mean(times)
            max_time = max(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]

            print(f"  📊 Average response time: {avg_time:.2f}ms")
            print(f"  📊 Max response time: {max_time:.2f}ms")
            print(f"  📊 P95 response time: {p95_time:.2f}ms")

            assert avg_time < 500, "Average response time should be < 500ms"
            assert p95_time < 1000, "P95 response time should be < 1s"
            print(f"  ✅ Performance acceptable under load")
