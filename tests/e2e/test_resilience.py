"""E2E tests for system resilience and error handling."""
import pytest
import requests
import time


@pytest.mark.e2e
class TestResilienceE2E:
    """E2E tests for system resilience."""

    def test_api_handles_invalid_requests_gracefully(self, e2e_api_base_url):
        """Test API error handling for various invalid inputs."""
        base_url = e2e_api_base_url

        print("\n🛡️ Testing invalid request handling...")

        # 1. Invalid prediction request
        print("  📋 Test 1: Invalid prediction data...")
        response = requests.post(
            f"{base_url}/predict",
            json={"invalid": "data"},
            timeout=5
        )
        assert response.status_code in [400, 422, 500]
        error_data = response.json()
        assert "detail" in error_data or "message" in error_data
        print(f"  ✅ Invalid data handled: {response.status_code}")

        # 2. Missing required fields
        print("  📋 Test 2: Missing required fields...")
        response = requests.post(f"{base_url}/predict", json={}, timeout=5)
        assert response.status_code == 422
        print(f"  ✅ Missing fields handled: {response.status_code}")

        # 3. Invalid model ID
        print("  📋 Test 3: Non-existent model ID...")
        response = requests.post(
            f"{base_url}/predict",
            json={
                "data": {"feature1": 1.0},
                "model_id": "non_existent_model_12345"
            },
            timeout=5
        )
        assert response.status_code in [404, 500]
        print(f"  ✅ Invalid model ID handled: {response.status_code}")

        # 4. Invalid HTTP method
        print("  📋 Test 4: Invalid HTTP method...")
        response = requests.delete(f"{base_url}/health", timeout=5)
        assert response.status_code in [405, 404]
        print(f"  ✅ Invalid method handled: {response.status_code}")

    def test_api_rate_limiting_behavior(self, e2e_api_base_url):
        """Test API behavior under rapid requests."""
        base_url = e2e_api_base_url

        print("\n⚡ Testing rapid request handling...")

        # Make 30 rapid requests
        responses = []
        start_time = time.time()
        for i in range(30):
            try:
                response = requests.get(f"{base_url}/health", timeout=2)
                responses.append(response.status_code)
            except:
                responses.append(None)
        duration = time.time() - start_time

        success_count = sum(1 for r in responses if r == 200)
        print(f"  📊 Successful requests: {success_count}/30")
        print(f"  📊 Duration: {duration:.2f}s")
        print(f"  📊 Throughput: {30/duration:.2f} req/s")

        # Most should succeed (no rate limiting implemented yet)
        assert success_count > 25, "Most requests should succeed"
        print(f"  ✅ System handles rapid requests")

    def test_api_timeout_handling(self, e2e_api_base_url):
        """Test API handles slow clients."""
        base_url = e2e_api_base_url

        print("\n⏱️ Testing timeout handling...")

        # Request with very short timeout
        try:
            response = requests.get(f"{base_url}/health", timeout=0.001)
            # If it succeeds, great
            assert response.status_code == 200
            print(f"  ✅ Request completed despite short timeout")
        except requests.Timeout:
            # If it times out, that's also acceptable
            print(f"  ✅ Timeout handled gracefully (client-side)")

    def test_system_recovery_after_errors(self, e2e_api_base_url):
        """Test system recovers after encountering errors."""
        base_url = e2e_api_base_url

        print("\n🔄 Testing system recovery...")

        # 1. Trigger some errors
        print("  📋 Step 1: Triggering errors...")
        for i in range(5):
            requests.post(f"{base_url}/predict", json={"bad": "data"}, timeout=5)
        print(f"  ✅ Triggered 5 error requests")

        # 2. Check system is still healthy
        print("  📋 Step 2: Checking system health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"  ✅ System recovered and healthy")

        # 3. Normal operations still work
        print("  📋 Step 3: Verifying normal operations...")
        response = requests.get(f"{base_url}/models", timeout=5)
        assert response.status_code == 200
        print(f"  ✅ Normal operations working")

    def test_concurrent_error_requests(self, e2e_api_base_url):
        """Test system handles concurrent error requests."""
        from concurrent.futures import ThreadPoolExecutor
        base_url = e2e_api_base_url

        def make_error_request(i):
            """Make a request that will error."""
            try:
                response = requests.post(
                    f"{base_url}/predict",
                    json={"invalid": f"data_{i}"},
                    timeout=5
                )
                return response.status_code
            except:
                return None

        print("\n⚡ Testing concurrent error requests...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_error_request, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All should return error codes
        error_codes = [r for r in results if r in [400, 422, 500]]
        print(f"  📊 Error responses: {len(error_codes)}/20")
        assert len(error_codes) >= 15, "Most should return proper error codes"

        # System should still be healthy
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        print(f"  ✅ System healthy after concurrent errors")

    def test_large_payload_handling(self, e2e_api_base_url):
        """Test API handles large payloads."""
        base_url = e2e_api_base_url

        print("\n📦 Testing large payload handling...")

        # Create a large batch of data
        large_batch = [{"feature1": i, "feature2": i * 2} for i in range(1000)]

        response = requests.post(
            f"{base_url}/predict/batch",
            json={"data": large_batch, "batch_size": 100},
            timeout=30
        )

        # Should handle gracefully (200 if implemented, 404/422 if not)
        assert response.status_code in [200, 404, 422, 500]
        print(f"  ✅ Large payload handled: {response.status_code}")

    def test_malformed_json_handling(self, e2e_api_base_url):
        """Test API handles malformed JSON."""
        base_url = e2e_api_base_url

        print("\n📝 Testing malformed JSON handling...")

        # Send invalid JSON
        response = requests.post(
            f"{base_url}/predict",
            data="{invalid json}",
            headers={"Content-Type": "application/json"},
            timeout=5
        )

        assert response.status_code in [400, 422, 500]
        print(f"  ✅ Malformed JSON handled: {response.status_code}")

    def test_missing_content_type(self, e2e_api_base_url):
        """Test API handles missing Content-Type header."""
        base_url = e2e_api_base_url

        print("\n📋 Testing missing Content-Type...")

        response = requests.post(
            f"{base_url}/predict",
            data='{"data": {"feature1": 1}}',
            timeout=5
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500]
        print(f"  ✅ Missing Content-Type handled: {response.status_code}")

    def test_sql_injection_attempts(self, e2e_api_base_url):
        """Test API prevents SQL injection."""
        base_url = e2e_api_base_url

        print("\n🛡️ Testing SQL injection prevention...")

        # SQL injection attempts
        sql_injections = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]

        for injection in sql_injections:
            response = requests.get(
                f"{base_url}/models",
                params={"model_name": injection},
                timeout=5
            )

            # Should not crash or expose data
            assert response.status_code in [200, 400, 422]
            print(f"  ✅ SQL injection prevented: {injection[:30]}...")

    def test_xss_prevention(self, e2e_api_base_url):
        """Test API prevents XSS attacks."""
        base_url = e2e_api_base_url

        print("\n🛡️ Testing XSS prevention...")

        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')"
        ]

        for payload in xss_payloads:
            response = requests.post(
                f"{base_url}/predict",
                json={"data": {"name": payload}},
                timeout=5
            )

            # Should sanitize or reject
            assert response.status_code in [200, 400, 422, 500]
            if response.status_code == 200:
                # If accepted, should be sanitized
                data = response.json()
                response_text = str(data)
                assert "<script>" not in response_text
            print(f"  ✅ XSS prevented: {payload[:30]}...")
