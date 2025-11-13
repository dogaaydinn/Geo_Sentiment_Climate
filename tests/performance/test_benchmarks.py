"""Performance benchmark tests."""
import pytest
import time
import statistics
from typing import Callable


class PerformanceBenchmark:
    """Helper class for performance benchmarking."""

    @staticmethod
    def measure_execution_time(
        func: Callable,
        iterations: int = 100
    ) -> dict:
        """Measure function execution time over multiple iterations."""
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "p95": sorted(times)[int(len(times) * 0.95)],
            "p99": sorted(times)[int(len(times) * 0.99)]
        }


@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_api_health_endpoint_benchmark(self, test_client):
        """Benchmark health endpoint performance."""
        benchmark = PerformanceBenchmark()

        def call_health():
            response = test_client.get("/health")
            assert response.status_code == 200

        print("\n📊 Benchmarking /health endpoint (1000 iterations)...")
        results = benchmark.measure_execution_time(call_health, iterations=1000)

        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Median: {results['median']:.2f}ms")
        print(f"  P95: {results['p95']:.2f}ms")
        print(f"  P99: {results['p99']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")

        # Performance target: < 50ms mean
        assert results['mean'] < 50.0, f"Mean response time {results['mean']:.2f}ms exceeds 50ms target"
        print(f"✅ Health endpoint performance acceptable")

    def test_models_list_endpoint_benchmark(self, test_client):
        """Benchmark models listing endpoint."""
        benchmark = PerformanceBenchmark()

        def call_models():
            response = test_client.get("/models")
            assert response.status_code == 200

        print("\n📊 Benchmarking /models endpoint (500 iterations)...")
        results = benchmark.measure_execution_time(call_models, iterations=500)

        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Median: {results['median']:.2f}ms")
        print(f"  P95: {results['p95']:.2f}ms")
        print(f"  P99: {results['p99']:.2f}ms")

        # Performance target: < 100ms mean
        assert results['mean'] < 100.0, f"Mean response time {results['mean']:.2f}ms exceeds 100ms target"
        print(f"✅ Models list performance acceptable")

    def test_metrics_endpoint_benchmark(self, test_client):
        """Benchmark metrics endpoint."""
        benchmark = PerformanceBenchmark()

        def call_metrics():
            response = test_client.get("/metrics")
            assert response.status_code == 200

        print("\n📊 Benchmarking /metrics endpoint (500 iterations)...")
        results = benchmark.measure_execution_time(call_metrics, iterations=500)

        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  P95: {results['p95']:.2f}ms")

        # Performance target: < 50ms mean
        assert results['mean'] < 50.0, f"Mean response time {results['mean']:.2f}ms exceeds 50ms target"
        print(f"✅ Metrics endpoint performance acceptable")

    def test_concurrent_requests_performance(self, test_client):
        """Test performance under concurrent load."""
        from concurrent.futures import ThreadPoolExecutor

        def make_request():
            start = time.perf_counter()
            response = test_client.get("/health")
            duration = (time.perf_counter() - start) * 1000
            return duration if response.status_code == 200 else None

        print("\n📊 Testing concurrent performance (50 requests, 10 workers)...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            times = [f.result() for f in futures if f.result() is not None]

        if times:
            mean_time = statistics.mean(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]

            print(f"  Mean: {mean_time:.2f}ms")
            print(f"  P95: {p95_time:.2f}ms")
            print(f"  Success rate: {len(times)}/50")

            assert mean_time < 100.0, "Concurrent mean response time should be < 100ms"
            assert len(times) >= 45, "At least 90% of concurrent requests should succeed"
            print(f"✅ Concurrent performance acceptable")

    def test_api_throughput(self, test_client):
        """Measure API throughput (requests per second)."""
        print("\n📊 Measuring API throughput (10 seconds)...")

        request_count = 0
        start_time = time.time()
        test_duration = 10  # seconds

        while time.time() - start_time < test_duration:
            response = test_client.get("/health")
            if response.status_code == 200:
                request_count += 1

        duration = time.time() - start_time
        throughput = request_count / duration

        print(f"  Total requests: {request_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        # Target: > 100 requests per second
        assert throughput > 50, f"Throughput {throughput:.2f} req/s is below 50 req/s minimum"
        print(f"✅ Throughput acceptable")

    def test_response_time_consistency(self, test_client):
        """Test response time consistency (low variance)."""
        benchmark = PerformanceBenchmark()

        def call_health():
            response = test_client.get("/health")
            assert response.status_code == 200

        print("\n📊 Testing response time consistency (100 iterations)...")
        results = benchmark.measure_execution_time(call_health, iterations=100)

        variance_coefficient = (results['stdev'] / results['mean']) * 100 if results['mean'] > 0 else 0

        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Std Dev: {results['stdev']:.2f}ms")
        print(f"  Coefficient of Variation: {variance_coefficient:.2f}%")

        # Variance should be reasonable
        assert variance_coefficient < 100, "Response time variance is too high"
        print(f"✅ Response time consistency acceptable")
