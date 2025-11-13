"""E2E test helper classes and functions."""
import requests
import time
from typing import Dict, Any, Optional, List


class E2ETestClient:
    """E2E test client for API testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for API to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.get("/health/ready")
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def get(self, path: str, **kwargs) -> requests.Response:
        """Make GET request."""
        url = f"{self.base_url}{path}"
        return self.session.get(url, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        """Make POST request."""
        url = f"{self.base_url}{path}"
        return self.session.post(url, **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        """Make PUT request."""
        url = f"{self.base_url}{path}"
        return self.session.put(url, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """Make DELETE request."""
        url = f"{self.base_url}{path}"
        return self.session.delete(url, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.get("/health")
        response.raise_for_status()
        return response.json()

    def train_model(
        self,
        data_path: str,
        model_type: str = "xgboost",
        **params
    ) -> Dict[str, Any]:
        """Trigger model training (if endpoint exists)."""
        payload = {
            "data_path": data_path,
            "model_type": model_type,
            **params
        }
        response = self.post("/train", json=payload)
        response.raise_for_status()
        return response.json()

    def predict(self, data: Dict[str, Any], **params) -> Dict[str, Any]:
        """Make prediction."""
        payload = {"data": data, **params}
        response = self.post("/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def batch_predict(
        self,
        data_list: List[Dict[str, Any]],
        **params
    ) -> Dict[str, Any]:
        """Make batch prediction."""
        payload = {"data": data_list, **params}
        response = self.post("/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()

    def list_models(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict]:
        """List available models."""
        params = {}
        if model_name:
            params["model_name"] = model_name
        if stage:
            params["stage"] = stage

        response = self.get("/models", params=params)
        response.raise_for_status()
        return response.json()

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        response = self.get(f"/models/{model_id}")
        response.raise_for_status()
        return response.json()

    def promote_model(self, model_id: str, stage: str) -> Dict[str, Any]:
        """Promote model to new stage."""
        response = self.post(
            f"/models/{model_id}/promote",
            params={"new_stage": stage}
        )
        response.raise_for_status()
        return response.json()

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        response = self.get("/metrics")
        response.raise_for_status()
        return response.json()


class ServiceHealthChecker:
    """Helper to check health of multiple services."""

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url

    def check_service(self, port: int, path: str = "/") -> bool:
        """Check if a service is healthy."""
        try:
            url = f"{self.base_url}:{port}{path}"
            response = requests.get(url, timeout=5)
            return response.status_code in [200, 404]  # 404 means service is up
        except:
            return False

    def check_all_services(self) -> Dict[str, bool]:
        """Check all common services."""
        services = {
            "api": (8000, "/health"),
            "postgres": (5432, "/"),  # Will fail but that's ok
            "redis": (6379, "/"),      # Will fail but that's ok
            "mlflow": (5000, "/"),
            "prometheus": (9090, "/-/healthy"),
            "grafana": (3000, "/api/health"),
        }

        results = {}
        for name, (port, path) in services.items():
            results[name] = self.check_service(port, path)

        return results


def create_sample_e2e_data(num_samples: int = 10) -> List[Dict[str, Any]]:
    """Create sample data for E2E testing."""
    import random

    data = []
    for i in range(num_samples):
        data.append({
            "pm25": random.uniform(10, 50),
            "temperature": random.uniform(15, 30),
            "humidity": random.uniform(40, 80),
            "wind_speed": random.uniform(0, 20),
            "co": random.uniform(0.1, 5.0),
            "no2": random.uniform(10, 80),
            "o3": random.uniform(20, 100)
        })

    return data


def measure_response_time(client: E2ETestClient, endpoint: str, num_requests: int = 10) -> Dict[str, float]:
    """Measure response time for an endpoint."""
    times = []

    for _ in range(num_requests):
        start = time.time()
        try:
            client.get(endpoint)
            duration = (time.time() - start) * 1000  # Convert to ms
            times.append(duration)
        except:
            pass

    if not times:
        return {"min": 0, "max": 0, "avg": 0, "p95": 0}

    times.sort()
    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
        "p95": times[int(len(times) * 0.95)] if times else 0
    }
