"""Helper classes and functions for API testing."""
from fastapi.testclient import TestClient
from typing import Dict, Any, Optional, List
import json


class APITestHelper:
    """Helper class for API testing."""

    def __init__(self, client: TestClient):
        self.client = client

    def make_prediction(
        self,
        data: Dict[str, Any],
        model_id: Optional[str] = None,
        expected_status: int = 200
    ) -> Dict:
        """Make a prediction request."""
        payload = {"data": data}
        if model_id:
            payload["model_id"] = model_id

        response = self.client.post("/predict", json=payload)
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
        return response.json()

    def batch_predict(
        self,
        data_list: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict:
        """Make a batch prediction request."""
        payload = {
            "data": data_list,
            "batch_size": batch_size
        }
        if model_id:
            payload["model_id"] = model_id

        response = self.client.post("/predict/batch", json=payload)
        assert response.status_code == 200, f"Batch predict failed: {response.text}"
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

        response = self.client.get("/models", params=params)
        assert response.status_code == 200, f"List models failed: {response.text}"
        return response.json()

    def get_model_info(self, model_id: str) -> Dict:
        """Get model information."""
        response = self.client.get(f"/models/{model_id}")
        assert response.status_code == 200, f"Get model info failed: {response.text}"
        return response.json()

    def promote_model(self, model_id: str, stage: str) -> Dict:
        """Promote model to new stage."""
        response = self.client.post(
            f"/models/{model_id}/promote",
            params={"new_stage": stage}
        )
        assert response.status_code == 200, f"Promote model failed: {response.text}"
        return response.json()

    def health_check(self) -> Dict:
        """Check API health."""
        response = self.client.get("/health")
        assert response.status_code == 200, f"Health check failed: {response.text}"
        return response.json()

    def get_metrics(self) -> Dict:
        """Get API metrics."""
        response = self.client.get("/metrics")
        assert response.status_code == 200, f"Get metrics failed: {response.text}"
        return response.json()


def assert_valid_prediction_response(response: Dict):
    """Assert that a prediction response has valid structure."""
    assert "predictions" in response, "Response missing 'predictions' field"
    assert isinstance(response["predictions"], list), "predictions should be a list"
    assert len(response["predictions"]) > 0, "predictions should not be empty"

    if "model_id" in response:
        assert isinstance(response["model_id"], str), "model_id should be a string"

    if "inference_time_ms" in response:
        assert isinstance(response["inference_time_ms"], (int, float)), "inference_time_ms should be numeric"
        assert response["inference_time_ms"] > 0, "inference_time_ms should be positive"


def assert_valid_model_info(model_info: Dict):
    """Assert that model info has valid structure."""
    required_fields = ["model_id", "model_name", "model_type"]
    for field in required_fields:
        assert field in model_info, f"Model info missing required field: {field}"

    if "metrics" in model_info:
        assert isinstance(model_info["metrics"], dict), "metrics should be a dictionary"

    if "stage" in model_info:
        assert model_info["stage"] in ["dev", "staging", "production"], f"Invalid stage: {model_info['stage']}"


def create_sample_prediction_input(include_all_features: bool = False) -> Dict[str, Any]:
    """Create sample input for prediction testing."""
    base_input = {
        "pm25": 25.5,
        "temperature": 20.0,
        "humidity": 60.0
    }

    if include_all_features:
        base_input.update({
            "co": 2.5,
            "no2": 45.0,
            "o3": 55.0,
            "so2": 10.0,
            "wind_speed": 5.0
        })

    return base_input
