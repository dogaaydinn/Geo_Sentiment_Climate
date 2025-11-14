"""
Main client for Geo Climate SDK.
"""

import requests
from typing import Dict, Any, List, Optional
import json
import time
from .exceptions import (
    GeoClimateError,
    AuthenticationError,
    RateLimitError,
    ValidationError
)


class GeoClimateClient:
    """
    Official Python client for Geo Climate API.

    Provides easy access to air quality predictions and model explanations.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.geo-climate.com",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Geo Climate client.

        Args:
            api_key: Your API key
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json',
            'User-Agent': f'geo-climate-python-sdk/1.0.0'
        })

    def predict(
        self,
        data: Dict[str, Any],
        model_id: Optional[str] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make a prediction.

        Args:
            data: Input features as dictionary
            model_id: Specific model to use (optional)
            explain: Whether to include explanation

        Returns:
            Prediction result with optional explanation

        Example:
            >>> data = {
            ...     "pm25": 35.5,
            ...     "temperature": 72,
            ...     "humidity": 65,
            ...     "wind_speed": 10
            ... }
            >>> result = client.predict(data)
            >>> print(result['prediction'])
        """
        endpoint = f"{self.base_url}/predict"

        payload = {"data": data}
        if model_id:
            payload["model_id"] = model_id

        response = self._request('POST', endpoint, json=payload)

        # Add explanation if requested
        if explain and response.get('prediction'):
            try:
                explanation = self.explain_prediction(data, model_id)
                response['explanation'] = explanation
            except Exception as e:
                response['explanation_error'] = str(e)

        return response

    def batch_predict(
        self,
        data_list: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Make batch predictions.

        Args:
            data_list: List of input dictionaries
            model_id: Specific model to use (optional)
            batch_size: Batch size for processing

        Returns:
            Batch prediction results

        Example:
            >>> data_list = [
            ...     {"pm25": 35.5, "temp": 72},
            ...     {"pm25": 42.0, "temp": 68}
            ... ]
            >>> results = client.batch_predict(data_list)
        """
        endpoint = f"{self.base_url}/predict/batch"

        payload = {
            "data": data_list,
            "batch_size": batch_size
        }
        if model_id:
            payload["model_id"] = model_id

        return self._request('POST', endpoint, json=payload)

    def explain_prediction(
        self,
        data: Dict[str, Any],
        model_id: Optional[str] = None,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Get explanation for prediction.

        Args:
            data: Input features
            model_id: Model to explain
            method: Explanation method ('shap' or 'lime')

        Returns:
            Explanation dictionary

        Example:
            >>> explanation = client.explain_prediction(data, method="shap")
            >>> print(explanation['top_features'])
        """
        endpoint = f"{self.base_url}/explain"

        payload = {
            "data": data,
            "method": method
        }
        if model_id:
            payload["model_id"] = model_id

        return self._request('POST', endpoint, json=payload)

    def list_models(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models.

        Args:
            model_name: Filter by model name
            stage: Filter by stage (dev/staging/production)

        Returns:
            List of model information

        Example:
            >>> models = client.list_models(stage="production")
            >>> for model in models:
            ...     print(model['model_id'], model['metrics'])
        """
        endpoint = f"{self.base_url}/models"

        params = {}
        if model_name:
            params["model_name"] = model_name
        if stage:
            params["stage"] = stage

        return self._request('GET', endpoint, params=params)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed model information.

        Args:
            model_id: Model ID

        Returns:
            Model information dictionary
        """
        endpoint = f"{self.base_url}/models/{model_id}"
        return self._request('GET', endpoint)

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.

        Returns:
            Health status
        """
        endpoint = f"{self.base_url}/health"
        return self._request('GET', endpoint)

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Usage statistics
        """
        endpoint = f"{self.base_url}/usage"
        return self._request('GET', endpoint)

    def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after}s"
                    )

                # Handle authentication errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")

                # Handle validation errors
                if response.status_code == 400:
                    error_msg = response.json().get('error', 'Validation error')
                    raise ValidationError(error_msg)

                # Raise for other HTTP errors
                response.raise_for_status()

                return response.json()

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        # All retries failed
        raise GeoClimateError(f"Request failed after {self.max_retries} attempts: {last_exception}")

    def __repr__(self):
        return f"GeoClimateClient(base_url='{self.base_url}')"
