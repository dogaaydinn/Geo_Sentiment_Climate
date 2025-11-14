# Geo Climate Python SDK

Official Python client for the Geo Sentiment Climate API.

## Installation

```bash
pip install geo-climate-sdk
```

## Quick Start

```python
from geo_climate_sdk import GeoClimateClient

# Initialize client
client = GeoClimateClient(api_key="your-api-key-here")

# Make a prediction
data = {
    "pm25": 35.5,
    "temperature": 72,
    "humidity": 65,
    "wind_speed": 10,
    "pressure": 1013
}

result = client.predict(data)
print(f"Predicted AQI: {result['prediction']}")

# Get explanation
explanation = client.explain_prediction(data, method="shap")
print(f"Top features: {explanation['top_features']}")
```

## Features

- **Simple API**: Easy-to-use interface for air quality predictions
- **Batch Predictions**: Process multiple inputs efficiently
- **Model Explanations**: Understand predictions with SHAP/LIME
- **Automatic Retries**: Built-in retry logic with exponential backoff
- **Type Hints**: Full type hint support for better IDE integration
- **Error Handling**: Comprehensive exception hierarchy

## Usage Examples

### Single Prediction

```python
# Make a prediction with explanation
result = client.predict(
    data={"pm25": 35.5, "temp": 72},
    explain=True
)

print(result['prediction'])
print(result['explanation'])
```

### Batch Predictions

```python
# Process multiple inputs
data_list = [
    {"pm25": 35.5, "temp": 72},
    {"pm25": 42.0, "temp": 68},
    {"pm25": 28.0, "temp": 75}
]

results = client.batch_predict(data_list)
for pred in results['predictions']:
    print(pred)
```

### Model Management

```python
# List available models
models = client.list_models(stage="production")
for model in models:
    print(f"{model['model_id']}: {model['metrics']}")

# Get specific model info
model_info = client.get_model_info("model-123")
print(model_info)
```

### Health Check

```python
# Check API health
health = client.health_check()
print(f"Status: {health['status']}")
```

## Configuration

```python
client = GeoClimateClient(
    api_key="your-api-key",
    base_url="https://api.geo-climate.com",  # Optional
    timeout=30,  # Request timeout in seconds
    max_retries=3  # Maximum retry attempts
)
```

## Error Handling

```python
from geo_climate_sdk import (
    AuthenticationError,
    RateLimitError,
    ValidationError
)

try:
    result = client.predict(data)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except ValidationError as e:
    print(f"Invalid input: {e}")
```

## API Reference

### GeoClimateClient

**Methods:**

- `predict(data, model_id=None, explain=False)` - Make a prediction
- `batch_predict(data_list, model_id=None, batch_size=1000)` - Batch predictions
- `explain_prediction(data, model_id=None, method="shap")` - Explain prediction
- `list_models(model_name=None, stage=None)` - List available models
- `get_model_info(model_id)` - Get model details
- `health_check()` - Check API health
- `get_usage_stats()` - Get usage statistics

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black geo_climate_sdk/

# Type checking
mypy geo_climate_sdk/
```

## License

Apache 2.0

## Support

- **Documentation**: https://docs.geo-climate.com
- **Issues**: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
- **Email**: dogaa882@gmail.com
