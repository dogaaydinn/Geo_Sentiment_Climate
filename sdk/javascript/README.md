# Geo Climate JavaScript/TypeScript SDK

Official JavaScript/TypeScript client for the Geo Sentiment Climate API.

## Installation

```bash
npm install @geo-climate/sdk
# or
yarn add @geo-climate/sdk
```

## Quick Start

```typescript
import { GeoClimateClient } from '@geo-climate/sdk';

// Initialize client
const client = new GeoClimateClient({
  apiKey: 'your-api-key-here'
});

// Make a prediction
const result = await client.predict({
  pm25: 35.5,
  temperature: 72,
  humidity: 65,
  wind_speed: 10
});

console.log(`Predicted AQI: ${result.prediction}`);
```

## Features

- ✅ **TypeScript Support**: Full type definitions included
- ✅ **Automatic Retries**: Built-in retry logic with exponential backoff
- ✅ **Error Handling**: Comprehensive error types
- ✅ **Promise-based**: Modern async/await API
- ✅ **Lightweight**: Minimal dependencies
- ✅ **Browser & Node.js**: Works in both environments

## Usage Examples

### TypeScript

```typescript
import { GeoClimateClient, PredictionInput } from '@geo-climate/sdk';

const client = new GeoClimateClient({
  apiKey: process.env.GEO_CLIMATE_API_KEY!,
  baseURL: 'https://api.geo-climate.com',
  timeout: 30000
});

// Single prediction with explanation
const data: PredictionInput = {
  pm25: 35.5,
  temperature: 72,
  humidity: 65
};

const result = await client.predict(data, { explain: true });
console.log(result.prediction);
console.log(result.explanation);
```

### JavaScript

```javascript
const { GeoClimateClient } = require('@geo-climate/sdk');

const client = new GeoClimateClient({
  apiKey: 'your-api-key'
});

// Batch predictions
const dataList = [
  { pm25: 35.5, temp: 72 },
  { pm25: 42.0, temp: 68 },
  { pm25: 28.0, temp: 75 }
];

const results = await client.batchPredict(dataList);
console.log(results.predictions);
```

### React Example

```tsx
import React, { useState } from 'react';
import { GeoClimateClient } from '@geo-climate/sdk';

const client = new GeoClimateClient({
  apiKey: process.env.REACT_APP_API_KEY!
});

function AirQualityPredictor() {
  const [prediction, setPrediction] = useState<number | null>(null);

  const handlePredict = async () => {
    const result = await client.predict({
      pm25: 35.5,
      temperature: 72,
      humidity: 65
    });

    setPrediction(result.prediction);
  };

  return (
    <div>
      <button onClick={handlePredict}>Get Prediction</button>
      {prediction && <p>Predicted AQI: {prediction}</p>}
    </div>
  );
}
```

### Node.js Express Example

```javascript
const express = require('express');
const { GeoClimateClient } = require('@geo-climate/sdk');

const app = express();
const client = new GeoClimateClient({
  apiKey: process.env.API_KEY
});

app.post('/predict', async (req, res) => {
  try {
    const result = await client.predict(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000);
```

## API Reference

### GeoClimateClient

**Constructor Options:**
- `apiKey` (required): Your API key
- `baseURL` (optional): API base URL (default: https://api.geo-climate.com)
- `timeout` (optional): Request timeout in ms (default: 30000)
- `maxRetries` (optional): Max retry attempts (default: 3)

**Methods:**

#### `predict(data, options?)`
Make a single prediction.

```typescript
const result = await client.predict(
  { pm25: 35.5, temp: 72 },
  {
    modelId: 'model-123',  // Optional
    explain: true          // Optional
  }
);
```

#### `batchPredict(dataList, options?)`
Process multiple predictions.

```typescript
const results = await client.batchPredict(
  [{ pm25: 35.5 }, { pm25: 42.0 }],
  {
    modelId: 'model-123',  // Optional
    batchSize: 1000        // Optional
  }
);
```

#### `explainPrediction(data, modelId?, method?)`
Get explanation for prediction.

```typescript
const explanation = await client.explainPrediction(
  { pm25: 35.5, temp: 72 },
  'model-123',  // Optional
  'shap'        // 'shap' or 'lime'
);
```

#### `listModels(filters?)`
List available models.

```typescript
const models = await client.listModels({
  modelName: 'xgboost',      // Optional
  stage: 'production'        // Optional: 'dev' | 'staging' | 'production'
});
```

#### `getModelInfo(modelId)`
Get detailed model information.

```typescript
const info = await client.getModelInfo('model-123');
```

#### `healthCheck()`
Check API health.

```typescript
const health = await client.healthCheck();
console.log(health.status);
```

## Error Handling

```typescript
import {
  GeoClimateClient,
  AuthenticationError,
  RateLimitError,
  ValidationError
} from '@geo-climate/sdk';

try {
  const result = await client.predict(data);
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof ValidationError) {
    console.error('Invalid input data');
  } else {
    console.error('Unexpected error:', error.message);
  }
}
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Lint
npm run lint
```

## License

Apache 2.0

## Links

- **Documentation**: https://docs.geo-climate.com
- **GitHub**: https://github.com/dogaaydinn/Geo_Sentiment_Climate
- **Issues**: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
