/**
 * Geo Climate JavaScript/TypeScript SDK
 *
 * Official JavaScript client for the Geo Sentiment Climate API.
 * Part of Phase 4: Advanced Features - API Marketplace.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export interface PredictionInput {
  [key: string]: number;
}

export interface PredictionResult {
  prediction: number;
  model_id: string;
  inference_time_ms: number;
  timestamp: string;
  cached?: boolean;
  explanation?: ExplanationResult;
}

export interface ExplanationResult {
  type: string;
  method: string;
  explanations: Array<{
    feature: string;
    value: number;
    shap_value?: number;
    contribution?: number;
  }>;
  top_features?: string[];
  trust_score?: number;
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  version: string;
  model_type: string;
  task_type: string;
  stage: string;
  metrics: { [key: string]: number };
  created_at: string;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
  uptime_seconds: number;
}

export interface GeoClimateClientConfig {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
}

export class GeoClimateError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'GeoClimateError';
  }
}

export class AuthenticationError extends GeoClimateError {
  constructor(message: string = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends GeoClimateError {
  constructor(message: string = 'Rate limit exceeded', public retryAfter?: number) {
    super(message, 429);
    this.name = 'RateLimitError';
  }
}

export class ValidationError extends GeoClimateError {
  constructor(message: string = 'Validation error') {
    super(message, 400);
    this.name = 'ValidationError';
  }
}

/**
 * Main client for Geo Climate API
 *
 * @example
 * ```typescript
 * const client = new GeoClimateClient({ apiKey: 'your-api-key' });
 *
 * const result = await client.predict({
 *   pm25: 35.5,
 *   temperature: 72,
 *   humidity: 65
 * });
 *
 * console.log(result.prediction);
 * ```
 */
export class GeoClimateClient {
  private axiosInstance: AxiosInstance;
  private maxRetries: number;

  constructor(config: GeoClimateClientConfig) {
    const {
      apiKey,
      baseURL = 'https://api.geo-climate.com',
      timeout = 30000,
      maxRetries = 3
    } = config;

    this.maxRetries = maxRetries;

    this.axiosInstance = axios.create({
      baseURL,
      timeout,
      headers: {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
        'User-Agent': '@geo-climate/sdk/1.0.0'
      }
    });

    // Response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      response => response,
      error => this.handleError(error)
    );
  }

  /**
   * Make a prediction
   */
  async predict(
    data: PredictionInput,
    options?: {
      modelId?: string;
      explain?: boolean;
    }
  ): Promise<PredictionResult> {
    const payload: any = { data };
    if (options?.modelId) {
      payload.model_id = options.modelId;
    }

    const response = await this.request<PredictionResult>('POST', '/predict', payload);

    // Add explanation if requested
    if (options?.explain && response.prediction) {
      try {
        const explanation = await this.explainPrediction(data, options.modelId);
        response.explanation = explanation;
      } catch (error) {
        console.warn('Failed to get explanation:', error);
      }
    }

    return response;
  }

  /**
   * Make batch predictions
   */
  async batchPredict(
    dataList: PredictionInput[],
    options?: {
      modelId?: string;
      batchSize?: number;
    }
  ): Promise<{ predictions: number[] }> {
    const payload: any = {
      data: dataList,
      batch_size: options?.batchSize || 1000
    };

    if (options?.modelId) {
      payload.model_id = options.modelId;
    }

    return this.request('POST', '/predict/batch', payload);
  }

  /**
   * Explain a prediction
   */
  async explainPrediction(
    data: PredictionInput,
    modelId?: string,
    method: 'shap' | 'lime' = 'shap'
  ): Promise<ExplanationResult> {
    const payload: any = { data, method };
    if (modelId) {
      payload.model_id = modelId;
    }

    return this.request('POST', '/explain', payload);
  }

  /**
   * List available models
   */
  async listModels(filters?: {
    modelName?: string;
    stage?: 'dev' | 'staging' | 'production';
  }): Promise<ModelInfo[]> {
    const params: any = {};
    if (filters?.modelName) params.model_name = filters.modelName;
    if (filters?.stage) params.stage = filters.stage;

    return this.request('GET', '/models', undefined, { params });
  }

  /**
   * Get model information
   */
  async getModelInfo(modelId: string): Promise<ModelInfo> {
    return this.request('GET', `/models/${modelId}`);
  }

  /**
   * Check API health
   */
  async healthCheck(): Promise<HealthStatus> {
    return this.request('GET', '/health');
  }

  /**
   * Get usage statistics
   */
  async getUsageStats(): Promise<any> {
    return this.request('GET', '/usage');
  }

  /**
   * Make HTTP request with retry logic
   */
  private async request<T = any>(
    method: string,
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    let lastError: any;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await this.axiosInstance.request({
          method,
          url,
          data,
          ...config
        });

        return response.data;
      } catch (error: any) {
        lastError = error;

        // Don't retry on client errors (except rate limit)
        if (error.statusCode && error.statusCode < 500 && error.statusCode !== 429) {
          throw error;
        }

        // Exponential backoff
        if (attempt < this.maxRetries - 1) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError;
  }

  /**
   * Handle API errors
   */
  private handleError(error: any): never {
    if (error.response) {
      const { status, data } = error.response;

      switch (status) {
        case 401:
          throw new AuthenticationError(data?.error || 'Invalid API key');

        case 429:
          const retryAfter = parseInt(error.response.headers['retry-after'] || '60');
          throw new RateLimitError(
            data?.error || 'Rate limit exceeded',
            retryAfter
          );

        case 400:
          throw new ValidationError(data?.error || 'Validation error');

        default:
          throw new GeoClimateError(
            data?.error || `Request failed with status ${status}`,
            status
          );
      }
    } else if (error.request) {
      throw new GeoClimateError('No response from server');
    } else {
      throw new GeoClimateError(error.message);
    }
  }
}

export default GeoClimateClient;
