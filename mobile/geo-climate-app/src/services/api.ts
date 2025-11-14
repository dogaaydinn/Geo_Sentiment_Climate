/**
 * API Service for Mobile App
 *
 * Interfaces with Geo Climate API using the JavaScript SDK.
 */

import { GeoClimateClient } from '@geo-climate/sdk';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_KEY_STORAGE = '@geo_climate:api_key';
const BASE_URL = 'https://api.geo-climate.com';

class ApiService {
  private client: GeoClimateClient | null = null;

  async initialize() {
    const apiKey = await AsyncStorage.getItem(API_KEY_STORAGE);

    if (apiKey) {
      this.client = new GeoClimateClient({
        apiKey,
        baseURL: BASE_URL
      });
    }
  }

  async setApiKey(apiKey: string) {
    await AsyncStorage.setItem(API_KEY_STORAGE, apiKey);
    this.client = new GeoClimateClient({
      apiKey,
      baseURL: BASE_URL
    });
  }

  async predict(data: any) {
    if (!this.client) {
      throw new Error('API client not initialized');
    }

    return await this.client.predict(data, { explain: true });
  }

  async getModels() {
    if (!this.client) {
      throw new Error('API client not initialized');
    }

    return await this.client.listModels({ stage: 'production' });
  }

  async checkHealth() {
    if (!this.client) {
      throw new Error('API client not initialized');
    }

    return await this.client.healthCheck();
  }

  isInitialized() {
    return this.client !== null;
  }
}

export default new ApiService();
