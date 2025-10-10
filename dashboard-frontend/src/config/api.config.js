/**
 * Configuración de la API
 */

export const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 60000,
  retries: 3,
  retryDelay: 1000,
};

export const ENDPOINTS = {
  // Forecast
  forecastDynamic: '/forecast/dynamic/',
  forecastYear: (year) => `/forecast/year/${year}`,
  
  // Health
  health: '/health',
  modelsInfo: '/models/info',
  
  // Data Quality
  qualityCheck: '/data/quality-check/',
  qualitySummary: '/data/quality-summary/',
  
  // Analysis
  anomalies: '/analysis/anomalies/',
  anomaliesYear: (year) => `/analysis/anomalies/year/${year}`,
  recommendations: '/analysis/recommendations/',
  recommendationsQuick: '/analysis/recommendations/quick',
};