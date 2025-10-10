/**
 * ApiService - Cliente HTTP base (SOLID: SRP - Solo gestión de HTTP)
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(` API Request: ${config.method.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(` API Response: ${response.config.url}`, response.data);
        return response;
      },
      (error) => {
        console.error(` API Error: ${error.config?.url}`, error.response?.data || error.message);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  handleError(error) {
    if (error.response) {
      // Error del servidor
      return {
        status: error.response.status,
        message: error.response.data?.detail || error.response.data?.error || 'Error del servidor',
        data: error.response.data,
      };
    } else if (error.request) {
      // No hubo respuesta
      return {
        status: 0,
        message: 'No se pudo conectar al servidor. Verifica tu conexión.',
        data: null,
      };
    } else {
      // Error en la configuración
      return {
        status: -1,
        message: error.message || 'Error desconocido',
        data: null,
      };
    }
  }

  async get(url, config = {}) {
    const response = await this.client.get(url, config);
    return response.data;
  }

  async post(url, data, config = {}) {
    const response = await this.client.post(url, data, config);
    return response.data;
  }

  async put(url, data, config = {}) {
    const response = await this.client.put(url, data, config);
    return response.data;
  }

  async delete(url, config = {}) {
    const response = await this.client.delete(url, config);
    return response.data;
  }
}

export default new ApiService();