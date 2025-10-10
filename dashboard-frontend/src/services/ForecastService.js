/**
 * ForecastService - Servicio de pronóstico (SOLID: SRP)
 */
import ApiService from './ApiService';

class ForecastService {
  /**
   * Obtiene pronóstico dinámico (monitoreo o proyección)
   */
  async getForecast(year, budget = null, dimensions = ['ORIGEN', 'CATEGORIA']) {
    return await ApiService.post('/forecast/dynamic/', {
      anio_objetivo: year,
      presupuesto_asignado: budget,
      dimensiones: dimensions,
      incluir_intervalos_confianza: true,
    });
  }

  /**
   * Obtiene pronóstico simple por año
   */
  async getForecastByYear(year, budget = null) {
    const params = budget ? `?presupuesto=${budget}` : '';
    return await ApiService.get(`/forecast/year/${year}${params}`);
  }

  /**
   * Obtiene información de modelos
   */
  async getModelsInfo() {
    return await ApiService.get('/models/info');
  }

  /**
   * Health check
   */
  async healthCheck() {
    return await ApiService.get('/health');
  }
}

export default new ForecastService();