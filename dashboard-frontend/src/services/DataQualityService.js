/**
 * DataQualityService - Servicio de calidad de datos (SOLID: SRP)
 */
import ApiService from './ApiService';

class DataQualityService {
  /**
   * Verifica calidad de datos completa
   */
  async checkQuality(includeDetails = true, limitExamples = 5) {
    return await ApiService.post('/data/quality-check/', {
      incluir_detalles: includeDetails,
      limite_ejemplos: limitExamples,
    });
  }

  /**
   * Obtiene resumen rápido de calidad
   */
  async getQualitySummary() {
    return await ApiService.get('/data/quality-summary/');
  }
}

export default new DataQualityService();