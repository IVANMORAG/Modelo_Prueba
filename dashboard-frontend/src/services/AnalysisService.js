/**
 * AnalysisService - Servicio de análisis (SOLID: SRP)
 */
import ApiService from './ApiService';

class AnalysisService {
  /**
   * Detecta anomalías en gastos
   */
  async detectAnomalies(contamination = 0.05, topN = 20) {
    return await ApiService.post('/analysis/anomalies/', {
      contamination,
      top_n: topN,
      incluir_contexto: true,
    });
  }

  /**
   * Obtiene anomalías por año
   */
  async getAnomaliesByYear(year, topN = 20) {
    return await ApiService.get(`/analysis/anomalies/year/${year}?top_n=${topN}`);
  }

  /**
   * Genera recomendaciones estratégicas
   */
  async getRecommendations(dimensions = ['ORIGEN', 'UN'], includeTrends = true) {
    return await ApiService.post('/analysis/recommendations/', {
      dimensiones_analisis: dimensions,
      incluir_tendencias: includeTrends,
    });
  }

  /**
   * Obtiene recomendaciones rápidas
   */
  async getQuickRecommendations() {
    return await ApiService.get('/analysis/recommendations/quick');
  }
}

export default new AnalysisService();