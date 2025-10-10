/**
 * useAnomalies - Custom Hook para anomalías
 */
import { useQuery } from '@tanstack/react-query';
import AnalysisService from '@services/AnalysisService';

export const useAnomalies = (contamination = 0.05, topN = 20, options = {}) => {
  return useQuery({
    queryKey: ['anomalies', contamination, topN],
    queryFn: () => AnalysisService.detectAnomalies(contamination, topN),
    staleTime: 10 * 60 * 1000, // 10 minutos
    ...options,
  });
};

export const useAnomaliesByYear = (year, topN = 20) => {
  return useQuery({
    queryKey: ['anomalies-year', year, topN],
    queryFn: () => AnalysisService.getAnomaliesByYear(year, topN),
    enabled: !!year,
  });
};