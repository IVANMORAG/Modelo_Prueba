/**
 * useForecast - Custom Hook para pronóstico (SOLID: ISP)
 */
import { useQuery } from '@tanstack/react-query';
import ForecastService from '@services/ForecastService';

export const useForecast = (year, budget = null, options = {}) => {
  return useQuery({
    queryKey: ['forecast', year, budget],
    queryFn: () => ForecastService.getForecast(year, budget),
    staleTime: 5 * 60 * 1000, // 5 minutos
    cacheTime: 30 * 60 * 1000, // 30 minutos
    ...options,
  });
};

export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => ForecastService.healthCheck(),
    refetchInterval: 30000, // Auto-refresh cada 30 segundos
    retry: 3,
  });
};

export const useModelsInfo = () => {
  return useQuery({
    queryKey: ['models-info'],
    queryFn: () => ForecastService.getModelsInfo(),
    staleTime: 60 * 60 * 1000, // 1 hora
  });
};