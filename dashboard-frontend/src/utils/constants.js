/**
 * Constantes de la aplicación
 */

export const APP_NAME = 'Forecast Dashboard';
export const APP_VERSION = '1.0.0';

export const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutos

export const YEARS = [2024, 2025, 2026, 2027];

export const DIMENSIONS = [
  { value: 'ORIGEN', label: 'Origen' },
  { value: 'UN', label: 'Unidad de Negocio' },
  { value: 'CATEGORIA', label: 'Categoría' },
  { value: 'CLASE', label: 'Clase' },
  { value: 'FAMILIA', label: 'Familia' },
  { value: 'DESTINO', label: 'Destino' },
];

export const SEVERITY_COLORS = {
  critico: 'red',
  alto: 'orange',
  medio: 'yellow',
  bajo: 'blue',
  info: 'gray',
};

export const CHART_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
];