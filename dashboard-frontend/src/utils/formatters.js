/**
 * Utilidades de formateo (SOLID: SRP)
 */

/**
 * Formatea un número a moneda mexicana
 */
export const formatCurrency = (value, options = {}) => {
  const {
    decimals = 0,
    currency = 'MXN',
    locale = 'es-MX'
  } = options;

  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

/**
 * Formatea números grandes con sufijos (B, M, K)
 */
export const formatLargeNumber = (value, decimals = 1) => {
  if (value >= 1e9) return `$${(value / 1e9).toFixed(decimals)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(decimals)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(decimals)}K`;
  return `$${value.toFixed(0)}`;
};

/**
 * Formatea porcentajes
 */
export const formatPercentage = (value, decimals = 1) => {
  return `${value.toFixed(decimals)}%`;
};

/**
 * Formatea fechas
 */
export const formatDate = (date, options = {}) => {
  const {
    locale = 'es-MX',
    dateStyle = 'medium'
  } = options;

  return new Intl.DateTimeFormat(locale, { dateStyle }).format(new Date(date));
};

/**
 * Formatea fecha y hora
 */
export const formatDateTime = (date) => {
  return new Intl.DateTimeFormat('es-MX', {
    dateStyle: 'medium',
    timeStyle: 'short'
  }).format(new Date(date));
};

/**
 * Formatea números con separadores de miles
 */
export const formatNumber = (value, decimals = 0) => {
  return new Intl.NumberFormat('es-MX', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

/**
 * Abrevia texto largo
 */
export const truncateText = (text, maxLength = 50) => {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength)}...`;
};

/**
 * Calcula diferencia entre dos valores
 */
export const calculateDifference = (current, previous) => {
  if (previous === 0) return 0;
  return ((current - previous) / previous) * 100;
};