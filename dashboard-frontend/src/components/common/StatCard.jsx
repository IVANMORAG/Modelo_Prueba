import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

/**
 * StatCard - Tarjeta de estadística elegante (SOLID: SRP)
 */
const StatCard = ({ 
  title, 
  value, 
  subtitle, 
  trend = null, 
  icon: Icon, 
  color = 'blue',
  prefix = '',
  suffix = '',
  loading = false 
}) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    red: 'from-red-500 to-red-600',
    yellow: 'from-yellow-500 to-yellow-600',
    purple: 'from-purple-500 to-purple-600',
  };

  const getTrendIcon = () => {
    if (!trend) return <Minus className="w-4 h-4" />;
    if (trend > 0) return <TrendingUp className="w-4 h-4 text-green-500" />;
    return <TrendingDown className="w-4 h-4 text-red-500" />;
  };

  const getTrendColor = () => {
    if (!trend) return 'text-gray-500';
    return trend > 0 ? 'text-green-500' : 'text-red-500';
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg animate-pulse">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="relative bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden group"
    >
      {/* Fondo gradiente animado */}
      <div className={`absolute inset-0 bg-gradient-to-br ${colorClasses[color]} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}></div>
      
      {/* Contenido */}
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
              {title}
            </p>
          </div>
          {Icon && (
            <div className={`p-3 rounded-lg bg-gradient-to-br ${colorClasses[color]} text-white`}>
              <Icon className="w-6 h-6" />
            </div>
          )}
        </div>

        <div className="space-y-2">
          <h3 className="text-3xl font-bold text-gray-900 dark:text-white">
            {prefix}{value}{suffix}
          </h3>
          
          {subtitle && (
            <div className="flex items-center gap-2">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {subtitle}
              </p>
              {trend !== null && (
                <div className={`flex items-center gap-1 ${getTrendColor()}`}>
                  {getTrendIcon()}
                  <span className="text-xs font-semibold">
                    {Math.abs(trend).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Brillo decorativo */}
      <div className="absolute -right-8 -top-8 w-32 h-32 bg-gradient-to-br from-white/20 to-transparent rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
    </motion.div>
  );
};

export default StatCard;