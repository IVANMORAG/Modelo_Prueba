import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, DollarSign, AlertTriangle, CheckCircle, 
  Calendar, BarChart3, PieChart, Activity
} from 'lucide-react';
import { 
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart as RePieChart, Pie, Cell
} from 'recharts';
import { useForecast } from '@hooks/useForecast';
import StatCard from '@components/common/StatCard';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const Dashboard = () => {
  const currentYear = new Date().getFullYear();
  const [selectedYear, setSelectedYear] = useState(currentYear);
  const budget = 70000000000; // 70B presupuesto

  const { data: forecast, isLoading, error } = useForecast(selectedYear, budget);

  // Formatear números a moneda
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('es-MX', {
      style: 'currency',
      currency: 'MXN',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Formatear números grandes (B, M, K)
  const formatLargeNumber = (value) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
    return `$${value.toFixed(0)}`;
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-64 animate-pulse"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-40 bg-gray-200 dark:bg-gray-700 rounded-xl animate-pulse"></div>
            ))}
          </div>
          <div className="h-96 bg-gray-200 dark:bg-gray-700 rounded-xl animate-pulse"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-8 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Error al cargar datos
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            {error.message || 'No se pudo conectar con el servidor'}
          </p>
        </div>
      </div>
    );
  }

  const isMonitoring = forecast?.modo === 'monitoreo';
  const isProjection = forecast?.modo === 'proyeccion';

  // Preparar datos para gráfica mensual
  const monthlyData = (forecast?.proyeccion_mensual || forecast?.gastos_mensuales || []).map(m => ({
    mes: m.mes_nombre,
    predicho: m.gasto_predicho / 1e9, // Convertir a billones
    real: m.gasto_real ? m.gasto_real / 1e9 : null,
    inferior: m.intervalo_inferior ? m.intervalo_inferior / 1e9 : null,
    superior: m.intervalo_superior ? m.intervalo_superior / 1e9 : null,
  }));

  // Preparar datos para desglose por ORIGEN
  const origenData = Object.entries(forecast?.desgloses?.ORIGEN || {})
    .slice(0, 6)
    .map(([_, item]) => ({
      name: item.valor,
      value: item.gasto_total / 1e9,
      percentage: item.porcentaje_total,
    }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header con selector de año */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row md:items-center md:justify-between gap-4"
        >
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-2">
              Dashboard de Pronóstico 📊
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Análisis y monitoreo presupuestal en tiempo real
            </p>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              className="px-4 py-2 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white font-semibold focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            >
              <option value={2024}>2024</option>
              <option value={2025}>2025</option>
              <option value={2026}>2026</option>
              <option value={2027}>2027</option>
            </select>
            
            <div className="px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-semibold shadow-lg">
              {isMonitoring ? '📍 Monitoreo' : '🔮 Proyección'}
            </div>
          </div>
        </motion.div>

        {/* Métricas principales */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title={isMonitoring ? "Gasto Real" : "Proyección Total"}
            value={formatLargeNumber(
              isMonitoring ? forecast.gasto_real_acumulado : forecast.gasto_proyectado_total
            )}
            subtitle={isMonitoring ? "Acumulado a la fecha" : `Año ${selectedYear}`}
            icon={DollarSign}
            color="blue"
            trend={isMonitoring ? forecast.desviacion_porcentual : null}
          />

          <StatCard
            title="Presupuesto Asignado"
            value={formatLargeNumber(budget)}
            subtitle={
              isMonitoring
                ? `Restante: ${formatLargeNumber(forecast.presupuesto_restante || 0)}`
                : `Diferencia: ${formatLargeNumber(forecast.diferencia_presupuesto || 0)}`
            }
            icon={TrendingUp}
            color="green"
          />

          <StatCard
            title={isMonitoring ? "Proyección Fin Año" : "Confianza"}
            value={
              isMonitoring
                ? formatLargeNumber(forecast.proyeccion_fin_anio)
                : `${(forecast.nivel_confianza * 100).toFixed(0)}%`
            }
            subtitle={isMonitoring ? "Estimado" : `${forecast.datos_historicos_meses} meses históricos`}
            icon={Calendar}
            color="purple"
          />

          <StatCard
            title="Alertas"
            value={forecast?.alertas?.length || 0}
            subtitle={
              forecast?.alertas?.length > 0
                ? forecast.alertas[0].severidad
                : "Sin alertas"
            }
            icon={forecast?.alertas?.length > 0 ? AlertTriangle : CheckCircle}
            color={forecast?.alertas?.length > 0 ? "red" : "green"}
          />
        </div>

        {/* Gráfica de tendencia mensual */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              📈 Distribución Mensual (Billones de Pesos)
            </h2>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-gray-600 dark:text-gray-400">Predicho</span>
              </div>
              {isMonitoring && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-gray-600 dark:text-gray-400">Real</span>
                </div>
              )}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={monthlyData}>
              <defs>
                <linearGradient id="colorPredicho" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="colorReal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
              <XAxis 
                dataKey="mes" 
                stroke="#9ca3af"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#9ca3af"
                style={{ fontSize: '12px' }}
                tickFormatter={(value) => `${value.toFixed(1)}B`}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#fff'
                }}
                formatter={(value) => `${value.toFixed(2)}B`}
              />
              <Area 
                type="monotone" 
                dataKey="predicho" 
                stroke="#3b82f6" 
                fillOpacity={1}
                fill="url(#colorPredicho)"
                strokeWidth={2}
              />
              {isMonitoring && (
                <Area 
                  type="monotone" 
                  dataKey="real" 
                  stroke="#10b981" 
                  fillOpacity={1}
                  fill="url(#colorReal)"
                  strokeWidth={2}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Grid de análisis por dimensión y alertas */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Desglose por ORIGEN (Pie Chart) */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
          >
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              🎯 Gasto por ORIGEN
            </h2>
            
            {origenData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <RePieChart>
                  <Pie
                    data={origenData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name}: ${percentage.toFixed(1)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {origenData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: 'none',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                    formatter={(value) => `${value.toFixed(2)}B`}
                  />
                </RePieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                No hay datos de desglose disponibles
              </div>
            )}
          </motion.div>

          {/* Alertas */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
          >
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
              🚨 Alertas y Notificaciones
            </h2>
            
            <div className="space-y-4">
              {forecast?.alertas && forecast.alertas.length > 0 ? (
                forecast.alertas.map((alerta, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border-l-4 ${
                      alerta.severidad === 'critico'
                        ? 'bg-red-50 dark:bg-red-900/20 border-red-500'
                        : alerta.severidad === 'alto'
                        ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-500'
                        : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <AlertTriangle className={`w-5 h-5 mt-0.5 ${
                        alerta.severidad === 'critico' ? 'text-red-500' :
                        alerta.severidad === 'alto' ? 'text-orange-500' :
                        'text-yellow-500'
                      }`} />
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                          {alerta.titulo}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {alerta.descripcion}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className={`text-xs px-2 py-1 rounded font-semibold ${
                            alerta.severidad === 'critico'
                              ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                              : alerta.severidad === 'alto'
                              ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300'
                              : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300'
                          }`}>
                            {alerta.severidad.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                  <CheckCircle className="w-16 h-16 text-green-500 mb-4" />
                  <p className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                    ¡Todo en orden!
                  </p>
                  <p className="text-sm text-gray-500">
                    No hay alertas por el momento
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Footer con info adicional */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center text-sm text-gray-500 dark:text-gray-400 py-4"
        >
          <p>
            Última actualización: {new Date().toLocaleString('es-MX')} • 
            Modo: {isMonitoring ? 'Monitoreo' : 'Proyección'} • 
            Año: {selectedYear}
          </p>
        </motion.div>

      </div>
    </div>
  );
};

export default Dashboard;