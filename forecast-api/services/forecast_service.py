"""
Servicio de Pronóstico y Monitoreo
Implementa Single Responsibility Principle (SRP) y Open/Closed Principle (OCP)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from abc import ABC, abstractmethod
import re

from services.model_service import model_service
from models.schemas import (
    ForecastMonitoreoResponse, ForecastProyeccionResponse,
    GastoMensual, DesgloseDimension, Alerta
)
from models.enums import (
    ModoOperacion, TipoAlerta, NivelSeveridad, 
    DimensionAnalisis, MESES_NOMBRES, FEATURES_MODELO
)
from utils.logger import logger
from utils.exceptions import ForecastException, InsufficientDataException


class ForecastStrategy(ABC):
    """Estrategia abstracta para pronóstico (Open/Closed Principle)"""
    
    @abstractmethod
    def execute(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Ejecuta la estrategia de pronóstico"""
        pass


class MonitoreoStrategy(ForecastStrategy):
    """Estrategia para monitoreo del año actual (2025)"""
    
    def execute(
        self, 
        df: pd.DataFrame,
        anio_objetivo: int,
        presupuesto_asignado: Optional[float] = None,
        dimensiones: List[DimensionAnalisis] = None,
        incluir_intervalos: bool = True
    ) -> ForecastMonitoreoResponse:
        """
        Ejecuta monitoreo comparando gasto real vs predicción inicial
        
        Args:
            df: DataFrame con datos históricos y avance año actual
            anio_objetivo: Año a monitorear (debe ser año actual)
            presupuesto_asignado: Presupuesto opcional
            dimensiones: Dimensiones para desglose
            incluir_intervalos: Incluir intervalos de confianza
            
        Returns:
            ForecastMonitoreoResponse
        """
        logger.info(f" Ejecutando monitoreo para año {anio_objetivo}")
        
        # Preparar datos
        df_preparado = self._prepare_data(df)
        
        # Filtrar datos del año objetivo
        df_anio = df_preparado[df_preparado['AÑO'] == anio_objetivo].copy()
        
        if len(df_anio) == 0:
            raise InsufficientDataException(
                f"No hay datos disponibles para el año {anio_objetivo}"
            )
        
        # Calcular gasto real acumulado
        gasto_real_acumulado = float(df_anio['TOTALPESOS'].sum())
        
        # Generar predicción total para el año (usando datos históricos)
        df_historico = df_preparado[df_preparado['AÑO'] < anio_objetivo]
        gasto_predicho_total = self._predict_annual_total(df_historico, anio_objetivo)
        
        # Calcular desviación
        desviacion_total = gasto_real_acumulado - gasto_predicho_total
        desviacion_porcentual = (desviacion_total / gasto_predicho_total * 100) if gasto_predicho_total > 0 else 0
        
        # Gastos mensuales
        gastos_mensuales = self._calculate_monthly_comparison(
            df_anio, df_historico, anio_objetivo, incluir_intervalos
        )
        
        # Desgloses por dimensión
        desgloses = self._calculate_dimension_breakdown(df_anio, dimensiones or [])
        
        # Proyección fin de año
        proyeccion_fin_anio = self._project_year_end(df_anio, gasto_predicho_total)
        
        # Generar alertas
        alertas = self._generate_monitoring_alerts(
            gasto_real_acumulado, gasto_predicho_total,
            desviacion_porcentual, presupuesto_asignado,
            proyeccion_fin_anio
        )
        
        # Calcular presupuesto restante
        presupuesto_restante = None
        if presupuesto_asignado:
            presupuesto_restante = presupuesto_asignado - gasto_real_acumulado
        
        return ForecastMonitoreoResponse(
            anio=anio_objetivo,
            gasto_real_acumulado=gasto_real_acumulado,
            gasto_predicho_total=gasto_predicho_total,
            desviacion_total=desviacion_total,
            desviacion_porcentual=desviacion_porcentual,
            gastos_mensuales=gastos_mensuales,
            desgloses=desgloses,
            alertas=alertas,
            presupuesto_asignado=presupuesto_asignado,
            presupuesto_restante=presupuesto_restante,
            proyeccion_fin_anio=proyeccion_fin_anio
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos para análisis"""
        df_prep = df.copy()
        
        # Convertir fechas
        if 'FECHAPEDIDO' in df_prep.columns:
            df_prep['FECHAPEDIDO'] = pd.to_datetime(df_prep['FECHAPEDIDO'], errors='coerce')
            df_prep['AÑO'] = df_prep['FECHAPEDIDO'].dt.year
            df_prep['MES'] = df_prep['FECHAPEDIDO'].dt.month
        
        # Filtrar registros válidos
        df_prep = df_prep[
            (df_prep['TOTALPESOS'] > 0) &
            (df_prep['AÑO'].notna())
        ].copy()
        
        return df_prep
    
    def _predict_annual_total(self, df_historico: pd.DataFrame, anio: int) -> float:
        """Predice gasto total anual usando promedio histórico ajustado"""
        if len(df_historico) == 0:
            raise InsufficientDataException("No hay datos históricos suficientes")
        
        # Calcular promedio anual histórico
        gastos_anuales = df_historico.groupby('AÑO')['TOTALPESOS'].sum()
        promedio_anual = float(gastos_anuales.mean())
        
        # Ajustar por tendencia (últimos 3 años)
        if len(gastos_anuales) >= 3:
            ultimos_3 = gastos_anuales.tail(3)
            tendencia = (ultimos_3.iloc[-1] - ultimos_3.iloc[0]) / 3
            promedio_anual += tendencia
        
        logger.info(f"Predicción total para {anio}: ${promedio_anual:,.2f}")
        return promedio_anual
    
    def _calculate_monthly_comparison(
        self,
        df_anio: pd.DataFrame,
        df_historico: pd.DataFrame,
        anio: int,
        incluir_intervalos: bool
    ) -> List[GastoMensual]:
        """Calcula comparación mensual real vs predicho"""
        gastos_mensuales = []
        
        # Gasto real por mes
        gasto_real_mes = df_anio.groupby('MES')['TOTALPESOS'].sum().to_dict()
        
        # Promedio histórico por mes
        gasto_hist_mes = df_historico.groupby('MES')['TOTALPESOS'].mean().to_dict()
        
        for mes in range(1, 13):
            gasto_real = gasto_real_mes.get(mes, 0.0)
            gasto_predicho = gasto_hist_mes.get(mes, 0.0)
            
            desviacion = gasto_real - gasto_predicho if gasto_real > 0 else None
            desviacion_pct = (desviacion / gasto_predicho * 100) if gasto_predicho > 0 and desviacion else None
            
            # Intervalos de confianza (±15% como estimación)
            intervalo_inf = gasto_predicho * 0.85 if incluir_intervalos else None
            intervalo_sup = gasto_predicho * 1.15 if incluir_intervalos else None
            
            gastos_mensuales.append(GastoMensual(
                mes=mes,
                mes_nombre=MESES_NOMBRES[mes - 1],
                gasto_predicho=float(gasto_predicho),
                gasto_real=float(gasto_real) if gasto_real > 0 else None,
                desviacion=float(desviacion) if desviacion else None,
                desviacion_porcentual=float(desviacion_pct) if desviacion_pct else None,
                intervalo_inferior=float(intervalo_inf) if intervalo_inf else None,
                intervalo_superior=float(intervalo_sup) if intervalo_sup else None
            ))
        
        return gastos_mensuales
    
    def _calculate_dimension_breakdown(
        self,
        df: pd.DataFrame,
        dimensiones: List[DimensionAnalisis]
    ) -> Dict[str, List[DesgloseDimension]]:
        """Calcula desglose por dimensiones"""
        desgloses = {}
        
        for dimension in dimensiones:
            dim_col = dimension.value
            
            if dim_col not in df.columns:
                continue
            
            desglose_list = []
            
            # Agrupar
            agrupado = df.groupby(dim_col)['TOTALPESOS'].agg(['sum', 'mean', 'count']).reset_index()
            agrupado.columns = [dim_col, 'gasto_total', 'gasto_promedio', 'cantidad']
            
            total_general = agrupado['gasto_total'].sum()
            
            for _, row in agrupado.iterrows():
                porcentaje = (row['gasto_total'] / total_general * 100) if total_general > 0 else 0
                
                desglose_list.append(DesgloseDimension(
                    dimension=dimension,
                    valor=str(row[dim_col]),
                    gasto_total=float(row['gasto_total']),
                    gasto_promedio=float(row['gasto_promedio']),
                    cantidad_pedidos=int(row['cantidad']),
                    porcentaje_total=float(porcentaje)
                ))
            
            desglose_list.sort(key=lambda x: x.gasto_total, reverse=True)
            desgloses[dim_col] = desglose_list[:10]
        
        return desgloses
    
    def _project_year_end(self, df_anio: pd.DataFrame, gasto_predicho_total: float) -> float:
        """Proyecta gasto al fin de año basado en avance actual"""
        mes_actual = datetime.now().month
        
        # Gasto real hasta la fecha
        gasto_real_acumulado = float(df_anio['TOTALPESOS'].sum())
        
        # Calcular tasa de ejecución
        meses_transcurridos = mes_actual
        tasa_mensual = gasto_real_acumulado / meses_transcurridos if meses_transcurridos > 0 else 0
        
        # Proyección lineal
        proyeccion_lineal = tasa_mensual * 12
        
        # Combinar con predicción histórica (promedio ponderado)
        proyeccion_final = (proyeccion_lineal * 0.6) + (gasto_predicho_total * 0.4)
        
        return float(proyeccion_final)
    
    def _generate_monitoring_alerts(
        self,
        gasto_real: float,
        gasto_predicho: float,
        desviacion_pct: float,
        presupuesto: Optional[float],
        proyeccion_fin: float
    ) -> List[Alerta]:
        """Genera alertas de monitoreo"""
        alertas = []
        
        # Alerta: Desviación significativa
        if abs(desviacion_pct) > 10:
            severidad = NivelSeveridad.CRITICO if abs(desviacion_pct) > 20 else NivelSeveridad.ALTO
            tipo_desv = "sobre" if desviacion_pct > 0 else "bajo"
            
            alertas.append(Alerta(
                tipo=TipoAlerta.DESVIACION_PRESUPUESTO,
                severidad=severidad,
                titulo=f"Desviación {tipo_desv} lo predicho",
                descripcion=f"El gasto real está {abs(desviacion_pct):.1f}% {tipo_desv} la predicción. "
                           f"Real: ${gasto_real:,.2f} vs Predicho: ${gasto_predicho:,.2f}",
                valor_observado=gasto_real,
                valor_esperado=gasto_predicho
            ))
        
        # Alerta: Exceso presupuestal
        if presupuesto and gasto_real > presupuesto * 0.9:
            alertas.append(Alerta(
                tipo=TipoAlerta.DESVIACION_PRESUPUESTO,
                severidad=NivelSeveridad.CRITICO,
                titulo="Presupuesto cerca de agotarse",
                descripcion=f"Se ha consumido el {gasto_real/presupuesto*100:.1f}% del presupuesto asignado. "
                           f"Presupuesto: ${presupuesto:,.2f}",
                valor_observado=gasto_real,
                valor_esperado=presupuesto
            ))
        
        # Alerta: Proyección excede presupuesto
        if presupuesto and proyeccion_fin > presupuesto:
            exceso = proyeccion_fin - presupuesto
            alertas.append(Alerta(
                tipo=TipoAlerta.DESVIACION_PRESUPUESTO,
                severidad=NivelSeveridad.ALTO,
                titulo="Proyección excede presupuesto",
                descripcion=f"La proyección de fin de año (${proyeccion_fin:,.2f}) "
                           f"excede el presupuesto en ${exceso:,.2f}",
                valor_observado=proyeccion_fin,
                valor_esperado=presupuesto
            ))
        
        return alertas


class ProyeccionStrategy(ForecastStrategy):
    """Estrategia para proyección de años futuros (2026+)"""
    
    def execute(
        self,
        df: pd.DataFrame,
        anio_objetivo: int,
        presupuesto_asignado: Optional[float] = None,
        dimensiones: List[DimensionAnalisis] = None,
        incluir_intervalos: bool = True
    ) -> ForecastProyeccionResponse:
        """
        Ejecuta proyección para año futuro usando modelo ML y Prophet
        
        Args:
            df: DataFrame con datos históricos
            anio_objetivo: Año futuro a proyectar
            presupuesto_asignado: Presupuesto opcional
            dimensiones: Dimensiones para desglose
            incluir_intervalos: Incluir intervalos de confianza
            
        Returns:
            ForecastProyeccionResponse
        """
        logger.info(f"🔮 Ejecutando proyección para año {anio_objetivo}")
        
        # Preparar datos
        df_preparado = self._prepare_data(df)
        
        # Validar datos suficientes
        if len(df_preparado) < 100:
            raise InsufficientDataException(
                f"Datos insuficientes para proyección: {len(df_preparado)} registros"
            )
        
        # Generar features para predicción
        df_features = self._generate_forecast_features(df_preparado, anio_objetivo)
        
        # Proyectar gasto total usando modelo ML
        gasto_proyectado_total = self._predict_with_ml_model(df_features)
        
        # Proyección mensual usando Prophet
        proyeccion_mensual = self._predict_monthly_with_prophet(
            df_preparado, anio_objetivo, incluir_intervalos
        )
        
        # Desgloses por dimensión
        desgloses = self._calculate_dimension_forecast(
            df_preparado, df_features, dimensiones or []
        )
        
        # Comparación con presupuesto
        diferencia_presupuesto = None
        diferencia_porcentual = None
        if presupuesto_asignado:
            diferencia_presupuesto = gasto_proyectado_total - presupuesto_asignado
            diferencia_porcentual = (diferencia_presupuesto / presupuesto_asignado * 100)
        
        # Generar alertas
        alertas = self._generate_projection_alerts(
            gasto_proyectado_total, presupuesto_asignado,
            diferencia_porcentual
        )
        
        # Métricas de confianza
        nivel_confianza = 0.95
        meses_historicos = len(df_preparado['AÑO'].unique()) * 12
        
        # Intervalos de confianza (±10% como estimación)
        intervalo_inferior = gasto_proyectado_total * 0.90 if incluir_intervalos else None
        intervalo_superior = gasto_proyectado_total * 1.10 if incluir_intervalos else None
        
        return ForecastProyeccionResponse(
            anio=anio_objetivo,
            gasto_proyectado_total=gasto_proyectado_total,
            intervalo_confianza_inferior=intervalo_inferior,
            intervalo_confianza_superior=intervalo_superior,
            proyeccion_mensual=proyeccion_mensual,
            desgloses=desgloses,
            presupuesto_asignado=presupuesto_asignado,
            diferencia_presupuesto=diferencia_presupuesto,
            diferencia_porcentual=diferencia_porcentual,
            alertas=alertas,
            nivel_confianza=nivel_confianza,
            datos_historicos_meses=meses_historicos
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos para proyección"""
        df_prep = df.copy()
        
        # Convertir fechas y crear features temporales
        if 'FECHAPEDIDO' in df_prep.columns:
            df_prep['FECHAPEDIDO'] = pd.to_datetime(df_prep['FECHAPEDIDO'], errors='coerce')
            df_prep['AÑO'] = df_prep['FECHAPEDIDO'].dt.year
            df_prep['MES'] = df_prep['FECHAPEDIDO'].dt.month
            df_prep['TRIMESTRE'] = df_prep['FECHAPEDIDO'].dt.quarter
            df_prep['DIA_SEMANA'] = df_prep['FECHAPEDIDO'].dt.dayofweek
            df_prep['DIA_MES'] = df_prep['FECHAPEDIDO'].dt.day
            df_prep['SEMANA_AÑO'] = df_prep['FECHAPEDIDO'].dt.isocalendar().week
        
        # Extraer cantidad
        if 'DETALLEFINAL' in df_prep.columns:
            df_prep['CANTIDAD'] = df_prep['DETALLEFINAL'].apply(self._extract_quantity)
        else:
            df_prep['CANTIDAD'] = 1
        
        # Crear flags temporales
        if 'DIA_MES' in df_prep.columns:
            df_prep['ES_FIN_MES'] = (df_prep['DIA_MES'] >= 25).astype(int)
            df_prep['ES_INICIO_MES'] = (df_prep['DIA_MES'] <= 5).astype(int)
        
        if 'MES' in df_prep.columns:
            df_prep['ES_FIN_TRIMESTRE'] = df_prep['MES'].isin([3, 6, 9, 12]).astype(int)
        
        # Filtrar válidos
        df_prep = df_prep[df_prep['TOTALPESOS'] > 0].copy()
        
        return df_prep
    
    def _extract_quantity(self, text) -> int:
        """Extrae cantidad del detalle"""
        if pd.isna(text):
            return 1
        
        patterns = [
            r'Cant[.:]?\s*(\d+)',
            r'Cantidad[.:]?\s*(\d+)',
            r'PZA[.:]?\s*(\d+)',
            r'UN[.:]?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text), re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 1
    
    def _generate_forecast_features(
        self, 
        df: pd.DataFrame, 
        anio: int
    ) -> pd.DataFrame:
        """Genera features para el año objetivo"""
        # Tomar muestra representativa del último año
        ultimo_anio = df['AÑO'].max()
        df_ultimo = df[df['AÑO'] == ultimo_anio].copy()
        
        if len(df_ultimo) == 0:
            df_ultimo = df.tail(1000).copy()
        
        # Actualizar año a proyectar
        df_forecast = df_ultimo.copy()
        df_forecast['AÑO'] = anio
        
        # Calcular features agregados
        df_forecast = self._calculate_aggregated_features(df, df_forecast)
        
        return df_forecast
    
    def _calculate_aggregated_features(
        self,
        df_historico: pd.DataFrame,
        df_forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Calcula features agregados necesarios para el modelo"""
        # Por CATEGORIA
        if 'CATEGORIA' in df_historico.columns:
            gasto_cat = df_historico.groupby('CATEGORIA')['TOTALPESOS'].agg(['mean', 'median', 'std']).reset_index()
            gasto_cat.columns = ['CATEGORIA', 'GASTO_PROM_CAT', 'GASTO_MED_CAT', 'GASTO_STD_CAT']
            df_forecast = df_forecast.merge(gasto_cat, on='CATEGORIA', how='left')
        
        # Por DESTINO
        if 'DESTINO' in df_historico.columns:
            gasto_dest = df_historico.groupby('DESTINO')['TOTALPESOS'].agg(['mean', 'median']).reset_index()
            gasto_dest.columns = ['DESTINO', 'GASTO_PROM_DEST', 'GASTO_MED_DEST']
            df_forecast = df_forecast.merge(gasto_dest, on='DESTINO', how='left')
        
        # Por ORIGEN
        if 'ORIGEN' in df_historico.columns:
            gasto_orig = df_historico.groupby('ORIGEN')['TOTALPESOS'].mean().reset_index()
            gasto_orig.columns = ['ORIGEN', 'GASTO_PROM_ORIG']
            df_forecast = df_forecast.merge(gasto_orig, on='ORIGEN', how='left')
        
        # Por UN
        if 'UN' in df_historico.columns:
            gasto_un = df_historico.groupby('UN')['TOTALPESOS'].mean().reset_index()
            gasto_un.columns = ['UN', 'GASTO_PROM_UN']
            df_forecast = df_forecast.merge(gasto_un, on='UN', how='left')
        
        # Ratios
        if 'GASTO_PROM_CAT' in df_forecast.columns:
            df_forecast['RATIO_GASTO_CAT'] = df_forecast['TOTALPESOS'] / (df_forecast['GASTO_PROM_CAT'] + 1)
        
        if 'GASTO_PROM_DEST' in df_forecast.columns:
            df_forecast['RATIO_GASTO_DEST'] = df_forecast['TOTALPESOS'] / (df_forecast['GASTO_PROM_DEST'] + 1)
        
        # Lag features (usar promedio)
        df_forecast['GASTO_MES_ANTERIOR'] = df_historico['TOTALPESOS'].median()
        
        # Rellenar nulos
        df_forecast = df_forecast.fillna(df_forecast.median(numeric_only=True))
        
        return df_forecast
    
    def _predict_with_ml_model(self, df_features: pd.DataFrame) -> float:
        """Predice usando el modelo XGBoost"""
        try:
            # Codificar variables categóricas
            df_encoded = df_features.copy()
            
            for col in ['CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN']:
                if col in df_encoded.columns:
                    try:
                        encoder = model_service.get_encoder(col)
                        df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                    except:
                        df_encoded[col] = 0
            
            # Seleccionar solo features del modelo
            features_disponibles = [f for f in FEATURES_MODELO if f in df_encoded.columns]
            X = df_encoded[features_disponibles]
            
            # Predecir
            predictions = model_service.predict_gasto(X)
            gasto_total = float(predictions.sum())
            
            logger.info(f" Predicción ML: ${gasto_total:,.2f}")
            return gasto_total
            
        except Exception as e:
            logger.warning(f" Error en predicción ML: {e}. Usando método alternativo.")
            # Fallback: usar promedio histórico con tendencia
            return float(df_features['TOTALPESOS'].sum() * 1.05)
    
    def _predict_monthly_with_prophet(
        self,
        df: pd.DataFrame,
        anio: int,
        incluir_intervalos: bool
    ) -> List[GastoMensual]:
        """Predice distribución mensual usando Prophet"""
        try:
            # Agregar datos mensuales
            df_mensual = df.groupby(pd.Grouper(key='FECHAPEDIDO', freq='M'))['TOTALPESOS'].sum().reset_index()
            df_mensual.columns = ['ds', 'y']
            df_mensual = df_mensual[df_mensual['ds'].notna()]
            
            # Calcular meses a proyectar
            ultimo_mes = df_mensual['ds'].max()
            meses_hasta_objetivo = ((anio - ultimo_mes.year) * 12) + (12 - ultimo_mes.month)
            
            if meses_hasta_objetivo <= 0:
                meses_hasta_objetivo = 12
            
            # Pronóstico con Prophet
            forecast = model_service.forecast_temporal(periods=meses_hasta_objetivo, freq='M')
            
            # Filtrar solo meses del año objetivo
            forecast['year'] = pd.to_datetime(forecast['ds']).dt.year
            forecast['month'] = pd.to_datetime(forecast['ds']).dt.month
            forecast_anio = forecast[forecast['year'] == anio]
            
            gastos_mensuales = []
            
            for _, row in forecast_anio.iterrows():
                mes = int(row['month'])
                gasto_pred = max(0, float(row['yhat']))
                
                intervalo_inf = max(0, float(row['yhat_lower'])) if incluir_intervalos else None
                intervalo_sup = max(0, float(row['yhat_upper'])) if incluir_intervalos else None
                
                gastos_mensuales.append(GastoMensual(
                    mes=mes,
                    mes_nombre=MESES_NOMBRES[mes - 1],
                    gasto_predicho=gasto_pred,
                    intervalo_inferior=intervalo_inf,
                    intervalo_superior=intervalo_sup
                ))
            
            # Completar meses faltantes
            meses_existentes = {gm.mes for gm in gastos_mensuales}
            for mes in range(1, 13):
                if mes not in meses_existentes:
                    # Usar promedio mensual histórico
                    gasto_prom = float(df[df['MES'] == mes]['TOTALPESOS'].mean())
                    gastos_mensuales.append(GastoMensual(
                        mes=mes,
                        mes_nombre=MESES_NOMBRES[mes - 1],
                        gasto_predicho=gasto_prom
                    ))
            
            return sorted(gastos_mensuales, key=lambda x: x.mes)
            
        except Exception as e:
            logger.warning(f" Error en Prophet: {e}. Usando distribución histórica.")
            return self._fallback_monthly_distribution(df)
    
    def _fallback_monthly_distribution(self, df: pd.DataFrame) -> List[GastoMensual]:
        """Distribución mensual de fallback basada en promedios históricos"""
        gastos_mensuales = []
        
        gasto_mensual_prom = df.groupby('MES')['TOTALPESOS'].mean().to_dict()
        
        for mes in range(1, 13):
            gasto_prom = gasto_mensual_prom.get(mes, df['TOTALPESOS'].mean())
            
            gastos_mensuales.append(GastoMensual(
                mes=mes,
                mes_nombre=MESES_NOMBRES[mes - 1],
                gasto_predicho=float(gasto_prom)
            ))
        
        return gastos_mensuales
    
    def _calculate_dimension_forecast(
        self,
        df_historico: pd.DataFrame,
        df_forecast: pd.DataFrame,
        dimensiones: List[DimensionAnalisis]
    ) -> Dict[str, List[DesgloseDimension]]:
        """Calcula desglose de proyección por dimensiones"""
        desgloses = {}
        
        for dimension in dimensiones:
            dim_col = dimension.value
            
            if dim_col not in df_forecast.columns:
                continue
            
            desglose_list = []
            
            # Agrupar por dimensión
            agrupado = df_forecast.groupby(dim_col)['TOTALPESOS'].agg(['sum', 'mean', 'count']).reset_index()
            agrupado.columns = [dim_col, 'gasto_total', 'gasto_promedio', 'cantidad']
            
            total_general = agrupado['gasto_total'].sum()
            
            for _, row in agrupado.iterrows():
                porcentaje = (row['gasto_total'] / total_general * 100) if total_general > 0 else 0
                
                desglose_list.append(DesgloseDimension(
                    dimension=dimension,
                    valor=str(row[dim_col]),
                    gasto_total=float(row['gasto_total']),
                    gasto_promedio=float(row['gasto_promedio']),
                    cantidad_pedidos=int(row['cantidad']),
                    porcentaje_total=float(porcentaje)
                ))
            
            # Ordenar por gasto total
            desglose_list.sort(key=lambda x: x.gasto_total, reverse=True)
            desgloses[dim_col] = desglose_list[:10]  # Top 10
        
        return desgloses
    
    def _generate_projection_alerts(
        self,
        gasto_proyectado: float,
        presupuesto: Optional[float],
        diferencia_pct: Optional[float]
    ) -> List[Alerta]:
        """Genera alertas para proyección"""
        alertas = []
        
        # Alerta: Exceso presupuestal proyectado
        if presupuesto and diferencia_pct and diferencia_pct > 5:
            severidad = NivelSeveridad.CRITICO if diferencia_pct > 15 else NivelSeveridad.ALTO
            
            alertas.append(Alerta(
                tipo=TipoAlerta.DESVIACION_PRESUPUESTO,
                severidad=severidad,
                titulo="Proyección excede presupuesto",
                descripcion=f"La proyección (${gasto_proyectado:,.2f}) excede el presupuesto "
                           f"en {diferencia_pct:.1f}% (${gasto_proyectado - presupuesto:,.2f})",
                valor_observado=gasto_proyectado,
                valor_esperado=presupuesto
            ))
        
        return alertas


class ForecastService:
    """Servicio principal de pronóstico que selecciona la estrategia adecuada"""
    
    def __init__(self):
        self.monitoreo_strategy = MonitoreoStrategy()
        self.proyeccion_strategy = ProyeccionStrategy()
    
    def forecast_dynamic(
        self,
        df: pd.DataFrame,
        anio_objetivo: int,
        presupuesto_asignado: Optional[float] = None,
        dimensiones: List[DimensionAnalisis] = None,
        incluir_intervalos_confianza: bool = True
    ):
        """
        Ejecuta pronóstico dinámico (monitoreo o proyección según el año)
        
        Args:
            df: DataFrame con datos
            anio_objetivo: Año a analizar
            presupuesto_asignado: Presupuesto opcional
            dimensiones: Dimensiones para desglose
            incluir_intervalos_confianza: Si incluir intervalos
            
        Returns:
            ForecastMonitoreoResponse o ForecastProyeccionResponse
        """
        anio_actual = datetime.now().year
        
        # Seleccionar estrategia según el año
        if anio_objetivo <= anio_actual:
            logger.info(f" Modo: MONITOREO (año actual/pasado)")
            return self.monitoreo_strategy.execute(
                df, anio_objetivo, presupuesto_asignado,
                dimensiones, incluir_intervalos_confianza
            )
        else:
            logger.info(f"🔮 Modo: PROYECCIÓN (año futuro)")
            return self.proyeccion_strategy.execute(
                df, anio_objetivo, presupuesto_asignado,
                dimensiones, incluir_intervalos_confianza
            )


# Singleton del servicio
forecast_service = ForecastService()