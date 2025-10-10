"""
Servicio de Detección de Anomalías
Implementa Single Responsibility Principle (SRP)
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from sklearn.ensemble import IsolationForest

from models.schemas import (
    AnomalyDetectionResponse, AnomaliaDetectada, Alerta
)
from models.enums import TipoAlerta, NivelSeveridad
from utils.logger import logger


class AnomalyDetector:
    """Detector de gastos anómalos usando Isolation Forest"""
    
    def __init__(self):
        self.model = None
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float = 0.05,
        top_n: int = 20,
        incluir_contexto: bool = True
    ) -> AnomalyDetectionResponse:
        """
        Detecta anomalías en los gastos
        
        Args:
            df: DataFrame con datos históricos
            contamination: Proporción esperada de anomalías
            top_n: Número de anomalías top a retornar
            incluir_contexto: Si incluir contexto adicional
            
        Returns:
            AnomalyDetectionResponse con anomalías detectadas
        """
        logger.info(f" Detectando anomalías en {len(df)} registros...")
        
        # Validar que TOTALPESOS exista
        if 'TOTALPESOS' not in df.columns:
            raise ValueError("Columna TOTALPESOS no encontrada")
        
        # Preparar datos
        df_clean = df[df['TOTALPESOS'] > 0].copy()
        
        if len(df_clean) == 0:
            logger.warning(" No hay datos válidos para análisis de anomalías")
            return self._empty_response()
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        df_clean['anomalia_score'] = iso_forest.fit_predict(
            df_clean[['TOTALPESOS']].values
        )
        
        # -1 es anomalía, 1 es normal
        df_anomalias = df_clean[df_clean['anomalia_score'] == -1].copy()
        df_normales = df_clean[df_clean['anomalia_score'] == 1].copy()
        
        # Calcular score de anomalía (distancia)
        decision_scores = iso_forest.decision_function(df_clean[['TOTALPESOS']].values)
        df_clean['anomalia_distance'] = -decision_scores  # Invertir para que mayor = más anómalo
        
        # Obtener top anomalías
        df_top_anomalias = df_anomalias.nlargest(top_n, 'TOTALPESOS')
        
        # Calcular estadísticas
        gasto_promedio_normal = float(df_normales['TOTALPESOS'].mean())
        gasto_promedio_anomalo = float(df_anomalias['TOTALPESOS'].mean())
        gasto_maximo_anomalo = float(df_anomalias['TOTALPESOS'].max())
        
        # Construir lista de anomalías
        anomalias_top = self._build_anomalies_list(
            df_top_anomalias, 
            gasto_promedio_normal,
            incluir_contexto
        )
        
        # Distribución por dimensión
        anomalias_por_origen = self._count_by_dimension(df_anomalias, 'ORIGEN')
        anomalias_por_categoria = self._count_by_dimension(df_anomalias, 'CATEGORIA')
        
        # Generar alertas
        alertas = self._generate_alerts(
            df_anomalias, 
            gasto_promedio_normal,
            gasto_maximo_anomalo
        )
        
        response = AnomalyDetectionResponse(
            total_registros_analizados=len(df_clean),
            total_anomalias=len(df_anomalias),
            porcentaje_anomalias=(len(df_anomalias) / len(df_clean) * 100),
            gasto_promedio_normal=gasto_promedio_normal,
            gasto_promedio_anomalo=gasto_promedio_anomalo,
            gasto_maximo_anomalo=gasto_maximo_anomalo,
            anomalias_top=anomalias_top,
            anomalias_por_origen=anomalias_por_origen,
            anomalias_por_categoria=anomalias_por_categoria,
            alertas=alertas
        )
        
        logger.info(
            f" Detección completada: {len(df_anomalias)} anomalías "
            f"({response.porcentaje_anomalias:.2f}%)"
        )
        
        return response
    
    def _build_anomalies_list(
        self,
        df_anomalias: pd.DataFrame,
        gasto_promedio_normal: float,
        incluir_contexto: bool
    ) -> List[AnomaliaDetectada]:
        """Construye lista de objetos AnomaliaDetectada"""
        anomalias = []
        
        for _, row in df_anomalias.iterrows():
            # Calcular desviación estándar
            desv_std = (row['TOTALPESOS'] - gasto_promedio_normal) / gasto_promedio_normal
            
            # Score de anomalía
            score = row.get('anomalia_distance', 0.0)
            
            # Contexto adicional
            contexto = None
            if incluir_contexto:
                contexto = {
                    'solicitud': str(row.get('SOLICITUD', 'N/A')),
                    'posicion': str(row.get('POSICION', 'N/A')),
                    'empresa': str(row.get('EMPRESA', 'N/A')),
                    'un': str(row.get('UN', 'N/A'))
                }
            
            anomalia = AnomaliaDetectada(
                fecha_pedido=pd.to_datetime(row['FECHAPEDIDO']).date(),
                origen=str(row.get('ORIGEN', 'N/A')),
                categoria=str(row.get('CATEGORIA', 'N/A')),
                destino=str(row.get('DESTINO', 'N/A')),
                totalpesos=float(row['TOTALPESOS']),
                desviacion_estandar=float(desv_std),
                score_anomalia=float(score),
                detalle=str(row.get('DETALLEFINAL', ''))[:200],  # Truncar
                contexto=contexto
            )
            anomalias.append(anomalia)
        
        return anomalias
    
    def _count_by_dimension(
        self, 
        df: pd.DataFrame, 
        dimension: str
    ) -> Dict[str, int]:
        """Cuenta anomalías por dimensión"""
        if dimension not in df.columns:
            return {}
        
        return df[dimension].value_counts().head(10).to_dict()
    
    def _generate_alerts(
        self,
        df_anomalias: pd.DataFrame,
        gasto_promedio_normal: float,
        gasto_maximo_anomalo: float
    ) -> List[Alerta]:
        """Genera alertas basadas en las anomalías detectadas"""
        alertas = []
        
        # Alerta: Volumen alto de anomalías
        if len(df_anomalias) > 100:
            alertas.append(Alerta(
                tipo=TipoAlerta.ANOMALIA,
                severidad=NivelSeveridad.ALTO,
                titulo="Alto volumen de anomalías detectadas",
                descripcion=f"Se detectaron {len(df_anomalias)} gastos anómalos. "
                           f"Revisar procesos de aprobación y validación.",
                valor_observado=float(len(df_anomalias)),
                valor_esperado=None
            ))
        
        # Alerta: Gasto máximo muy elevado
        if gasto_maximo_anomalo > gasto_promedio_normal * 50:
            alertas.append(Alerta(
                tipo=TipoAlerta.GASTO_ELEVADO,
                severidad=NivelSeveridad.CRITICO,
                titulo="Gasto anómalo extremadamente elevado",
                descripcion=f"Gasto máximo anómalo: ${gasto_maximo_anomalo:,.2f} "
                           f"(50x superior al promedio normal de ${gasto_promedio_normal:,.2f})",
                valor_observado=gasto_maximo_anomalo,
                valor_esperado=gasto_promedio_normal,
                contexto={"ratio": gasto_maximo_anomalo / gasto_promedio_normal}
            ))
        
        # Alerta: Concentración en un ORIGEN específico
        if 'ORIGEN' in df_anomalias.columns:
            origen_counts = df_anomalias['ORIGEN'].value_counts()
            if len(origen_counts) > 0:
                origen_max = origen_counts.idxmax()
                count_max = origen_counts.max()
                
                if count_max > len(df_anomalias) * 0.5:  # Más del 50%
                    alertas.append(Alerta(
                        tipo=TipoAlerta.PATRON_INUSUAL,
                        severidad=NivelSeveridad.MEDIO,
                        titulo=f"Concentración de anomalías en {origen_max}",
                        descripcion=f"{count_max} de {len(df_anomalias)} anomalías "
                                   f"provienen de {origen_max}. Revisar proceso de compra.",
                        valor_observado=float(count_max),
                        dimension="ORIGEN",
                        contexto={"origen": origen_max}
                    ))
        
        return alertas
    
    def _empty_response(self) -> AnomalyDetectionResponse:
        """Retorna respuesta vacía cuando no hay datos"""
        return AnomalyDetectionResponse(
            total_registros_analizados=0,
            total_anomalias=0,
            porcentaje_anomalias=0.0,
            gasto_promedio_normal=0.0,
            gasto_promedio_anomalo=0.0,
            gasto_maximo_anomalo=0.0,
            anomalias_top=[],
            anomalias_por_origen={},
            anomalias_por_categoria={},
            alertas=[]
        )


# Singleton del detector
anomaly_detector = AnomalyDetector()