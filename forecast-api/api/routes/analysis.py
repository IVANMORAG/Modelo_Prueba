"""
Rutas de API para Análisis (Anomalías y Recomendaciones)
"""
from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas import (
    AnomalyDetectionRequest, AnomalyDetectionResponse,
    RecommendationRequest, RecommendationResponse
)
from services.anomaly_detector import AnomalyDetector
from services.recommendation_engine import RecommendationEngine
from repositories.data_repository import DataRepository
from api.dependencies import (
    get_anomaly_detector,
    get_recommendation_engine,
    get_data_repository
)
from utils.logger import logger


router = APIRouter(prefix="/analysis", tags=["Análisis"])


@router.post(
    "/anomalies/",
    response_model=AnomalyDetectionResponse,
    summary="Detección de Anomalías en Gastos",
    description="""
    **Detecta gastos anómalos usando Isolation Forest.**
    
    **Características:**
    - Identificación automática de gastos inusuales
    - Top N anomalías por monto
    - Distribución de anomalías por ORIGEN y CATEGORIA
    - Alertas automáticas para patrones sospechosos
    - Contexto detallado de cada anomalía
    
    **Casos de uso:**
    - Auditoría de gastos
    - Detección de fraude
    - Identificación de errores de captura
    - Control presupuestal
    """
)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    anomaly_detector: AnomalyDetector = Depends(get_anomaly_detector),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Detecta anomalías en gastos
    
    - **contamination**: Proporción esperada de anomalías (0.01-0.20)
    - **top_n**: Número de anomalías top a retornar
    - **incluir_contexto**: Si incluir información contextual
    """
    try:
        logger.info("🔍 Iniciando detección de anomalías")
        
        # Obtener datos del último año
        from datetime import datetime
        anio_actual = datetime.now().year
        df = data_repo.get_compras_año(anio_actual)
        
        if len(df) == 0:
            # Si no hay datos actuales, usar año anterior
            df = data_repo.get_compras_año(anio_actual - 1)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay datos disponibles para análisis"
            )
        
        # Detectar anomalías
        resultado = anomaly_detector.detect_anomalies(
            df=df,
            contamination=request.contamination,
            top_n=request.top_n,
            incluir_contexto=request.incluir_contexto
        )
        
        logger.info(
            f"✅ Detección completada: {resultado.total_anomalias} anomalías "
            f"({resultado.porcentaje_anomalias:.2f}%)"
        )
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error en detección de anomalías: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/anomalies/year/{anio}",
    summary="Anomalías de un año específico",
    description="Detecta anomalías para un año en particular"
)
async def detect_anomalies_year(
    anio: int,
    top_n: int = 20,
    anomaly_detector: AnomalyDetector = Depends(get_anomaly_detector),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Detecta anomalías para un año específico
    
    - **anio**: Año a analizar
    - **top_n**: Número de anomalías top
    """
    try:
        # Obtener datos del año
        df = data_repo.get_compras_año(anio)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No hay datos disponibles para el año {anio}"
            )
        
        # Detectar anomalías con configuración por defecto
        resultado = anomaly_detector.detect_anomalies(
            df=df,
            contamination=0.05,
            top_n=top_n,
            incluir_contexto=True
        )
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/recommendations/",
    response_model=RecommendationResponse,
    summary="Recomendaciones Estratégicas",
    description="""
    **Genera recomendaciones estratégicas basadas en análisis de datos.**
    
    **Análisis incluido:**
    - Optimización de costos por ORIGEN
    - Consolidación de compras
    - Eficiencia operacional por UN
    - Estrategias por categoría
    - Oportunidades de ahorro cuantificadas
    - Tendencias identificadas
    
    **Salidas:**
    - Recomendaciones priorizadas
    - Análisis por ORIGEN y UN
    - Tendencias significativas
    - Oportunidades de ahorro con impacto estimado
    - Resumen ejecutivo
    """
)
async def generate_recommendations(
    request: RecommendationRequest,
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Genera recomendaciones estratégicas
    
    - **dimensiones_analisis**: Dimensiones a analizar
    - **incluir_tendencias**: Si incluir análisis de tendencias
    """
    try:
        logger.info("🎯 Generando recomendaciones estratégicas")
        
        # Obtener datos históricos (últimos 2 años)
        from datetime import datetime
        anio_actual = datetime.now().year
        df = data_repo.get_compras_rango_años(anio_actual - 2, anio_actual)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay datos suficientes para generar recomendaciones"
            )
        
        # Generar recomendaciones
        resultado = recommendation_engine.generate_recommendations(
            df=df,
            dimensiones_analisis=request.dimensiones_analisis,
            incluir_tendencias=request.incluir_tendencias
        )
        
        logger.info(
            f"✅ Recomendaciones generadas: {len(resultado.recomendaciones)} "
            f"recomendaciones, {len(resultado.oportunidades_ahorro)} oportunidades"
        )
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error en generación de recomendaciones: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/recommendations/quick",
    summary="Recomendaciones rápidas",
    description="Obtiene recomendaciones con configuración por defecto"
)
async def quick_recommendations(
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """Genera recomendaciones rápidas con configuración por defecto"""
    try:
        from datetime import datetime
        from models.enums import DimensionAnalisis
        
        anio_actual = datetime.now().year
        df = data_repo.get_compras_rango_años(anio_actual - 1, anio_actual)
        
        if len(df) == 0:
            return {
                "mensaje": "No hay datos suficientes",
                "recomendaciones": []
            }
        
        # Generar con configuración básica
        resultado = recommendation_engine.generate_recommendations(
            df=df,
            dimensiones_analisis=[DimensionAnalisis.ORIGEN, DimensionAnalisis.UN],
            incluir_tendencias=False
        )
        
        # Retornar solo lo esencial
        return {
            "total_recomendaciones": len(resultado.recomendaciones),
            "recomendaciones_top": resultado.recomendaciones[:5],
            "oportunidades_ahorro": resultado.oportunidades_ahorro[:3],
            "resumen_ejecutivo": resultado.resumen_ejecutivo
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )