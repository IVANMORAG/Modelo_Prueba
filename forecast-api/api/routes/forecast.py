"""
Rutas de API para Pronóstico
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Union

from models.schemas import (
    ForecastDynamicRequest,
    ForecastMonitoreoResponse,
    ForecastProyeccionResponse
)
from services.forecast_service import ForecastService
from repositories.data_repository import DataRepository
from api.dependencies import (
    get_forecast_service,
    get_data_repository,
    get_validated_year
)
from utils.logger import logger
from utils.exceptions import ForecastException, InsufficientDataException


router = APIRouter(prefix="/forecast", tags=["Pronóstico"])


@router.post(
    "/dynamic/",
    response_model=Union[ForecastMonitoreoResponse, ForecastProyeccionResponse],
    summary="Pronóstico Dinámico (Monitoreo o Proyección)",
    description="""
    **Endpoint principal para pronóstico y monitoreo presupuestal.**
    
    **Modos de operación:**
    - **Monitoreo (año actual ≤ 2025)**: Compara gasto real acumulado vs predicción inicial
    - **Proyección (año futuro > 2025)**: Predice gasto total del año futuro
    
    **Funcionalidades:**
    - Pronóstico de gasto total anual
    - Distribución mensual
    - Desglose por dimensiones (ORIGEN, UN, CATEGORIA, etc.)
    - Comparación con presupuesto asignado
    - Alertas automáticas de desviaciones
    - Intervalos de confianza
    """
)
async def forecast_dynamic(
    request: ForecastDynamicRequest,
    forecast_service: ForecastService = Depends(get_forecast_service),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Ejecuta pronóstico dinámico según el año objetivo
    
    - **anio_objetivo**: Año a analizar (2025 para monitoreo, 2026+ para proyección)
    - **presupuesto_asignado**: Presupuesto opcional para comparación
    - **dimensiones**: Dimensiones para desglose (ORIGEN, UN, CATEGORIA, etc.)
    - **incluir_intervalos_confianza**: Si incluir intervalos de confianza
    """
    try:
        logger.info(
            f"🚀 Iniciando pronóstico dinámico para año {request.anio_objetivo}"
        )
        
        # Validar año
        anio = get_validated_year(request.anio_objetivo)
        
        # Obtener datos históricos (últimos 5 años + año objetivo)
        anio_inicio = max(2016, anio - 5)
        anio_fin = anio
        
        logger.info(f"📥 Obteniendo datos desde {anio_inicio} hasta {anio_fin}")
        df = data_repo.get_compras_rango_años(anio_inicio, anio_fin)
        
        if len(df) == 0:
            raise InsufficientDataException(
                f"No hay datos disponibles para el rango {anio_inicio}-{anio_fin}"
            )
        
        # Ejecutar pronóstico
        resultado = forecast_service.forecast_dynamic(
            df=df,
            anio_objetivo=anio,
            presupuesto_asignado=request.presupuesto_asignado,
            dimensiones=request.dimensiones,
            incluir_intervalos_confianza=request.incluir_intervalos_confianza
        )
        
        logger.info(
            f"✅ Pronóstico completado para año {anio} "
            f"(Modo: {resultado.modo.value})"
        )
        
        return resultado
        
    except InsufficientDataException as e:
        logger.error(f"❌ Datos insuficientes: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except ForecastException as e:
        logger.error(f"❌ Error en pronóstico: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )


@router.get(
    "/year/{anio}",
    summary="Obtener pronóstico para un año específico",
    description="Atajo para obtener pronóstico de un año sin body complejo"
)
async def forecast_year(
    anio: int,
    presupuesto: float = None,
    forecast_service: ForecastService = Depends(get_forecast_service),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Obtiene pronóstico para un año específico (versión simplificada)
    
    - **anio**: Año objetivo
    - **presupuesto**: Presupuesto opcional
    """
    try:
        # Validar año
        anio = get_validated_year(anio)
        
        # Obtener datos
        anio_inicio = max(2016, anio - 5)
        df = data_repo.get_compras_rango_años(anio_inicio, anio)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No hay datos disponibles para el año {anio}"
            )
        
        # Ejecutar pronóstico con configuración por defecto
        resultado = forecast_service.forecast_dynamic(
            df=df,
            anio_objetivo=anio,
            presupuesto_asignado=presupuesto,
            dimensiones=[],
            incluir_intervalos_confianza=True
        )
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )