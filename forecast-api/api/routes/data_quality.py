"""
Rutas de API para Calidad de Datos
"""
from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas import DataQualityRequest, DataQualityResponse
from services.data_validator import DataValidator
from repositories.data_repository import DataRepository
from api.dependencies import get_data_validator, get_data_repository
from utils.logger import logger


router = APIRouter(prefix="/data", tags=["Calidad de Datos"])


@router.post(
    "/quality-check/",
    response_model=DataQualityResponse,
    summary="Verificación de Calidad de Datos",
    description="""
    **Analiza la calidad de los datos de compras.**
    
    **Verificaciones realizadas:**
    - Completitud (valores nulos)
    - Consistencia (TOTALPESOS negativos, fechas inválidas)
    - Valores "Desconocido" en columnas clave
    - Registros duplicados o inválidos
    
    **Retorna:**
    - Puntuación de calidad (0-100)
    - Estado general (Excelente, Buena, Aceptable, Deficiente)
    - Problemas identificados con ejemplos
    - Recomendaciones de limpieza
    """
)
async def quality_check(
    request: DataQualityRequest,
    data_validator: DataValidator = Depends(get_data_validator),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """
    Ejecuta verificación de calidad de datos
    
    - **incluir_detalles**: Si incluir ejemplos de problemas
    - **limite_ejemplos**: Número máximo de ejemplos por problema
    """
    try:
        logger.info("🔍 Iniciando verificación de calidad de datos")
        
        # Obtener datos del último año para análisis
        from datetime import datetime
        anio_actual = datetime.now().year
        df = data_repo.get_compras_año(anio_actual)
        
        if len(df) == 0:
            # Si no hay datos del año actual, usar año anterior
            df = data_repo.get_compras_año(anio_actual - 1)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay datos disponibles para análisis de calidad"
            )
        
        # Ejecutar validación
        resultado = data_validator.validate_data_quality(
            df=df,
            incluir_detalles=request.incluir_detalles,
            limite_ejemplos=request.limite_ejemplos
        )
        
        logger.info(
            f"✅ Análisis completado: {resultado.estado_general.value} "
            f"(Puntuación: {resultado.puntuacion_calidad:.2f})"
        )
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de calidad: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/quality-summary/",
    summary="Resumen rápido de calidad",
    description="Obtiene un resumen rápido sin detalles"
)
async def quality_summary(
    data_validator: DataValidator = Depends(get_data_validator),
    data_repo: DataRepository = Depends(get_data_repository)
):
    """Obtiene resumen rápido de calidad de datos"""
    try:
        from datetime import datetime
        anio_actual = datetime.now().year
        df = data_repo.get_compras_año(anio_actual)
        
        if len(df) == 0:
            df = data_repo.get_compras_año(anio_actual - 1)
        
        if len(df) == 0:
            return {
                "status": "sin_datos",
                "mensaje": "No hay datos disponibles"
            }
        
        # Validación básica sin detalles
        resultado = data_validator.validate_data_quality(
            df=df,
            incluir_detalles=False,
            limite_ejemplos=0
        )
        
        return {
            "total_registros": resultado.total_registros,
            "estado_general": resultado.estado_general,
            "puntuacion_calidad": resultado.puntuacion_calidad,
            "requiere_limpieza": resultado.requiere_limpieza,
            "problemas_count": len(resultado.problemas)
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )