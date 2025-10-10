"""
API de Pronóstico de Gasto y Monitoreo Presupuestal
Implementa principios SOLID y arquitectura modular
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import uvicorn

from config.settings import settings
from config.database import db_manager
from services.model_service import model_service
from api.routes import forecast, data_quality, analysis
from utils.logger import logger
from utils.exceptions import APIBaseException
from models.schemas import HealthResponse, ModelsInfoResponse, ModelInfo


# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación"""
    # Startup
    logger.info("=" * 70)
    logger.info(" INICIANDO API DE PRONÓSTICO Y MONITOREO PRESUPUESTAL")
    logger.info("=" * 70)
    
    # Validar artefactos
    artifacts_status = settings.validate_artifacts_exist()
    logger.info(f" Artefactos: {artifacts_status}")
    
    if not all(artifacts_status.values()):
        logger.warning(" Algunos artefactos no están disponibles")
        missing = [k for k, v in artifacts_status.items() if not v]
        logger.warning(f"   Faltantes: {missing}")
    
    # Probar conexión a base de datos
    try:
        if db_manager.test_connection():
            logger.info(" Conexión a base de datos establecida")
        else:
            logger.warning(" No se pudo conectar a la base de datos")
    except Exception as e:
        logger.error(f" Error al conectar a base de datos: {e}")
    
    # Verificar modelos cargados
    models_info = model_service.get_models_info()
    logger.info(" Modelos cargados:")
    for model_name, info in models_info.items():
        status_icon = "✅" if info.get('loaded', False) else "❌"
        logger.info(f"   {status_icon} {model_name}")
    
    logger.info("=" * 70)
    logger.info(f" API lista en http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f" Documentación: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info(" Cerrando API...")
    logger.info(" API cerrada correctamente")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    # API de Pronóstico de Gasto y Monitoreo Presupuestal
    
    API RESTful para análisis predictivo y monitoreo de gastos corporativos usando Machine Learning.
    
    ##  Funcionalidades Principales
    
    ###  Pronóstico Dinámico
    - **Monitoreo 2025**: Compara gasto real vs predicción inicial
    - **Proyección 2026+**: Predice gasto total para años futuros
    - Distribución mensual con intervalos de confianza
    - Desglose multidimensional (ORIGEN, UN, CATEGORIA, etc.)
    - Comparación con presupuesto asignado
    
    ###  Calidad de Datos
    - Verificación de completitud y consistencia
    - Identificación de problemas (nulos, negativos, duplicados)
    - Puntuación de calidad (0-100)
    - Recomendaciones de limpieza
    
    ###  Detección de Anomalías
    - Isolation Forest para detectar gastos inusuales
    - Top N anomalías con contexto
    - Alertas automáticas de patrones sospechosos
    - Distribución por dimensiones
    
    ###  Recomendaciones Estratégicas
    - Optimización de costos por proveedor
    - Oportunidades de consolidación
    - Análisis de tendencias
    - Ahorro potencial cuantificado
    
    ##  Arquitectura
    
    La API está construida siguiendo principios SOLID:
    - **SRP**: Cada servicio tiene una responsabilidad única
    - **OCP**: Extensible mediante estrategias
    - **DIP**: Abstracción de capas de datos y modelos
    
    ##  Modelos ML
    
    - **XGBoost Regressor** (R² = 99.43%): Predicción de gasto
    - **Prophet**: Series temporales y estacionalidad
    - **Isolation Forest**: Detección de anomalías
    
    ##  Guía de Uso
    
    1. Verificar salud: `GET /health`
    2. Validar calidad de datos: `POST /data/quality-check/`
    3. Generar pronóstico: `POST /forecast/dynamic/`
    4. Detectar anomalías: `POST /analysis/anomalies/`
    5. Obtener recomendaciones: `POST /analysis/recommendations/`
    """,
    lifespan=lifespan,
    debug=settings.DEBUG
)


# ============================================================================
# Middleware
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(APIBaseException)
async def api_exception_handler(request: Request, exc: APIBaseException):
    """Handler para excepciones personalizadas de la API"""
    logger.error(f" API Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "type": exc.__class__.__name__
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler para errores de validación de Pydantic"""
    logger.error(f" Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Error de validación de datos",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler general para excepciones no capturadas"""
    logger.error(f" Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc) if settings.DEBUG else "Contacte al administrador"
        }
    )


# ============================================================================
# Routes
# ============================================================================

# Incluir routers
app.include_router(forecast.router)
app.include_router(data_quality.router)
app.include_router(analysis.router)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get(
    "/",
    summary="Root",
    description="Información básica de la API"
)
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "api": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica el estado de la API y sus dependencias"
)
async def health_check():
    """
    Verifica el estado de salud de la API
    
    Retorna:
    - Estado general
    - Conexión a base de datos
    - Estado de modelos ML
    - Versión de la API
    """
    try:
        # Verificar base de datos
        db_connected = db_manager.test_connection()
        
        # Verificar modelos
        models_info = model_service.get_models_info()
        models_loaded = {
            name: info.get('loaded', False)
            for name, info in models_info.items()
        }
        
        # Determinar estado general
        all_healthy = db_connected and all(models_loaded.values())
        status_str = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            status=status_str,
            database_connected=db_connected,
            models_loaded=models_loaded,
            api_version=settings.API_VERSION
        )
        
    except Exception as e:
        logger.error(f" Error en health check: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            models_loaded={},
            api_version=settings.API_VERSION
        )


@app.get(
    "/models/info",
    response_model=ModelsInfoResponse,
    summary="Información de Modelos",
    description="Obtiene información detallada de los modelos ML cargados"
)
async def models_info():
    """
    Retorna información de los modelos ML
    
    Incluye:
    - Tipo de modelo
    - Métricas de rendimiento
    - Fecha de carga
    - Estado de carga
    """
    try:
        models_info_dict = model_service.get_models_info()
        
        modelos = []
        total_cargados = 0
        
        for nombre, info in models_info_dict.items():
            if nombre == 'encoders':
                continue
                
            cargado = info.get('loaded', False)
            if cargado:
                total_cargados += 1
            
            modelo = ModelInfo(
                nombre=nombre,
                tipo=info.get('type', 'Unknown'),
                version=None,
                metricas=info.get('metrics'),
                cargado=cargado,
                fecha_carga=info.get('timestamp')
            )
            modelos.append(modelo)
        
        return ModelsInfoResponse(
            modelos=modelos,
            total_modelos=len(modelos),
            modelos_cargados=total_cargados,
            artifacts_path=str(settings.ARTIFACTS_DIR)
        )
        
    except Exception as e:
        logger.error(f" Error al obtener info de modelos: {e}")
        raise


@app.post(
    "/models/reload",
    summary="Recargar Modelos",
    description="Recarga todos los modelos ML desde disco (solo para mantenimiento)"
)
async def reload_models():
    """
    Recarga todos los modelos ML
    
    **Advertencia**: Esta operación puede tardar varios segundos
    """
    try:
        logger.info(" Recargando modelos...")
        model_service.reload_models()
        logger.info(" Modelos recargados exitosamente")
        
        return {
            "status": "success",
            "mensaje": "Modelos recargados exitosamente",
            "modelos": model_service.get_models_info()
        }
        
    except Exception as e:
        logger.error(f" Error al recargar modelos: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )