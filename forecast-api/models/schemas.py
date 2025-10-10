"""
Esquemas Pydantic para Request/Response
Implementa Interface Segregation Principle (ISP)
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from models.enums import (
    ModoOperacion, TipoAlerta, NivelSeveridad, 
    EstadoCalidadDatos, DimensionAnalisis
)


# ============================================================================
# Request Models
# ============================================================================

class ForecastDynamicRequest(BaseModel):
    """Request para pronóstico dinámico (monitoreo o proyección)"""
    anio_objetivo: int = Field(..., ge=2025, le=2030, description="Año objetivo para el pronóstico")
    presupuesto_asignado: Optional[float] = Field(None, gt=0, description="Presupuesto asignado (opcional)")
    dimensiones: List[DimensionAnalisis] = Field(
        default=[DimensionAnalisis.ORIGEN, DimensionAnalisis.CATEGORIA],
        description="Dimensiones para el desglose"
    )
    incluir_intervalos_confianza: bool = Field(default=True, description="Incluir intervalos de confianza")
    
    @field_validator('anio_objetivo')
    def validar_anio(cls, v):
        current_year = datetime.now().year
        if v < current_year:
            raise ValueError(f"El año objetivo no puede ser menor al año actual ({current_year})")
        return v


class DataQualityRequest(BaseModel):
    """Request para verificación de calidad de datos"""
    incluir_detalles: bool = Field(default=True, description="Incluir detalles de registros problemáticos")
    limite_ejemplos: int = Field(default=10, ge=1, le=100, description="Límite de ejemplos a mostrar")


class AnomalyDetectionRequest(BaseModel):
    """Request para detección de anomalías"""
    contamination: float = Field(default=0.05, ge=0.01, le=0.2, description="Proporción esperada de anomalías")
    top_n: int = Field(default=20, ge=5, le=100, description="Número de anomalías top a retornar")
    incluir_contexto: bool = Field(default=True, description="Incluir contexto de las anomalías")


class RecommendationRequest(BaseModel):
    """Request para generación de recomendaciones"""
    dimensiones_analisis: List[DimensionAnalisis] = Field(
        default=[DimensionAnalisis.ORIGEN, DimensionAnalisis.UN],
        description="Dimensiones para análisis"
    )
    incluir_tendencias: bool = Field(default=True, description="Incluir análisis de tendencias")


# ============================================================================
# Response Models - Componentes
# ============================================================================

class GastoMensual(BaseModel):
    """Representación de gasto mensual"""
    mes: int = Field(..., ge=1, le=12)
    mes_nombre: str
    gasto_predicho: float
    gasto_real: Optional[float] = None
    desviacion: Optional[float] = None
    desviacion_porcentual: Optional[float] = None
    intervalo_inferior: Optional[float] = None
    intervalo_superior: Optional[float] = None


class DesgloseDimension(BaseModel):
    """Desglose de gasto por dimensión específica"""
    dimension: DimensionAnalisis
    valor: str
    gasto_total: float
    gasto_promedio: float
    cantidad_pedidos: int
    porcentaje_total: float


class Alerta(BaseModel):
    """Modelo de alerta del sistema"""
    tipo: TipoAlerta
    severidad: NivelSeveridad
    titulo: str
    descripcion: str
    valor_observado: float
    valor_esperado: Optional[float] = None
    fecha_deteccion: datetime = Field(default_factory=datetime.now)
    dimension: Optional[str] = None
    contexto: Optional[Dict[str, Any]] = None


class Recomendacion(BaseModel):
    """Modelo de recomendación"""
    titulo: str
    descripcion: str
    prioridad: NivelSeveridad
    categoria: str
    impacto_estimado: Optional[str] = None
    acciones_sugeridas: List[str]


# ============================================================================
# Response Models - Principales
# ============================================================================

class ForecastMonitoreoResponse(BaseModel):
    """Response para modo monitoreo (año actual)"""
    modo: ModoOperacion = ModoOperacion.MONITOREO
    anio: int
    fecha_generacion: datetime = Field(default_factory=datetime.now)
    
    # Resumen general
    gasto_real_acumulado: float
    gasto_predicho_total: float
    desviacion_total: float
    desviacion_porcentual: float
    
    # Detalles mensuales
    gastos_mensuales: List[GastoMensual]
    
    # Desgloses por dimensión
    desgloses: Dict[str, List[DesgloseDimension]]
    
    # Alertas y análisis
    alertas: List[Alerta]
    presupuesto_asignado: Optional[float] = None
    presupuesto_restante: Optional[float] = None
    proyeccion_fin_anio: float


class ForecastProyeccionResponse(BaseModel):
    """Response para modo proyección (año futuro)"""
    modo: ModoOperacion = ModoOperacion.PROYECCION
    anio: int
    fecha_generacion: datetime = Field(default_factory=datetime.now)
    
    # Proyección total
    gasto_proyectado_total: float
    intervalo_confianza_inferior: Optional[float] = None
    intervalo_confianza_superior: Optional[float] = None
    
    # Distribución mensual
    proyeccion_mensual: List[GastoMensual]
    
    # Desgloses por dimensión
    desgloses: Dict[str, List[DesgloseDimension]]
    
    # Comparación con presupuesto
    presupuesto_asignado: Optional[float] = None
    diferencia_presupuesto: Optional[float] = None
    diferencia_porcentual: Optional[float] = None
    
    # Alertas
    alertas: List[Alerta]
    
    # Métricas de confianza
    nivel_confianza: float
    datos_historicos_meses: int


class ProblemaCalidad(BaseModel):
    """Representación de un problema de calidad de datos"""
    columna: str
    tipo_problema: str
    cantidad: int
    porcentaje: float
    ejemplos: Optional[List[Dict[str, Any]]] = None


class DataQualityResponse(BaseModel):
    """Response para verificación de calidad de datos"""
    fecha_analisis: datetime = Field(default_factory=datetime.now)
    total_registros: int
    estado_general: EstadoCalidadDatos
    puntuacion_calidad: float = Field(..., ge=0, le=100)
    
    # Análisis de completitud
    valores_nulos: Dict[str, int]
    porcentaje_nulos: Dict[str, float]
    
    # Análisis de consistencia
    registros_invalidos: int
    totalpesos_negativos: int
    fechas_invalidas: int
    
    # Problemas identificados
    problemas: List[ProblemaCalidad]
    
    # Recomendaciones
    requiere_limpieza: bool
    recomendaciones: List[str]


class AnomaliaDetectada(BaseModel):
    """Representación de una anomalía detectada"""
    fecha_pedido: date
    origen: str
    categoria: str
    destino: str
    totalpesos: float
    desviacion_estandar: float
    score_anomalia: float
    detalle: Optional[str] = None
    contexto: Optional[Dict[str, Any]] = None


class AnomalyDetectionResponse(BaseModel):
    """Response para detección de anomalías"""
    fecha_analisis: datetime = Field(default_factory=datetime.now)
    total_registros_analizados: int
    total_anomalias: int
    porcentaje_anomalias: float
    
    # Estadísticas
    gasto_promedio_normal: float
    gasto_promedio_anomalo: float
    gasto_maximo_anomalo: float
    
    # Anomalías top
    anomalias_top: List[AnomaliaDetectada]
    
    # Distribución por dimensión
    anomalias_por_origen: Dict[str, int]
    anomalias_por_categoria: Dict[str, int]
    
    # Alertas generadas
    alertas: List[Alerta]


class TendenciaDimension(BaseModel):
    """Tendencia para una dimensión específica"""
    dimension: str
    valor: str
    tendencia: str  # "creciente", "decreciente", "estable"
    variacion_porcentual: float
    gasto_actual: float
    gasto_anterior: float


class RecommendationResponse(BaseModel):
    """Response para recomendaciones estratégicas"""
    fecha_generacion: datetime = Field(default_factory=datetime.now)
    
    # Recomendaciones principales
    recomendaciones: List[Recomendacion]
    
    # Análisis por dimensión
    analisis_origen: Dict[str, Dict[str, Any]]
    analisis_un: Dict[str, Dict[str, Any]]
    
    # Tendencias
    tendencias: List[TendenciaDimension]
    
    # Oportunidades identificadas
    oportunidades_ahorro: List[Dict[str, Any]]
    
    # Resumen ejecutivo
    resumen_ejecutivo: str


# ============================================================================
# Health & Info Models
# ============================================================================

class ModelInfo(BaseModel):
    """Información de un modelo cargado"""
    nombre: str
    tipo: str
    version: Optional[str] = None
    metricas: Optional[Dict[str, float]] = None
    cargado: bool
    fecha_carga: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Response para health check"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    database_connected: bool
    models_loaded: Dict[str, bool]
    api_version: str


class ModelsInfoResponse(BaseModel):
    """Response con información de todos los modelos"""
    modelos: List[ModelInfo]
    total_modelos: int
    modelos_cargados: int
    artifacts_path: str