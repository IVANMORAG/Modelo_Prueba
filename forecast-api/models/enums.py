"""
Enumeraciones y Constantes del Dominio
"""
from enum import Enum


class ModoOperacion(str, Enum):
    """Modos de operación de la API"""
    MONITOREO = "monitoreo"
    PROYECCION = "proyeccion"


class DimensionAnalisis(str, Enum):
    """Dimensiones disponibles para análisis"""
    ORIGEN = "ORIGEN"
    UN = "UN"
    CATEGORIA = "CATEGORIA"
    CLASE = "CLASE"
    FAMILIA = "FAMILIA"
    DESTINO = "DESTINO"
    EMPRESA = "EMPRESA"


class TipoAlerta(str, Enum):
    """Tipos de alertas del sistema"""
    ANOMALIA = "anomalia"
    DESVIACION_PRESUPUESTO = "desviacion_presupuesto"
    GASTO_ELEVADO = "gasto_elevado"
    PATRON_INUSUAL = "patron_inusual"


class NivelSeveridad(str, Enum):
    """Niveles de severidad para alertas"""
    CRITICO = "critico"
    ALTO = "alto"
    MEDIO = "medio"
    BAJO = "bajo"
    INFO = "info"


class EstadoCalidadDatos(str, Enum):
    """Estados de calidad de datos"""
    EXCELENTE = "excelente"
    BUENA = "buena"
    ACEPTABLE = "aceptable"
    DEFICIENTE = "deficiente"


# Constantes
COLUMNAS_REQUERIDAS = [
    "ORIGEN", "EMPRESA", "UN", "FECHAPEDIDO", "CENTROCOSTO",
    "DESTINO", "TOTALPESOS", "DETALLEFINAL", "SOLICITUD",
    "POSICION", "CATEGORIA", "CLASE", "FAMILIA"
]

COLUMNAS_NUMERICAS = ["CENTROCOSTO", "TOTALPESOS", "SOLICITUD", "POSICION"]

COLUMNAS_CATEGORICAS = [
    "ORIGEN", "EMPRESA", "UN", "DESTINO", "CATEGORIA", "CLASE", "FAMILIA"
]

COLUMNAS_FECHA = ["FECHASOLICITUD", "FECHAPEDIDO"]

# Features para el modelo (sin PRECIO_UNITARIO para evitar data leakage)
FEATURES_MODELO = [
    'CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN',
    'CENTROCOSTO', 'AÑO', 'MES', 'TRIMESTRE', 'DIA_SEMANA',
    'SEMANA_AÑO', 'ES_FIN_MES', 'ES_INICIO_MES', 'ES_FIN_TRIMESTRE',
    'CANTIDAD', 'GASTO_PROM_CAT', 'GASTO_MED_CAT', 'GASTO_STD_CAT',
    'GASTO_PROM_DEST', 'GASTO_MED_DEST', 'GASTO_PROM_ORIG',
    'GASTO_PROM_UN', 'RATIO_GASTO_CAT', 'RATIO_GASTO_DEST',
    'GASTO_MES_ANTERIOR'
]

MESES_NOMBRES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]