"""
Servicio de Validación y Calidad de Datos
Implementa Single Responsibility Principle (SRP)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from models.enums import (
    EstadoCalidadDatos, COLUMNAS_REQUERIDAS, 
    COLUMNAS_NUMERICAS, COLUMNAS_FECHA
)
from models.schemas import DataQualityResponse, ProblemaCalidad
from utils.logger import logger
from utils.exceptions import DataValidationException


class DataValidator:
    """Validador de calidad de datos"""
    
    def __init__(self):
        self.columnas_requeridas = COLUMNAS_REQUERIDAS
        self.columnas_numericas = COLUMNAS_NUMERICAS
        self.columnas_fecha = COLUMNAS_FECHA
    
    def validate_data_quality(
        self, 
        df: pd.DataFrame, 
        incluir_detalles: bool = True,
        limite_ejemplos: int = 10
    ) -> DataQualityResponse:
        """
        Valida la calidad de los datos de entrada
        
        Args:
            df: DataFrame a validar
            incluir_detalles: Si incluir ejemplos de problemas
            limite_ejemplos: Límite de ejemplos por problema
            
        Returns:
            DataQualityResponse con el análisis completo
        """
        logger.info(f" Iniciando validación de calidad de datos: {len(df)} registros")
        
        # Validaciones básicas
        self._validate_required_columns(df)
        
        # Análisis de completitud
        valores_nulos = self._analyze_missing_values(df)
        porcentaje_nulos = {
            col: (count / len(df) * 100) if len(df) > 0 else 0
            for col, count in valores_nulos.items()
        }
        
        # Análisis de consistencia
        totalpesos_negativos = self._count_negative_values(df, 'TOTALPESOS')
        fechas_invalidas = self._count_invalid_dates(df)
        registros_invalidos = self._count_invalid_records(df)
        
        # Identificar problemas
        problemas = self._identify_problems(
            df, valores_nulos, totalpesos_negativos, 
            fechas_invalidas, incluir_detalles, limite_ejemplos
        )
        
        # Calcular puntuación de calidad
        puntuacion_calidad = self._calculate_quality_score(
            df, valores_nulos, totalpesos_negativos, fechas_invalidas
        )
        
        # Determinar estado general
        estado_general = self._determine_quality_state(puntuacion_calidad)
        
        # Generar recomendaciones
        recomendaciones = self._generate_recommendations(problemas, estado_general)
        
        # Determinar si requiere limpieza
        requiere_limpieza = estado_general in [
            EstadoCalidadDatos.DEFICIENTE, 
            EstadoCalidadDatos.ACEPTABLE
        ]
        
        response = DataQualityResponse(
            total_registros=len(df),
            estado_general=estado_general,
            puntuacion_calidad=puntuacion_calidad,
            valores_nulos=valores_nulos,
            porcentaje_nulos=porcentaje_nulos,
            registros_invalidos=registros_invalidos,
            totalpesos_negativos=totalpesos_negativos,
            fechas_invalidas=fechas_invalidas,
            problemas=problemas,
            requiere_limpieza=requiere_limpieza,
            recomendaciones=recomendaciones
        )
        
        logger.info(f" Validación completada: Estado {estado_general.value}, Puntuación {puntuacion_calidad:.2f}")
        
        return response
    
    def _validate_required_columns(self, df: pd.DataFrame):
        """Valida que existan las columnas requeridas"""
        missing_columns = set(self.columnas_requeridas) - set(df.columns)
        if missing_columns:
            raise DataValidationException(
                f"Columnas requeridas faltantes: {missing_columns}"
            )
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analiza valores nulos por columna"""
        return df[self.columnas_requeridas].isnull().sum().to_dict()
    
    def _count_negative_values(self, df: pd.DataFrame, column: str) -> int:
        """Cuenta valores negativos en una columna"""
        if column not in df.columns:
            return 0
        return int((df[column] < 0).sum())
    
    def _count_invalid_dates(self, df: pd.DataFrame) -> int:
        """Cuenta fechas inválidas"""
        invalid_count = 0
        for col in self.columnas_fecha:
            if col in df.columns:
                invalid_count += df[col].isnull().sum()
        return int(invalid_count)
    
    def _count_invalid_records(self, df: pd.DataFrame) -> int:
        """Cuenta registros con múltiples problemas"""
        mask = (
            (df['TOTALPESOS'] <= 0) |
            (df['CATEGORIA'] == 'Desconocido') |
            (df['DESTINO'] == 'Desconocido')
        )
        return int(mask.sum())
    
    def _identify_problems(
        self, 
        df: pd.DataFrame,
        valores_nulos: Dict[str, int],
        totalpesos_negativos: int,
        fechas_invalidas: int,
        incluir_detalles: bool,
        limite_ejemplos: int
    ) -> List[ProblemaCalidad]:
        """Identifica problemas específicos en los datos"""
        problemas = []
        
        # Problema: Valores nulos críticos
        for col, count in valores_nulos.items():
            if count > 0 and col in ['TOTALPESOS', 'CATEGORIA', 'DESTINO']:
                ejemplos = None
                if incluir_detalles:
                    ejemplos_df = df[df[col].isnull()].head(limite_ejemplos)
                    ejemplos = ejemplos_df.to_dict('records')
                
                problemas.append(ProblemaCalidad(
                    columna=col,
                    tipo_problema="valores_nulos",
                    cantidad=count,
                    porcentaje=(count / len(df) * 100) if len(df) > 0 else 0,
                    ejemplos=ejemplos
                ))
        
        # Problema: TOTALPESOS negativos o cero
        if totalpesos_negativos > 0:
            ejemplos = None
            if incluir_detalles:
                ejemplos_df = df[df['TOTALPESOS'] <= 0].head(limite_ejemplos)
                ejemplos = ejemplos_df[['FECHAPEDIDO', 'ORIGEN', 'TOTALPESOS']].to_dict('records')
            
            problemas.append(ProblemaCalidad(
                columna='TOTALPESOS',
                tipo_problema="valores_negativos_o_cero",
                cantidad=totalpesos_negativos,
                porcentaje=(totalpesos_negativos / len(df) * 100) if len(df) > 0 else 0,
                ejemplos=ejemplos
            ))
        
        # Problema: Fechas inválidas
        if fechas_invalidas > 0:
            problemas.append(ProblemaCalidad(
                columna='FECHAS',
                tipo_problema="fechas_invalidas",
                cantidad=fechas_invalidas,
                porcentaje=(fechas_invalidas / len(df) * 100) if len(df) > 0 else 0
            ))
        
        # Problema: Valores "Desconocido"
        if 'CATEGORIA' in df.columns:
            desconocidos = (df['CATEGORIA'] == 'Desconocido').sum()
            if desconocidos > 0:
                problemas.append(ProblemaCalidad(
                    columna='CATEGORIA',
                    tipo_problema="valores_desconocidos",
                    cantidad=int(desconocidos),
                    porcentaje=(desconocidos / len(df) * 100) if len(df) > 0 else 0
                ))
        
        return problemas
    
    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        valores_nulos: Dict[str, int],
        totalpesos_negativos: int,
        fechas_invalidas: int
    ) -> float:
        """Calcula puntuación de calidad (0-100)"""
        if len(df) == 0:
            return 0.0
        
        # Componentes de la puntuación
        score = 100.0
        
        # Penalizar por valores nulos (máximo -30 puntos)
        total_nulos = sum(valores_nulos.values())
        penalizacion_nulos = min((total_nulos / len(df)) * 100, 30)
        score -= penalizacion_nulos
        
        # Penalizar por valores negativos (máximo -25 puntos)
        penalizacion_negativos = min((totalpesos_negativos / len(df)) * 100, 25)
        score -= penalizacion_negativos
        
        # Penalizar por fechas inválidas (máximo -20 puntos)
        penalizacion_fechas = min((fechas_invalidas / len(df)) * 100, 20)
        score -= penalizacion_fechas
        
        # Penalizar por "Desconocido" (máximo -15 puntos)
        if 'CATEGORIA' in df.columns:
            desconocidos = (df['CATEGORIA'] == 'Desconocido').sum()
            penalizacion_desconocidos = min((desconocidos / len(df)) * 100, 15)
            score -= penalizacion_desconocidos
        
        return max(0.0, min(100.0, score))
    
    def _determine_quality_state(self, score: float) -> EstadoCalidadDatos:
        """Determina el estado de calidad según la puntuación"""
        if score >= 90:
            return EstadoCalidadDatos.EXCELENTE
        elif score >= 75:
            return EstadoCalidadDatos.BUENA
        elif score >= 60:
            return EstadoCalidadDatos.ACEPTABLE
        else:
            return EstadoCalidadDatos.DEFICIENTE
    
    def _generate_recommendations(
        self, 
        problemas: List[ProblemaCalidad],
        estado: EstadoCalidadDatos
    ) -> List[str]:
        """Genera recomendaciones basadas en los problemas encontrados"""
        recomendaciones = []
        
        if estado == EstadoCalidadDatos.DEFICIENTE:
            recomendaciones.append(
                " CRÍTICO: Los datos requieren limpieza exhaustiva antes de análisis"
            )
        
        for problema in problemas:
            if problema.tipo_problema == "valores_nulos":
                if problema.porcentaje > 10:
                    recomendaciones.append(
                        f"Revisar proceso de captura para {problema.columna} "
                        f"({problema.porcentaje:.1f}% nulos)"
                    )
                else:
                    recomendaciones.append(
                        f"Imputar o eliminar registros nulos en {problema.columna}"
                    )
            
            elif problema.tipo_problema == "valores_negativos_o_cero":
                recomendaciones.append(
                    f"Filtrar registros con TOTALPESOS ≤ 0 ({problema.cantidad} registros)"
                )
            
            elif problema.tipo_problema == "valores_desconocidos":
                recomendaciones.append(
                    f"Enriquecer clasificación de {problema.columna} "
                    f"({problema.cantidad} desconocidos)"
                )
        
        if not recomendaciones:
            recomendaciones.append(" Calidad de datos aceptable para análisis")
        
        return recomendaciones


# Singleton del validador
data_validator = DataValidator()