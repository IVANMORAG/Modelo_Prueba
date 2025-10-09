"""
Dependencias comunes de FastAPI
"""
from fastapi import Depends, HTTPException, status
from typing import Optional
import pandas as pd

from services.model_service import model_service
from services.data_validator import data_validator
from services.anomaly_detector import anomaly_detector
from services.recommendation_engine import recommendation_engine
from services.forecast_service import forecast_service
from repositories.data_repository import data_repository
from config.database import get_db
from utils.logger import logger


def get_model_service():
    """Dependency para obtener el servicio de modelos"""
    return model_service


def get_data_validator():
    """Dependency para obtener el validador de datos"""
    return data_validator


def get_anomaly_detector():
    """Dependency para obtener el detector de anomalías"""
    return anomaly_detector


def get_recommendation_engine():
    """Dependency para obtener el motor de recomendaciones"""
    return recommendation_engine


def get_forecast_service():
    """Dependency para obtener el servicio de pronóstico"""
    return forecast_service


def get_data_repository():
    """Dependency para obtener el repositorio de datos"""
    return data_repository


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que el DataFrame tenga datos válidos
    
    Args:
        df: DataFrame a validar
        
    Returns:
        DataFrame validado
        
    Raises:
        HTTPException: Si el DataFrame es inválido
    """
    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="DataFrame vacío o nulo"
        )
    
    return df


def get_validated_year(year: int) -> int:
    """
    Valida que el año esté en rango válido
    
    Args:
        year: Año a validar
        
    Returns:
        Año validado
        
    Raises:
        HTTPException: Si el año es inválido
    """
    from datetime import datetime
    from config.settings import settings
    
    current_year = datetime.now().year
    max_year = current_year + settings.MAX_FORECAST_YEARS
    
    if year < 2016 or year > max_year:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Año debe estar entre 2016 y {max_year}"
        )
    
    return year