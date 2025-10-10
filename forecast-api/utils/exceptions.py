"""
Excepciones Personalizadas de la API
Implementa Open/Closed Principle (OCP) - Extensible sin modificar código base
"""
from fastapi import HTTPException, status


class APIBaseException(HTTPException):
    """Excepción base para todas las excepciones personalizadas de la API"""
    
    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)


class ModelNotLoadedException(APIBaseException):
    """Se lanza cuando un modelo ML no ha sido cargado correctamente"""
    
    def __init__(self, model_name: str):
        super().__init__(
            detail=f"El modelo '{model_name}' no ha sido cargado correctamente",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class DataValidationException(APIBaseException):
    """Se lanza cuando los datos de entrada no cumplen con las validaciones"""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Error de validación de datos: {detail}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class DatabaseConnectionException(APIBaseException):
    """Se lanza cuando hay problemas de conexión a la base de datos"""
    
    def __init__(self, detail: str = "Error de conexión a la base de datos"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class ForecastException(APIBaseException):
    """Se lanza cuando ocurre un error durante el proceso de pronóstico"""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Error en proceso de pronóstico: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class InvalidYearException(APIBaseException):
    """Se lanza cuando el año objetivo no es válido"""
    
    def __init__(self, year: int, max_year: int):
        super().__init__(
            detail=f"Año objetivo {year} inválido. Debe estar entre 2025 y {max_year}",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class InsufficientDataException(APIBaseException):
    """Se lanza cuando no hay suficientes datos históricos para el pronóstico"""
    
    def __init__(self, detail: str = "Datos históricos insuficientes para realizar el pronóstico"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST
        )