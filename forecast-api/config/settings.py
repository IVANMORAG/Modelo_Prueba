"""
Configuración Global de la API
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno"""
   
    # Database Configuration
    DB_SERVER: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
   
    # API Configuration
    API_TITLE: str = "API de Pronóstico de Gasto y Monitoreo Presupuestal"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
   
    # Model Artifacts Paths
    ARTIFACTS_DIR: Path = Path("./artifacts")
    MODEL_PREDICCION_PATH: Path = Path("./artifacts/modelo_prediccion_limimo.pkl")
    MODEL_TEMPORAL_PATH: Path = Path("./artifacts/modelo_demanda_temporal.pkl")
    ENCODERS_PATH: Path = Path("./artifacts/encoders.pkl")
    ANALISIS_PATH: Path = Path("./artifacts/analisis.pkl")
   
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/api.log")
   
    # Forecasting Parameters
    DEFAULT_ANOMALY_THRESHOLD: float = 0.05
    DEFAULT_CONFIDENCE_LEVEL: float = 0.95
    MAX_FORECAST_YEARS: int = 5
   
    # Cache Configuration
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    CACHE_TTL: int = 3600
   
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = 'ignore'
   
    @property
    def oledb_connection_string(self) -> str:
        """Cadena de conexión OLEDB exactamente como en Excel"""
        return (
            f"Provider=SQLOLEDB.1;"
            f"Data Source={self.DB_SERVER};"
            f"Initial Catalog={self.DB_NAME};"
            f"User ID={self.DB_USER};"
            f"Password={self.DB_PASSWORD};"
            f"Persist Security Info=True"
        )
   
    def validate_artifacts_exist(self) -> dict[str, bool]:
        """Valida que todos los artefactos necesarios existan"""
        artifacts = {
            "modelo_prediccion": self.MODEL_PREDICCION_PATH.exists(),
            "modelo_temporal": self.MODEL_TEMPORAL_PATH.exists(),
            "encoders": self.ENCODERS_PATH.exists(),
            "analisis": self.ANALISIS_PATH.exists()
        }
        return artifacts

# Singleton de configuración
settings = Settings()