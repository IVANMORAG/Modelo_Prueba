"""
Sistema de Logging Centralizado
Implementa Single Responsibility Principle (SRP)
"""
from loguru import logger
from pathlib import Path
from config.settings import settings
import sys


def setup_logger():
    """Configura el sistema de logging de la aplicación"""
    
    # Crear directorio de logs si no existe
    log_dir = settings.LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remover configuración por defecto
    logger.remove()
    
    # Console handler con formato colorizado
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # File handler con rotación
    logger.add(
        settings.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True
    )
    
    logger.info(" Sistema de logging inicializado")
    return logger


# Inicializar logger
logger = setup_logger()