"""
Servicio de Gestión de Modelos ML
Implementa Single Responsibility Principle (SRP) - Solo gestión de modelos
"""
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import settings
from utils.logger import logger
from utils.exceptions import ModelNotLoadedException
from models.enums import FEATURES_MODELO


class ModelService:
    """Servicio para carga y gestión de modelos de Machine Learning"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.analisis: Dict[str, Any] = {}
        self.load_timestamps: Dict[str, datetime] = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """Carga todos los modelos y artefactos necesarios"""
        try:
            # Cargar modelo de predicción (XGBoost limpio)
            self._load_prediccion_model()
            
            # Cargar modelo temporal (Prophet)
            self._load_temporal_model()
            
            # Cargar encoders
            self._load_encoders()
            
            # Cargar análisis previos
            self._load_analisis()
            
            logger.info(" Todos los modelos cargados correctamente")
            
        except Exception as e:
            logger.error(f" Error al cargar modelos: {e}")
            raise
    
    def _load_prediccion_model(self):
        """Carga el modelo de predicción de gasto"""
        model_path = settings.MODEL_PREDICCION_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo de predicción no encontrado: {model_path}")
        
        self.models['prediccion_gasto'] = joblib.load(model_path)
        self.load_timestamps['prediccion_gasto'] = datetime.now()
        logger.info(f" Modelo de predicción cargado desde: {model_path}")
    
    def _load_temporal_model(self):
        """Carga el modelo de series temporales (Prophet)"""
        model_path = settings.MODEL_TEMPORAL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo temporal no encontrado: {model_path}")
        
        self.models['demanda_temporal'] = joblib.load(model_path)
        self.load_timestamps['demanda_temporal'] = datetime.now()
        logger.info(f" Modelo temporal cargado desde: {model_path}")
    
    def _load_encoders(self):
        """Carga los encoders para variables categóricas"""
        encoders_path = settings.ENCODERS_PATH
        
        if not encoders_path.exists():
            raise FileNotFoundError(f"Encoders no encontrados: {encoders_path}")
        
        self.encoders = joblib.load(encoders_path)
        logger.info(f" Encoders cargados: {list(self.encoders.keys())}")
    
    def _load_analisis(self):
        """Carga análisis previos (por ORIGEN, UN, alertas)"""
        analisis_path = settings.ANALISIS_PATH
        
        if not analisis_path.exists():
            logger.warning(f" Análisis previos no encontrados: {analisis_path}")
            return
        
        self.analisis = joblib.load(analisis_path)
        logger.info(f" Análisis previos cargados")
    
    def get_model(self, model_name: str) -> Any:
        """
        Obtiene un modelo cargado
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Modelo solicitado
            
        Raises:
            ModelNotLoadedException: Si el modelo no está cargado
        """
        if model_name not in self.models:
            raise ModelNotLoadedException(model_name)
        
        return self.models[model_name]
    
    def get_encoder(self, column_name: str) -> Any:
        """Obtiene un encoder para una columna específica"""
        if column_name not in self.encoders:
            raise ValueError(f"Encoder no encontrado para columna: {column_name}")
        
        return self.encoders[column_name]
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Verifica si un modelo está cargado"""
        return model_name in self.models
    
    def get_models_info(self) -> Dict[str, Any]:
        """Retorna información sobre los modelos cargados"""
        return {
            'prediccion_gasto': {
                'loaded': self.is_model_loaded('prediccion_gasto'),
                'type': 'XGBoost Regressor',
                'timestamp': self.load_timestamps.get('prediccion_gasto'),
                'metrics': {
                    'R2': 0.9943,
                    'RMSE': 6284.91,
                    'MAE': 776.23
                }
            },
            'demanda_temporal': {
                'loaded': self.is_model_loaded('demanda_temporal'),
                'type': 'Prophet',
                'timestamp': self.load_timestamps.get('demanda_temporal')
            },
            'encoders': {
                'loaded': len(self.encoders) > 0,
                'count': len(self.encoders),
                'columns': list(self.encoders.keys())
            }
        }
    
    def predict_gasto(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicción de gasto usando el modelo de regresión
        
        Args:
            df_features: DataFrame con features preparados
            
        Returns:
            Array con predicciones
        """
        model = self.get_model('prediccion_gasto')
        
        # Validar que todas las features necesarias estén presentes
        missing_features = set(FEATURES_MODELO) - set(df_features.columns)
        if missing_features:
            raise ValueError(f"Features faltantes: {missing_features}")
        
        # Seleccionar solo las features del modelo
        X = df_features[FEATURES_MODELO]
        
        # Realizar predicción
        predictions = model.predict(X)
        
        logger.info(f" Predicción realizada para {len(df_features)} registros")
        
        return predictions
    
    def forecast_temporal(self, periods: int = 12, freq: str = 'M') -> pd.DataFrame:
        """
        Realiza pronóstico usando el modelo Prophet
        
        Args:
            periods: Número de períodos a pronosticar
            freq: Frecuencia ('M' para mensual, 'D' para diario)
            
        Returns:
            DataFrame con pronóstico
        """
        model = self.get_model('demanda_temporal')
        
        # Crear dataframe futuro
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Realizar pronóstico
        forecast = model.predict(future)
        
        logger.info(f" Pronóstico temporal realizado para {periods} períodos")
        
        return forecast
    
    def reload_models(self):
        """Recarga todos los modelos desde disco"""
        logger.info(" Recargando modelos...")
        self.models.clear()
        self.encoders.clear()
        self.analisis.clear()
        self.load_timestamps.clear()
        self._load_all_models()
        logger.info(" Modelos recargados exitosamente")


# Singleton del servicio de modelos
model_service = ModelService()