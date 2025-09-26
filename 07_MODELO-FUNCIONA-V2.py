#!/usr/bin/env python3
"""
Sistema Avanzado de Predicción de Compras - Grupo Salinas
Modelo mejorado con técnicas avanzadas de ML y redes neuronales
Predice: Cantidad, Costo y Momento de compras por categoría

Características principales:
- Modelos ensemble avanzados (XGBoost, LightGBM, CatBoost)
- Redes neuronales con TensorFlow
- Optimización automática de hiperparámetros
- Feature engineering avanzado
- Validación temporal robusta
- Manejo inteligente de outliers
- Modo para datos escasos con ARIMA, Gaussian Processes y transfer learning

Autor: Sistema IA Avanzado
Versión: 2.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar visualización
plt.style.use('default')
sns.set_palette("husl")

# ML Libraries - Modelos Avanzados
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            ExtraTreesRegressor, VotingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression)
from sklearn.svm import SVR
from sklearn.model_selection import (TimeSeriesSplit, GridSearchCV, 
                                   RandomizedSearchCV, cross_val_score)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Modelos avanzados externos (instalar si no están disponibles)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost no disponible")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM no disponible")

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("⚠️ CatBoost no disponible")

# TensorFlow para redes neuronales
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("✅ TensorFlow disponible para redes neuronales")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow no disponible")

# Modelos para small data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Otras librerías
from datetime import datetime, timedelta
import json
import os
import joblib
from scipy import stats
from scipy.stats import zscore
import itertools

# =============================================================================
# CONFIGURACIÓN GLOBAL AVANZADA
# =============================================================================

class ConfiguracionAvanzada:
    def __init__(self):
        self.random_state = 42
        self.test_size = 0.25
        self.min_meses_entrenamiento = 6  # Reducido para manejar más categorías
        self.cv_folds = 5
        self.outlier_threshold = 3.5  # Z-score threshold
        self.feature_selection_k = 15  # Top K features
        self.ensemble_weights = 'uniform'
        self.neural_epochs = 100
        self.neural_batch_size = 32
        self.neural_patience = 15
        
        # Configuración de modelos
        self.models_config = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9]
            } if XGB_AVAILABLE else {},
            'lgb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9]
            } if LGB_AVAILABLE else {},
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }

config = ConfiguracionAvanzada()

# Variables globales
modelos_entrenados = {}
datos_temporales = {}
metricas_globales = {}
feature_importance_global = {}

# =============================================================================
# UTILIDADES AVANZADAS
# =============================================================================

def detectar_y_limpiar_outliers(df, columnas=['TOTALPESOS'], metodo='zscore', threshold=3.5):
    """
    Detecta y limpia outliers usando múltiples métodos
    """
    print(f"🧹 Detectando outliers en {columnas} con método {metodo}...")
    
    df_clean = df.copy()
    outliers_removidos = 0
    
    for columna in columnas:
        if columna not in df_clean.columns:
            continue
            
        antes = len(df_clean)
        
        if metodo == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df_clean[columna].dropna()))
            df_clean = df_clean[z_scores < threshold]
            
        elif metodo == 'iqr':
            # Interquartile Range method
            Q1 = df_clean[columna].quantile(0.25)
            Q3 = df_clean[columna].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[columna] >= lower_bound) & 
                               (df_clean[columna] <= upper_bound)]
            
        elif metodo == 'percentile':
            # Percentile method
            lower_percentile = df_clean[columna].quantile(0.01)
            upper_percentile = df_clean[columna].quantile(0.99)
            df_clean = df_clean[(df_clean[columna] >= lower_percentile) & 
                               (df_clean[columna] <= upper_percentile)]
        
        despues = len(df_clean)
        outliers_col = antes - despues
        outliers_removidos += outliers_col
        
        print(f"   📊 {columna}: {outliers_col:,} outliers removidos")
    
    print(f"✅ Total outliers removidos: {outliers_removidos:,}")
    return df_clean

def crear_features_avanzados(df_serie, categoria):
    """
    Crea features avanzados de ingeniería temporal y específicos del dominio
    """
    df = df_serie.copy()
    print(f"🔬 Creando features avanzados para {categoria}...")
    
    # Validar que las columnas requeridas existen
    required_metrics = ['gasto_total', 'num_transacciones', 'gasto_promedio']
    available_metrics = [m for m in required_metrics if m in df.columns]
    if not available_metrics:
        print(f"❌ No se encontraron métricas requeridas: {required_metrics}")
        return df
    
    # Verificar que 'fecha' es una columna válida
    if 'fecha' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        print(f"❌ Columna 'fecha' no válida o no encontrada")
        return df
    
    # 1. FEATURES TEMPORALES BÁSICOS
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['año'] = df['fecha'].dt.year
    df['dias_mes'] = df['fecha'].dt.days_in_month
    
    # 2. FEATURES ESTACIONALES AVANZADOS
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trim_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trim_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    
    df['es_fin_año'] = ((df['mes'] == 12) | (df['mes'] == 1)).astype(int)
    df['es_medio_año'] = ((df['mes'] >= 6) & (df['mes'] <= 8)).astype(int)
    df['es_inicio_año'] = ((df['mes'] >= 1) & (df['mes'] <= 3)).astype(int)
    
    # 3. LAGS Y VENTANAS MÓVILES AVANZADOS
    lags = [1, 2, 3, 6, 12, 24]
    windows = [3, 6, 12, 24]
    
    for metric in available_metrics:
        for lag in lags:
            df[f'{metric}_lag_{lag}'] = df[metric].shift(lag)
        
        for window in windows:
            df[f'{metric}_ma_{window}'] = df[metric].rolling(window=window, min_periods=1).mean()
            df[f'{metric}_std_{window}'] = df[metric].rolling(window=window, min_periods=1).std()
            df[f'{metric}_min_{window}'] = df[metric].rolling(window=window, min_periods=1).min()
            df[f'{metric}_max_{window}'] = df[metric].rolling(window=window, min_periods=1).max()
            df[f'{metric}_median_{window}'] = df[metric].rolling(window=window, min_periods=1).median()
            df[f'{metric}_q25_{window}'] = df[metric].rolling(window=window, min_periods=1).quantile(0.25)
            df[f'{metric}_q75_{window}'] = df[metric].rolling(window=window, min_periods=1).quantile(0.75)
    
    # 4. FEATURES DE TENDENCIA Y MOMENTUM
    df['tendencia_lineal'] = range(len(df))
    
    for metric in available_metrics:
        df[f'{metric}_pct_change_1'] = df[metric].pct_change(1)
        df[f'{metric}_pct_change_3'] = df[metric].pct_change(3)
        df[f'{metric}_pct_change_12'] = df[metric].pct_change(12)
        df[f'{metric}_accel'] = df[f'{metric}_pct_change_1'].diff()
        
        # Momentum solo si las columnas de medias móviles existen
        if f'{metric}_ma_3' in df.columns and f'{metric}_ma_12' in df.columns:
            df[f'{metric}_momentum_3_12'] = (df[f'{metric}_ma_3'] / df[f'{metric}_ma_12']) - 1
        if f'{metric}_ma_6' in df.columns and f'{metric}_ma_24' in df.columns:
            df[f'{metric}_momentum_6_24'] = (df[f'{metric}_ma_6'] / df[f'{metric}_ma_24']) - 1
    
    # 5. FEATURES ESPECÍFICOS DEL DOMINIO DE COMPRAS
    if 'gasto_total' in df.columns and 'num_transacciones' in df.columns:
        df['gasto_por_transaccion'] = df['gasto_total'] / (df['num_transacciones'] + 1e-8)
    if 'gasto_total' in df.columns and 'ordenes_unicas' in df.columns:
        df['gasto_por_orden'] = df['gasto_total'] / (df['ordenes_unicas'] + 1e-8)
        df['transacciones_por_orden'] = df['num_transacciones'] / (df['ordenes_unicas'] + 1e-8)
        df['diversidad_centros'] = df['centros_costo'] / (df['ordenes_unicas'] + 1e-8)
        df['diversidad_solicitantes'] = df['solicitantes_unicos'] / (df['ordenes_unicas'] + 1e-8)
    
    # 6. VOLATILIDAD Y ESTABILIDAD
    if 'gasto_total' in df.columns and f'gasto_total_ma_6' in df.columns:
        df['volatilidad_gasto'] = df['gasto_total'].rolling(window=6, min_periods=1).std()
        df['coef_variacion_gasto'] = df['volatilidad_gasto'] / (df['gasto_total_ma_6'] + 1e-8)
        df['estabilidad_gasto'] = 1 / (1 + df['coef_variacion_gasto'])
    
    # 7. FEATURES DE ACTIVIDAD Y PERIODICIDAD
    if 'gasto_total' in df.columns:
        df['meses_sin_actividad'] = (df['gasto_total'] == 0).astype(int)
        df['racha_sin_actividad'] = df.groupby((df['gasto_total'] > 0).cumsum())['meses_sin_actividad'].cumsum()
        df['compra_activa'] = (df['gasto_total'] > 0).astype(int)
        df['frecuencia_compra_6m'] = df['compra_activa'].rolling(window=6, min_periods=1).sum()
        df['frecuencia_compra_12m'] = df['compra_activa'].rolling(window=12, min_periods=1).sum()
    
    # 8. FEATURES COMPARATIVOS
    for metric in available_metrics:
        media_historica = df[metric].expanding(min_periods=1).mean()
        df[f'{metric}_vs_historico'] = df[metric] / (media_historica + 1e-8)
        df[f'{metric}_desv_historico'] = (df[metric] - media_historica) / (media_historica + 1e-8)
    
    # 9. FEATURES DE ANOMALÍAS
    for metric in available_metrics:
        rolling_mean = df[metric].rolling(window=12, min_periods=1).mean()
        rolling_std = df[metric].rolling(window=12, min_periods=1).std()
        df[f'{metric}_z_score'] = (df[metric] - rolling_mean) / (rolling_std + 1e-8)
        df[f'{metric}_es_anomalia'] = (np.abs(df[f'{metric}_z_score']) > 2).astype(int)
    
    # 10. LIMPIEZA FINAL
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['fecha', 'año', 'mes', 'trimestre']:
            df[col] = (df[col]
                      .fillna(method='ffill')
                      .fillna(method='bfill')
                      .fillna(df[col].rolling(window=6, min_periods=1).mean())
                      .fillna(0))
    
    print(f"✅ Features creados: {len([col for col in df.columns if any(x in col for x in ['_lag_', '_ma_', '_std_', '_pct_', 'sin', 'cos'])])} features temporales")
    
    return df

# =============================================================================
# MODELOS AVANZADOS
# =============================================================================

class ModeloEnsembleAvanzado:
    """
    Clase para manejar modelos ensemble avanzados
    """
    def __init__(self, target_name, config):
        self.target_name = target_name
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_model = None
        self.best_score = -np.inf
        
    def crear_modelos_base(self):
        """
        Crea los modelos base para el ensemble
        """
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.9,
                random_state=self.config.random_state
            ),
            'et': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config.random_state)
        }
        
        # Agregar modelos externos si están disponibles
        if XGB_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.9,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if LGB_AVAILABLE:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.9,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        if CB_AVAILABLE:
            models['cb'] = cb.CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=8,
                random_state=self.config.random_state,
                verbose=False
            )
        
        return models
    
    def optimizar_hiperparametros(self, X_train, y_train, model_name, model):
        """
        Optimiza hiperparámetros usando RandomizedSearchCV
        """
        if model_name not in self.config.models_config or not self.config.models_config[model_name]:
            return model
        
        print(f"🔧 Optimizando {model_name}...")
        
        param_grid = self.config.models_config[model_name]
        
        # Usar TimeSeriesSplit para validación temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,  # Número limitado de iteraciones para velocidad
            cv=tscv,
            scoring='neg_mean_absolute_error',
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        try:
            search.fit(X_train, y_train)
            return search.best_estimator_
        except Exception as e:
            print(f"⚠️ Error optimizando {model_name}: {str(e)}")
            return model
    
    def entrenar_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Entrena ensemble de modelos con optimización automática
        """
        print(f"🚀 Entrenando ensemble para {self.target_name}...")
        
        # Selección de features
        if X_train.shape[1] > self.config.feature_selection_k:
            print(f"🎯 Seleccionando top {self.config.feature_selection_k} features...")
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
        else:
            X_train_selected = X_train
            X_test_selected = X_test
        
        # Crear y entrenar modelos base
        base_models = self.crear_modelos_base()
        trained_models = []
        model_scores = {}
        
        for name, model in base_models.items():
            try:
                print(f"   Entrenando {name}...")
                
                # Escalar datos para modelos que lo requieren
                if name in ['ridge', 'elastic', 'svr']:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_selected)
                    X_test_scaled = scaler.transform(X_test_selected)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train_selected
                    X_test_scaled = X_test_selected
                    self.scalers[name] = None
                
                # Optimizar hiperparámetros
                optimized_model = self.optimizar_hiperparametros(X_train_scaled, y_train, name, model)
                
                # Entrenar modelo optimizado
                optimized_model.fit(X_train_scaled, y_train)
                
                # Evaluar
                y_pred = optimized_model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[name] = {'r2': score, 'mae': mae}
                
                if score > 0.1:  # Solo incluir modelos con desempeño mínimo aceptable
                    self.models[name] = optimized_model
                    trained_models.append((name, optimized_model))
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = optimized_model
                
                print(f"   {name}: R² = {score:.3f}, MAE = {mae:,.0f}")
                
            except Exception as e:
                print(f"   ⚠️ Error entrenando {name}: {str(e)}")
        
        # Crear ensemble final si hay múltiples modelos exitosos
        if len(trained_models) > 1:
            try:
                print("🎯 Creando modelo ensemble...")
                ensemble_models = [(name, model) for name, model in trained_models 
                                 if model_scores[name]['r2'] > 0.2]
                
                if len(ensemble_models) > 1:
                    voting_regressor = VotingRegressor(
                        estimators=ensemble_models,
                        weights=None  # Pesos uniformes por simplicidad
                    )
                    
                    # Preparar datos para ensemble (usar el escalador del mejor modelo individual)
                    if self.best_model in [model for _, model in ensemble_models]:
                        best_name = next(name for name, model in trained_models if model == self.best_model)
                        if self.scalers[best_name]:
                            X_train_ensemble = self.scalers[best_name].transform(X_train_selected)
                            X_test_ensemble = self.scalers[best_name].transform(X_test_selected)
                        else:
                            X_train_ensemble = X_train_selected
                            X_test_ensemble = X_test_selected
                        
                        voting_regressor.fit(X_train_ensemble, y_train)
                        y_pred_ensemble = voting_regressor.predict(X_test_ensemble)
                        ensemble_score = r2_score(y_test, y_pred_ensemble)
                        ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
                        
                        print(f"   Ensemble: R² = {ensemble_score:.3f}, MAE = {ensemble_mae:,.0f}")
                        
                        if ensemble_score > self.best_score:
                            self.models['ensemble'] = voting_regressor
                            self.best_model = voting_regressor
                            self.best_score = ensemble_score
            
            except Exception as e:
                print(f"   ⚠️ Error creando ensemble: {str(e)}")
        
        return model_scores
    
    def predecir(self, X):
        """
        Realiza predicciones usando el mejor modelo
        """
        if self.best_model is None:
            return np.zeros(len(X))
        
        # Aplicar selección de features
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        # Aplicar escalado si es necesario
        best_name = next((name for name, model in self.models.items() if model == self.best_model), 'ensemble')
        if best_name in self.scalers and self.scalers[best_name]:
            X_scaled = self.scalers[best_name].transform(X_selected)
        else:
            X_scaled = X_selected
        
        return self.best_model.predict(X_scaled)

class RedNeuronalAvanzada:
    """
    Red neuronal avanzada para predicción de series temporales
    """
    def __init__(self, target_name, config):
        self.target_name = target_name
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def crear_modelo(self, input_dim):
        """
        Crea arquitectura de red neuronal avanzada
        """
        if not TF_AVAILABLE:
            return None
        
        model = keras.Sequential([
            # Capa de entrada con regularización
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Capas ocultas con regularización progresiva
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            # Capa de salida
            layers.Dense(1, activation='linear')
        ])
        
        # Optimizador con learning rate adaptativo
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def entrenar(self, X_train, y_train, X_test, y_test):
        """
        Entrena la red neuronal con callbacks avanzados
        """
        if not TF_AVAILABLE:
            print(f"⚠️ TensorFlow no disponible para {self.target_name}")
            return None
        
        print(f"🧠 Entrenando red neuronal para {self.target_name}...")
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear modelo
        self.model = self.crear_modelo(X_train.shape[1])
        
        if self.model is None:
            return None
        
        # Callbacks avanzados
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.neural_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        try:
            # Entrenar
            self.history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.config.neural_epochs,
                batch_size=self.config.neural_batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluar
            y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"   Red Neuronal: R² = {r2:.3f}, MAE = {mae:,.0f}")
            
            return {'r2': r2, 'mae': mae}
            
        except Exception as e:
            print(f"   ⚠️ Error entrenando red neuronal: {str(e)}")
            return None
    
    def predecir(self, X):
        """
        Realiza predicciones con la red neuronal
        """
        if self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

# =============================================================================
# FUNCIONES PARA SMALL DATA
# =============================================================================

def entrenar_modo_small_data(df_categoria, nombre_categoria):
    print(f"🚀 Modo Small Data/Few-Shot para {nombre_categoria} con {len(df_categoria)} meses")
    
    # Crear features (ya optimizado)
    df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    
    # Preparar datos (usa 'fecha' como index)
    df_features.set_index('fecha', inplace=True)
    
    # Definir features y targets
    exclude_cols = [
        'categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
        'num_transacciones', 'ordenes_unicas', 'centros_costo', 
        'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año'
    ]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # 1. ARIMA para Gasto y Cantidad (simple para short series)
    targets = ['gasto_total', 'num_transacciones']
    modelos_small = {}
    resultados_small = {}
    for target in targets:
        if target in df_features.columns:
            try:
                model = ARIMA(df_features[target], order=(1,1,1))  # Orden simple
                model_fit = model.fit()
                modelos_small[f'arima_{target}'] = model_fit
                # Evaluar con split simple (últimos 20% como test)
                split = int(len(df_features) * 0.8)
                train, test = df_features[target][:split], df_features[target][split:]
                pred = model_fit.forecast(len(test))
                r2 = r2_score(test, pred)
                mae = mean_absolute_error(test, pred)
                resultados_small[target] = {'r2': r2, 'mae': mae}
                print(f"   ARIMA {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
            except Exception as e:
                print(f"   ⚠️ Error ARIMA {target}: {str(e)}")
    
    # 2. Gaussian Process para multivariado/irregular (gasto como target, features como X)
    if 'gasto_total' in df_features.columns:
        try:
            # Usar tiempo como X (índice numérico), gasto como y
            X = np.arange(len(df_features)).reshape(-1, 1)  # Tiempo lineal
            y = df_features['gasto_total'].values
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp.fit(X, y)
            modelos_small['gp_gasto'] = gp
            # Predicción de prueba
            X_test = np.array([[len(df_features) + i] for i in range(1, 4)])
            pred, sigma = gp.predict(X_test, return_std=True)
            print(f"   GP Gasto: Predicciones = {pred}")
        except Exception as e:
            print(f"   ⚠️ Error GP Gasto: {str(e)}")
    
    # 3. Clasificación para Momento (probabilidad de compra, si temporal débil)
    if 'gasto_total' in df_features.columns:
        try:
            df_features['compra_activa'] = (df_features['gasto_total'] > 0).astype(int)
            X = df_features[feature_cols].fillna(0)  # Features simples
            y = df_features['compra_activa']
            clf = LogisticRegression()
            clf.fit(X, y)
            modelos_small['clf_momento'] = clf
            prob = clf.predict_proba(X)[:, 1].mean()
            resultados_small['momento'] = {
                'accuracy': accuracy_score(y, clf.predict(X)),
                'auc': roc_auc_score(y, clf.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else 0.5,
                'probabilidad_promedio': prob
            }
            print(f"   Clasificador Momento: Probabilidad promedio = {prob:.3f}")
        except Exception as e:
            print(f"   ⚠️ Error Clasificador Momento: {str(e)}")
    
    # 4. Transfer Learning (few-shot): Pre-entrena en categoría similar grande
    large_category = 'Tecnología'  # Asume esta como grande; ajusta si necesario
    if large_category in modelos_entrenados:
        pre_model = modelos_entrenados[large_category]['modelos'].get('gasto')
        if pre_model:
            # Fine-tune: Fit parcial en small data (si es regressor)
            X_small = df_features[feature_cols].fillna(0)
            y_small = df_features['gasto_total']
            pre_model.best_model.fit(X_small, y_small)  # Re-fit simple para transfer
            modelos_small['transfer_gasto'] = pre_model
            print("   Transfer Learning aplicado desde 'Tecnología'")
    
    return {
        'categoria': nombre_categoria,
        'modelos': modelos_small,
        'resultados': resultados_small,
        'features': feature_cols,
        'datos_usados': len(df_features)
    }

# =============================================================================
# SISTEMA PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

def entrenar_categoria_avanzada(df_categoria, nombre_categoria, verbose=True):
    """
    Entrena modelos avanzados para una categoría específica
    """
    print(f"\n🎯 Procesando categoría: {nombre_categoria}")
    print("="*60)
    
    # Validar datos mínimos para modo estándar
    if len(df_categoria) < 18:  # Umbral para small data (anterior min era 18, ahora usamos 6 en config pero distinguimos)
        if verbose:
            print(f"❌ Datos insuficientes para modo estándar: {len(df_categoria)} < 18 meses. Usando modo small data.")
        return entrenar_modo_small_data(df_categoria, nombre_categoria)
    
    # Crear features avanzados
    try:
        df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    except Exception as e:
        if verbose:
            print(f"❌ Error creando features: {str(e)}")
        return None
    
    # Eliminar filas con demasiados NaN
    threshold = len(df_features.columns) * 0.5  # Al menos 50% de datos válidos
    df_features = df_features.dropna(thresh=int(threshold))
    
    if len(df_features) < config.min_meses_entrenamiento:
        if verbose:
            print(f"❌ Datos insuficientes después de limpieza: {len(df_features)}")
        return None
    
    # Definir features y targets
    exclude_cols = [
        'categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
        'num_transacciones', 'ordenes_unicas', 'centros_costo', 
        'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año'
    ]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    if len(feature_cols) < 5:
        if verbose:
            print(f"❌ Features insuficientes: {len(feature_cols)}")
        return None
    
    # Preparar datos
    X = df_features[feature_cols].fillna(0)
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']
    y_momento = (df_features['gasto_total'] > 0).astype(int)  # Probabilidad de compra
    
    # Split temporal más robusto
    split_idx = int(len(X) * (1 - config.test_size))
    if split_idx < config.min_meses_entrenamiento * 0.7:  # Al menos 70% para entrenamiento
        if verbose:
            print(f"❌ Split insuficiente: {split_idx}")
        return None
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    y_momento_train, y_momento_test = y_momento.iloc[:split_idx], y_momento.iloc[split_idx:]
    
    # Entrenar modelos para cada target
    resultados = {}
    modelos_categoria = {}
    
    # 1. MODELO PARA GASTO
    print("💰 Entrenando modelos para GASTO...")
    modelo_gasto = ModeloEnsembleAvanzado('gasto', config)
    scores_gasto = modelo_gasto.entrenar_ensemble(X_train, y_gasto_train, X_test, y_gasto_test)
    
    if modelo_gasto.best_model is not None:
        modelos_categoria['gasto'] = modelo_gasto
        resultados['gasto'] = {
            'r2': modelo_gasto.best_score,
            'mae': mean_absolute_error(y_gasto_test, modelo_gasto.predecir(X_test)),
            'scores_individuales': scores_gasto
        }
    
    # Red neuronal para gasto (si TensorFlow disponible)
    if TF_AVAILABLE and len(X_train) > 50:  # Solo si hay suficientes datos
        nn_gasto = RedNeuronalAvanzada('gasto', config)
        nn_scores = nn_gasto.entrenar(X_train, y_gasto_train, X_test, y_gasto_test)
        if nn_scores and nn_scores['r2'] > modelo_gasto.best_score:
            modelos_categoria['gasto_nn'] = nn_gasto
            resultados['gasto']['red_neuronal'] = nn_scores
    
    # 2. MODELO PARA CANTIDAD
    print("📦 Entrenando modelos para CANTIDAD...")
    modelo_cantidad = ModeloEnsembleAvanzado('cantidad', config)
    scores_cantidad = modelo_cantidad.entrenar_ensemble(X_train, y_cantidad_train, X_test, y_cantidad_test)
    
    if modelo_cantidad.best_model is not None:
        modelos_categoria['cantidad'] = modelo_cantidad
        resultados['cantidad'] = {
            'r2': modelo_cantidad.best_score,
            'mae': mean_absolute_error(y_cantidad_test, modelo_cantidad.predecir(X_test)),
            'scores_individuales': scores_cantidad
        }
    
    # Red neuronal para cantidad
    if TF_AVAILABLE and len(X_train) > 50:
        nn_cantidad = RedNeuronalAvanzada('cantidad', config)
        nn_scores = nn_cantidad.entrenar(X_train, y_cantidad_train, X_test, y_cantidad_test)
        if nn_scores and nn_scores['r2'] > modelo_cantidad.best_score:
            modelos_categoria['cantidad_nn'] = nn_cantidad
            resultados['cantidad']['red_neuronal'] = nn_scores
    
    # 3. MODELO PARA MOMENTO (Probabilidad de compra)
    print("⏰ Entrenando modelos para MOMENTO...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    try:
        # Usar clasificador para probabilidad de compra
        clf_momento = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=config.random_state,
            n_jobs=-1
        )
        
        clf_momento.fit(X_train, y_momento_train)
        y_momento_pred = clf_momento.predict(X_test)
        y_momento_prob = clf_momento.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_momento_test, y_momento_pred)
        try:
            auc = roc_auc_score(y_momento_test, y_momento_prob)
        except:
            auc = 0.5
        
        modelos_categoria['momento'] = clf_momento
        resultados['momento'] = {
            'accuracy': accuracy,
            'auc': auc,
            'probabilidad_promedio': y_momento_prob.mean()
        }
        
        print(f"   Momento: Accuracy = {accuracy:.3f}, AUC = {auc:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Error entrenando modelo de momento: {str(e)}")
    
    # Guardar información de features importantes
    if 'gasto' in modelos_categoria:
        try:
            # Feature importance del mejor modelo de gasto
            best_model = modelos_categoria['gasto'].best_model
            if hasattr(best_model, 'feature_importances_'):
                feature_names = feature_cols
                if modelos_categoria['gasto'].feature_selector:
                    selected_indices = modelos_categoria['gasto'].feature_selector.get_support(indices=True)
                    feature_names = [feature_cols[i] for i in selected_indices]
                
                importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                feature_importance_global[nombre_categoria] = importance_dict
        except:
            pass
    
    # Resumen de resultados
    if verbose and resultados:
        print(f"\n✅ RESULTADOS para {nombre_categoria}:")
        for target, metrics in resultados.items():
            if target in ['gasto', 'cantidad']:
                print(f"   {target.upper()}: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:,.0f}")
            elif target == 'momento':
                print(f"   {target.upper()}: Accuracy = {metrics['accuracy']:.3f}, AUC = {metrics['auc']:.3f}")
    
    if not modelos_categoria:
        if verbose:
            print(f"❌ No se pudo entrenar ningún modelo para {nombre_categoria}")
        return None
    
    return {
        'categoria': nombre_categoria,
        'modelos': modelos_categoria,
        'resultados': resultados,
        'features': feature_cols,
        'datos_usados': len(df_features)
    }

def predecir_categoria_avanzada(modelo_info, meses_adelante=1):
    """
    Genera predicciones avanzadas para una categoría
    """
    if modelo_info is None:
        return None
    
    categoria = modelo_info['categoria']
    modelos = modelo_info['modelos']
    features = modelo_info['features']
    
    # Obtener últimos datos para predicción
    df_categoria = datos_temporales.get('CATEGORIA', pd.DataFrame())
    if df_categoria.empty:
        return None
    
    df_cat = df_categoria[df_categoria['categoria'] == categoria].copy()
    if len(df_cat) == 0:
        return None
    
    # Crear features para el último período
    df_features = crear_features_avanzados(df_cat, categoria)
    
    if len(df_features) == 0:
        return None
    
    # Usar los últimos datos disponibles como base para predicción
    ultimo_punto = df_features.iloc[-1:][features].fillna(0)
    
    predicciones = {}
    
    # Predicción de gasto
    if 'gasto' in modelos:
        pred_gasto = modelos['gasto'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)  # No permitir gastos negativos
    elif 'gasto_nn' in modelos:
        pred_gasto = modelos['gasto_nn'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'arima_gasto_total' in modelos:
        pred_gasto = modelos['arima_gasto_total'].forecast(meses_adelante)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gp_gasto' in modelos:
        X_pred = np.array([[len(df_features) + i] for i in range(1, meses_adelante + 1)])
        pred_gasto = modelos['gp_gasto'].predict(X_pred)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'transfer_gasto' in modelos:
        pred_gasto = modelos['transfer_gasto'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    else:
        predicciones['gasto_predicho'] = 0
    
    # Predicción de cantidad
    if 'cantidad' in modelos:
        pred_cantidad = modelos['cantidad'].predecir(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'cantidad_nn' in modelos:
        pred_cantidad = modelos['cantidad_nn'].predecir(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'arima_num_transacciones' in modelos:
        pred_cantidad = modelos['arima_num_transacciones'].forecast(meses_adelante)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    else:
        predicciones['cantidad_predicha'] = 0
    
    # Predicción de momento (probabilidad de compra)
    if 'momento' in modelos:
        prob_compra = modelos['momento'].predict_proba(ultimo_punto)[0, 1]
        predicciones['probabilidad_compra'] = prob_compra
    elif 'clf_momento' in modelos:
        prob_compra = modelos['clf_momento'].predict_proba(ultimo_punto)[0, 1]
        predicciones['probabilidad_compra'] = prob_compra
    else:
        predicciones['probabilidad_compra'] = 0.5  # Neutral
    
    # Métricas de confianza
    if 'gasto' in modelo_info['resultados']:
        predicciones['confianza_gasto'] = modelo_info['resultados']['gasto']['r2']
    else:
        predicciones['confianza_gasto'] = 0
    
    if 'cantidad' in modelo_info['resultados']:
        predicciones['confianza_cantidad'] = modelo_info['resultados']['cantidad']['r2']
    else:
        predicciones['confianza_cantidad'] = 0
    
    predicciones['categoria'] = categoria
    
    return predicciones

# =============================================================================
# FUNCIONES PARA CREAR SERIES TEMPORALES
# =============================================================================

def crear_series_temporales(df, nivel='CATEGORIA'):
    """
    Convierte datos transaccionales en series temporales mensuales
    """
    print(f"\n📊 Creando series temporales por {nivel}...")
    
    # Agregación mensual con múltiples métricas
    agg_dict = {
        'TOTALPESOS': ['sum', 'mean', 'std', 'count'],
        'SOLICITUD': 'nunique',
        'CENTROCOSTO': 'nunique',
        'SOLICITANTE': 'nunique'
    }
    
    df_agregado = df.groupby([nivel, 'año_mes']).agg(agg_dict).reset_index()
    
    # Aplanar nombres de columnas
    df_agregado.columns = [
        nivel.lower(), 'año_mes', 'gasto_total', 'gasto_promedio', 
        'gasto_std', 'num_transacciones', 'ordenes_unicas', 
        'centros_costo', 'solicitantes_unicos'
    ]
    
    # Convertir período a fecha
    df_agregado['fecha'] = df_agregado['año_mes'].dt.to_timestamp()
    df_agregado = df_agregado.sort_values([nivel.lower(), 'fecha'])
    
    # Rellenar meses faltantes para cada categoría
    print("🔄 Completando series temporales...")
    df_completo = []
    
    categorias_unicas = df_agregado[nivel.lower()].unique()
    print(f"   Procesando {len(categorias_unicas)} {nivel.lower()}s...")
    
    for i, categoria in enumerate(categorias_unicas):
        if i % 10 == 0 and i > 0:
            print(f"   Progreso: {i}/{len(categorias_unicas)}")
            
        df_cat = df_agregado[df_agregado[nivel.lower()] == categoria].copy()
        
        # Crear rango completo de fechas
        fecha_min = df_cat['fecha'].min()
        fecha_max = df_cat['fecha'].max()
        fechas_completas = pd.date_range(fecha_min, fecha_max, freq='MS')
        
        # Reindexar y rellenar
        df_cat = df_cat.set_index('fecha').reindex(fechas_completas)
        df_cat[nivel.lower()] = categoria
        df_cat = df_cat.fillna(0)  # Llenar huecos con 0
        df_cat['fecha'] = df_cat.index
        
        df_completo.append(df_cat.reset_index(drop=True))
    
    df_final = pd.concat(df_completo, ignore_index=True)
    
    print(f"✅ Series temporales creadas:")
    print(f"   📊 {len(df_final):,} puntos de datos")
    print(f"   🏷️ {df_final[nivel.lower()].nunique()} {nivel.lower()}s únicas")
    print(f"   📅 {df_final['fecha'].nunique()} períodos temporales")
    
    # Guardar en variable global
    global datos_temporales
    datos_temporales[nivel] = df_final
    
    return df_final

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def cargar_y_procesar_datos_avanzado(ruta_csv):
    """
    Versión avanzada de carga y procesamiento de datos
    """
    print("🚀 SISTEMA AVANZADO DE PREDICCIÓN - GRUPO SALINAS")
    print("="*60)
    print("📊 Cargando y procesando datos...")
    
    # Cargar datos
    df = pd.read_csv(ruta_csv)
    print(f"✅ Datos originales: {len(df):,} registros, {len(df.columns)} columnas")
    
    # Limpieza básica
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
    df = df[df['TOTALPESOS'] > 0]
    
    # Limpieza avanzada de outliers
    df = detectar_y_limpiar_outliers(df, ['TOTALPESOS'], metodo='zscore', threshold=3.0)
    
    # Normalización de categorías
    print("🧹 Normalizando categorías...")
    df['CATEGORIA'] = df['CATEGORIA'].fillna('Sin Categoría')
    
    # Unificar categorías similares
    categoria_mapping = {
        'Construccion': 'Construcción',
        'Tecnologia': 'Tecnología',
        'Produccion': 'Producción',
        'Servicios ': 'Servicios',
        'Insumos ': 'Insumos',
        'Ensamblika ': 'Ensamblika'
    }
    
    df['CATEGORIA'] = df['CATEGORIA'].replace(categoria_mapping)
    
    # Filtrar categorías con datos suficientes
    categoria_counts = df['CATEGORIA'].value_counts()
    categorias_validas = categoria_counts[categoria_counts >= 100].index  # Al menos 100 transacciones
    df = df[df['CATEGORIA'].isin(categorias_validas)]
    
    # Crear columnas temporales
    df['año'] = df['FECHAPEDIDO'].dt.year
    df['mes'] = df['FECHAPEDIDO'].dt.month
    df['año_mes'] = df['FECHAPEDIDO'].dt.to_period('M')
    df['trimestre'] = df['FECHAPEDIDO'].dt.quarter
    
    # Limpiar otras columnas
    df['FAMILIA'] = df['FAMILIA'].fillna('Sin Familia')
    df['CLASE'] = df['CLASE'].fillna('Sin Clase')
    df['CENTROCOSTO'] = df['CENTROCOSTO'].fillna('Sin Centro')
    df['SOLICITANTE'] = df['SOLICITANTE'].fillna('Sin Solicitante')
    
    print(f"✅ Datos procesados: {len(df):,} registros")
    print(f"📅 Período: {df['FECHAPEDIDO'].min().strftime('%Y-%m-%d')} a {df['FECHAPEDIDO'].max().strftime('%Y-%m-%d')}")
    print(f"💰 Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
    print(f"🏷️ Categorías válidas: {df['CATEGORIA'].nunique()}")
    
    return df

def ejecutar_entrenamiento_avanzado(df, nivel='CATEGORIA'):
    """
    Ejecuta el entrenamiento avanzado completo
    """
    print(f"\n🎯 INICIANDO ENTRENAMIENTO AVANZADO POR {nivel}")
    print("="*60)
    
    # Crear series temporales
    df_temporal = crear_series_temporales(df, nivel)
    
    # Obtener lista de categorías para procesar
    categorias = df_temporal[nivel.lower()].unique()
    print(f"📋 Categorías a procesar: {len(categorias)}")
    
    # Entrenar modelos por categoría
    modelos_exitosos = {}
    resultados_globales = []
    
    for i, categoria in enumerate(categorias):
        print(f"\n🔄 Progreso: {i+1}/{len(categorias)} - {categoria}")
        
        # Filtrar datos de la categoría
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        df_cat = df_cat.sort_values('fecha')
        
        # Entrenar modelos para esta categoría
        resultado = entrenar_categoria_avanzada(df_cat, categoria)
        
        if resultado is not None:
            modelos_exitosos[categoria] = resultado
            
            # Agregar a resultados globales
            resultado_resumen = {
                'categoria': categoria,
                'datos_usados': resultado['datos_usados']
            }
            
            if 'gasto' in resultado['resultados']:
                resultado_resumen.update({
                    'r2_gasto': resultado['resultados']['gasto']['r2'],
                    'mae_gasto': resultado['resultados']['gasto']['mae']
                })
            
            if 'cantidad' in resultado['resultados']:
                resultado_resumen.update({
                    'r2_cantidad': resultado['resultados']['cantidad']['r2'],
                    'mae_cantidad': resultado['resultados']['cantidad']['mae']
                })
            
            if 'momento' in resultado['resultados']:
                resultado_resumen.update({
                    'accuracy_momento': resultado['resultados']['momento']['accuracy'],
                    'auc_momento': resultado['resultados']['momento']['auc']
                })
            
            resultados_globales.append(resultado_resumen)
    
    # Guardar modelos en variable global
    global modelos_entrenados
    modelos_entrenados = modelos_exitosos
    
    return modelos_exitosos, resultados_globales

def generar_predicciones_avanzadas(modelos, meses_adelante=1):
    """
    Genera predicciones avanzadas para todas las categorías
    """
    print(f"\n🔮 GENERANDO PREDICCIONES PARA {meses_adelante} MES(ES) ADELANTE")
    print("="*60)
    
    predicciones = []
    
    for categoria, modelo_info in modelos.items():
        pred = predecir_categoria_avanzada(modelo_info, meses_adelante)
        if pred is not None:
            predicciones.append(pred)
    
    if not predicciones:
        print("❌ No se pudieron generar predicciones")
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df_predicciones = pd.DataFrame(predicciones)
    
    # Calcular totales
    total_gasto = df_predicciones['gasto_predicho'].sum()
    total_cantidad = df_predicciones['cantidad_predicha'].sum()
    confianza_promedio_gasto = df_predicciones['confianza_gasto'].mean()
    confianza_promedio_cantidad = df_predicciones['confianza_cantidad'].mean()
    prob_promedio_compra = df_predicciones['probabilidad_compra'].mean()
    
    print(f"💰 Gasto total predicho: ${total_gasto:,.2f}")
    print(f"📦 Cantidad total predicha: {total_cantidad:,} transacciones")
    print(f"🎯 Confianza promedio (Gasto): {confianza_promedio_gasto:.3f}")
    print(f"🎯 Confianza promedio (Cantidad): {confianza_promedio_cantidad:.3f}")
    print(f"📊 Probabilidad promedio de compra: {prob_promedio_compra:.3f}")
    
    return df_predicciones.sort_values('gasto_predicho', ascending=False)

def mostrar_resumen_avanzado(resultados_globales):
    """
    Muestra resumen completo de resultados
    """
    print(f"\n📊 RESUMEN FINAL DEL ENTRENAMIENTO AVANZADO")
    print("="*60)
    
    if not resultados_globales:
        print("❌ No hay resultados para mostrar")
        return
    
    df_resultados = pd.DataFrame(resultados_globales)
    
    # Estadísticas generales
    total_categorias = len(df_resultados)
    print(f"🎯 Categorías procesadas exitosamente: {total_categorias}")
    
    # Estadísticas de calidad - Gasto
    if 'r2_gasto' in df_resultados.columns:
        r2_gasto_mean = df_resultados['r2_gasto'].mean()
        r2_gasto_median = df_resultados['r2_gasto'].median()
        modelos_buenos_gasto = len(df_resultados[df_resultados['r2_gasto'] > 0.5])
        modelos_regulares_gasto = len(df_resultados[(df_resultados['r2_gasto'] > 0.3) & (df_resultados['r2_gasto'] <= 0.5)])
        
        print(f"\n💰 CALIDAD MODELOS DE GASTO:")
        print(f"   R² promedio: {r2_gasto_mean:.3f}")
        print(f"   R² mediana: {r2_gasto_median:.3f}")
        print(f"   Modelos buenos (R² > 0.5): {modelos_buenos_gasto}/{total_categorias} ({100*modelos_buenos_gasto/total_categorias:.1f}%)")
        print(f"   Modelos regulares (R² 0.3-0.5): {modelos_regulares_gasto}/{total_categorias} ({100*modelos_regulares_gasto/total_categorias:.1f}%)")
    
    # Estadísticas de calidad - Cantidad
    if 'r2_cantidad' in df_resultados.columns:
        r2_cantidad_mean = df_resultados['r2_cantidad'].mean()
        r2_cantidad_median = df_resultados['r2_cantidad'].median()
        modelos_buenos_cantidad = len(df_resultados[df_resultados['r2_cantidad'] > 0.5])
        
        print(f"\n📦 CALIDAD MODELOS DE CANTIDAD:")
        print(f"   R² promedio: {r2_cantidad_mean:.3f}")
        print(f"   R² mediana: {r2_cantidad_median:.3f}")
        print(f"   Modelos buenos (R² > 0.5): {modelos_buenos_cantidad}/{total_categorias} ({100*modelos_buenos_cantidad/total_categorias:.1f}%)")
    
    # Top mejores modelos
    print(f"\n🏆 TOP 5 MEJORES MODELOS (por R² Gasto):")
    if 'r2_gasto' in df_resultados.columns:
        top_5_gasto = df_resultados.nlargest(5, 'r2_gasto')[['categoria', 'r2_gasto', 'mae_gasto']]
        for _, row in top_5_gasto.iterrows():
            print(f"   {row['categoria']}: R² = {row['r2_gasto']:.3f}, MAE = ${row['mae_gasto']:,.0f}")
    
    # Mostrar feature importance global
    if feature_importance_global:
        print(f"\n🎯 FEATURES MÁS IMPORTANTES GLOBALMENTE:")
        # Promediar importancias across categorías
        all_features = {}
        for cat, features in feature_importance_global.items():
            for feature, importance in features.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        # Calcular importancia promedio
        avg_importance = {feature: np.mean(importances) for feature, importances in all_features.items()}
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.3f}")

def guardar_modelos_avanzados(modelos, directorio="modelos_avanzados"):
    """
    Guarda todos los modelos entrenados
    """
    print(f"\n💾 GUARDANDO MODELOS AVANZADOS...")
    
    # Crear directorio con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_final = f"{directorio}_{timestamp}"
    os.makedirs(dir_final, exist_ok=True)
    
    modelos_guardados = 0
    
    for categoria, modelo_info in modelos.items():
        try:
            # Crear subdirectorio para cada categoría
            cat_dir = os.path.join(dir_final, categoria.replace("/", "_").replace(" ", "_"))
            os.makedirs(cat_dir, exist_ok=True)
            
            # Guardar cada modelo de la categoría
            for tipo_modelo, modelo_obj in modelo_info['modelos'].items():
                archivo_modelo = os.path.join(cat_dir, f"{tipo_modelo}.joblib")
                joblib.dump(modelo_obj, archivo_modelo)
                modelos_guardados += 1
            
            # Guardar metadatos
            metadata = {
                'categoria': modelo_info['categoria'],
                'resultados': modelo_info['resultados'],
                'features': modelo_info['features'],
                'datos_usados': modelo_info['datos_usados'],
                'timestamp': timestamp
            }
            
            metadata_file = os.path.join(cat_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Error guardando {categoria}: {str(e)}")
    
    # Guardar configuración global
    config_global = {
        'total_categorias': len(modelos),
        'modelos_guardados': modelos_guardados,
        'configuracion': {
            'random_state': config.random_state,
            'min_meses_entrenamiento': config.min_meses_entrenamiento,
            'test_size': config.test_size
        },
        'feature_importance_global': feature_importance_global,
        'timestamp': timestamp
    }
    
    config_file = os.path.join(dir_final, "configuracion_global.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_global, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ Modelos guardados exitosamente:")
    print(f"   📁 Directorio: {dir_final}")
    print(f"   🔧 Modelos guardados: {modelos_guardados}")
    print(f"   📊 Categorías: {len(modelos)}")

# =============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =============================================================================

def ejecutar_sistema_completo(ruta_csv, meses_prediccion=1):
    """
    Ejecuta el sistema completo de predicción avanzada
    """
    print("🚀 INICIANDO SISTEMA AVANZADO DE PREDICCIÓN DE COMPRAS")
    print("="*80)
    
    # 1. Cargar y procesar datos
    df = cargar_y_procesar_datos_avanzado(ruta_csv)
    
    # 2. Entrenar modelos
    modelos, resultados = ejecutar_entrenamiento_avanzado(df)
    
    # 3. Mostrar resumen
    mostrar_resumen_avanzado(resultados)
    
    # 4. Generar predicciones
    if modelos:
        predicciones = generar_predicciones_avanzadas(modelos, meses_prediccion)
        
        print(f"\n🔮 TOP 10 PREDICCIONES POR GASTO:")
        print("-" * 80)
        if not predicciones.empty:
            display(predicciones.head(10))
    
    # 5. Guardar modelos
    if modelos:
        guardar_modelos_avanzados(modelos)
    
    print(f"\n✅ SISTEMA COMPLETADO EXITOSAMENTE!")
    print("="*80)
    
    return modelos, predicciones, resultados

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

"""
# Para usar el sistema:

# 1. Cargar datos y ejecutar sistema completo
ruta_archivo = "path/to/your/data.csv"
modelos, predicciones, resultados = ejecutar_sistema_completo(ruta_archivo)

# 2. Para predicciones adicionales
nuevas_predicciones = generar_predicciones_avanzadas(modelos, meses_adelante=3)

# 3. Para cargar modelos previamente guardados
# modelos_cargados = joblib.load("modelos_avanzados_YYYYMMDD_HHMM/categoria/modelo.joblib")
"""
