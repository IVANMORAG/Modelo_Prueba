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
- Modo rescate para categorías problemáticas

Autor: Sistema IA Avanzado
Versión: 2.2 (con correcciones de errores y mejoras de robustez)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import os
import joblib
import json
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import zscore
import itertools

# Configurar logging
logging.basicConfig(
    filename='predicciones.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# ML Libraries
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            ExtraTreesRegressor, VotingRegressor, RandomForestClassifier,
                            IsolationForest, StackingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression)
from sklearn.svm import SVR
from sklearn.model_selection import (TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           mean_absolute_percentage_error, accuracy_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Modelos avanzados externos

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logging.info("Prophet disponible para small data")
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet no disponible. Instala con 'pip install prophet' para mejorar small data")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost no disponible")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.warning("LightGBM no disponible")

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    logging.warning("CatBoost no disponible")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    logging.info("TensorFlow disponible para redes neuronales")
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow no disponible")

# Configuración global
class ConfiguracionAvanzada:
    def __init__(self):
        self.random_state = 42
        self.test_size = 0.25
        self.min_meses_entrenamiento = 6
        self.cv_folds = 5
        self.outlier_threshold = 3.5
        self.feature_selection_k = 15
        self.ensemble_weights = 'uniform'
        self.neural_epochs = 100
        self.neural_batch_size = 32
        self.neural_patience = 15
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
modelos_entrenados = {}
datos_temporales = {}
feature_importance_global = {}
UMBRAL_GASTO_MALO = 0.4
UMBRAL_CANTIDAD_MALO = 0.6
CATEGORIAS_PROBLEMATICAS = [
    'EPC Propio', 'Gestión y Comercialización (G&C)', 'O&M Terceros',
    'Construcción ', 'Transportes'
]

import unicodedata

def normalize_category_name(name):
    """Normaliza nombres de categorías: elimina acentos, espacios extra y unifica mayúsculas/minúsculas."""
    # Convertir a minúsculas
    name = str(name).lower()
    # Eliminar acentos
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Eliminar espacios extra y caracteres especiales
    name = ' '.join(name.split())
    # Diccionario para unificar nombres similares
    category_mapping = {
        'construccion': 'Construcción',
        'construccion ': 'Construcción',  # Captura "Construcción " con espacio
        'construcción': 'Construcción',
        'construcción ': 'Construcción',
        'electromecanico': 'Electromecánico',
        'electromecánico': 'Electromecánico',
        'impresos y publicidad': 'Impresos Y Publicidad',
        'gastos de fin de ano': 'Gastos De Fin De Año',
        'o&m propio': 'O&M Propio',
        'o&m terceros': 'O&M Terceros',
        'epc tercero': 'EPC Tercero',
        'obsequios y atenciones': 'Obsequios Y Atenciones',
        'tecnologia': 'Tecnología',
        'tecnología': 'Tecnología',
        'produccion': 'Producción',
        'producción': 'Producción',
        'servicios': 'Servicios',
        'servicios ': 'Servicios',
        'insumos': 'Insumos',
        'insumos ': 'Insumos',
        'ensamblika': 'Ensamblika',
        'ensamblika ': 'Ensamblika'
    }
    return category_mapping.get(name, name.capitalize())

def cargar_y_procesar_datos_avanzado(ruta_csv):
    """Carga y procesa los datos, normalizando nombres de categorías."""
    print(" SISTEMA AVANZADO DE PREDICCIÓN - GRUPO SALINAS")
    print("="*60)
    print(" Cargando y procesando datos...")
    logging.info("Cargando datos desde CSV")
    
    try:
        df = pd.read_csv(ruta_csv, encoding='latin-1')
        df.columns = df.columns.str.upper()
        logging.info(f"Datos originales: {len(df):,} registros, {len(df.columns)} columnas")
        
        # Verificar columnas requeridas
        required_columns = ['FECHAPEDIDO', 'TOTALPESOS', 'CATEGORIA', 'SOLICITUD', 'CENTROCOSTO', 'SOLICITANTE']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
        
        # Normalizar nombres de categorías
        df['CATEGORIA'] = df['CATEGORIA'].apply(normalize_category_name)
        
        # Convertir fechas y limpiar datos
        df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
        df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS', 'CATEGORIA'])
        df = df[df['TOTALPESOS'] > 0]
        
        # Limpiar outliers
        df = detectar_y_limpiar_outliers(df, ['TOTALPESOS'], metodo='zscore', threshold=3.0)
        
        # Rellenar valores nulos en otras columnas
        df['FAMILIA'] = df['FAMILIA'].fillna('Sin Familia')
        df['CLASE'] = df['CLASE'].fillna('Sin Clase')
        df['CENTROCOSTO'] = df['CENTROCOSTO'].fillna('Sin Centro')
        df['SOLICITANTE'] = df['SOLICITANTE'].fillna('Sin Solicitante')
        
        # Crear columnas temporales
        df['AÑO'] = df['FECHAPEDIDO'].dt.year
        df['MES'] = df['FECHAPEDIDO'].dt.month
        df['AÑO_MES'] = df['FECHAPEDIDO'].dt.to_period('M')
        df['TRIMESTRE'] = df['FECHAPEDIDO'].dt.quarter
        df['MES_SIN'] = np.sin(2 * np.pi * df['MES'] / 12)
        df['MES_COS'] = np.cos(2 * np.pi * df['MES'] / 12)
        
        # Filtrar categorías con pocos registros
        categoria_counts = df['CATEGORIA'].value_counts()
        categorias_validas = categoria_counts[categoria_counts >= 100].index
        df = df[df['CATEGORIA'].isin(categorias_validas)]
        
        # Agregar datos por categoría y mes
        global datos_temporales
        datos_temporales = df.groupby(['CATEGORIA', 'AÑO_MES']).agg({
            'TOTALPESOS': 'sum',
            'SOLICITUD': 'count',
            'CENTROCOSTO': 'nunique',
            'SOLICITANTE': 'nunique',
            'MES_SIN': 'mean',
            'MES_COS': 'mean'
        }).reset_index().rename(columns={
            'TOTALPESOS': 'gasto_total',
            'SOLICITUD': 'num_transacciones',
            'CENTROCOSTO': 'centros_costo',
            'SOLICITANTE': 'solicitantes_unicos'
        })
        
        logging.info(f"Datos procesados: {len(df):,} registros")
        logging.info(f"Período: {df['FECHAPEDIDO'].min().strftime('%Y-%m-%d')} a {df['FECHAPEDIDO'].max().strftime('%Y-%m-%d')}")
        logging.info(f"Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
        logging.info(f"Categorías válidas: {df['CATEGORIA'].nunique()}")
        
        return df
    except Exception as e:
        logging.error(f"Error cargando datos: {str(e)}")
        raise

# Utilidades avanzadas
def detectar_y_limpiar_outliers(df, columnas=['TOTALPESOS'], metodo='zscore', threshold=3.5):
    logging.info(f"Detectando outliers en {columnas} con método {metodo}")
    df_clean = df.copy()
    outliers_removidos = 0
    for columna in columnas:
        if columna not in df_clean.columns:
            continue
        antes = len(df_clean)
        if metodo == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[columna].dropna()))
            df_clean = df_clean[z_scores < threshold]
        elif metodo == 'iqr':
            Q1 = df_clean[columna].quantile(0.25)
            Q3 = df_clean[columna].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[columna] >= lower_bound) & (df_clean[columna] <= upper_bound)]
        elif metodo == 'percentile':
            lower_percentile = df_clean[columna].quantile(0.01)
            upper_percentile = df_clean[columna].quantile(0.99)
            df_clean = df_clean[(df_clean[columna] >= lower_percentile) & (df_clean[columna] <= upper_percentile)]
        despues = len(df_clean)
        outliers_col = antes - despues
        outliers_removidos += outliers_col
        logging.info(f"{columna}: {outliers_col:,} outliers removidos")
    logging.info(f"Total outliers removidos: {outliers_removidos:,}")
    return df_clean

def crear_features_avanzados(df_serie, categoria):
    df = df_serie.copy()
    logging.info(f"Creando features avanzados para {categoria}")
    required_metrics = ['gasto_total', 'num_transacciones', 'gasto_promedio']
    available_metrics = [m for m in required_metrics if m in df.columns]
    if not available_metrics:
        logging.error(f"No se encontraron métricas requeridas: {required_metrics}")
        return df
    if 'fecha' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        logging.error(f"Columna 'fecha' no válida o no encontrada")
        return df

    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['año'] = df['fecha'].dt.year
    df['dias_mes'] = df['fecha'].dt.days_in_month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trim_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trim_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    df['es_fin_año'] = ((df['mes'] == 12) | (df['mes'] == 1)).astype(int)
    df['es_medio_año'] = ((df['mes'] >= 6) & (df['mes'] <= 8)).astype(int)
    df['es_inicio_año'] = ((df['mes'] >= 1) & (df['mes'] <= 3)).astype(int)

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

    df['tendencia_lineal'] = range(len(df))
    for metric in available_metrics:
        df[f'{metric}_pct_change_1'] = df[metric].pct_change(1)
        df[f'{metric}_pct_change_3'] = df[metric].pct_change(3)
        df[f'{metric}_pct_change_12'] = df[metric].pct_change(12)
        df[f'{metric}_accel'] = df[f'{metric}_pct_change_1'].diff()
        if f'{metric}_ma_3' in df.columns and f'{metric}_ma_12' in df.columns:
            df[f'{metric}_momentum_3_12'] = (df[f'{metric}_ma_3'] / df[f'{metric}_ma_12']) - 1
        if f'{metric}_ma_6' in df.columns and f'{metric}_ma_24' in df.columns:
            df[f'{metric}_momentum_6_24'] = (df[f'{metric}_ma_6'] / df[f'{metric}_ma_24']) - 1

    if 'gasto_total' in df.columns and 'num_transacciones' in df.columns:
        df['gasto_por_transaccion'] = df['gasto_total'] / (df['num_transacciones'] + 1e-8)
    if 'gasto_total' in df.columns and 'ordenes_unicas' in df.columns:
        df['gasto_por_orden'] = df['gasto_total'] / (df['ordenes_unicas'] + 1e-8)
        df['transacciones_por_orden'] = df['num_transacciones'] / (df['ordenes_unicas'] + 1e-8)
        df['diversidad_centros'] = df['centros_costo'] / (df['ordenes_unicas'] + 1e-8)
        df['diversidad_solicitantes'] = df['solicitantes_unicos'] / (df['ordenes_unicas'] + 1e-8)

    if 'gasto_total' in df.columns and f'gasto_total_ma_6' in df.columns:
        df['volatilidad_gasto'] = df['gasto_total'].rolling(window=6, min_periods=1).std()
        df['coef_variacion_gasto'] = df['volatilidad_gasto'] / (df['gasto_total_ma_6'] + 1e-8)
        df['estabilidad_gasto'] = 1 / (1 + df['coef_variacion_gasto'])

    if 'gasto_total' in df.columns:
        df['meses_sin_actividad'] = (df['gasto_total'] == 0).astype(int)
        df['racha_sin_actividad'] = df.groupby((df['gasto_total'] > 0).cumsum())['meses_sin_actividad'].cumsum()
        df['compra_activa'] = (df['gasto_total'] > 0).astype(int)
        df['frecuencia_compra_6m'] = df['compra_activa'].rolling(window=6, min_periods=1).sum()
        df['frecuencia_compra_12m'] = df['compra_activa'].rolling(window=12, min_periods=1).sum()

    for metric in available_metrics:
        media_historica = df[metric].expanding(min_periods=1).mean()
        df[f'{metric}_vs_historico'] = df[metric] / (media_historica + 1e-8)
        df[f'{metric}_desv_historico'] = (df[metric] - media_historica) / (media_historica + 1e-8)
        rolling_mean = df[metric].rolling(window=12, min_periods=1).mean()
        rolling_std = df[metric].rolling(window=12, min_periods=1).std()
        df[f'{metric}_z_score'] = (df[metric] - rolling_mean) / (rolling_std + 1e-8)
        df[f'{metric}_es_anomalia'] = (np.abs(df[f'{metric}_z_score']) > 2).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['fecha', 'año', 'mes', 'trimestre']:
            df[col] = (df[col]
                      .fillna(method='ffill')
                      .fillna(method='bfill')
                      .fillna(df[col].rolling(window=6, min_periods=1).mean())
                      .fillna(0))
    logging.info(f"Features creados: {len([col for col in df.columns if any(x in col for x in ['_lag_', '_ma_', '_std_', '_pct_', 'sin', 'cos'])])} features temporales")
    return df

# Modelos avanzados
class ModeloEnsembleAvanzado:
    def __init__(self, target_name, config):
        self.target_name = target_name
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_model = None
        self.best_score = -np.inf

    def crear_modelos_base(self):
        models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=self.config.random_state, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.9, random_state=self.config.random_state),
            'et': ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=self.config.random_state, n_jobs=-1),
            'ridge': Ridge(alpha=10.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config.random_state)
        }
        if XGB_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.9, random_state=self.config.random_state, n_jobs=-1)
        if LGB_AVAILABLE:
            models['lgb'] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.9, random_state=self.config.random_state, n_jobs=-1, verbose=-1)
        if CB_AVAILABLE:
            models['cb'] = cb.CatBoostRegressor(iterations=200, learning_rate=0.1, depth=8, random_state=self.config.random_state, verbose=False)
        return models

    def optimizar_hiperparametros(self, X_train, y_train, model_name, model):
        if model_name not in self.config.models_config or not self.config.models_config[model_name]:
            return model
        logging.info(f"Optimizando {model_name}")
        param_grid = self.config.models_config[model_name]
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=tscv, scoring='neg_mean_absolute_error', random_state=self.config.random_state, n_jobs=-1, verbose=0)
        try:
            search.fit(X_train, y_train)
            return search.best_estimator_
        except Exception as e:
            logging.error(f"Error optimizando {model_name}: {str(e)}")
            return model

    def entrenar_ensemble(self, X_train, y_train, X_test, y_test):
        logging.info(f"Entrenando ensemble para {self.target_name}")
        if X_train.shape[1] > self.config.feature_selection_k:
            logging.info(f"Seleccionando top {self.config.feature_selection_k} features")
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
        else:
            X_train_selected = X_train
            X_test_selected = X_test

        base_models = self.crear_modelos_base()
        trained_models = []
        model_scores = {}
        for name, model in base_models.items():
            try:
                logging.info(f"Entrenando {name}")
                if name in ['ridge', 'elastic', 'svr']:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_selected)
                    X_test_scaled = scaler.transform(X_test_selected)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train_selected
                    X_test_scaled = X_test_selected
                    self.scalers[name] = None
                optimized_model = self.optimizar_hiperparametros(X_train_scaled, y_train, name, model)
                optimized_model.fit(X_train_scaled, y_train)
                y_pred = optimized_model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                model_scores[name] = {'r2': score, 'mae': mae}
                if score > 0.1:
                    self.models[name] = optimized_model
                    trained_models.append((name, optimized_model))
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = optimized_model
                logging.info(f"{name}: R² = {score:.3f}, MAE = {mae:,.0f}")
            except Exception as e:
                logging.error(f"Error entrenando {name}: {str(e)}")

        if len(trained_models) > 1:
            try:
                logging.info("Creando modelo ensemble")
                ensemble_models = [(name, model) for name, model in trained_models if model_scores[name]['r2'] > 0.2]
                if len(ensemble_models) > 1:
                    voting_regressor = VotingRegressor(estimators=ensemble_models, weights=None)
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
                    logging.info(f"Ensemble: R² = {ensemble_score:.3f}, MAE = {ensemble_mae:,.0f}")
                    if ensemble_score > self.best_score:
                        self.models['ensemble'] = voting_regressor
                        self.best_model = voting_regressor
                        self.best_score = ensemble_score
            except Exception as e:
                logging.error(f"Error creando ensemble: {str(e)}")
        return model_scores

    def predecir(self, X):
        if self.best_model is None:
            return np.zeros(len(X))
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        best_name = next((name for name, model in self.models.items() if model == self.best_model), 'ensemble')
        if best_name in self.scalers and self.scalers[best_name]:
            X_scaled = self.scalers[best_name].transform(X_selected)
        else:
            X_scaled = X_selected
        return self.best_model.predict(X_scaled)

class RedNeuronalAvanzada:
    def __init__(self, target_name, config):
        self.target_name = target_name
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def crear_modelo(self, input_dim):
        if not TF_AVAILABLE:
            return None
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(1, activation='linear')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    def entrenar(self, X_train, y_train, X_test, y_test):
        if not TF_AVAILABLE:
            logging.error(f"TensorFlow no disponible para {self.target_name}")
            return None
        logging.info(f"Entrenando red neuronal para {self.target_name}")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = self.crear_modelo(X_train.shape[1])
        if self.model is None:
            return None
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.neural_patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        try:
            self.history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.config.neural_epochs,
                batch_size=self.config.neural_batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=callbacks,
                verbose=0
            )
            y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            logging.info(f"Red Neuronal: R² = {r2:.3f}, MAE = {mae:,.0f}")
            return {'r2': r2, 'mae': mae}
        except Exception as e:
            logging.error(f"Error entrenando red neuronal: {str(e)}")
            return None

    def predecir(self, X):
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

# Funciones para small data
def augmentar_datos_small_data(df_features, target='gasto_total'):
    df_augmented = df_features.copy()
    if len(df_features) >= 6:
        return df_augmented
    n_augment = 6 - len(df_features)
    for _ in range(n_augment):
        ruido = np.random.normal(0, 0.1, len(df_features))
        df_temp = df_features.copy()
        for col in df_temp.select_dtypes(include=[np.number]).columns:
            if col not in ['fecha', 'año', 'mes', 'trimestre']:
                df_temp[col] = df_temp[col] + ruido * df_temp[col].std()
        df_augmented = pd.concat([df_augmented, df_temp], ignore_index=True)
    logging.info(f"Datos aumentados: {len(df_features)} → {len(df_augmented)}")
    return df_augmented


def entrenar_modo_small_data(df_categoria, nombre_categoria):
    """
    Entrena modelos para categorías con pocos datos (<=11 meses) usando Prophet con covariates,
    ARIMA, Gaussian Process y Transfer Learning como respaldo.
    
    Args:
        df_categoria (pd.DataFrame): Datos de la categoría.
        nombre_categoria (str): Nombre de la categoría.
    
    Returns:
        dict: Modelos entrenados, resultados, características usadas y datos usados.
    """
    logging.info(f"Modo Small Data para {nombre_categoria} con {len(df_categoria)} meses")
    print(f"   Procesando {nombre_categoria} en modo small data ({len(df_categoria)} meses)")
    
    # Crear features avanzados
    df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    df_features.set_index('fecha', inplace=True)
    
    # Definir columnas a excluir de las features
    exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
                    'num_transacciones', 'ordenes_unicas', 'centros_costo', 
                    'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    modelos_small = {}
    resultados_small = {}
    
    # Entrenar Prophet (o ARIMA como fallback) para gasto_total y num_transacciones
    targets = ['gasto_total', 'num_transacciones']
    for target in targets:
        if target in df_features.columns:
            df_prophet = pd.DataFrame({
                'ds': df_features.index,  # Fecha para Prophet
                'y': df_features[target],  # Variable objetivo
                'mes_sin': df_features['mes_sin'],  # Covariate estacional
                'mes_cos': df_features['mes_cos']   # Covariate estacional
            })
            
            if PROPHET_AVAILABLE:
                try:
                    # Deshabilitar yearly seasonality si <12 meses
                    seasonality = len(df_features) >= 12
                    model = Prophet(
                        yearly_seasonality=seasonality,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.01  # Menos sensibilidad para small data
                    )
                    # Agregar covariates
                    model.add_regressor('mes_sin')
                    model.add_regressor('mes_cos')
                    model.fit(df_prophet)
                    modelos_small[f'prophet_{target}'] = model
                    
                    # Evaluar el modelo
                    split = int(len(df_prophet) * 0.8)
                    train = df_prophet.iloc[:split]
                    test = df_prophet.iloc[split:]
                    future = model.make_future_dataframe(periods=len(test), freq='MS')
                    future['mes_sin'] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
                    future['mes_cos'] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
                    forecast = model.predict(future)
                    pred = forecast['yhat'].iloc[-len(test):]
                    r2 = r2_score(test['y'], pred)
                    mae = mean_absolute_error(test['y'], pred)
                    resultados_small[target] = {'r2': r2, 'mae': mae}
                    logging.info(f"Prophet {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                    print(f"   Prophet {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                except Exception as e:
                    logging.error(f"Error Prophet {target}: {str(e)}")
                    # Fallback a ARIMA
                    try:
                        model = ARIMA(df_features[target], order=(1,1,1))
                        model_fit = model.fit()
                        modelos_small[f'arima_{target}'] = model_fit
                        pred = model_fit.forecast(len(test['y']))
                        r2 = r2_score(test['y'], pred)
                        mae = mean_absolute_error(test['y'], pred)
                        resultados_small[target] = {'r2': r2, 'mae': mae}
                        logging.info(f"ARIMA fallback {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                        print(f"   ARIMA fallback {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                    except Exception as e:
                        logging.error(f"Error ARIMA {target}: {str(e)}")
                        # Fallback a promedio ponderado
                        pred = np.average(df_features[target], weights=np.linspace(1, 0.1, len(df_features[target])))
                        resultados_small[target] = {'r2': -999, 'mae': np.mean(np.abs(df_features[target] - pred))}
                        modelos_small[f'promedio_{target}'] = lambda x: pred
                        logging.info(f"Promedio fallback {target}: Valor = {pred:,.0f}")
                        print(f"   Promedio fallback {target}: Valor = {pred:,.0f}")
            else:
                # Si Prophet no está disponible, usar ARIMA
                try:
                    model = ARIMA(df_features[target], order=(1,1,1))
                    model_fit = model.fit()
                    modelos_small[f'arima_{target}'] = model_fit
                    split = int(len(df_features) * 0.8)
                    train, test = df_features[target][:split], df_features[target][split:]
                    pred = model_fit.forecast(len(test))
                    r2 = r2_score(test, pred)
                    mae = mean_absolute_error(test, pred)
                    resultados_small[target] = {'r2': r2, 'mae': mae}
                    logging.info(f"ARIMA {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                    print(f"   ARIMA {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                except Exception as e:
                    logging.error(f"Error ARIMA {target}: {str(e)}")
                    # Fallback a promedio ponderado
                    pred = np.average(df_features[target], weights=np.linspace(1, 0.1, len(df_features[target])))
                    resultados_small[target] = {'r2': -999, 'mae': np.mean(np.abs(df_features[target] - pred))}
                    modelos_small[f'promedio_{target}'] = lambda x: pred
                    logging.info(f"Promedio fallback {target}: Valor = {pred:,.0f}")
                    print(f"   Promedio fallback {target}: Valor = {pred:,.0f}")
    
    # Gaussian Process para gasto_total
    if 'gasto_total' in df_features.columns:
        try:
            X = np.arange(len(df_features)).reshape(-1, 1)
            y = df_features['gasto_total'].values
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp.fit(X, y)
            modelos_small['gp_gasto'] = gp
            X_test = np.array([[len(df_features) + i] for i in range(1, 4)])
            pred, _ = gp.predict(X_test, return_std=True)
            logging.info(f"GP Gasto: Predicciones = {pred}")
            print(f"   GP Gasto: Predicciones = {pred}")
        except Exception as e:
            logging.error(f"Error GP Gasto: {str(e)}")
    
    # Clasificador para Momento
    if 'gasto_total' in df_features.columns:
        try:
            df_features['compra_activa'] = (df_features['gasto_total'] > 0).astype(int)
            if len(np.unique(df_features['compra_activa'])) > 1:
                X = df_features[feature_cols].fillna(0)
                y = df_features['compra_activa']
                clf = LogisticRegression(class_weight='balanced', random_state=config.random_state)
                clf.fit(X, y)
                modelos_small['clf_momento'] = clf
                prob = clf.predict_proba(X)[:, 1].mean()
                resultados_small['momento'] = {
                    'accuracy': accuracy_score(y, clf.predict(X)),
                    'auc': roc_auc_score(y, clf.predict_proba(X)[:, 1]),
                    'probabilidad_promedio': prob
                }
                logging.info(f"Clasificador Momento: Probabilidad promedio = {prob:.3f}")
                print(f"   Clasificador Momento: Probabilidad promedio = {prob:.3f}")
            else:
                logging.warning(f"Clasificador Momento: Solo una clase encontrada")
                print(f"   Clasificador Momento: Solo una clase encontrada")
                resultados_small['momento'] = {'accuracy': 1.0, 'auc': 0.5, 'probabilidad_promedio': 0.5}
        except Exception as e:
            logging.error(f"Error Clasificador Momento: {str(e)}")
            resultados_small['momento'] = {'accuracy': 0.0, 'auc': 0.5, 'probabilidad_promedio': 0.5}
    
    # Transfer Learning desde categoría grande (Tecnología)
    large_category = 'Tecnología'
    if large_category in modelos_entrenados:
        pre_model = modelos_entrenados[large_category]['modelos'].get('gasto')
        if pre_model:
            try:
                X_small = df_features[feature_cols].fillna(0)
                y_small = df_features['gasto_total']
                pre_model.best_model.fit(X_small, y_small)
                modelos_small['transfer_gasto'] = pre_model
                logging.info("Transfer Learning aplicado desde 'Tecnología'")
                print(f"   Transfer Learning aplicado desde 'Tecnología'")
            except Exception as e:
                logging.error(f"Error Transfer Learning: {str(e)}")
    
    return {
        'categoria': nombre_categoria,
        'modelos': modelos_small,
        'resultados': resultados_small,
        'features': feature_cols,
        'datos_usados': len(df_features)
    }

# Sistema principal
def entrenar_categoria_avanzada(df_categoria, nombre_categoria, verbose=True):
    logging.info(f"Procesando categoría: {nombre_categoria}")
    if len(df_categoria) < 18:
        logging.info(f"Datos insuficientes para modo estándar: {len(df_categoria)} < 18 meses")
        return entrenar_modo_small_data(df_categoria, nombre_categoria)
    try:
        df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    except Exception as e:
        logging.error(f"Error creando features: {str(e)}")
        return None

    threshold = len(df_features.columns) * 0.5
    df_features = df_features.dropna(thresh=int(threshold))
    if len(df_features) < config.min_meses_entrenamiento:
        logging.error(f"Datos insuficientes después de limpieza: {len(df_features)}")
        return None

    exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 'num_transacciones', 'ordenes_unicas', 'centros_costo', 'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    if len(feature_cols) < 5:
        logging.error(f"Features insuficientes: {len(feature_cols)}")
        return None

    X = df_features[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']
    y_momento = (df_features['gasto_total'] > 0).astype(int)

    split_idx = int(len(X) * (1 - config.test_size))
    if split_idx < config.min_meses_entrenamiento * 0.7:
        logging.error(f"Split insuficiente: {split_idx}")
        return None

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    y_momento_train, y_momento_test = y_momento.iloc[:split_idx], y_momento.iloc[split_idx:]

    resultados = {}
    modelos_categoria = {}
    print("Entrenando modelos para GASTO...")
    modelo_gasto = ModeloEnsembleAvanzado('gasto', config)
    scores_gasto = modelo_gasto.entrenar_ensemble(X_train, y_gasto_train, X_test, y_gasto_test)
    if modelo_gasto.best_model is not None:
        modelos_categoria['gasto'] = modelo_gasto
        resultados['gasto'] = {
            'r2': modelo_gasto.best_score,
            'mae': mean_absolute_error(y_gasto_test, modelo_gasto.predecir(X_test)),
            'scores_individuales': scores_gasto
        }

    if TF_AVAILABLE and len(X_train) > 50:
        nn_gasto = RedNeuronalAvanzada('gasto', config)
        nn_scores = nn_gasto.entrenar(X_train, y_gasto_train, X_test, y_gasto_test)
        if nn_scores and nn_scores['r2'] > modelo_gasto.best_score:
            modelos_categoria['gasto_nn'] = nn_gasto
            resultados['gasto']['red_neuronal'] = nn_scores

    print("Entrenando modelos para CANTIDAD...")
    modelo_cantidad = ModeloEnsembleAvanzado('cantidad', config)
    scores_cantidad = modelo_cantidad.entrenar_ensemble(X_train, y_cantidad_train, X_test, y_cantidad_test)
    if modelo_cantidad.best_model is not None:
        modelos_categoria['cantidad'] = modelo_cantidad
        resultados['cantidad'] = {
            'r2': modelo_cantidad.best_score,
            'mae': mean_absolute_error(y_cantidad_test, modelo_cantidad.predecir(X_test)),
            'scores_individuales': scores_cantidad
        }

    if TF_AVAILABLE and len(X_train) > 50:
        nn_cantidad = RedNeuronalAvanzada('cantidad', config)
        nn_scores = nn_cantidad.entrenar(X_train, y_cantidad_train, X_test, y_cantidad_test)
        if nn_scores and nn_scores['r2'] > modelo_cantidad.best_score:
            modelos_categoria['cantidad_nn'] = nn_cantidad
            resultados['cantidad']['red_neuronal'] = nn_scores

    print("Entrenando modelos para MOMENTO...")
    try:
        if len(np.unique(y_momento_train)) > 1:
            clf_momento = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=config.random_state, n_jobs=-1, class_weight='balanced')
            clf_momento.fit(X_train, y_momento_train)
            y_momento_pred = clf_momento.predict(X_test)
            y_momento_prob = clf_momento.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_momento_test, y_momento_pred)
            auc = roc_auc_score(y_momento_test, y_momento_prob)
            modelos_categoria['momento'] = clf_momento
            resultados['momento'] = {
                'accuracy': accuracy,
                'auc': auc,
                'probabilidad_promedio': y_momento_prob.mean()
            }
            logging.info(f"Momento: Accuracy = {accuracy:.3f}, AUC = {auc:.3f}")
        else:
            logging.warning(f"Momento: Solo una clase encontrada")
            resultados['momento'] = {'accuracy': 1.0, 'auc': 0.5, 'probabilidad_promedio': 0.5}
    except Exception as e:
        logging.error(f"Error entrenando modelo de momento: {str(e)}")

    if 'gasto' in modelos_categoria:
        try:
            best_model = modelos_categoria['gasto'].best_model
            if hasattr(best_model, 'feature_importances_'):
                feature_names = feature_cols
                if modelos_categoria['gasto'].feature_selector:
                    selected_indices = modelos_categoria['gasto'].feature_selector.get_support(indices=True)
                    feature_names = [feature_cols[i] for i in selected_indices]
                feature_importance_global[nombre_categoria] = dict(zip(feature_names, best_model.feature_importances_))
        except:
            pass

    if verbose and resultados:
        print(f"\nRESULTADOS para {nombre_categoria}:")
        for target, metrics in resultados.items():
            if target in ['gasto', 'cantidad']:
                print(f"   {target.upper()}: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:,.0f}")
            elif target == 'momento':
                print(f"   {target.upper()}: Accuracy = {metrics['accuracy']:.3f}, AUC = {metrics['auc']:.3f}")

    if not modelos_categoria:
        logging.error(f"No se pudo entrenar ningún modelo para {nombre_categoria}")
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
    Genera predicciones para una categoría usando los modelos entrenados.
    
    Args:
        modelo_info (dict): Información del modelo entrenado.
        meses_adelante (int): Número de meses a predecir.
    
    Returns:
        dict: Predicciones para gasto, cantidad y probabilidad de compra.
    """
    if modelo_info is None:
        return None
    categoria = modelo_info['categoria']
    modelos = modelo_info['modelos']
    features = modelo_info['features']
    df_categoria = datos_temporales.get('CATEGORIA', pd.DataFrame())
    if df_categoria.empty:
        logging.error(f"No hay datos temporales para {categoria}")
        return None
    df_cat = df_categoria[df_categoria['categoria'] == categoria].copy()
    if len(df_cat) == 0:
        logging.error(f"No hay datos para la categoría {categoria}")
        return None
    df_features = crear_features_avanzados(df_cat, categoria)
    if len(df_features) == 0:
        logging.error(f"No se pudieron crear features para {categoria}")
        return None

    # Manejo de rescate
    is_rescued = modelo_info.get('is_rescued', False)
    if is_rescued:
        logging.info(f"Aplicando transformaciones de rescate en predicción para {categoria}")
        power_transformers = modelo_info.get('power_transformers', {})
        for col, pt in power_transformers.items():
            if col in df_features.columns:
                try:
                    valores = df_features[col].values.reshape(-1, 1)
                    df_features[f'{col}_power'] = pt.transform(valores).flatten()
                except Exception as e:
                    logging.error(f"Error aplicando power transform en predict para {col}: {str(e)}")
        df_features = crear_features_estacionales_fuertes(df_features)
        top_features = modelo_info.get('top_features', [])
        if top_features:
            df_features = crear_features_interaccion(df_features, top_features)

    available_features = [f for f in features if f in df_features.columns]
    if len(available_features) < len(features):
        logging.warning(f"Features faltantes en predicción para {categoria}: {set(features) - set(available_features)}")
        for f in features:
            if f not in df_features.columns:
                df_features[f] = 0
    ultimo_punto = df_features.iloc[-1:][features].fillna(0)

    pca = modelo_info.get('pca', None)
    if pca:
        try:
            ultimo_punto = pca.transform(ultimo_punto)
        except Exception as e:
            logging.error(f"Error aplicando PCA en predicción: {str(e)}")
            return None

    predicciones = {}
    
    # Predicciones para gasto
    if 'gasto' in modelos:
        pred_gasto = modelos['gasto'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gasto_nn' in modelos:
        pred_gasto = modelos['gasto_nn'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gasto_intermitente' in modelos:
        clf, reg = modelos['gasto_intermitente']
        prob_compra = clf.predict_proba(ultimo_punto)[:, 1][0]
        if reg:
            monto_pred = reg.predict(ultimo_punto)[0]
            pred_gasto = prob_compra * monto_pred
        else:
            pred_gasto = prob_compra * df_features['gasto_total'].mean()
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gasto_stack' in modelos:
        pred_gasto = modelos['gasto_stack'].predict(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'prophet_gasto_total' in modelos:
        model = modelos['prophet_gasto_total']
        future = model.make_future_dataframe(periods=meses_adelante, freq='MS')
        future['mes_sin'] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
        future['mes_cos'] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
        forecast = model.predict(future)
        pred_gasto = forecast['yhat'].iloc[-1]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
        predicciones['gasto_lower'] = forecast['yhat_lower'].iloc[-1]
        predicciones['gasto_upper'] = forecast['yhat_upper'].iloc[-1]
    elif 'arima_gasto_total' in modelos:
        pred_gasto = modelos['arima_gasto_total'].predict(n_periods=meses_adelante)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gp_gasto' in modelos:
        X_pred = np.array([[len(df_features) + i] for i in range(1, meses_adelante + 1)])
        pred_gasto = modelos['gp_gasto'].predict(X_pred)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'transfer_gasto' in modelos:
        pred_gasto = modelos['transfer_gasto'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'promedio_gasto_total' in modelos:
        pred_gasto = modelos['promedio_gasto_total'](None)
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    else:
        predicciones['gasto_predicho'] = 0
        predicciones['gasto_lower'] = 0
        predicciones['gasto_upper'] = 0

    # Predicciones para cantidad
    if 'cantidad' in modelos:
        pred_cantidad = modelos['cantidad'].predecir(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'cantidad_nn' in modelos:
        pred_cantidad = modelos['cantidad_nn'].predecir(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'cantidad_intermitente' in modelos:
        clf, reg = modelos['cantidad_intermitente']
        prob_compra = clf.predict_proba(ultimo_punto)[:, 1][0]
        if reg:
            monto_pred = reg.predict(ultimo_punto)[0]
            pred_cantidad = prob_compra * monto_pred
        else:
            pred_cantidad = prob_compra * df_features['num_transacciones'].mean()
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'cantidad_stack' in modelos:
        pred_cantidad = modelos['cantidad_stack'].predict(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'prophet_num_transacciones' in modelos:
        model = modelos['prophet_num_transacciones']
        future = model.make_future_dataframe(periods=meses_adelante, freq='MS')
        future['mes_sin'] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
        future['mes_cos'] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
        forecast = model.predict(future)
        pred_cantidad = forecast['yhat'].iloc[-1]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
        predicciones['cantidad_lower'] = forecast['yhat_lower'].iloc[-1]
        predicciones['cantidad_upper'] = forecast['yhat_upper'].iloc[-1]
    elif 'arima_num_transacciones' in modelos:
        pred_cantidad = modelos['arima_num_transacciones'].predict(n_periods=meses_adelante)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    elif 'promedio_num_transacciones' in modelos:
        pred_cantidad = modelos['promedio_num_transacciones'](None)
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    else:
        predicciones['cantidad_predicha'] = 0
        predicciones['cantidad_lower'] = 0
        predicciones['cantidad_upper'] = 0

    # Predicciones para momento
    if 'momento' in modelos:
        prob_compra = modelos['momento'].predict_proba(ultimo_punto)[:, 1][0]
        predicciones['probabilidad_compra'] = prob_compra
    elif 'clf_momento' in modelos:
        prob_compra = modelos['clf_momento'].predict_proba(ultimo_punto)[:, 1][0]
        predicciones['probabilidad_compra'] = prob_compra
    else:
        predicciones['probabilidad_compra'] = 0.5

    # Confianza
    predicciones['confianza_gasto'] = modelo_info['resultados'].get('gasto_total', {}).get('r2', 0)
    predicciones['confianza_cantidad'] = modelo_info['resultados'].get('num_transacciones', {}).get('r2', 0)

    predicciones['categoria'] = categoria
    return predicciones

# Funciones para series temporales
def crear_series_temporales(df, nivel='CATEGORIA'):
    logging.info(f"Creando series temporales por {nivel}")
    agg_dict = {
        'TOTALPESOS': ['sum', 'mean', 'std', 'count'],
        'SOLICITUD': 'nunique',
        'CENTROCOSTO': 'nunique',
        'SOLICITANTE': 'nunique'
    }
    df_agregado = df.groupby([nivel, 'año_mes']).agg(agg_dict).reset_index()
    df_agregado.columns = [nivel.lower(), 'año_mes', 'gasto_total', 'gasto_promedio', 'gasto_std', 'num_transacciones', 'ordenes_unicas', 'centros_costo', 'solicitantes_unicos']
    df_agregado['fecha'] = df_agregado['año_mes'].dt.to_timestamp()
    df_agregado = df_agregado.sort_values([nivel.lower(), 'fecha'])
    df_completo = []
    categorias_unicas = df_agregado[nivel.lower()].unique()
    logging.info(f"Procesando {len(categorias_unicas)} {nivel.lower()}s")
    for i, categoria in enumerate(categorias_unicas):
        if i % 10 == 0 and i > 0:
            logging.info(f"Progreso: {i}/{len(categorias_unicas)}")
        df_cat = df_agregado[df_agregado[nivel.lower()] == categoria].copy()
        fecha_min = df_cat['fecha'].min()
        fecha_max = df_cat['fecha'].max()
        fechas_completas = pd.date_range(fecha_min, fecha_max, freq='MS')
        df_cat = df_cat.set_index('fecha').reindex(fechas_completas)
        df_cat[nivel.lower()] = categoria
        df_cat = df_cat.fillna(0)
        df_cat['fecha'] = df_cat.index
        df_completo.append(df_cat.reset_index(drop=True))
    df_final = pd.concat(df_completo, ignore_index=True)
    logging.info(f"Series temporales creadas: {len(df_final):,} puntos, {df_final[nivel.lower()].nunique()} {nivel.lower()}s, {df_final['fecha'].nunique()} períodos")
    global datos_temporales
    datos_temporales[nivel] = df_final
    return df_final

# Funciones de rescate
def crear_features_estacionales_fuertes(df):
    df_seasonal = df.copy()
    if 'mes' in df.columns:
        df_seasonal['mes_sin_2'] = np.sin(4 * np.pi * df['mes'] / 12)
        df_seasonal['mes_cos_2'] = np.cos(4 * np.pi * df['mes'] / 12)
        df_seasonal['mes_sin_3'] = np.sin(6 * np.pi * df['mes'] / 12)
        df_seasonal['mes_cos_3'] = np.cos(6 * np.pi * df['mes'] / 12)
        df_seasonal['es_enero'] = (df['mes'] == 1).astype(int)
        df_seasonal['es_diciembre'] = (df['mes'] == 12).astype(int)
        df_seasonal['es_junio'] = (df['mes'] == 6).astype(int)
        df_seasonal['cuadrante_año'] = pd.cut(df['mes'], bins=[0, 3, 6, 9, 12], labels=[1,2,3,4]).astype(int)
    if 'trimestre' in df.columns:
        for t in range(1, 5):
            df_seasonal[f'trim_{t}'] = (df['trimestre'] == t).astype(int)
    return df_seasonal

def crear_features_interaccion(df, top_features, max_interacciones=10):
    df_interact = df.copy()
    if len(top_features) < 2:
        return df_interact
    available_features = [f for f in top_features if f in df.columns][:5]
    contador = 0
    for i, feat1 in enumerate(available_features):
        for feat2 in available_features[i+1:]:
            if contador >= max_interacciones:
                break
            df_interact[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            denominador = df[feat2].replace(0, 1e-10)
            df_interact[f'{feat1}_div_{feat2}'] = df[feat1] / denominador
            contador += 2
    return df_interact

def reducir_dimensionalidad_adaptativa(X, varianza_objetivo=0.95):
    if X.shape[1] <= 15:
        return X, None
    try:
        pca = PCA(n_components=varianza_objetivo, random_state=42)
        X_reduced = pca.fit_transform(X)
        logging.info(f"PCA: {X.shape[1]} → {X_reduced.shape[1]} features (varianza: {pca.explained_variance_ratio_.sum():.3f})")
        return X_reduced, pca
    except:
        return X, None

def detectar_patron_intermitente(y):
    prop_ceros = (y == 0).sum() / len(y)
    return prop_ceros > 0.3

def modelo_para_intermitentes(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train_binary)
    prob_compra = clf.predict_proba(X_test)[:, 1]
    mask_train = y_train > 0
    if mask_train.sum() > 5:
        X_train_nonzero = X_train[mask_train]
        y_train_nonzero = y_train[mask_train]
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        reg.fit(X_train_nonzero, y_train_nonzero)
        monto_pred = reg.predict(X_test)
        y_pred = prob_compra * monto_pred
    else:
        y_pred = prob_compra * y_train[y_train > 0].mean() if (y_train > 0).any() else np.zeros(len(X_test))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {'r2': r2, 'mae': mae, 'model': (clf, reg if 'reg' in locals() else None)}

def modelo_stacking_robusto(X_train, y_train, X_test, y_test, modelos_base):
    from sklearn.linear_model import Ridge
    estimators = [(name, model) for name, model in modelos_base.items() if model is not None and hasattr(model, 'predict')]
    if len(estimators) < 2:
        return None
    try:
        stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=3)
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return {'r2': r2, 'mae': mae, 'model': stacking}
    except:
        return None

def rescatar_categoria_problematica(df_categoria, nombre_categoria, modelo_anterior=None):
    logging.info(f"MODO RESCATE activado para: {nombre_categoria}")
    print(f"\n MODO RESCATE activado para: {nombre_categoria}")
    print("="*70)

    es_intermitente = detectar_patron_intermitente(df_categoria['gasto_total'])
    logging.info(f"Patrón intermitente: {'SÍ' if es_intermitente else 'NO'}, Datos disponibles: {len(df_categoria)} meses")
    print(f"   Patrón intermitente: {'SÍ' if es_intermitente else 'NO'}")
    print(f"   Datos disponibles: {len(df_categoria)} meses")

    df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    df_features = augmentar_datos_small_data(df_features)

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['fecha', 'año', 'mes', 'trimestre']]
    power_transformers = {}
    for col in numeric_cols:
        if col in df_features.columns:
            try:
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                valores = df_features[col].values.reshape(-1, 1)
                df_features[f'{col}_power'] = pt.fit_transform(valores).flatten()
                power_transformers[col] = pt
            except:
                logging.warning(f"No se pudo aplicar PowerTransformer a {col}")

    df_features = crear_features_estacionales_fuertes(df_features)
    top_features = ['gasto_total_lag_1', 'gasto_total_ma_3', 'num_transacciones_lag_1', 'gasto_promedio_lag_1', 'mes_sin', 'mes_cos']
    df_features = crear_features_interaccion(df_features, top_features)

    exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 'num_transacciones', 'ordenes_unicas', 'centros_costo', 'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    X = df_features[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']

    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]

    X_train_red, pca = reducir_dimensionalidad_adaptativa(X_train)
    X_test_red = pca.transform(X_test) if pca else X_test

    resultados_rescate = {}
    modelos_rescate = {}
    modelos_base = ModeloEnsembleAvanzado('gasto', config).crear_modelos_base()

    print("   Entrenando modelos de rescate...")
    if es_intermitente:
        logging.info("Aplicando modelo para intermitentes")
        gasto_result = modelo_para_intermitentes(X_train_red, y_gasto_train, X_test_red, y_gasto_test)
        cantidad_result = modelo_para_intermitentes(X_train_red, y_cantidad_train, X_test_red, y_cantidad_test)
        if gasto_result:
            resultados_rescate['gasto'] = {'r2': gasto_result['r2'], 'mae': gasto_result['mae']}
            modelos_rescate['gasto_intermitente'] = gasto_result['model']
        if cantidad_result:
            resultados_rescate['cantidad'] = {'r2': cantidad_result['r2'], 'mae': cantidad_result['mae']}
            modelos_rescate['cantidad_intermitente'] = cantidad_result['model']
    else:
        for name, model in modelos_base.items():
            try:
                model.fit(X_train_red, y_gasto_train)
                y_pred = model.predict(X_test_red)
                r2 = r2_score(y_gasto_test, y_pred)
                mae = mean_absolute_error(y_gasto_test, y_pred)
                modelos_base[name] = model
                logging.info(f"{name} (gasto): R² = {r2:.3f}, MAE = {mae:,.0f}")
            except Exception as e:
                logging.error(f"Error entrenando {name} para gasto: {str(e)}")
                modelos_base[name] = None
        stack_result = modelo_stacking_robusto(X_train_red, y_gasto_train, X_test_red, y_gasto_test, modelos_base)
        if stack_result:
            resultados_rescate['gasto'] = {'r2': stack_result['r2'], 'mae': stack_result['mae']}
            modelos_rescate['gasto_stack'] = stack_result['model']

        for name, model in modelos_base.items():
            try:
                model.fit(X_train_red, y_cantidad_train)
                y_pred = model.predict(X_test_red)
                r2 = r2_score(y_cantidad_test, y_pred)
                mae = mean_absolute_error(y_cantidad_test, y_pred)
                modelos_base[name] = model
                logging.info(f"{name} (cantidad): R² = {r2:.3f}, MAE = {mae:,.0f}")
            except Exception as e:
                logging.error(f"Error entrenando {name} para cantidad: {str(e)}")
                modelos_base[name] = None
        stack_result = modelo_stacking_robusto(X_train_red, y_cantidad_train, X_test_red, y_cantidad_test, modelos_base)
        if stack_result:
            resultados_rescate['cantidad'] = {'r2': stack_result['r2'], 'mae': stack_result['mae']}
            modelos_rescate['cantidad_stack'] = stack_result['model']

    if not resultados_rescate:
        logging.warning("Rescate no logró mejoras significativas")
        print("   ⚠️  Rescate no logró mejoras significativas")
        return None

    print(f"\n RESCATE COMPLETADO:")
    if 'gasto' in resultados_rescate:
        print(f"   GASTO: R² = {resultados_rescate['gasto']['r2']:.3f}, MAE = {resultados_rescate['gasto']['mae']:,.0f}")
    if 'cantidad' in resultados_rescate:
        print(f"   CANTIDAD: R² = {resultados_rescate['cantidad']['r2']:.3f}, MAE = {resultados_rescate['cantidad']['mae']:,.0f}")

    return {
        'categoria': nombre_categoria,
        'modelos': modelos_rescate,
        'resultados': resultados_rescate,
        'features': feature_cols,
        'pca': pca,
        'is_rescued': True,
        'power_transformers': power_transformers,
        'top_features': top_features,
        'datos_usados': len(df_features)
    }

def entrenar_categoria_avanzada_con_rescate(df_categoria, nombre_categoria, verbose=True):
    logging.info(f"Iniciando entrenamiento avanzado con rescate para {nombre_categoria} ({len(df_categoria)} meses)")
    if len(df_categoria) < 6:
        logging.warning(f"{nombre_categoria} tiene solo {len(df_categoria)} meses, intentando modo small data")
        resultado_small = entrenar_modo_small_data(df_categoria, nombre_categoria)
        if resultado_small is not None:
            if verbose:
                print(f" {nombre_categoria}: OK, entrenado en modo small data")
            logging.info(f"{nombre_categoria}: Entrenado en modo small data con éxito")
            return resultado_small
        else:
            logging.error(f"{nombre_categoria}: Falló el modo small data")
            if verbose:
                print(f" ⚠️  {nombre_categoria}: Falló el modo small data")
            return None

    try:
        resultado_normal = entrenar_categoria_avanzada(df_categoria, nombre_categoria, verbose=False)
    except Exception as e:
        logging.error(f"Error en entrenamiento normal para {nombre_categoria}: {str(e)}")
        resultado_normal = None

    necesita_rescate = False
    razon = ""
    if resultado_normal is None:
        necesita_rescate = True
        razon = "entrenamiento normal falló"
    elif nombre_categoria in CATEGORIAS_PROBLEMATICAS:
        necesita_rescate = True
        razon = "categoría en lista problemática"
    else:
        r2_gasto = resultado_normal.get('resultados', {}).get('gasto', {}).get('r2', -999)
        r2_cantidad = resultado_normal.get('resultados', {}).get('cantidad', {}).get('r2', -999)
        if r2_gasto < UMBRAL_GASTO_MALO or r2_cantidad < UMBRAL_CANTIDAD_MALO:
            necesita_rescate = True
            razon = f"R² bajo (gasto={r2_gasto:.3f}, cantidad={r2_cantidad:.3f})"

    if necesita_rescate:
        if verbose:
            print(f"\n ⚠️  {nombre_categoria} necesita rescate: {razon}")
        logging.info(f"{nombre_categoria} necesita rescate: {razon}")
        try:
            resultado_rescate = rescatar_categoria_problematica(df_categoria, nombre_categoria, resultado_normal)
            if resultado_rescate is not None:
                logging.info(f"Rescate exitoso para {nombre_categoria}")
                if verbose:
                    print(f" {nombre_categoria}: Rescate exitoso")
                return resultado_rescate
            elif resultado_normal is not None:
                logging.warning(f"Rescate no mejoró para {nombre_categoria}, usando resultado normal")
                if verbose:
                    print(f"   Usando resultado normal (rescate no mejoró)")
                return resultado_normal
            else:
                logging.error(f"Todo falló para {nombre_categoria}")
                if verbose:
                    print(f" ⚠️  {nombre_categoria}: Todo falló")
                return None
        except Exception as e:
            logging.error(f"Error en modo rescate para {nombre_categoria}: {str(e)}")
            if resultado_normal is not None:
                logging.warning(f"Rescate no mejoró para {nombre_categoria}, usando resultado normal")
                if verbose:
                    print(f"   Usando resultado normal (rescate no mejoró)")
                return resultado_normal
            return None
    else:
        logging.info(f"{nombre_categoria}: OK, no necesita rescate")
        if verbose:
            print(f" {nombre_categoria}: OK, no necesita rescate")
        return resultado_normal

# Funciones principales
def cargar_y_procesar_datos_avanzado(ruta_csv):
    print(" SISTEMA AVANZADO DE PREDICCIÓN - GRUPO SALINAS")
    print("="*60)
    print(" Cargando y procesando datos...")
    df = pd.read_csv(ruta_csv)
    logging.info(f"Datos originales: {len(df):,} registros, {len(df.columns)} columnas")
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
    df = df[df['TOTALPESOS'] > 0]
    df = detectar_y_limpiar_outliers(df, ['TOTALPESOS'], metodo='zscore', threshold=3.0)
    df['CATEGORIA'] = df['CATEGORIA'].fillna('Sin Categoría')
    categoria_mapping = {
        'Construccion': 'Construcción',
        'Tecnologia': 'Tecnología',
        'Produccion': 'Producción',
        'Servicios ': 'Servicios',
        'Insumos ': 'Insumos',
        'Ensamblika ': 'Ensamblika'
    }
    df['CATEGORIA'] = df['CATEGORIA'].replace(categoria_mapping)
    categoria_counts = df['CATEGORIA'].value_counts()
    categorias_validas = categoria_counts[categoria_counts >= 100].index
    df = df[df['CATEGORIA'].isin(categorias_validas)]
    df['año'] = df['FECHAPEDIDO'].dt.year
    df['mes'] = df['FECHAPEDIDO'].dt.month
    df['año_mes'] = df['FECHAPEDIDO'].dt.to_period('M')
    df['trimestre'] = df['FECHAPEDIDO'].dt.quarter
    df['FAMILIA'] = df['FAMILIA'].fillna('Sin Familia')
    df['CLASE'] = df['CLASE'].fillna('Sin Clase')
    df['CENTROCOSTO'] = df['CENTROCOSTO'].fillna('Sin Centro')
    df['SOLICITANTE'] = df['SOLICITANTE'].fillna('Sin Solicitante')
    logging.info(f"Datos procesados: {len(df):,} registros")
    logging.info(f"Período: {df['FECHAPEDIDO'].min().strftime('%Y-%m-%d')} a {df['FECHAPEDIDO'].max().strftime('%Y-%m-%d')}")
    logging.info(f"Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
    logging.info(f"Categorías válidas: {df['CATEGORIA'].nunique()}")
    return df

def ejecutar_entrenamiento_avanzado_con_rescate(df, nivel='CATEGORIA'):
    print(f"\n INICIANDO ENTRENAMIENTO AVANZADO CON RESCATE AUTOMÁTICO")
    print("="*80)
    df_temporal = crear_series_temporales(df, nivel)
    categorias = df_temporal[nivel.lower()].unique()
    logging.info(f"Categorías a procesar: {len(categorias)}")
    logging.info(f"Categorías en lista problemática: {len([c for c in categorias if c in CATEGORIAS_PROBLEMATICAS])}")
    modelos_exitosos = {}
    resultados_globales = []
    for i, categoria in enumerate(categorias):
        print(f"\n Progreso: {i+1}/{len(categorias)} - {categoria}")
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        df_cat = df_cat.sort_values('fecha')
        resultado = entrenar_categoria_avanzada_con_rescate(df_cat, categoria)
        if resultado is not None:
            modelos_exitosos[categoria] = resultado
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
    global modelos_entrenados
    modelos_entrenados = modelos_exitosos
    return modelos_exitosos, resultados_globales

def generar_predicciones_avanzadas(modelos, meses_adelante=1):
    print(f"\n GENERANDO PREDICCIONES PARA {meses_adelante} MES(ES) ADELANTE")
    print("="*60)
    predicciones = []
    for categoria, modelo_info in modelos.items():
        pred = predecir_categoria_avanzada(modelo_info, meses_adelante)
        if pred is not None:
            predicciones.append(pred)
    if not predicciones:
        print(" No se pudieron generar predicciones")
        return pd.DataFrame()
    df_predicciones = pd.DataFrame(predicciones)
    total_gasto = df_predicciones['gasto_predicho'].sum()
    total_cantidad = df_predicciones['cantidad_predicha'].sum()
    confianza_promedio_gasto = df_predicciones['confianza_gasto'].mean()
    confianza_promedio_cantidad = df_predicciones['confianza_cantidad'].mean()
    prob_promedio_compra = df_predicciones['probabilidad_compra'].mean()
    print(f" Gasto total predicho: ${total_gasto:,.2f}")
    print(f" Cantidad total predicha: {total_cantidad:,} transacciones")
    print(f" Confianza promedio (Gasto): {confianza_promedio_gasto:.3f}")
    print(f" Confianza promedio (Cantidad): {confianza_promedio_cantidad:.3f}")
    print(f" Probabilidad promedio de compra: {prob_promedio_compra:.3f}")
    return df_predicciones.sort_values('gasto_predicho', ascending=False)

def mostrar_resumen_avanzado(resultados_globales):
    print(f"\n RESUMEN FINAL DEL ENTRENAMIENTO AVANZADO")
    print("="*60)
    if not resultados_globales:
        print(" No hay resultados para mostrar")
        return
    df_resultados = pd.DataFrame(resultados_globales)
    total_categorias = len(df_resultados)
    print(f" Categorías procesadas exitosamente: {total_categorias}")
    if 'r2_gasto' in df_resultados.columns:
        r2_gasto_mean = df_resultados['r2_gasto'].mean()
        r2_gasto_median = df_resultados['r2_gasto'].median()
        modelos_buenos_gasto = len(df_resultados[df_resultados['r2_gasto'] > 0.5])
        modelos_regulares_gasto = len(df_resultados[(df_resultados['r2_gasto'] > 0.3) & (df_resultados['r2_gasto'] <= 0.5)])
        print(f"\n CALIDAD MODELOS DE GASTO:")
        print(f"   R² promedio: {r2_gasto_mean:.3f}")
        print(f"   R² mediana: {r2_gasto_median:.3f}")
        print(f"   Modelos buenos (R² > 0.5): {modelos_buenos_gasto}/{total_categorias} ({100*modelos_buenos_gasto/total_categorias:.1f}%)")
        print(f"   Modelos regulares (R² 0.3-0.5): {modelos_regulares_gasto}/{total_categorias} ({100*modelos_regulares_gasto/total_categorias:.1f}%)")
    if 'r2_cantidad' in df_resultados.columns:
        r2_cantidad_mean = df_resultados['r2_cantidad'].mean()
        r2_cantidad_median = df_resultados['r2_cantidad'].median()
        modelos_buenos_cantidad = len(df_resultados[df_resultados['r2_cantidad'] > 0.5])
        print(f"\n CALIDAD MODELOS DE CANTIDAD:")
        print(f"   R² promedio: {r2_cantidad_mean:.3f}")
        print(f"   R² mediana: {r2_cantidad_median:.3f}")
        print(f"   Modelos buenos (R² > 0.5): {modelos_buenos_cantidad}/{total_categorias} ({100*modelos_buenos_cantidad/total_categorias:.1f}%)")
    if 'r2_gasto' in df_resultados.columns:
        print(f"\n TOP 5 MEJORES MODELOS (por R² Gasto):")
        top_5_gasto = df_resultados.nlargest(5, 'r2_gasto')[['categoria', 'r2_gasto', 'mae_gasto']]
        for _, row in top_5_gasto.iterrows():
            print(f"   {row['categoria']}: R² = {row['r2_gasto']:.3f}, MAE = ${row['mae_gasto']:,.0f}")
    if feature_importance_global:
        print(f"\n FEATURES MÁS IMPORTANTES GLOBALMENTE:")
        all_features = {}
        for cat, features in feature_importance_global.items():
            for feature, importance in features.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        avg_importance = {feature: np.mean(importances) for feature, importances in all_features.items()}
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.3f}")

def guardar_modelos_avanzados(modelos, directorio="modelos_avanzados"):
    logging.info("Guardando modelos avanzados")
    print(f"\n GUARDANDO MODELOS AVANZADOS...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_final = f"{directorio}_{timestamp}"
    os.makedirs(dir_final, exist_ok=True)
    modelos_guardados = 0
    for categoria, modelo_info in modelos.items():
        try:
            cat_dir = os.path.join(dir_final, categoria.replace("/", "_").replace(" ", "_"))
            os.makedirs(cat_dir, exist_ok=True)
            for tipo_modelo, modelo_obj in modelo_info['modelos'].items():
                archivo_modelo = os.path.join(cat_dir, f"{tipo_modelo}.joblib")
                joblib.dump(modelo_obj, archivo_modelo)
                modelos_guardados += 1
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
            logging.error(f"Error guardando {categoria}: {str(e)}")
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
    print(f" Modelos guardados exitosamente:")
    print(f"    Directorio: {dir_final}")
    print(f"    Modelos guardados: {modelos_guardados}")
    print(f"    Categorías: {len(modelos)}")

def ejecutar_sistema_completo_con_rescate(ruta_csv, meses_prediccion=1):
    print("🚀 INICIANDO SISTEMA AVANZADO DE PREDICCIÓN DE COMPRAS CON RESCATE")
    print("="*80)
    df = cargar_y_procesar_datos_avanzado(ruta_csv)
    modelos, resultados = ejecutar_entrenamiento_avanzado_con_rescate(df)
    mostrar_resumen_avanzado(resultados)
    if modelos:
        predicciones = generar_predicciones_avanzadas(modelos, meses_prediccion)
        print(f"\n📊 TOP 10 PREDICCIONES POR GASTO:")
        print("-" * 80)
        if not predicciones.empty:
            print(predicciones.head(10))
    if modelos:
        guardar_modelos_avanzados(modelos)
    print(f"\n✅ SISTEMA COMPLETADO EXITOSAMENTE!")
    print("="*80)
    return modelos, predicciones, resultados
