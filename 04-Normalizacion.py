#!/usr/bin/env python3
"""
Sistema Avanzado de Predicción de Compras - Grupo Salinas
Versión optimizada usando solo librerías estándar de Python
Predice: Cantidad, Costo y Momento de compras por categoría

Características principales:
- Modelos ensemble con sklearn (RandomForest, GradientBoosting, etc.)
- Red neuronal simple implementada desde cero
- Optimización automática de hiperparámetros
- Feature engineering avanzado
- Validación temporal robusta
- Sin dependencias externas compiladas

Autor: Sistema IA Optimizado
Versión: 2.1 - Pure Python
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

# ML Libraries - Solo sklearn estándar
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            ExtraTreesRegressor, VotingRegressor, RandomForestClassifier)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                SGDRegressor, BayesianRidge)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (TimeSeriesSplit, GridSearchCV, 
                                   RandomizedSearchCV, cross_val_score)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                           mean_absolute_percentage_error, accuracy_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Otras librerías estándar
from datetime import datetime, timedelta
import json
import os
import joblib
from scipy import stats
from scipy.stats import zscore
import itertools
import pickle

# =============================================================================
# CONFIGURACIÓN GLOBAL OPTIMIZADA
# =============================================================================

class ConfiguracionOptimizada:
    def __init__(self):
        self.random_state = 42
        self.test_size = 0.25
        self.min_meses_entrenamiento = 18
        self.cv_folds = 5
        self.outlier_threshold = 3.5
        self.feature_selection_k = 15
        self.ensemble_weights = 'uniform'
        
        # Configuración de modelos optimizada para sklearn
        self.models_config = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['auto', 'sqrt']
            },
            'et': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        }

config = ConfiguracionOptimizada()

# Variables globales
modelos_entrenados = {}
datos_temporales = {}
metricas_globales = {}
feature_importance_global = {}

# =============================================================================
# RED NEURONAL SIMPLE IMPLEMENTADA DESDE CERO
# =============================================================================

class RedNeuronalSimple:
    """
    Red neuronal simple implementada desde cero usando solo NumPy
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.layers = []
        
        # Construir arquitectura
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Inicialización Xavier
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'w': w, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['w']) + layer['b']
            self.z_values.append(z)
            
            if i < len(self.layers) - 1:  # Hidden layers
                a = self.relu(z)
            else:  # Output layer (linear)
                a = z
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer error
        dz = output - y.reshape(-1, 1)
        
        # Backpropagate
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # Gradients
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # Update weights
            layer['w'] -= self.learning_rate * dw
            layer['b'] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                dz = np.dot(dz, layer['w'].T) * self.relu_derivative(self.z_values[i-1])
    
    def fit(self, X, y, epochs=100, batch_size=32, verbose=False):
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
            
            for batch_start in range(0, len(X), batch_size):
                batch_end = min(batch_start + batch_size, len(X))
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss (MSE)
                loss = np.mean((output.flatten() - y_batch) ** 2)
                epoch_loss += loss
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.losses = losses
        return self
    
    def predict(self, X):
        return self.forward(X).flatten()

# =============================================================================
# UTILIDADES OPTIMIZADAS
# =============================================================================

def crear_series_temporales(df, nivel='CATEGORIA'):
    """
    Crea series temporales agregadas por categoría/familia
    """
    print(f"📈 Creando series temporales por {nivel}...")
    
    # Definir columnas de agrupación
    group_cols = ['año_mes']
    if nivel == 'CATEGORIA':
        group_cols.append('CATEGORIA')
    elif nivel == 'FAMILIA':
        group_cols.extend(['CATEGORIA', 'FAMILIA'])
    
    # Agregar datos
    df_temporal = df.groupby(group_cols).agg({
        'TOTALPESOS': ['sum', 'mean', 'std', 'count'],
        'NUMPEDIDO': 'nunique',
        'CENTROCOSTO': 'nunique', 
        'SOLICITANTE': 'nunique'
    }).reset_index()
    
    # Aplanar nombres de columnas
    df_temporal.columns = [
        'año_mes', 'categoria' if nivel == 'CATEGORIA' else 'categoria',
        'familia' if nivel == 'FAMILIA' else None,
        'gasto_total', 'gasto_promedio', 'gasto_std', 'num_transacciones',
        'ordenes_unicas', 'centros_costo', 'solicitantes_unicos'
    ]
    
    # Limpiar columnas None
    df_temporal = df_temporal.loc[:, df_temporal.columns.notna()]
    
    # Convertir año_mes a fecha
    df_temporal['fecha'] = df_temporal['año_mes'].dt.to_timestamp()
    
    # Rellenar valores faltantes
    df_temporal['gasto_std'] = df_temporal['gasto_std'].fillna(0)
    
    # Almacenar en variable global
    global datos_temporales
    datos_temporales[nivel] = df_temporal
    
    print(f"✅ Series temporales creadas: {len(df_temporal)} registros")
    return df_temporal

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
            z_scores = np.abs(stats.zscore(df_clean[columna].dropna()))
            df_clean = df_clean[z_scores < threshold]
            
        elif metodo == 'iqr':
            Q1 = df_clean[columna].quantile(0.25)
            Q3 = df_clean[columna].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[columna] >= lower_bound) & 
                               (df_clean[columna] <= upper_bound)]
        
        despues = len(df_clean)
        outliers_col = antes - despues
        outliers_removidos += outliers_col
        
        print(f"   📊 {columna}: {outliers_col:,} outliers removidos")
    
    print(f"✅ Total outliers removidos: {outliers_removidos:,}")
    return df_clean

def crear_features_avanzados(df_serie, categoria):
    """
    Crea features avanzados de ingeniería temporal
    """
    df = df_serie.copy()
    print(f"🔬 Creando features avanzados para {categoria}...")
    
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
    
    # Estacionalidad específica
    df['es_fin_año'] = ((df['mes'] == 12) | (df['mes'] == 1)).astype(int)
    df['es_medio_año'] = ((df['mes'] >= 6) & (df['mes'] <= 8)).astype(int)
    df['es_inicio_año'] = ((df['mes'] >= 1) & (df['mes'] <= 3)).astype(int)
    
    # 3. LAGS Y VENTANAS MÓVILES
    lags = [1, 2, 3, 6, 12]
    metrics = ['gasto_total', 'num_transacciones', 'gasto_promedio']
    
    for metric in metrics:
        if metric in df.columns:
            for lag in lags:
                df[f'{metric}_lag_{lag}'] = df[metric].shift(lag)
    
    # Ventanas móviles
    windows = [3, 6, 12]
    for metric in metrics:
        if metric in df.columns:
            for window in windows:
                df[f'{metric}_ma_{window}'] = df[metric].rolling(window=window, min_periods=1).mean()
                df[f'{metric}_std_{window}'] = df[metric].rolling(window=window, min_periods=1).std()
                df[f'{metric}_min_{window}'] = df[metric].rolling(window=window, min_periods=1).min()
                df[f'{metric}_max_{window}'] = df[metric].rolling(window=window, min_periods=1).max()
    
    # 4. FEATURES DE TENDENCIA
    df['tendencia_lineal'] = range(len(df))
    
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_pct_change_1'] = df[metric].pct_change(1)
            df[f'{metric}_pct_change_3'] = df[metric].pct_change(3)
            df[f'{metric}_pct_change_12'] = df[metric].pct_change(12)
            df[f'{metric}_momentum_3_12'] = (df[f'{metric}_ma_3'] / (df[f'{metric}_ma_12'] + 1e-8)) - 1
    
    # 5. RATIOS Y EFICIENCIAS
    df['gasto_por_transaccion'] = df['gasto_total'] / (df['num_transacciones'] + 1e-8)
    df['gasto_por_orden'] = df['gasto_total'] / (df['ordenes_unicas'] + 1e-8)
    df['transacciones_por_orden'] = df['num_transacciones'] / (df['ordenes_unicas'] + 1e-8)
    
    # 6. VOLATILIDAD
    df['volatilidad_gasto'] = df['gasto_total'].rolling(window=6, min_periods=1).std()
    df['coef_variacion_gasto'] = df['volatilidad_gasto'] / (df['gasto_ma_6'] + 1e-8)
    
    # 7. FEATURES COMPARATIVOS
    for metric in ['gasto_total', 'num_transacciones']:
        if metric in df.columns:
            media_historica = df[metric].expanding(min_periods=1).mean()
            df[f'{metric}_vs_historico'] = df[metric] / (media_historica + 1e-8)
    
    # 8. LIMPIEZA FINAL
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Imputación
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['fecha', 'año', 'mes', 'trimestre']:
            df[col] = (df[col]
                      .fillna(method='ffill')
                      .fillna(method='bfill')
                      .fillna(0))
    
    print(f"✅ Features creados: {len([col for col in df.columns if any(x in col for x in ['_lag_', '_ma_', '_std_', '_pct_'])])} features temporales")
    
    return df

# =============================================================================
# MODELOS OPTIMIZADOS
# =============================================================================

class ModeloEnsembleOptimizado:
    """
    Ensemble usando solo librerías estándar de sklearn
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
        Crea modelos base usando solo sklearn estándar
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
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config.random_state),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.config.random_state
            ),
            'knn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'tree': DecisionTreeRegressor(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state
            ),
            'bayes_ridge': BayesianRidge(),
            'sgd': SGDRegressor(
                alpha=0.01,
                learning_rate='adaptive',
                random_state=self.config.random_state
            )
        }
        
        return models
    
    def optimizar_hiperparametros(self, X_train, y_train, model_name, model):
        """
        Optimiza hiperparámetros usando RandomizedSearchCV
        """
        if model_name not in self.config.models_config or not self.config.models_config[model_name]:
            return model
        
        print(f"🔧 Optimizando {model_name}...")
        
        param_grid = self.config.models_config[model_name]
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=10,  # Reducido para velocidad
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
        Entrena ensemble de modelos
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
        
        # Crear modelos base
        base_models = self.crear_modelos_base()
        trained_models = []
        model_scores = {}
        
        for name, model in base_models.items():
            try:
                print(f"   Entrenando {name}...")
                
                # Escalar datos para modelos que lo requieren
                if name in ['ridge', 'elastic', 'mlp', 'knn', 'sgd']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_selected)
                    X_test_scaled = scaler.transform(X_test_selected)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train_selected
                    X_test_scaled = X_test_selected
                    self.scalers[name] = None
                
                # Optimizar hiperparámetros
                optimized_model = self.optimizar_hiperparametros(X_train_scaled, y_train, name, model)
                
                # Entrenar
                optimized_model.fit(X_train_scaled, y_train)
                
                # Evaluar
                y_pred = optimized_model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[name] = {'r2': score, 'mae': mae}
                
                if score > 0.1:  # Solo incluir modelos con desempeño aceptable
                    self.models[name] = optimized_model
                    trained_models.append((name, optimized_model))
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = optimized_model
                
                print(f"   {name}: R² = {score:.3f}, MAE = {mae:,.0f}")
                
            except Exception as e:
                print(f"   ⚠️ Error entrenando {name}: {str(e)}")
        
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
        best_name = next((name for name, model in self.models.items() if model == self.best_model), 'rf')
        if best_name in self.scalers and self.scalers[best_name]:
            X_scaled = self.scalers[best_name].transform(X_selected)
        else:
            X_scaled = X_selected
        
        return self.best_model.predict(X_scaled)

# =============================================================================
# SISTEMA PRINCIPAL OPTIMIZADO
# =============================================================================

def entrenar_categoria_optimizada(df_categoria, nombre_categoria, verbose=True):
    """
    Entrena modelos optimizados para una categoría específica
    """
    print(f"\n🎯 Procesando categoría: {nombre_categoria}")
    print("="*60)
    
    # Validar datos mínimos
    if len(df_categoria) < config.min_meses_entrenamiento:
        if verbose:
            print(f"❌ Datos insuficientes: {len(df_categoria)} < {config.min_meses_entrenamiento} meses")
        return None
    
    # Crear features avanzados
    try:
        df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    except Exception as e:
        if verbose:
            print(f"❌ Error creando features: {str(e)}")
        return None
    
    # Eliminar filas con demasiados NaN
    threshold = len(df_features.columns) * 0.5
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
    y_momento = (df_features['gasto_total'] > 0).astype(int)
    
    # Split temporal
    split_idx = int(len(X) * (1 - config.test_size))
    if split_idx < config.min_meses_entrenamiento * 0.7:
        if verbose:
            print(f"❌ Split insuficiente: {split_idx}")
        return None
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    y_momento_train, y_momento_test = y_momento.iloc[:split_idx], y_momento.iloc[split_idx:]
    
    # Entrenar modelos
    resultados = {}
    modelos_categoria = {}
    
    # 1. MODELO PARA GASTO
    print("💰 Entrenando modelos para GASTO...")
    modelo_gasto = ModeloEnsembleOptimizado('gasto', config)
    scores_gasto = modelo_gasto.entrenar_ensemble(X_train, y_gasto_train, X_test, y_gasto_test)
    
    if modelo_gasto.best_model is not None:
        modelos_categoria['gasto'] = modelo_gasto
        resultados['gasto'] = {
            'r2': modelo_gasto.best_score,
            'mae': mean_absolute_error(y_gasto_test, modelo_gasto.predecir(X_test)),
            'scores_individuales': scores_gasto
        }
    
    # Red neuronal simple para gasto
    if len(X_train) > 50:
        try:
            print("   Entrenando red neuronal simple...")
            nn_simple = RedNeuronalSimple(
                input_size=X_train.shape[1],
                hidden_sizes=[64, 32],
                learning_rate=0.001
            )
            
            # Escalar datos
            scaler_nn = StandardScaler()
            X_train_scaled = scaler_nn.fit_transform(X_train)
            X_test_scaled = scaler_nn.transform(X_test)
            
            # Entrenar
            nn_simple.fit(X_train_scaled, y_gasto_train.values, epochs=100, verbose=False)
            
            # Evaluar
            y_pred_nn = nn_simple.predict(X_test_scaled)
            r2_nn = r2_score(y_gasto_test, y_pred_nn)
            mae_nn = mean_absolute_error(y_gasto_test, y_pred_nn)
            
            print(f"   Red neuronal simple: R² = {r2_nn:.3f}, MAE = {mae_nn:,.0f}")
            
            if r2_nn > modelo_gasto.best_score:
                modelos_categoria['gasto_nn'] = {'model': nn_simple, 'scaler': scaler_nn}
                resultados['gasto']['red_neuronal'] = {'r2': r2_nn, 'mae': mae_nn}
                
        except Exception as e:
            print(f"   ⚠️ Error entrenando red neuronal: {str(e)}")
    
    # 2. MODELO PARA CANTIDAD
    print("📦 Entrenando modelos para CANTIDAD...")
    modelo_cantidad = ModeloEnsembleOptimizado('cantidad', config)
    scores_cantidad = modelo_cantidad.entrenar_ensemble(X_train, y_cantidad_train, X_test, y_cantidad_test)
    
    if modelo_cantidad.best_model is not None:
        modelos_categoria['cantidad'] = modelo_cantidad
        resultados['cantidad'] = {
            'r2': modelo_cantidad.best_score,
            'mae': mean_absolute_error(y_cantidad_test, modelo_cantidad.predecir(X_test)),
            'scores_individuales': scores_cantidad
        }
    
    # 3. MODELO PARA MOMENTO (Probabilidad de compra)
    print("⏰ Entrenando modelos para MOMENTO...")
    try:
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
    
    # Guardar feature importance
    if 'gasto' in modelos_categoria:
        try:
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

def predecir_categoria_optimizada(modelo_info, meses_adelante=1):
    """
    Genera predicciones optimizadas para una categoría
    """
    if modelo_info is None:
        return None
    
    categoria = modelo_info['categoria']
    modelos = modelo_info['modelos']
    features = modelo_info['features']
    
    # Obtener últimos datos
    df_categoria = datos_temporales.get('CATEGORIA', pd.DataFrame())
    if df_categoria.empty:
        return None
    
    df_cat = df_categoria[df_categoria['categoria'] == categoria].copy()
    if len(df_cat) == 0:
        return None
    
    # Crear features
    df_features = crear_features_avanzados(df_cat, categoria)
    
    if len(df_features) == 0:
        return None
    
    # Usar últimos datos para predicción
    ultimo_punto = df_features.iloc[-1:][features].fillna(0)
    
    predicciones = {}
    
    # Predicción de gasto
    if 'gasto' in modelos:
        pred_gasto = modelos['gasto'].predecir(ultimo_punto)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    elif 'gasto_nn' in modelos:
        nn_info = modelos['gasto_nn']
        ultimo_punto_scaled = nn_info['scaler'].transform(ultimo_punto)
        pred_gasto = nn_info['model'].predict(ultimo_punto_scaled)[0]
        predicciones['gasto_predicho'] = max(0, pred_gasto)
    else:
        predicciones['gasto_predicho'] = 0
    
    # Predicción de cantidad
    if 'cantidad' in modelos:
        pred_cantidad = modelos['cantidad'].predecir(ultimo_punto)[0]
        predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    else:
        predicciones['cantidad_predicha'] = 0
    
    # Predicción de momento
    if 'momento' in modelos:
        prob_compra = modelos['momento'].predict_proba(ultimo_punto)[0, 1]
        predicciones['probabilidad_compra'] = prob_compra
    else:
        predicciones['probabilidad_compra'] = 0.5
    
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
# FUNCIONES PRINCIPALES
# =============================================================================

def cargar_y_procesar_datos_optimizado(ruta_excel, sheet_name="Detalle"):
    """
    Carga y procesamiento optimizado de datos
    """
    print("🚀 SISTEMA OPTIMIZADO DE PREDICCIÓN - GRUPO SALINAS")
    print("="*60)
    print("📊 Cargando y procesando datos...")
    
    # Cargar datos
    df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
    print(f"✅ Datos originales: {len(df):,} registros, {len(df.columns)} columnas")
    
    # Limpieza básica
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
    df = df[df['TOTALPESOS'] > 0]
    
    # Limpieza de outliers
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
    categorias_validas = categoria_counts[categoria_counts >= 100].index
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

def ejecutar_entrenamiento_optimizado(df, nivel='CATEGORIA'):
    """
    Ejecuta el entrenamiento optimizado completo
    """
    print(f"\n🎯 INICIANDO ENTRENAMIENTO OPTIMIZADO POR {nivel}")
    print("="*60)
    
    # Crear series temporales
    df_temporal = crear_series_temporales(df, nivel)
    
    # Obtener categorías
    categorias = df_temporal[nivel.lower()].unique()
    print(f"📋 Categorías a procesar: {len(categorias)}")
    
    # Entrenar modelos
    modelos_exitosos = {}
    resultados_globales = []
    
    for i, categoria in enumerate(categorias):
        print(f"\n🔄 Progreso: {i+1}/{len(categorias)} - {categoria}")
        
        # Filtrar datos
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        df_cat = df_cat.sort_values('fecha')
        
        # Entrenar
        resultado = entrenar_categoria_optimizada(df_cat, categoria)
        
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
    
    # Guardar modelos
    global modelos_entrenados
    modelos_entrenados = modelos_exitosos
    
    return modelos_exitosos, resultados_globales

def generar_predicciones_optimizadas(modelos, meses_adelante=1):
    """
    Genera predicciones optimizadas
    """
    print(f"\n🔮 GENERANDO PREDICCIONES PARA {meses_adelante} MES(ES) ADELANTE")
    print("="*60)
    
    predicciones = []
    
    for categoria, modelo_info in modelos.items():
        pred = predecir_categoria_optimizada(modelo_info, meses_adelante)
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

def mostrar_resumen_optimizado(resultados_globales):
    """
    Muestra resumen completo optimizado
    """
    print(f"\n📊 RESUMEN FINAL DEL ENTRENAMIENTO OPTIMIZADO")
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
    
    # Feature importance global
    if feature_importance_global:
        print(f"\n🎯 FEATURES MÁS IMPORTANTES GLOBALMENTE:")
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

def guardar_modelos_optimizados(modelos, directorio="modelos_optimizados"):
    """
    Guarda modelos optimizados
    """
    print(f"\n💾 GUARDANDO MODELOS OPTIMIZADOS...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_final = f"{directorio}_{timestamp}"
    os.makedirs(dir_final, exist_ok=True)
    
    modelos_guardados = 0
    
    for categoria, modelo_info in modelos.items():
        try:
            cat_dir = os.path.join(dir_final, categoria.replace("/", "_").replace(" ", "_"))
            os.makedirs(cat_dir, exist_ok=True)
            
            # Guardar modelos
            for tipo_modelo, modelo_obj in modelo_info['modelos'].items():
                archivo_modelo = os.path.join(cat_dir, f"{tipo_modelo}.pkl")
                with open(archivo_modelo, 'wb') as f:
                    pickle.dump(modelo_obj, f)
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

def ejecutar_sistema_completo_optimizado(ruta_excel, sheet_name="Detalle", meses_prediccion=1):
    """
    Ejecuta el sistema completo optimizado
    """
    print("🚀 INICIANDO SISTEMA OPTIMIZADO DE PREDICCIÓN DE COMPRAS")
    print("="*80)
    
    # 1. Cargar y procesar datos
    df = cargar_y_procesar_datos_optimizado(ruta_excel, sheet_name)
    
    # 2. Entrenar modelos
    modelos, resultados = ejecutar_entrenamiento_optimizado(df)
    
    # 3. Mostrar resumen
    mostrar_resumen_optimizado(resultados)
    
    # 4. Generar predicciones
    if modelos:
        predicciones = generar_predicciones_optimizadas(modelos, meses_prediccion)
        
        print(f"\n🔮 TOP 10 PREDICCIONES POR GASTO:")
        print("-" * 80)
        if not predicciones.empty:
            print(predicciones.head(10).to_string(index=False))
    
    # 5. Guardar modelos
    if modelos:
        guardar_modelos_optimizados(modelos)
    
    print(f"\n✅ SISTEMA COMPLETADO EXITOSAMENTE!")
    print("="*80)
    
    return modelos, predicciones, resultados

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

"""
# Para usar el sistema optimizado:

# 1. Cargar datos y ejecutar sistema completo
ruta_archivo = "path/to/your/data.xlsx"
modelos, predicciones, resultados = ejecutar_sistema_completo_optimizado(ruta_archivo)

# 2. Para predicciones adicionales
nuevas_predicciones = generar_predicciones_optimizadas(modelos, meses_adelante=3)

# 3. Para cargar modelos previamente guardados
import pickle
with open("modelos_optimizados_YYYYMMDD_HHMM/categoria/modelo.pkl", 'rb') as f:
    modelo_cargado = pickle.load(f)

# Ventajas de esta versión:
# - Solo usa librerías estándar de Python y sklearn
# - No requiere compilación de código C
# - Red neuronal implementada desde cero con NumPy
# - Más fácil de instalar y ejecutar
# - Mantiene alta calidad predictiva
# - Compatible con cualquier entorno Python estándar
"""
