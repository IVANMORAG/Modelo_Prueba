#!/usr/bin/env python3
"""
Sistema Avanzado de Predicción de Compras - Grupo Salinas
Versión con Prophet para small data

Versión: 2.3
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

logging.basicConfig(filename='predicciones.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# ML Libraries (mismo que antes)
# ... (incluye todas las imports anteriores)

# Import Prophet (opcional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logging.info("Prophet disponible para small data")
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet no disponible. Instala con 'pip install prophet' para mejorar small data")

# Configuración global (mismo)
# ...

# Utilidades avanzadas (mismo)
# ...

# Modelos avanzados (mismo)
# ...

# Funciones para small data - ACTUALIZADO CON PROPHET
def entrenar_modo_small_data(df_categoria, nombre_categoria):
    logging.info(f"Modo Small Data para {nombre_categoria} con {len(df_categoria)} meses")
    df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    df_features.set_index('fecha', inplace=True)
    exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 'num_transacciones', 'ordenes_unicas', 'centros_costo', 'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    modelos_small = {}
    resultados_small = {}

    targets = ['gasto_total', 'num_transacciones']
    for target in targets:
        if target in df_features.columns:
            df_prophet = pd.DataFrame({
                'ds': df_features.index,
                'y': df_features[target]
            })
            if PROPHET_AVAILABLE:
                try:
                    # Para small data: Deshabilitar seasonality si <12 meses
                    seasonality = len(df_features) >= 12
                    model = Prophet(
                        yearly_seasonality=seasonality,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05  # Flexible para non-linear
                    )
                    model.fit(df_prophet)
                    modelos_small[f'prophet_{target}'] = model
                    # Evaluar
                    split = int(len(df_prophet) * 0.8)
                    train = df_prophet.iloc[:split]
                    test = df_prophet.iloc[split:]
                    future = model.make_future_dataframe(periods=len(test), freq='MS')
                    forecast = model.predict(future)
                    pred = forecast['yhat'].iloc[-len(test):]
                    r2 = r2_score(test['y'], pred)
                    mae = mean_absolute_error(test['y'], pred)
                    resultados_small[target] = {'r2': r2, 'mae': mae}
                    logging.info(f"Prophet {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                    print(f"   Prophet {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                except Exception as e:
                    logging.error(f"Error Prophet {target}: {str(e)}")
                    # Fallback a ARIMA si Prophet falla
                    model = ARIMA(df_features[target], order=(1,1,1))
                    model_fit = model.fit()
                    modelos_small[f'arima_{target}'] = model_fit
                    pred = model_fit.forecast(len(test['y']))
                    r2 = r2_score(test['y'], pred)
                    mae = mean_absolute_error(test['y'], pred)
                    resultados_small[target] = {'r2': r2, 'mae': mae}
                    logging.info(f"ARIMA fallback {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
                    print(f"   ARIMA fallback {target}: R² = {r2:.3f}, MAE = {mae:,.0f}")
            else:
                # Si no Prophet, fallback a ARIMA
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

    # Gaussian Process (mismo)
    # ...

    # Clasificador Momento (mismo, con fix para una clase)
    # ...

    # Transfer Learning (mismo)
    # ...

    return {
        'categoria': nombre_categoria,
        'modelos': modelos_small,
        'resultados': resultados_small,
        'features': feature_cols,
        'datos_usados': len(df_features)
    }

# Predecir con Prophet en predicción
# En predecir_categoria_avanzada, agregar:
elif f'prophet_gasto_total' in modelos:
    model = modelos['prophet_gasto_total']
    future = model.make_future_dataframe(periods=meses_adelante, freq='MS')
    forecast = model.predict(future)
    pred_gasto = forecast['yhat'].iloc[-1]
    predicciones['gasto_predicho'] = max(0, pred_gasto)
    # Uncertainty (opcional)
    predicciones['gasto_lower'] = forecast['yhat_lower'].iloc[-1]
    predicciones['gasto_upper'] = forecast['yhat_upper'].iloc[-1]
elif f'prophet_num_transacciones' in modelos:
    model = modelos['prophet_num_transacciones']
    future = model.make_future_dataframe(periods=meses_adelante, freq='MS')
    forecast = model.predict(future)
    pred_cantidad = forecast['yhat'].iloc[-1]
    predicciones['cantidad_predicha'] = max(0, int(pred_cantidad))
    predicciones['cantidad_lower'] = forecast['yhat_lower'].iloc[-1]
    predicciones['cantidad_upper'] = forecast['yhat_upper'].iloc[-1]

# Resto del código (mismo, con logging)
# ...

# Ejemplo de uso
if __name__ == "__main__":
    ruta_csv = "/content/drive/MyDrive/DATASET/datos_limpios_categoria.csv"  # Reemplaza con tu ruta
    modelos, predicciones, resultados = ejecutar_sistema_completo_con_rescate(ruta_csv, meses_prediccion=1)
