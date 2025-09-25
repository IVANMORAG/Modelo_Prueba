import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Librerías de Machine Learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # Habilitar HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Utilidades
from datetime import datetime, timedelta
import json
import os

# Variables globales para almacenar resultados
modelos_entrenados = {}
metricas_globales = {'gasto': [], 'cantidad': []}

def cargar_y_limpiar_datos(ruta_excel, sheet_name="Detalle"):
    """
    Carga y limpia los datos, unificando y agrupando categorías.
    """
    print("📂 Cargando y limpiando datos (versión 3.0)...")
    try:
        df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado en la ruta '{ruta_excel}'.")
        return None
    
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
    df = df[df['TOTALPESOS'] > 0]
    
    # MEJORA: Unificar categorías para consolidar datos.
    mapeo_categorias = {
        'Construccion': 'Construcción',
        'Tecnologia': 'Tecnología',
        'Produccion': 'Producción',
        'Impresos y publicidad': 'Impresos Y Publicidad',
        'Gastos de Fin de Año': 'Gastos De Fin De Año',
        'Servicios ': 'Servicios',
        'Ensamblika ': 'Ensamblika',
        'O&M propio': 'O&M Propio',
        'O&M terceros': 'O&M Terceros',
        'EPC propio': 'EPC Propio',
        'EPC tercero': 'EPC Tercero',
        'Obsequios y atenciones': 'Obsequios Y Atenciones',
        'Servicios a la operación': 'Servicios a la Operación',
        'Electromecanico': 'Electromecánico',
        'Gestión y comercialización (G&C)': 'Gestión y Comercialización (G&C)'
    }
    df['CATEGORIA'] = df['CATEGORIA'].replace(mapeo_categorias).fillna('Sin Categoría')
    df['CATEGORIA'] = df['CATEGORIA'].str.strip()

    # MEJORA: Filtrar outliers para evitar valores extremos que sesguen el modelo.
    q_low = df['TOTALPESOS'].quantile(0.01)
    q_high = df['TOTALPESOS'].quantile(0.99)
    df = df[(df['TOTALPESOS'] > q_low) & (df['TOTALPESOS'] < q_high)]
    
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'])
    df['año_mes'] = df['FECHAPEDIDO'].dt.to_period('M')
    
    print(f"✅ Datos limpios: {len(df):,} registros")
    print(f"📅 Período: {df['FECHAPEDIDO'].min().strftime('%Y-%m-%d')} a {df['FECHAPEDIDO'].max().strftime('%Y-%m-%d')}")
    print(f"💰 Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
    
    return df

def crear_series_temporales(df, nivel='CATEGORIA'):
    """
    Convierte datos transaccionales en series temporales mensuales y agrupa categorías con pocos datos.
    """
    print(f"\n📊 Creando series temporales por {nivel}...")
    df_agregado = df.groupby([nivel, 'año_mes']).agg(
        gasto_total=('TOTALPESOS', 'sum'),
        num_transacciones=('TOTALPESOS', 'count')
    ).reset_index()
    
    df_agregado['fecha'] = df_agregado['año_mes'].dt.to_timestamp()
    df_agregado = df_agregado.sort_values([nivel, 'fecha'])

    # MEJORA: Agrupar categorías con menos de 12 meses de datos en 'Otros'.
    conteo_meses = df_agregado.groupby(nivel)['fecha'].count()
    categorias_a_agrupar = conteo_meses[conteo_meses < 12].index
    df_agregado['CATEGORIA'] = df_agregado['CATEGORIA'].apply(lambda x: 'Otros' if x in categorias_a_agrupar else x)

    df_completo = []
    categorias_unicas = df_agregado[nivel].unique()
    for categoria in categorias_unicas:
        df_cat = df_agregado[df_agregado[nivel] == categoria].copy()
        fecha_min = df_cat['fecha'].min()
        fecha_max = df_cat['fecha'].max()
        fechas_completas = pd.date_range(fecha_min, fecha_max, freq='MS')
        df_cat = df_cat.set_index('fecha').reindex(fechas_completas).fillna(0)
        df_cat[nivel] = categoria
        df_cat['fecha'] = df_cat.index
        df_completo.append(df_cat.reset_index(drop=True))
        
    df_final = pd.concat(df_completo, ignore_index=True)
    
    print(f"✅ Series temporales creadas: {len(df_final):,} puntos")
    print(f"🏷️ Categorías finales: {df_final[nivel].nunique()}")
    
    return df_final

def crear_features_temporales(df_serie):
    """
    Crea features de ingeniería temporal (lags, rolling stats, etc.).
    """
    df = df_serie.copy()
    
    # Lags y rolling stats
    for lag in [1, 3, 6, 12]:
        df[f'gasto_lag_{lag}'] = df['gasto_total'].shift(lag)
        df[f'cantidad_lag_{lag}'] = df['num_transacciones'].shift(lag)
    for window in [3, 6, 12]:
        df[f'gasto_rolling_mean_{window}'] = df['gasto_total'].rolling(window=window).mean()
        df[f'cantidad_rolling_mean_{window}'] = df['num_transacciones'].rolling(window=window).mean()
    
    # Features estacionales
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    
    # Limpiar NaN y valores extremos
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

def entrenar_modelo_categoria(df_categoria, nombre_categoria):
    """
    Entrena y selecciona el mejor modelo para una categoría.
    """
    df_features = crear_features_temporales(df_categoria)
    df_features = df_features.dropna().reset_index(drop=True)
    
    if len(df_features) < 12: # Mínimo 12 meses después de features
        print(f"⚠️ Datos insuficientes para {nombre_categoria} después de features.")
        return None
    
    exclude_cols = ['CATEGORIA', 'fecha', 'año_mes', 'gasto_total', 'num_transacciones']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols]
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']

    split_idx = int(len(X) * 0.8) # Usamos 80% para entrenamiento
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]

    # MEJORA: Selección del mejor modelo entre candidatos
    candidatos_gasto = {
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    candidatos_cantidad = {
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }

    resultados = {'gasto': {'r2': -np.inf, 'modelo': None, 'nombre': ''}, 'cantidad': {'r2': -np.inf, 'modelo': None, 'nombre': ''}}

    # Evaluación para Gasto
    for nombre, modelo in candidatos_gasto.items():
        modelo.fit(X_train, y_gasto_train)
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_gasto_test, y_pred)
        if r2 > resultados['gasto']['r2']:
            resultados['gasto']['r2'] = r2
            resultados['gasto']['modelo'] = modelo
            resultados['gasto']['nombre'] = nombre
    
    # Evaluación para Cantidad
    for nombre, modelo in candidatos_cantidad.items():
        modelo.fit(X_train, y_cantidad_train)
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_cantidad_test, y_pred)
        if r2 > resultados['cantidad']['r2']:
            resultados['cantidad']['r2'] = r2
            resultados['cantidad']['modelo'] = modelo
            resultados['cantidad']['nombre'] = nombre
            
    print(f"✅ Resultados para {nombre_categoria}:")
    print(f"   Gasto: R²={resultados['gasto']['r2']:.3f} (Modelo: {resultados['gasto']['nombre']})")
    print(f"   Cantidad: R²={resultados['cantidad']['r2']:.3f} (Modelo: {resultados['cantidad']['nombre']})")

    return {
        'modelos': {'gasto': resultados['gasto']['modelo'], 'cantidad': resultados['cantidad']['modelo']},
        'features': feature_cols
    }

def generar_predicciones(modelos_entrenados, df_completo):
    """
    Genera predicciones para cada categoría usando los modelos entrenados.
    """
    print("\n🔮 Generando predicciones para el próximo mes...")
    predicciones = []
    for categoria, modelo_info in modelos_entrenados.items():
        df_cat = df_completo[df_completo['CATEGORIA'] == categoria].copy()
        
        ultima_fila = df_cat.iloc[-1:]
        proximo_mes_fecha = ultima_fila['fecha'].iloc[0] + pd.DateOffset(months=1)
        
        df_futuro = ultima_fila.copy()
        df_futuro['fecha'] = proximo_mes_fecha
        df_futuro = crear_features_temporales(df_futuro)
        
        # Asegurarse de que el DataFrame futuro tenga las mismas columnas que el de entrenamiento
        X_futuro = df_futuro[modelo_info['features']]
        
        gasto_predicho = modelo_info['modelos']['gasto'].predict(X_futuro)[0]
        cantidad_predicha = modelo_info['modelos']['cantidad'].predict(X_futuro)[0]
        
        predicciones.append({
            'categoria': categoria,
            'gasto_predicho': max(0, gasto_predicho),
            'cantidad_predicha': max(0, cantidad_predicha)
        })
        
    return pd.DataFrame(predicciones)

def main(ruta_excel):
    """
    Ejecuta el pipeline completo.
    """
    df = cargar_y_limpiar_datos(ruta_excel)
    if df is None:
        return
        
    df_series = crear_series_temporales(df)
    
    categorias_unicas = df_series['CATEGORIA'].unique()
    
    print("\n🚀 Entrenando modelos para cada categoría...")
    global modelos_entrenados
    for categoria in categorias_unicas:
        df_cat = df_series[df_series['CATEGORIA'] == categoria]
        modelo_info = entrenar_modelo_categoria(df_cat, categoria)
        if modelo_info:
            modelos_entrenados[categoria] = modelo_info
            
    resumen_predicciones = generar_predicciones(modelos_entrenados, df_series)
    
    print("\n" + "="*50)
    print("RESUMEN DE PREDICCIONES")
    print("="*50)
    print(resumen_predicciones.sort_values('gasto_predicho', ascending=False).head(15).to_string())
    print(f"\n💰 Gasto total predicho: ${resumen_predicciones['gasto_predicho'].sum():,.2f}")
    
# Ejecución del pipeline
# Cambia 'tu_archivo.xlsx' a la ruta de tu archivo de datos
if __name__ == '__main__':
    # ¡IMPORTANTE! Cambia la ruta del archivo aquí.
    ruta_del_archivo = 'tu_archivo.xlsx' 
    main(ruta_del_archivo)
