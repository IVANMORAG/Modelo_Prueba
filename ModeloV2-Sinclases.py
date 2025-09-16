#!/usr/bin/env python3
"""
Modelo de Predicción de Compras - Grupo Salinas
Versión optimizada para Jupyter Notebook (sin clases)
Predice: Cantidad, Costo y Momento de compras por categoría/familia

Autor: Asistente Claude
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para notebooks
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

# Librerías para Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Librerías opcionales
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("✅ Prophet disponible")
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet no disponible. Solo se usarán modelos clásicos.")

# Utilidades
from datetime import datetime, timedelta
import json
import os

# Variables globales para almacenar resultados
modelos_entrenados = {}
datos_temporales = {}
metricas_globales = {}
configuracion = {
    'modo_debug': True,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10
}

# =============================================================================
# FUNCIONES DE CARGA Y LIMPIEZA DE DATOS
# =============================================================================

def cargar_y_limpiar_datos(ruta_excel, sheet_name="Detalle"):
    """
    Carga y limpia los datos de compras desde Excel
    """
    print("📂 Cargando datos...")
    
    # Cargar datos
    df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
    print(f"✅ Datos cargados: {len(df):,} registros, {len(df.columns)} columnas")
    
    # Mostrar primeras filas para verificar
    print("\n📋 Primeras 3 filas:")
    display(df.head(3))
    
    # Información de columnas
    print(f"\n📊 Columnas encontradas: {list(df.columns)}")
    
    # Limpiar fechas
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
    
    # Filtrar datos válidos
    antes_limpieza = len(df)
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
    df = df[df['TOTALPESOS'] > 0]  # Solo montos positivos
    
    print(f"🧹 Registros eliminados por datos faltantes: {antes_limpieza - len(df):,}")
    
    # Crear columnas temporales
    df['año'] = df['FECHAPEDIDO'].dt.year
    df['mes'] = df['FECHAPEDIDO'].dt.month
    df['año_mes'] = df['FECHAPEDIDO'].dt.to_period('M')
    df['trimestre'] = df['FECHAPEDIDO'].dt.quarter
    df['dia_semana'] = df['FECHAPEDIDO'].dt.dayofweek
    df['semana_año'] = df['FECHAPEDIDO'].dt.isocalendar().week
    
    # Limpiar categorías
    df['CATEGORIA'] = df['CATEGORIA'].fillna('Sin Categoría')
    df['FAMILIA'] = df['FAMILIA'].fillna('Sin Familia')
    df['CLASE'] = df['CLASE'].fillna('Sin Clase')
    
    print(f"\n✅ Datos limpios: {len(df):,} registros válidos")
    print(f"📅 Período: {df['FECHAPEDIDO'].min().strftime('%Y-%m-%d')} a {df['FECHAPEDIDO'].max().strftime('%Y-%m-%d')}")
    print(f"💰 Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
    print(f"🏷️ Categorías únicas: {df['CATEGORIA'].nunique()}")
    print(f"👥 Familias únicas: {df['FAMILIA'].nunique()}")
    
    return df

def mostrar_estadisticas_basicas(df):
    """
    Muestra estadísticas descriptivas básicas
    """
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS BÁSICAS")
    print("="*60)
    
    # Por categoría
    stats_categoria = df.groupby('CATEGORIA').agg({
        'TOTALPESOS': ['sum', 'mean', 'count'],
        'SOLICITUD': 'nunique'
    }).round(2)
    
    stats_categoria.columns = ['Gasto_Total', 'Gasto_Promedio', 'Num_Transacciones', 'Ordenes_Unicas']
    stats_categoria = stats_categoria.sort_values('Gasto_Total', ascending=False)
    
    print("\n🏆 TOP 10 CATEGORÍAS POR GASTO TOTAL:")
    display(stats_categoria.head(10))
    
    # Gráfico de barras
    plt.figure(figsize=(12, 6))
    top_10_cat = stats_categoria.head(10)
    plt.bar(range(len(top_10_cat)), top_10_cat['Gasto_Total'])
    plt.title('Top 10 Categorías por Gasto Total')
    plt.xlabel('Categorías')
    plt.ylabel('Gasto Total ($)')
    plt.xticks(range(len(top_10_cat)), top_10_cat.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Evolución temporal
    evolucion_mensual = df.groupby('año_mes')['TOTALPESOS'].sum()
    
    plt.figure(figsize=(12, 6))
    evolucion_mensual.plot(kind='line', marker='o')
    plt.title('Evolución del Gasto Total por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Gasto Total ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return stats_categoria

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

def crear_features_temporales(df_serie):
    """
    Crea features de ingeniería temporal para machine learning
    """
    df = df_serie.copy()
    
    print("🔧 Creando features temporales...")
    
    # 1. LAGS (valores históricos)
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        df[f'gasto_lag_{lag}'] = df['gasto_total'].shift(lag)
        df[f'transacciones_lag_{lag}'] = df['num_transacciones'].shift(lag)
    
    # 2. ROLLING STATISTICS (ventanas móviles)
    windows = [3, 6, 12]
    for window in windows:
        df[f'gasto_rolling_mean_{window}'] = df['gasto_total'].rolling(window=window).mean()
        df[f'gasto_rolling_std_{window}'] = df['gasto_total'].rolling(window=window).std()
        df[f'gasto_rolling_max_{window}'] = df['gasto_total'].rolling(window=window).max()
        df[f'transacciones_rolling_mean_{window}'] = df['num_transacciones'].rolling(window=window).mean()
    
    # 3. FEATURES ESTACIONALES
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    
    # 4. TENDENCIA Y CAMBIOS
    df['tendencia'] = range(len(df))
    df['gasto_cambio_mes'] = df['gasto_total'].pct_change()
    df['gasto_cambio_acelerado'] = df['gasto_cambio_mes'].diff()
    df['transacciones_cambio_mes'] = df['num_transacciones'].pct_change()
    
    # 5. FEATURES ESPECÍFICOS PARA MOMENTO DE COMPRA
    df['dias_sin_compra'] = (df['num_transacciones'] == 0).astype(int)
    df['meses_sin_compra'] = df.groupby((df['num_transacciones'] > 0).cumsum())['dias_sin_compra'].cumsum()
    
    # 6. RATIOS Y EFICIENCIAS
    df['gasto_por_transaccion'] = df['gasto_total'] / (df['num_transacciones'] + 1)
    df['gasto_por_orden'] = df['gasto_total'] / (df['ordenes_unicas'] + 1)
    df['transacciones_por_orden'] = df['num_transacciones'] / (df['ordenes_unicas'] + 1)
    
    # 7. FEATURES DE VOLATILIDAD
    df['volatilidad_gasto'] = df['gasto_total'].rolling(window=6).std()
    df['volatilidad_transacciones'] = df['num_transacciones'].rolling(window=6).std()
    
    # Limpiar valores infinitos y NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Estrategia de llenado más inteligente
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['fecha']:
            # Primero forward fill, luego backward fill, finalmente 0
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ Features creados: {len([col for col in df.columns if 'lag' in col or 'rolling' in col or 'sin' in col or 'cos' in col])} features temporales")
    
    return df

# =============================================================================
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def entrenar_modelo_categoria(df_categoria, nombre_categoria, verbose=True):
    """
    Entrena modelos de ML para una categoría específica
    """
    if len(df_categoria) < 12:  # Menos de 1 año de datos
        if verbose:
            print(f"⚠️ Pocos datos para {nombre_categoria}: {len(df_categoria)} meses")
        return None
    
    # Crear features
    df_features = crear_features_temporales(df_categoria)
    
    # Eliminar filas con muchos NaN (especialmente por los lags iniciales)
    min_datos_requeridos = max(12, len(df_features) * 0.3)  # Al menos 30% de los datos o 12 meses
    df_features = df_features.dropna(thresh=len(df_features.columns)*0.7)  # Mantener filas con al menos 70% datos válidos
    
    if len(df_features) < min_datos_requeridos:
        if verbose:
            print(f"⚠️ Datos insuficientes después de crear features para {nombre_categoria}: {len(df_features)} < {min_datos_requeridos}")
        return None
    
    # Definir columnas de features (excluir targets y metadatos)
    exclude_cols = [
        'categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
        'num_transacciones', 'ordenes_unicas', 'centros_costo', 
        'solicitantes_unicos', 'gasto_promedio', 'gasto_std'
    ]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # Preparar datos
    X = df_features[feature_cols]
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']
    y_momento = df_features['dias_sin_compra']
    
    # Split temporal más robusto (75% entrenamiento, 25% prueba)
    split_idx = int(len(X) * 0.75)
    if split_idx < 6:  # Mínimo 6 puntos para entrenar
        if verbose:
            print(f"⚠️ Datos insuficientes para split temporal en {nombre_categoria}: {split_idx}")
        return None
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    y_momento_train, y_momento_test = y_momento.iloc[:split_idx], y_momento.iloc[split_idx:]
    
    # Configurar modelos con mejor configuración
    config = configuracion
    modelos = {
        'gasto': RandomForestRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'], 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config['random_state']
        ),
        'cantidad': RandomForestRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config['random_state']
        ),
        'momento': RandomForestRegressor(
            n_estimators=50,  # Menos árboles para problema binario
            max_depth=5,
            min_samples_split=3,
            random_state=config['random_state']
        )
    }
    
    # Entrenar modelos
    try:
        modelos['gasto'].fit(X_train, y_gasto_train)
        modelos['cantidad'].fit(X_train, y_cantidad_train) 
        modelos['momento'].fit(X_train, y_momento_train)
    except Exception as e:
        if verbose:
            print(f"❌ Error entrenando {nombre_categoria}: {e}")
        return None
    
    # Hacer predicciones de prueba
    pred_gasto = modelos['gasto'].predict(X_test)
    pred_cantidad = modelos['cantidad'].predict(X_test)
    pred_momento = modelos['momento'].predict(X_test)
    
    # Calcular métricas
    metricas = {
        'mae_gasto': mean_absolute_error(y_gasto_test, pred_gasto),
        'rmse_gasto': np.sqrt(mean_squared_error(y_gasto_test, pred_gasto)),
        'r2_gasto': r2_score(y_gasto_test, pred_gasto),
        'mae_cantidad': mean_absolute_error(y_cantidad_test, pred_cantidad),
        'rmse_cantidad': np.sqrt(mean_squared_error(y_cantidad_test, pred_cantidad)),
        'r2_cantidad': r2_score(y_cantidad_test, pred_cantidad),
        'mae_momento': mean_absolute_error(y_momento_test, pred_momento),
        'r2_momento': r2_score(y_momento_test, pred_momento),
        'num_datos_entrenamiento': len(X_train),
        'num_datos_prueba': len(X_test),
        'num_features': len(feature_cols)
    }
    
    if verbose and configuracion['modo_debug']:
        print(f"🎯 {nombre_categoria}:")
        print(f"   💰 Gasto - MAE: ${metricas['mae_gasto']:,.0f}, R²: {metricas['r2_gasto']:.3f}")
        print(f"   📊 Cantidad - MAE: {metricas['mae_cantidad']:.1f}, R²: {metricas['r2_cantidad']:.3f}")
        print(f"   🕐 Momento - MAE: {metricas['mae_momento']:.2f}, R²: {metricas['r2_momento']:.3f}")
    
    return {
        'modelos': modelos,
        'feature_cols': feature_cols,
        'metricas': metricas,
        'datos_entrenamiento': df_features,
        'split_idx': split_idx,
        'fecha_ultimo_dato': df_features['fecha'].max()
    }

def entrenar_todos_modelos(df, nivel='CATEGORIA', limite_categorias=None):
    """
    Entrena modelos para todas las categorías/familias
    """
    print(f"\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS POR {nivel}")
    print("="*60)
    
    # Crear series temporales
    df_temporal = crear_series_temporales(df, nivel)
    
    # Obtener categorías únicas
    categorias = df_temporal[nivel.lower()].unique()
    
    # Limitar número de categorías si se especifica (útil para pruebas)
    if limite_categorias:
        categorias = categorias[:limite_categorias]
        print(f"🔧 Limitando a las primeras {limite_categorias} categorías para pruebas")
    
    print(f"📋 Categorías a procesar: {len(categorias)}")
    
    # Contadores de progreso
    modelos_exitosos = 0
    modelos_fallidos = 0
    errores_detallados = []
    
    # Entrenar cada categoría
    for i, categoria in enumerate(categorias):
        print(f"\n📈 Progreso: {i+1}/{len(categorias)} - {categoria}")
        
        # Filtrar datos de la categoría
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        
        # Entrenar modelo
        resultado = entrenar_modelo_categoria(df_cat, categoria, verbose=True)
        
        if resultado:
            # Guardar en variables globales
            modelos_entrenados[categoria] = resultado
            metricas_globales[categoria] = resultado['metricas']
            modelos_exitosos += 1
        else:
            modelos_fallidos += 1
            errores_detallados.append(categoria)
    
    # Resumen final
    print(f"\n" + "="*60)
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"="*60)
    print(f"🎯 Modelos exitosos: {modelos_exitosos}/{len(categorias)} ({modelos_exitosos/len(categorias)*100:.1f}%)")
    print(f"❌ Modelos fallidos: {modelos_fallidos}")
    
    if errores_detallados and configuracion['modo_debug']:
        print(f"\n⚠️ Categorías con errores:")
        for categoria in errores_detallados[:10]:  # Mostrar solo las primeras 10
            print(f"   - {categoria}")
        if len(errores_detallados) > 10:
            print(f"   ... y {len(errores_detallados)-10} más")
    
    # Estadísticas de calidad de modelos
    if modelos_exitosos > 0:
        r2_gastos = [metricas_globales[cat]['r2_gasto'] for cat in modelos_entrenados.keys()]
        r2_cantidades = [metricas_globales[cat]['r2_cantidad'] for cat in modelos_entrenados.keys()]
        
        print(f"\n📊 ESTADÍSTICAS DE CALIDAD:")
        print(f"   R² promedio (Gasto): {np.mean(r2_gastos):.3f}")
        print(f"   R² promedio (Cantidad): {np.mean(r2_cantidades):.3f}")
        print(f"   Modelos con R² > 0.5 (Gasto): {sum(1 for r2 in r2_gastos if r2 > 0.5)}")
        print(f"   Modelos con R² > 0.3 (Cantidad): {sum(1 for r2 in r2_cantidades if r2 > 0.3)}")
    
    return modelos_exitosos

# =============================================================================
# FUNCIONES DE PREDICCIÓN
# =============================================================================

def predecir_categoria(categoria, meses_adelante=1):
    """
    Genera predicciones para una categoría específica
    """
    if categoria not in modelos_entrenados:
        return {"error": f"No hay modelo entrenado para {categoria}"}
    
    modelo_info = modelos_entrenados[categoria]
    
    # Obtener últimos datos para hacer la predicción
    df_cat = modelo_info['datos_entrenamiento']
    if len(df_cat) == 0:
        return {"error": f"No hay datos disponibles para {categoria}"}
    
    ultima_fila = df_cat.iloc[-1]
    
    # Crear features para predicción futura
    features_pred = []
    for col in modelo_info['feature_cols']:
        if 'lag_1' in col:
            # Lag 1 = valor actual (lo que queremos predecir se basa en el valor actual)
            if 'gasto' in col:
                features_pred.append(ultima_fila['gasto_total'])
            elif 'transacciones' in col:
                features_pred.append(ultima_fila['num_transacciones'])
            else:
                features_pred.append(ultima_fila.get(col, 0))
                
        elif 'lag_' in col:
            # Otros lags - usar valores históricos
            features_pred.append(ultima_fila.get(col, 0))
            
        elif 'rolling' in col:
            # Rolling features - usar último valor calculado
            features_pred.append(ultima_fila.get(col, 0))
            
        elif 'tendencia' in col:
            # Tendencia - incrementar según meses adelante
            features_pred.append(ultima_fila.get(col, 0) + meses_adelante)
            
        elif col in ['mes_sin', 'mes_cos', 'trimestre_sin', 'trimestre_cos']:
            # Features estacionales - calcular para el mes futuro
            fecha_futura = ultima_fila['fecha'] + timedelta(days=30 * meses_adelante)
            mes_futuro = fecha_futura.month
            trimestre_futuro = (mes_futuro - 1) // 3 + 1
            
            if 'mes_sin' in col:
                features_pred.append(np.sin(2 * np.pi * mes_futuro / 12))
            elif 'mes_cos' in col:
                features_pred.append(np.cos(2 * np.pi * mes_futuro / 12))
            elif 'trimestre_sin' in col:
                features_pred.append(np.sin(2 * np.pi * trimestre_futuro / 4))
            elif 'trimestre_cos' in col:
                features_pred.append(np.cos(2 * np.pi * trimestre_futuro / 4))
                
        elif 'cambio' in col:
            # Features de cambio - usar último valor
            features_pred.append(ultima_fila.get(col, 0))
            
        else:
            # Otros features - usar último valor disponible
            features_pred.append(ultima_fila.get(col, 0))
    
    # Convertir a array numpy
    X_pred = np.array(features_pred).reshape(1, -1)
    
    # Verificar que tenemos el número correcto de features
    if X_pred.shape[1] != len(modelo_info['feature_cols']):
        return {"error": f"Mismatch en número de features: esperado {len(modelo_info['feature_cols'])}, obtenido {X_pred.shape[1]}"}
    
    try:
        # Hacer predicciones
        pred_gasto = modelo_info['modelos']['gasto'].predict(X_pred)[0]
        pred_cantidad = modelo_info['modelos']['cantidad'].predict(X_pred)[0]
        pred_momento = modelo_info['modelos']['momento'].predict(X_pred)[0]
        
        # Asegurar valores no negativos
        pred_gasto = max(0, pred_gasto)
        pred_cantidad = max(0, pred_cantidad)
        pred_momento = max(0, min(1, pred_momento))  # Momento entre 0 y 1
        
        # Calcular fecha de predicción
        fecha_ultima = modelo_info['fecha_ultimo_dato']
        fecha_prediccion = fecha_ultima + timedelta(days=30 * meses_adelante)
        
        # Calcular intervalos de confianza (aproximados)
        mae_gasto = modelo_info['metricas']['mae_gasto']
        mae_cantidad = modelo_info['metricas']['mae_cantidad']
        
        return {
            "categoria": categoria,
            "fecha_prediccion": fecha_prediccion.strftime("%Y-%m-%d"),
            "gasto_predicho": round(pred_gasto, 2),
            "gasto_min_estimado": round(max(0, pred_gasto - mae_gasto), 2),
            "gasto_max_estimado": round(pred_gasto + mae_gasto, 2),
            "cantidad_predicha": round(pred_cantidad),
            "cantidad_min_estimada": round(max(0, pred_cantidad - mae_cantidad)),
            "cantidad_max_estimada": round(pred_cantidad + mae_cantidad),
            "probabilidad_compra": round(1 - pred_momento, 3),  # 1 - probabilidad de NO comprar
            "confianza_gasto": round(modelo_info['metricas']['r2_gasto'], 3),
            "confianza_cantidad": round(modelo_info['metricas']['r2_cantidad'], 3),
            "error_promedio_gasto": round(mae_gasto, 2),
            "meses_datos_entrenamiento": len(modelo_info['datos_entrenamiento']),
            "ultima_actualizacion": modelo_info['fecha_ultimo_dato'].strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        return {"error": f"Error en predicción para {categoria}: {str(e)}"}

def predecir_todas_categorias(meses_adelante=1, top_n=None):
    """
    Genera predicciones para todas las categorías entrenadas
    """
    if not modelos_entrenados:
        print("❌ No hay modelos entrenados. Ejecuta primero entrenar_todos_modelos()")
        return None
    
    print(f"🔮 Generando predicciones para {meses_adelante} mes(es) adelante...")
    print(f"📊 Categorías disponibles: {len(modelos_entrenados)}")
    
    resultados = []
    errores = []
    
    for i, categoria in enumerate(modelos_entrenados.keys()):
        if i % 20 == 0 and i > 0:
            print(f"   Progreso: {i}/{len(modelos_entrenados)}")
            
        pred = predecir_categoria(categoria, meses_adelante)
        
        if "error" not in pred:
            resultados.append(pred)
        else:
            errores.append((categoria, pred["error"]))
    
    if not resultados:
        print("❌ No se pudieron generar predicciones")
        return None
    
    # Convertir a DataFrame
    df_pred = pd.DataFrame(resultados)
    
    # Ordenar por gasto predicho (descendente)
    df_pred = df_pred.sort_values('gasto_predicho', ascending=False)
    
    # Limitar a top N si se especifica
    if top_n:
        df_pred = df_pred.head(top_n)
        print(f"📊 Mostrando top {top_n} categorías por gasto predicho")
    
    # Estadísticas de resumen
    total_gasto = df_pred['gasto_predicho'].sum()
    total_transacciones = df_pred['cantidad_predicha'].sum()
    confianza_promedio_gasto = df_pred['confianza_gasto'].mean()
    confianza_promedio_cantidad = df_pred['confianza_cantidad'].mean()
    probabilidad_compra_promedio = df_pred['probabilidad_compra'].mean()
    
    print(f"\n" + "="*60)
    print(f"📋 RESUMEN DE PREDICCIONES")
    print(f"="*60)
    print(f"💰 Gasto total predicho: ${total_gasto:,.2f}")
    print(f"📈 Transacciones totales predichas: {total_transacciones:,.0f}")
    print(f"🎯 Confianza promedio (Gasto): {confianza_promedio_gasto:.3f}")
    print(f"🎯 Confianza promedio (Cantidad): {confianza_promedio_cantidad:.3f}")
    print(f"🛒 Probabilidad promedio de compra: {probabilidad_compra_promedio:.3f}")
    
    if errores and configuracion['modo_debug']:
        print(f"\n⚠️ Errores en {len(errores)} categorías:")
        for categoria, error in errores[:5]:
            print(f"   - {categoria}: {error}")
    
    return df_pred

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS
# =============================================================================

def mostrar_top_predicciones(df_predicciones, top_n=10):
    """
    Muestra y visualiza las top predicciones
    """
    if df_predicciones is None or len(df_predicciones) == 0:
        print("❌ No hay predicciones para mostrar")
        return
    
    print(f"\n🏆 TOP {top_n} PREDICCIONES POR GASTO")
    print("="*80)
    
    top_pred = df_predicciones.head(top_n)
    
    # Mostrar tabla
    display(top_pred[['categoria', 'gasto_predicho', 'cantidad_predicha', 
                     'probabilidad_compra', 'confianza_gasto']].round(2))
    
    # Gráfico de barras para gasto
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Gasto predicho
    plt.subplot(2, 2, 1)
    plt.bar(range(len(top_pred)), top_pred['gasto_predicho'])
    plt.title(f'Top {top_n} - Gasto Predicho')
    plt.xlabel('Categorías')
    plt.ylabel('Gasto Predicho ($)')
    plt.xticks(range(len(top_pred)), 
               [cat[:15] + '...' if len(cat) > 15 else cat for cat in top_pred['categoria']], 
               rotation=45, ha='right')
    
    # Subplot 2: Cantidad predicha
    plt.subplot(2, 2, 2)
    plt.bar(range(len(top_pred)), top_pred['cantidad_predicha'], color='orange')
    plt.title(f'Top {top_n} - Cantidad Predicha')
    plt.xlabel('Categorías')
    plt.ylabel('Número de Transacciones')
    plt.xticks(range(len(top_pred)), 
               [cat[:15] + '...' if len(cat) > 15 else cat for cat in top_pred['categoria']], 
               rotation=45, ha='right')
    
    # Subplot 3: Confianza del modelo
    plt.subplot(2, 2, 3)
    plt.bar(range(len(top_pred)), top_pred['confianza_gasto'], color='green')
    plt.title(f'Top {top_n} - Confianza del Modelo (R²)')
    plt.xlabel('Categorías')
    plt.ylabel('R² Score')
    plt.xticks(range(len(top_pred)), 
               [cat[:15] + '...' if len(cat) > 15 else cat for cat in top_pred['categoria']], 
               rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Subplot 4: Probabilidad de compra
    plt.subplot(2, 2, 4)
    plt.bar(range(len(top_pred)), top_pred['probabilidad_compra'], color='purple')
    plt.title(f'Top {top_n} - Probabilidad de Compra')
    plt.xlabel('Categorías')
    plt.ylabel('Probabilidad')
    plt.xticks(range(len(top_pred)), 
               [cat[:15] + '...' if len(cat) > 15 else cat for cat in top_pred['categoria']], 
               rotation=45, ha='right')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def analizar_calidad_modelos():
    """
    Analiza la calidad de todos los modelos entrenados
    """
    if not metricas_globales:
        print("❌ No hay modelos entrenados para analizar")
        return
    
    print("\n📊 ANÁLISIS DE CALIDAD DE MODELOS")
    print("="*60)
    
    # Crear DataFrame con métricas
    metricas_df = pd.DataFrame(metricas_globales).T
    
    # Estadísticas generales
    print(f"📈 Estadísticas de R² (Gasto):")
    print(f"   Promedio: {metricas_df['r2_gasto'].mean():.3f}")
    print(f"   Mediana: {metricas_df['r2_gasto'].median():.3f}")
    print(f"   Mínimo: {metricas_df['r2_gasto'].min():.3f}")
    print(f"   Máximo: {metricas_df['r2_gasto'].max():.3f}")
    
    print(f"\n📊 Estadísticas de R² (Cantidad):")
    print(f"   Promedio: {metricas_df['r2_cantidad'].mean():.3f}")
    print(f"   Mediana: {metricas_df['r2_cantidad'].median():.3f}")
    print(f"   Mínimo: {metricas_df['r2_cantidad'].min():.3f}")
    print(f"   Máximo: {metricas_df['r2_cantidad'].max():.3f}")
    
    # Distribución de calidad
    excelentes = len(metricas_df[metricas_df['r2_gasto'] > 0.8])
    buenos = len(metricas_df[(metricas_df['r2_gasto'] > 0.5) & (metricas_df['r2_gasto'] <= 0.8)])
    regulares = len(metricas_df[(metricas_df['r2_gasto'] > 0.3) & (metricas_df['r2_gasto'] <= 0.5)])
    pobres = len(metricas_df[metricas_df['r2_gasto'] <= 0.3])
    
    print(f"\n🎯 Distribución de Calidad (Gasto):")
    print(f"   Excelentes (R² > 0.8): {excelentes} ({excelentes/len(metricas_df)*100:.1f}%)")
    print(f"   Buenos (R² 0.5-0.8): {buenos} ({buenos/len(metricas_df)*100:.1f}%)")
    print(f"   Regulares (R² 0.3-0.5): {regulares} ({regulares/len(metricas_df)*100:.1f}%)")
    print(f"   Pobres (R² < 0.3): {pobres} ({pobres/len(metricas_df)*100:.1f}%)")
    
    # Top y bottom modelos
    print(f"\n🏆 TOP 5 MEJORES MODELOS (por R² Gasto):")
    top_modelos = metricas_df.nlargest(5, 'r2_gasto')
    for idx, row in top_modelos.iterrows():
        print(f"   {idx}: R² = {row['r2_gasto']:.3f}, MAE = ${row['mae_gasto']:,.0f}")
    
    print(f"\n⚠️ TOP 5 MODELOS PARA MEJORAR (por R² Gasto):")
    bottom_modelos = metricas_df.nsmallest(5, 'r2_gasto')
    for idx, row in bottom_modelos.iterrows():
        print(f"   {idx}: R² = {row['r2_gasto']:.3f}, MAE = ${row['mae_gasto']:,.0f}")
    
    # Visualización
    plt.figure(figsize=(15, 10))
    
    # Histograma de R² Gasto
    plt.subplot(2, 3, 1)
    plt.hist(metricas_df['r2_gasto'], bins=20, alpha=0.7, color='blue')
    plt.title('Distribución R² (Gasto)')
    plt.xlabel('R² Score')
    plt.ylabel('Frecuencia')
    
    # Histograma de R² Cantidad
    plt.subplot(2, 3, 2)
    plt.hist(metricas_df['r2_cantidad'], bins=20, alpha=0.7, color='orange')
    plt.title('Distribución R² (Cantidad)')
    plt.xlabel('R² Score')
    plt.ylabel('Frecuencia')
    
    # Scatter plot R² Gasto vs Cantidad
    plt.subplot(2, 3, 3)
    plt.scatter(metricas_df['r2_gasto'], metricas_df['r2_cantidad'], alpha=0.6)
    plt.xlabel('R² Gasto')
    plt.ylabel('R² Cantidad')
    plt.title('R² Gasto vs Cantidad')
    
    # MAE Gasto
    plt.subplot(2, 3, 4)
    plt.hist(metricas_df['mae_gasto'], bins=20, alpha=0.7, color='green')
    plt.title('Distribución MAE (Gasto)')
    plt.xlabel('MAE ($)')
    plt.ylabel('Frecuencia')
    
    # Número de datos de entrenamiento
    plt.subplot(2, 3, 5)
    plt.hist(metricas_df['num_datos_entrenamiento'], bins=20, alpha=0.7, color='purple')
    plt.title('Datos de Entrenamiento por Modelo')
    plt.xlabel('Número de Observaciones')
    plt.ylabel('Frecuencia')
    
    # Box plot de R² por categorías
    plt.subplot(2, 3, 6)
    r2_data = [metricas_df['r2_gasto'], metricas_df['r2_cantidad'], metricas_df['r2_momento']]
    plt.boxplot(r2_data, labels=['Gasto', 'Cantidad', 'Momento'])
    plt.title('Distribución R² por Tipo de Predicción')
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.show()
    
    return metricas_df

# =============================================================================
# FUNCIONES DE GUARDADO Y CARGA
# =============================================================================

def guardar_modelos(ruta_base="modelos_salinas"):
    """
    Guarda todos los modelos entrenados y metadatos
    """
    if not modelos_entrenados:
        print("❌ No hay modelos para guardar")
        return False
    
    print(f"💾 Guardando modelos en '{ruta_base}/'...")
    
    # Crear directorio
    os.makedirs(ruta_base, exist_ok=True)
    
    modelos_guardados = 0
    
    for categoria, modelo_info in modelos_entrenados.items():
        try:
            categoria_clean = categoria.replace(" ", "_").replace("/", "_").replace("\\", "_")
            categoria_clean = categoria_clean.replace("(", "").replace(")", "").replace("&", "y")
            
            # Guardar modelos individuales
            joblib.dump(modelo_info['modelos']['gasto'], 
                       f"{ruta_base}/{categoria_clean}_gasto.pkl")
            joblib.dump(modelo_info['modelos']['cantidad'], 
                       f"{ruta_base}/{categoria_clean}_cantidad.pkl")
            joblib.dump(modelo_info['modelos']['momento'], 
                       f"{ruta_base}/{categoria_clean}_momento.pkl")
            
            # Guardar features y metadatos
            joblib.dump(modelo_info['feature_cols'], 
                       f"{ruta_base}/{categoria_clean}_features.pkl")
            
            modelos_guardados += 1
            
        except Exception as e:
            print(f"⚠️ Error guardando {categoria}: {e}")
    
    # Guardar metadatos generales
    try:
        metadata = {
            'categorias': list(modelos_entrenados.keys()),
            'fecha_entrenamiento': datetime.now().isoformat(),
            'num_modelos': len(modelos_entrenados),
            'modelos_guardados': modelos_guardados,
            'metricas': metricas_globales,
            'configuracion': configuracion,
            'version': '2.0'
        }
        
        with open(f"{ruta_base}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Guardado completo:")
        print(f"   📁 Directorio: {ruta_base}/")
        print(f"   🎯 Modelos guardados: {modelos_guardados}/{len(modelos_entrenados)}")
        print(f"   📋 Archivo de metadatos: metadata.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando metadatos: {e}")
        return False

def cargar_modelos(ruta_base="modelos_salinas"):
    """
    Carga modelos previamente guardados
    """
    print(f"📂 Cargando modelos desde '{ruta_base}/'...")
    
    try:
        # Cargar metadatos
        with open(f"{ruta_base}/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"📋 Metadatos encontrados:")
        print(f"   🗓️ Fecha entrenamiento: {metadata.get('fecha_entrenamiento', 'N/A')}")
        print(f"   🎯 Número de modelos: {metadata.get('num_modelos', 0)}")
        print(f"   📊 Versión: {metadata.get('version', '1.0')}")
        
    except FileNotFoundError:
        print(f"❌ No se encontró archivo de metadatos en {ruta_base}/")
        return 0
    except Exception as e:
        print(f"❌ Error leyendo metadatos: {e}")
        return 0
    
    # Limpiar variables globales
    global modelos_entrenados, metricas_globales
    modelos_entrenados.clear()
    metricas_globales.clear()
    
    # Cargar configuración si está disponible
    if 'configuracion' in metadata:
        configuracion.update(metadata['configuracion'])
    
    # Cargar cada modelo
    modelos_cargados = 0
    for categoria in metadata['categorias']:
        categoria_clean = categoria.replace(" ", "_").replace("/", "_").replace("\\", "_")
        categoria_clean = categoria_clean.replace("(", "").replace(")", "").replace("&", "y")
        
        try:
            # Cargar modelos
            modelos = {
                'gasto': joblib.load(f"{ruta_base}/{categoria_clean}_gasto.pkl"),
                'cantidad': joblib.load(f"{ruta_base}/{categoria_clean}_cantidad.pkl"),
                'momento': joblib.load(f"{ruta_base}/{categoria_clean}_momento.pkl")
            }
            
            # Cargar features
            feature_cols = joblib.load(f"{ruta_base}/{categoria_clean}_features.pkl")
            
            # Reconstruir info del modelo
            modelos_entrenados[categoria] = {
                'modelos': modelos,
                'feature_cols': feature_cols,
                'metricas': metadata['metricas'].get(categoria, {}),
                'datos_entrenamiento': None,  # No guardamos los datos de entrenamiento
                'split_idx': None,
                'fecha_ultimo_dato': datetime.now()  # Placeholder
            }
            
            # Cargar métricas
            metricas_globales[categoria] = metadata['metricas'].get(categoria, {})
            
            modelos_cargados += 1
            
        except FileNotFoundError:
            print(f"⚠️ Archivos no encontrados para {categoria}")
        except Exception as e:
            print(f"⚠️ Error cargando {categoria}: {e}")
    
    print(f"✅ Carga completa:")
    print(f"   🎯 Modelos cargados: {modelos_cargados}/{len(metadata['categorias'])}")
    print(f"   📊 Listos para hacer predicciones")
    
    return modelos_cargados

# =============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR TODO EL PIPELINE
# =============================================================================

def ejecutar_pipeline_completo(ruta_excel, sheet_name="Detalle", limite_categorias=None, 
                              meses_prediccion=1, guardar_resultados=True):
    """
    Ejecuta todo el pipeline de ML de principio a fin
    """
    print("🏢 MODELO DE PREDICCIÓN DE COMPRAS - GRUPO SALINAS")
    print("="*80)
    print("📋 Versión optimizada para Jupyter Notebook")
    print("="*80)
    
    try:
        # 1. CARGAR Y LIMPIAR DATOS
        print("\n🔄 PASO 1: CARGA Y LIMPIEZA DE DATOS")
        df = cargar_y_limpiar_datos(ruta_excel, sheet_name)
        
        # 2. ANÁLISIS EXPLORATORIO
        print("\n🔄 PASO 2: ANÁLISIS EXPLORATORIO")
        stats_basicas = mostrar_estadisticas_basicas(df)
        
        # 3. ENTRENAMIENTO DE MODELOS
        print("\n🔄 PASO 3: ENTRENAMIENTO DE MODELOS")
        modelos_exitosos = entrenar_todos_modelos(df, nivel='CATEGORIA', 
                                                limite_categorias=limite_categorias)
        
        if modelos_exitosos == 0:
            print("❌ No se pudieron entrenar modelos")
            return None, None, None
        
        # 4. ANÁLISIS DE CALIDAD
        print("\n🔄 PASO 4: ANÁLISIS DE CALIDAD DE MODELOS")
        metricas_df = analizar_calidad_modelos()
        
        # 5. GENERAR PREDICCIONES
        print(f"\n🔄 PASO 5: GENERACIÓN DE PREDICCIONES")
        df_predicciones = predecir_todas_categorias(meses_adelante=meses_prediccion)
        
        # 6. MOSTRAR RESULTADOS
        print(f"\n🔄 PASO 6: VISUALIZACIÓN DE RESULTADOS")
        mostrar_top_predicciones(df_predicciones, top_n=15)
        
        # 7. GUARDAR MODELOS
        if guardar_resultados:
            print(f"\n🔄 PASO 7: GUARDADO DE MODELOS")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            ruta_guardado = f"modelos_salinas_{timestamp}"
            guardar_modelos(ruta_guardado)
        
        print(f"\n" + "="*80)
        print("🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        print("="*80)
        print(f"✅ Datos procesados: {len(df):,} registros")
        print(f"✅ Modelos entrenados: {modelos_exitosos}")
        print(f"✅ Predicciones generadas: {len(df_predicciones) if df_predicciones is not None else 0}")
        
        return df, df_predicciones, metricas_df
        
    except Exception as e:
        print(f"\n❌ ERROR EN EL PIPELINE:")
        print(f"   {str(e)}")
        import traceback
        if configuracion['modo_debug']:
            print("\n🔍 TRACEBACK COMPLETO:")
            traceback.print_exc()
        return None, None, None

# =============================================================================
# FUNCIONES DE USO RÁPIDO PARA JUPYTER
# =============================================================================

def prediccion_rapida(categoria, mostrar_detalles=True):
    """
    Función rápida para hacer una predicción individual
    """
    if not modelos_entrenados:
        print("❌ No hay modelos cargados. Ejecuta primero el pipeline o carga modelos existentes.")
        return None
    
    pred = predecir_categoria(categoria)
    
    if "error" in pred:
        print(f"❌ {pred['error']}")
        return None
    
    if mostrar_detalles:
        print(f"🎯 PREDICCIÓN PARA: {categoria}")
        print(f"="*50)
        print(f"💰 Gasto predicho: ${pred['gasto_predicho']:,.2f}")
        print(f"   Rango estimado: ${pred['gasto_min_estimado']:,.2f} - ${pred['gasto_max_estimado']:,.2f}")
        print(f"📊 Transacciones predichas: {pred['cantidad_predicha']:,.0f}")
        print(f"   Rango estimado: {pred['cantidad_min_estimada']:,.0f} - {pred['cantidad_max_estimada']:,.0f}")
        print(f"🛒 Probabilidad de compra: {pred['probabilidad_compra']:.1%}")
        print(f"📅 Fecha predicción: {pred['fecha_prediccion']}")
        print(f"🎯 Confianza del modelo: {pred['confianza_gasto']:.3f}")
        print(f"📈 Error promedio: ${pred['error_promedio_gasto']:,.2f}")
    
    return pred

def mostrar_categorias_disponibles():
    """
    Muestra las categorías para las que hay modelos entrenados
    """
    if not modelos_entrenados:
        print("❌ No hay modelos cargados.")
        return []
    
    categorias = list(modelos_entrenados.keys())
    print(f"📋 CATEGORÍAS DISPONIBLES ({len(categorias)}):")
    print("="*50)
    
    for i, cat in enumerate(sorted(categorias), 1):
        confianza = metricas_globales.get(cat, {}).get('r2_gasto', 0)
        print(f"{i:2d}. {cat} (Confianza: {confianza:.3f})")
    
    return categorias

def resumen_rapido():
    """
    Muestra un resumen rápido del estado actual
    """
    print("📊 RESUMEN RÁPIDO DEL MODELO")
    print("="*40)
    
    if not modelos_entrenados:
        print("❌ No hay modelos cargados")
        return
    
    num_modelos = len(modelos_entrenados)
    if metricas_globales:
        r2_promedio = np.mean([m.get('r2_gasto', 0) for m in metricas_globales.values()])
        mae_promedio = np.mean([m.get('mae_gasto', 0) for m in metricas_globales.values()])
    else:
        r2_promedio = mae_promedio = 0
    
    print(f"🎯 Modelos entrenados: {num_modelos}")
    print(f"📈 R² promedio: {r2_promedio:.3f}")
    print(f"💰 Error promedio: ${mae_promedio:,.0f}")
    print(f"🔧 Modo debug: {'Activado' if configuracion['modo_debug'] else 'Desactivado'}")
    
    # Top 3 mejores modelos
    if metricas_globales:
        mejores = sorted(metricas_globales.items(), 
                        key=lambda x: x[1].get('r2_gasto', 0), 
                        reverse=True)[:3]
        print(f"\n🏆 Top 3 mejores modelos:")
        for i, (cat, met) in enumerate(mejores, 1):
            print(f"   {i}. {cat}: R² = {met.get('r2_gasto', 0):.3f}")

# =============================================================================
# CONFIGURACIÓN INICIAL PARA JUPYTER
# =============================================================================

print("🚀 MODELO DE PREDICCIÓN DE COMPRAS - GRUPO SALINAS")
print("📋 Versión optimizada para Jupyter Notebook")
print("="*60)
print("✅ Librerías cargadas correctamente")
print("✅ Funciones definidas")
print("\n📖 FUNCIONES PRINCIPALES DISPONIBLES:")
print("   • ejecutar_pipeline_completo() - Ejecuta todo el proceso")
print("   • cargar_modelos() - Carga modelos guardados")
print("   • prediccion_rapida() - Predicción individual")
print("   • mostrar_categorias_disponibles() - Lista categorías")
print("   • resumen_rapido() - Estado actual del modelo")
print("\n🔧 Para empezar, usa:")
print("   df, pred, metrics = ejecutar_pipeline_completo('tu_archivo.xlsx')")
print("="*60)