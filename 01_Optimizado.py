#!/usr/bin/env python3
"""
Mejoras al Modelo de Predicción de Compras - Grupo Salinas
Versión 2.0 - Optimizaciones sin romper modelos exitosos (SIN CLASES)

Autor: Asistente Claude (adaptado para Jupyter)
Fecha: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import json
import os
from datetime import datetime

# ============================================================================
# VARIABLES GLOBALES (en lugar de self)
# ============================================================================

def inicializar_variables_globales(modo_debug=True):
    """
    Inicializa todas las variables que antes estaban en self.__init__
    """
    global modelos, modelos_backup, datos_temporales, categorias, modo_debug, metricas, umbrales_calidad
    
    modelos = {}
    modelos_backup = {}
    datos_temporales = {}
    categorias = []
    modo_debug = modo_debug
    metricas = {}
    umbrales_calidad = {
        'r2_minimo': 0.3,
        'datos_minimos': 24,  # 2 años mínimo
        'mae_maximo_relativo': 0.5  # 50% del promedio
    }
    
    print("✅ Variables globales inicializadas")
    return True

# ============================================================================
# FUNCIONES DE LIMPIEZA DE DATOS
# ============================================================================

def limpiar_categorias_duplicadas(df):
    """
    Limpia y normaliza nombres de categorías duplicadas
    """
    print("🧹 Limpiando categorías duplicadas...")
    
    # Mapeo de categorías duplicadas
    mapeo_categorias = {
        'Tecnologia': 'Tecnología',
        'Construccion': 'Construcción', 
        'Construcción ': 'Construcción',
        'Electromecanico': 'Electromecánico',
        'Produccion': 'Producción',
        'Insumos ': 'Insumos',
        'Servicios ': 'Servicios',
        'Gastos de Fin de Año': 'Gastos De Fin De Año',
        'Gastos de fin de año': 'Gastos De Fin De Año',
        'Impresos y publicidad': 'Impresos Y Publicidad',
        'Obsequios y atenciones': 'Obsequios Y Atenciones',
        'Gestión y comercialización (G&C)': 'Gestión y Comercialización (G&C)',
        'O&M propio': 'O&M Propio',
        'O&M terceros': 'O&M Terceros',
        'EPC propio': 'EPC Propio',
        'EPC tercero': 'EPC Tercero',
        'Ensamblika ': 'Ensamblika'
    }
    
    # Aplicar mapeo
    df['CATEGORIA_ORIGINAL'] = df['CATEGORIA'].copy()
    df['CATEGORIA'] = df['CATEGORIA'].replace(mapeo_categorias)
    
    # Estadísticas de limpieza
    duplicadas_encontradas = len([k for k, v in mapeo_categorias.items() if k in df['CATEGORIA_ORIGINAL'].values])
    print(f"   ✅ {duplicadas_encontradas} categorías duplicadas corregidas")
    print(f"   📊 Categorías únicas antes: {df['CATEGORIA_ORIGINAL'].nunique()}")
    print(f"   📊 Categorías únicas después: {df['CATEGORIA'].nunique()}")
    
    return df

def filtrar_outliers_inteligente(df):
    """
    Filtra outliers extremos pero preserva datos importantes
    """
    print("🎯 Filtrando outliers inteligentemente...")
    
    original_len = len(df)
    
    # Filtrar por percentiles para TOTALPESOS
    q1 = df['TOTALPESOS'].quantile(0.001)  # Más permisivo
    q99 = df['TOTALPESOS'].quantile(0.999)
    
    # Mantener el 99.8% de los datos
    df_filtrado = df[(df['TOTALPESOS'] >= q1) & (df['TOTALPESOS'] <= q99)]
    
    filtrados = original_len - len(df_filtrado)
    print(f"   🗑️ Outliers filtrados: {filtrados:,} ({filtrados/original_len*100:.1f}%)")
    print(f"   💰 Rango final: ${q1:,.2f} a ${q99:,.2f}")
    
    return df_filtrado

# ============================================================================
# FUNCIONES DE EVALUACIÓN Y FEATURES
# ============================================================================

def evaluar_calidad_categoria(df_categoria, nombre_categoria):
    """
    Evalúa si una categoría tiene datos suficientes y de calidad
    """
    # Criterios de calidad
    criterios = {
        'suficientes_meses': len(df_categoria) >= umbrales_calidad['datos_minimos'],
        'variabilidad': df_categoria['gasto_total'].std() > 0,
        'no_solo_ceros': (df_categoria['gasto_total'] > 0).sum() > len(df_categoria) * 0.1,
        'tendencia_estable': True  # Por ahora siempre True
    }
    
    # Calcular score de calidad
    calidad_score = sum(criterios.values()) / len(criterios)
    
    if modo_debug and calidad_score < 1.0:
        print(f"   ⚠️ {nombre_categoria} - Calidad: {calidad_score:.2f}")
        for criterio, cumple in criterios.items():
            if not cumple:
                print(f"      ❌ {criterio}")
    
    return calidad_score >= 0.75, criterios

def crear_features_temporales_mejoradas(df_serie):
    """
    Crea features más robustas y menos propensos a overfitting
    """
    df = df_serie.copy()
    
    # Features básicos de lag (menos lags para evitar overfitting)
    for lag in [1, 3, 6, 12]:  # Reducido de 1,2,3,6,12
        df[f'gasto_lag_{lag}'] = df['gasto_total'].shift(lag)
        df[f'transacciones_lag_{lag}'] = df['num_transacciones'].shift(lag)
    
    # Rolling statistics más conservadores
    for window in [3, 6, 12]:
        df[f'gasto_ma_{window}'] = df['gasto_total'].rolling(window=window, min_periods=2).mean()
        df[f'gasto_std_{window}'] = df['gasto_total'].rolling(window=window, min_periods=2).std()
        df[f'transacciones_ma_{window}'] = df['num_transacciones'].rolling(window=window, min_periods=2).mean()
    
    # Features estacionales mejorados
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['semestre'] = ((df['mes'] - 1) // 6) + 1
    
    # Encodings circulares más robustos
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    
    # Tendencia suavizada
    df['tendencia'] = range(len(df))
    df['tendencia_normalizada'] = (df['tendencia'] - df['tendencia'].mean()) / df['tendencia'].std()
    
    # Ratios más estables
    df['gasto_cambio_mes'] = df['gasto_total'].pct_change().fillna(0)
    df['gasto_cambio_suavizado'] = df['gasto_cambio_mes'].rolling(3, min_periods=1).mean()
    
    # Features de volatilidad
    df['gasto_volatilidad'] = df['gasto_total'].rolling(6, min_periods=2).std() / df['gasto_total'].rolling(6, min_periods=2).mean()
    df['gasto_volatilidad'] = df['gasto_volatilidad'].fillna(df['gasto_volatilidad'].median())
    
    # Limpiar infinitos y NaN de forma más robusta
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    # Imputación más inteligente
    for col in numeric_columns:
        if df[col].isnull().any():
            if 'lag' in col or 'ma' in col:
                # Para lags y medias móviles, usar forward fill limitado
                df[col] = df[col].fillna(method='ffill').fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())
    
    return df

def seleccionar_mejor_modelo(X_train, y_train, X_test, y_test, nombre_categoria):
    """
    Prueba múltiples modelos y selecciona el mejor
    """
    modelos_candidatos = {
        'rf_conservative': RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'rf_balanced': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        ),
        'gbm_conservative': GradientBoostingRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42
        )
    }
    
    mejor_modelo = None
    mejor_score = -np.inf
    mejor_nombre = None
    
    for nombre_modelo, modelo in modelos_candidatos.items():
        try:
            # Entrenar
            modelo.fit(X_train, y_train)
            
            # Predecir
            pred = modelo.predict(X_test)
            
            # Evaluar
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            
            # Score combinado (penalizar R2 negativos fuertemente)
            if r2 < 0:
                score = r2 * 2  # Penalización doble para R2 negativos
            else:
                score = r2
            
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo
                mejor_nombre = nombre_modelo
                
        except Exception as e:
            if modo_debug:
                print(f"      ❌ Error en {nombre_modelo}: {str(e)[:50]}")
            continue
    
    return mejor_modelo, mejor_score, mejor_nombre

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def crear_series_temporales(df, nivel='CATEGORIA'):
    """
    Crea las series temporales agrupadas por categoría
    """
    print(f"📈 Creando series temporales por {nivel}...")
    
    # Agrupar por período y categoría
    df_series = df.groupby([df['FECHAPEDIDO'].dt.to_period('M'), 'CATEGORIA']).agg({
        'TOTALPESOS': ['sum', 'count'],
        'ORDEN': 'nunique'
    }).reset_index()
    
    # Renombrar columnas
    df_series.columns = ['año_mes', 'categoria', 'gasto_total', 'num_transacciones', 'ordenes_unicas']
    
    # Convertir período a fecha
    df_series['fecha'] = df_series['año_mes'].dt.to_timestamp()
    
    # Ordenar
    df_series = df_series.sort_values(['categoria', 'fecha']).reset_index(drop=True)
    
    print(f"   ✅ {len(df_series)} registros temporales creados")
    print(f"   📊 Categorías: {df_series['categoria'].nunique()}")
    
    return df_series

def entrenar_modelo_individual_mejorado(df_categoria, nombre_categoria):
    """
    Versión mejorada del entrenamiento individual
    """
    # Evaluar calidad de datos
    es_viable, criterios = evaluar_calidad_categoria(df_categoria, nombre_categoria)
    
    if not es_viable:
        if modo_debug:
            print(f"   ❌ {nombre_categoria}: Datos insuficientes o de baja calidad")
        return None
    
    # Crear features mejoradas
    df_features = crear_features_temporales_mejoradas(df_categoria)
    
    # Eliminar filas con muchos NaN
    df_features = df_features.dropna()
    
    if len(df_features) < 12:  # Mínimo 1 año después de crear features
        if modo_debug:
            print(f"   ❌ {nombre_categoria}: Muy pocos datos después de features ({len(df_features)})")
        return None
    
    # Definir features (más selectivo)
    exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
                   'num_transacciones', 'ordenes_unicas', 'mes', 'trimestre', 'semestre']
    feature_cols = [col for col in df_features.columns 
                   if col not in exclude_cols and not col.endswith('_cambio_mes')]
    
    X = df_features[feature_cols]
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']
    
    # Split temporal más conservador (75% entrenamiento, 25% prueba)
    split_idx = max(int(len(X) * 0.75), len(X) - 6)  # Máximo 6 meses para test
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    
    # Seleccionar mejores modelos
    modelo_gasto, score_gasto, tipo_gasto = seleccionar_mejor_modelo(
        X_train, y_gasto_train, X_test, y_gasto_test, nombre_categoria
    )
    
    modelo_cantidad, score_cantidad, tipo_cantidad = seleccionar_mejor_modelo(
        X_train, y_cantidad_train, X_test, y_cantidad_test, nombre_categoria
    )
    
    if modelo_gasto is None or modelo_cantidad is None:
        if modo_debug:
            print(f"   ❌ {nombre_categoria}: No se pudo entrenar ningún modelo viable")
        return None
    
    # Evaluar modelos finales
    pred_gasto = modelo_gasto.predict(X_test)
    pred_cantidad = modelo_cantidad.predict(X_test)
    
    mae_gasto = mean_absolute_error(y_gasto_test, pred_gasto)
    r2_gasto = r2_score(y_gasto_test, pred_gasto)
    mae_cantidad = mean_absolute_error(y_cantidad_test, pred_cantidad)
    r2_cantidad = r2_score(y_cantidad_test, pred_cantidad)
    
    # Verificar calidad mínima
    if r2_gasto < -5 or r2_cantidad < -5:  # Evitar modelos extremadamente malos
        if modo_debug:
            print(f"   ❌ {nombre_categoria}: Modelos con rendimiento extremadamente pobre")
        return None
    
    if modo_debug:
        print(f"   ✅ {nombre_categoria} ({tipo_gasto}/{tipo_cantidad}):")
        print(f"      💰 Gasto - MAE: ${mae_gasto:,.0f}, R²: {r2_gasto:.3f}")
        print(f"      📊 Cantidad - MAE: {mae_cantidad:.1f}, R²: {r2_cantidad:.3f}")
    
    return {
        'modelos': {
            'gasto': modelo_gasto,
            'cantidad': modelo_cantidad,
            'momento': modelo_cantidad  # Usar mismo modelo para momento
        },
        'tipos_modelo': {
            'gasto': tipo_gasto,
            'cantidad': tipo_cantidad
        },
        'feature_cols': feature_cols,
        'metricas': {
            'mae_gasto': mae_gasto,
            'r2_gasto': r2_gasto,
            'mae_cantidad': mae_cantidad,
            'r2_cantidad': r2_cantidad,
            'mae_momento': mae_cantidad,
            'r2_momento': r2_cantidad
        },
        'datos_entrenamiento': df_features,
        'split_idx': split_idx,
        'calidad_datos': criterios
    }

def cargar_y_mejorar_datos(ruta_excel, sheet_name="Detalle"):
    """
    Carga datos con todas las mejoras aplicadas
    """
    print("📂 Cargando datos con mejoras...")
    
    # Cargar datos
    df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
    print(f"   📊 Datos originales: {len(df):,} registros")
    
    # Limpiar fechas
    df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
    
    # Filtros básicos
    df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS', 'CATEGORIA'])
    df = df[df['TOTALPESOS'] > 0]
    print(f"   🧹 Después de filtros básicos: {len(df):,} registros")
    
    # Limpiar categorías duplicadas
    df = limpiar_categorias_duplicadas(df)
    
    # Filtrar outliers inteligentemente
    df = filtrar_outliers_inteligente(df)
    
    # Crear columnas temporales
    df['año'] = df['FECHAPEDIDO'].dt.year
    df['mes'] = df['FECHAPEDIDO'].dt.month
    df['año_mes'] = df['FECHAPEDIDO'].dt.to_period('M')
    df['trimestre'] = df['FECHAPEDIDO'].dt.quarter
    
    print(f"   ✅ Datos finales: {len(df):,} registros")
    print(f"   📅 Período: {df['FECHAPEDIDO'].min()} a {df['FECHAPEDIDO'].max()}")
    print(f"   📊 Categorías únicas: {df['CATEGORIA'].nunique()}")
    
    return df

def backup_modelos_exitosos():
    """
    Hace backup de modelos que ya funcionan bien
    """
    print("💾 Creando backup de modelos exitosos...")
    
    modelos_buenos = {}
    for categoria, modelo_info in modelos.items():
        r2_gasto = modelo_info['metricas']['r2_gasto']
        r2_cantidad = modelo_info['metricas']['r2_cantidad']
        
        # Criterios para considerar un modelo "bueno"
        if r2_gasto > 0.5 and r2_cantidad > 0.3:
            modelos_buenos[categoria] = modelo_info.copy()
    
    global modelos_backup
    modelos_backup = modelos_buenos
    print(f"   ✅ {len(modelos_buenos)} modelos buenos respaldados")
    return len(modelos_buenos)

def restaurar_modelo_si_empeora(categoria, modelo_nuevo, modelo_anterior):
    """
    Restaura modelo anterior si el nuevo es peor
    """
    if modelo_anterior is None:
        return modelo_nuevo
    
    # Comparar métricas
    r2_nuevo_gasto = modelo_nuevo['metricas']['r2_gasto']
    r2_anterior_gasto = modelo_anterior['metricas']['r2_gasto']
    
    # Si el nuevo modelo es significativamente peor, mantener el anterior
    if r2_nuevo_gasto < r2_anterior_gasto - 0.1:  # Tolerancia del 10%
        if modo_debug:
            print(f"      🔄 {categoria}: Manteniendo modelo anterior (R² {r2_anterior_gasto:.3f} vs {r2_nuevo_gasto:.3f})")
        return modelo_anterior
    
    return modelo_nuevo

def entrenar_todos_modelos_mejorado(df, nivel='CATEGORIA'):
    """
    Entrenamiento mejorado que preserva modelos exitosos
    """
    print(f"🚀 Entrenamiento mejorado por {nivel}...")
    
    # Crear series temporales
    df_temporal = crear_series_temporales(df, nivel)
    global datos_temporales
    datos_temporales = df_temporal
    
    # Backup de modelos existentes si los hay
    if modelos:
        backup_modelos_exitosos()
    
    # Obtener categorías
    categorias = df_temporal[nivel.lower()].unique()
    categorias_exitosas = 0
    categorias_mejoradas = 0
    categorias_preservadas = 0
    
    print(f"   📊 Procesando {len(categorias)} categorías...")
    
    for i, categoria in enumerate(categorias, 1):
        if modo_debug and i % 5 == 0:
            print(f"   📈 Progreso: {i}/{len(categorias)}")
        
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        modelo_anterior = modelos_backup.get(categoria)
        
        # Entrenar nuevo modelo
        resultado = entrenar_modelo_individual_mejorado(df_cat, categoria)
        
        if resultado:
            # Decidir si mantener nuevo o anterior
            modelo_final = restaurar_modelo_si_empeora(categoria, resultado, modelo_anterior)
            
            modelos[categoria] = modelo_final
            metricas[categoria] = modelo_final['metricas']
            
            if modelo_final == resultado:
                if modelo_anterior:
                    categorias_mejoradas += 1
                else:
                    categorias_exitosas += 1
            else:
                categorias_preservadas += 1
        
        elif modelo_anterior:
            # Si no se pudo entrenar nuevo pero había uno anterior bueno
            modelos[categoria] = modelo_anterior
            metricas[categoria] = modelo_anterior['metricas']
            categorias_preservadas += 1
            if modo_debug:
                print(f"      🔒 {categoria}: Preservando modelo anterior exitoso")
    
    total_modelos = len(modelos)
    print(f"\n✅ ENTRENAMIENTO MEJORADO COMPLETO:")
    print(f"   🆕 Modelos nuevos exitosos: {categorias_exitosas}")
    print(f"   ⬆️ Modelos mejorados: {categorias_mejoradas}")
    print(f"   🔒 Modelos preservados: {categorias_preservadas}")
    print(f"   📊 Total modelos finales: {total_modelos}/{len(categorias)} ({total_modelos/len(categorias)*100:.1f}%)")
    
    return total_modelos

# ============================================================================
# FUNCIONES DE GUARDADO Y CARGA
# ============================================================================

def guardar_modelos(ruta_guardado):
    """
    Guarda todos los modelos entrenados
    """
    print(f"💾 Guardando modelos en {ruta_guardado}...")
    
    # Crear directorio
    os.makedirs(ruta_guardado, exist_ok=True)
    
    # Guardar modelos individuales
    for categoria, modelo_info in modelos.items():
        # Guardar modelos
        joblib.dump(modelo_info['modelos'], f"{ruta_guardado}/modelo_{categoria.replace(' ', '_')}.pkl")
        
        # Guardar features
        with open(f"{ruta_guardado}/features_{categoria.replace(' ', '_')}.json", 'w') as f:
            json.dump(modelo_info['feature_cols'], f)
    
    # Guardar metadata
    metadata = {
        'modelos': {cat: info['metricas'] for cat, info in modelos.items()},
        'total_modelos': len(modelos),
        'fecha_entrenamiento': datetime.now().isoformat(),
        'version': '2.0_sin_clases'
    }
    
    with open(f"{ruta_guardado}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Modelos guardados exitosamente")
    print(f"   📁 Archivos creados: {len(os.listdir(ruta_guardado))}")

def cargar_modelos(ruta_modelos):
    """
    Carga modelos previamente guardados
    """
    global modelos
    
    print(f"📂 Cargando modelos desde {ruta_modelos}...")
    
    if not os.path.exists(f"{ruta_modelos}/metadata.json"):
        print("   ❌ No se encontró metadata.json")
        return 0
    
    # Cargar metadata
    with open(f"{ruta_modelos}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Cargar modelos
    modelos_cargados = 0
    for categoria, metricas_cat in metadata['modelos'].items():
        try:
            # Cargar modelo
            ruta_modelo = f"{ruta_modelos}/modelo_{categoria.replace(' ', '_')}.pkl"
            if os.path.exists(ruta_modelo):
                modelos_cargados += 1
                modelos[categoria] = {
                    'modelos': joblib.load(ruta_modelo),
                    'metricas': metricas_cat,
                    'feature_cols': []  # Se cargarán cuando sea necesario
                }
        except Exception as e:
            print(f"   ⚠️ Error cargando {categoria}: {e}")
            continue
    
    print(f"   ✅ {modelos_cargados} modelos cargados")
    return modelos_cargados

def predecir_todas_categorias(meses_futuro=1):
    """
    Genera predicciones para todas las categorías
    """
    print(f"🔮 Generando predicciones para {meses_futuro} mes(es) futuro...")
    
    if not modelos:
        print("❌ No hay modelos entrenados")
        return pd.DataFrame()
    
    predicciones = []
    
    for categoria, modelo_info in modelos.items():
        try:
            df_cat = datos_temporales[datos_temporales['categoria'] == categoria]
            if len(df_cat) == 0:
                continue
            
            # Crear features para el último período
            df_features = crear_features_temporales_mejoradas(df_cat)
            ultima_fila = df_features.iloc[-1:]
            
            # Crear features para predicción futura
            feature_cols = modelo_info['feature_cols']
            X_pred = ultima_fila[feature_cols].iloc[[0]]
            
            # Predecir
            pred_gasto = modelo_info['modelos']['gasto'].predict(X_pred)[0]
            pred_cantidad = modelo_info['modelos']['cantidad'].predict(X_pred)[0]
            
            # Fecha futura
            ultima_fecha = df_cat['fecha'].max()
            fecha_pred = ultima_fecha + pd.DateOffset(months=meses_futuro)
            
            predicciones.append({
                'categoria': categoria,
                'fecha_prediccion': fecha_pred,
                'gasto_predicho': max(0, pred_gasto),  # No negativos
                'transacciones_predichas': max(0, pred_cantidad),
                'r2_gasto': modelo_info['metricas']['r2_gasto'],
                'r2_cantidad': modelo_info['metricas']['r2_cantidad']
            })
            
        except Exception as e:
            if modo_debug:
                print(f"   ⚠️ Error prediciendo {categoria}: {e}")
            continue
    
    df_predicciones = pd.DataFrame(predicciones)
    
    if len(df_predicciones) > 0:
        print(f"   ✅ {len(df_predicciones)} predicciones generadas")
        print(f"   💰 Total predicho: ${df_predicciones['gasto_predicho'].sum():,.2f}")
    
    return df_predicciones

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def mejorar_modelo_existente(ruta_excel, ruta_modelos_anterior=None):
    """
    Función principal para mejorar el modelo existente (SIN CLASES)
    """
    print("🔧 MEJORANDO MODELO EXISTENTE - GRUPO SALINAS")
    print("="*60)
    
    # Inicializar variables globales
    inicializar_variables_globales(modo_debug=True)
    
    # Cargar modelos anteriores si existen
    if ruta_modelos_anterior and os.path.exists(f"{ruta_modelos_anterior}/metadata.json"):
        print("📂 Cargando modelos anteriores para preservar los exitosos...")
        try:
            num_cargados = cargar_modelos(ruta_modelos_anterior)
            print(f"   ✅ {num_cargados} modelos anteriores cargados")
        except Exception as e:
            print(f"   ⚠️ No se pudieron cargar modelos anteriores: {e}")
    
    try:
        # 1. Cargar datos con mejoras
        df = cargar_y_mejorar_datos(ruta_excel, "Detalle")
        
        # 2. Entrenar con preservación de modelos exitosos
        total_modelos = entrenar_todos_modelos_mejorado(df, 'CATEGORIA')
        
        if total_modelos == 0:
            print("❌ No se pudieron entrenar modelos")
            return None, pd.DataFrame()
        
        # 3. Análisis de calidad
        r2_gastos = [m['metricas']['r2_gasto'] for m in modelos.values()]
        r2_cantidades = [m['metricas']['r2_cantidad'] for m in modelos.values()]
        
        # Filtrar R² extremos para estadísticas
        r2_gastos_filtrados = [r2 for r2 in r2_gastos if r2 > -10]
        r2_cantidades_filtrados = [r2 for r2 in r2_cantidades if r2 > -10]
        
        print(f"\n📊 CALIDAD DE MODELOS MEJORADOS:")
        print(f"   💰 R² Gasto promedio: {np.mean(r2_gastos_filtrados):.3f}")
        print(f"   📈 R² Cantidad promedio: {np.mean(r2_cantidades_filtrados):.3f}")
        print(f"   ✅ Modelos con R² > 0.5 (Gasto): {sum(1 for r2 in r2_gastos if r2 > 0.5)}")
        print(f"   ✅ Modelos con R² > 0.3 (Cantidad): {sum(1 for r2 in r2_cantidades if r2 > 0.3)}")
        
        # 4. Generar predicciones
        df_predicciones = predecir_todas_categorias(1)
        
        # 5. Guardar modelos mejorados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        ruta_guardado = f"modelos_mejorados_{timestamp}"
        guardar_modelos(ruta_guardado)
        
        print(f"\n✅ MEJORAS COMPLETADAS:")
        print(f"   📁 Modelos guardados en: {ruta_guardado}")
        print(f"   🎯 Total modelos: {total_modelos}")
        print(f"   💰 Predicción total: ${df_predicciones['gasto_predicho'].sum():,.2f}")
        
        return modelos, df_predicciones
        
    except Exception as e:
        print(f"❌ Error en el proceso de mejora: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, pd.DataFrame()

# ============================================================================
# EJEMPLO DE USO EN JUPYTER
# ============================================================================

# Para usar las mejoras en tu notebook:
"""
# CELDA 1: Inicializar y ejecutar
modelos_entrenados, predicciones = mejorar_modelo_existente(
    "Compras/Compras Totales Ene - Dic '24 (20).xlsx",
    "modelos_entrenados20250919_1419"  # Ruta de tus modelos anteriores (opcional)
)

# CELDA 2: Ver predicciones
if len(predicciones) > 0:
    print("\n🎯 TOP 10 PREDICCIONES:")
    print(predicciones.nlargest(10, 'gasto_predicho')[['categoria', 'gasto_predicho', 'r2_gasto']])
    
    # Gráfico rápido
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    top_categorias = predicciones.nlargest(10, 'gasto_predicho')['categoria'].values
    top_predicciones = predicciones[predicciones['categoria'].isin(top_categorias)]['gasto_predicho']
    plt.barh(top_categorias, top_predicciones)
    plt.title('Top 10 Categorías por Gasto Predicho')
    plt.xlabel('Gasto Predicho ($)')
    plt.tight_layout()
    plt.show()
"""


# Copia todo el código de arriba aquí
# Luego ejecuta:
modelos_entrenados, predicciones = mejorar_modelo_existente(
    "ruta/a/tu/archivo.xlsx"
)

# Ver las predicciones
print(f"Total predicho: ${predicciones['gasto_predicho'].sum():,.2f}")
predicciones.head(10)
