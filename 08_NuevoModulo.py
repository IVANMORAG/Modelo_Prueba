#!/usr/bin/env python3
"""
MÓDULO DE RESCATE PARA CATEGORÍAS PROBLEMÁTICAS
Este módulo solo se activa para categorías con bajo rendimiento
NO MODIFICA las categorías que ya funcionan bien

Agregar este código AL FINAL del script original, antes de ejecutar_sistema_completo()
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, StackingRegressor
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
# CONFIGURACIÓN DE UMBRALES PARA RESCATE
# =============================================================================

UMBRAL_GASTO_MALO = 0.4  # R² menor a esto activa rescate para gasto
UMBRAL_CANTIDAD_MALO = 0.6  # R² menor a esto activa rescate para cantidad
CATEGORIAS_PROBLEMATICAS = [
    'EPC Propio', 
    'Gestión y Comercialización (G&C)',
    'O&M Terceros',
    'Construcción ',  # Con espacio
    'Transportes'
]

# =============================================================================
# TÉCNICAS DE RESCATE AVANZADAS
# =============================================================================

def aplicar_transformacion_potencia(df, columnas_numericas):
    """
    Aplica transformación de potencia para normalizar distribuciones sesgadas
    Útil para datos con outliers o distribuciones no normales
    """
    df_transformed = df.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    
    for col in columnas_numericas:
        if col in df.columns:
            try:
                valores = df[col].values.reshape(-1, 1)
                df_transformed[f'{col}_power'] = pt.fit_transform(valores).flatten()
            except:
                pass
    
    return df_transformed

def crear_features_interaccion(df, top_features, max_interacciones=10):
    """
    Crea features de interacción entre las más importantes
    Captura relaciones no lineales
    """
    df_interact = df.copy()
    
    if len(top_features) < 2:
        return df_interact
    
    # Limitar a top features disponibles
    available_features = [f for f in top_features if f in df.columns][:5]
    
    contador = 0
    for i, feat1 in enumerate(available_features):
        for feat2 in available_features[i+1:]:
            if contador >= max_interacciones:
                break
            
            # Multiplicación
            df_interact[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            # División (evitar división por cero)
            denominador = df[feat2].replace(0, 1e-10)
            df_interact[f'{feat1}_div_{feat2}'] = df[feat1] / denominador
            
            contador += 2
    
    return df_interact

def reducir_dimensionalidad_adaptativa(X, varianza_objetivo=0.95):
    """
    Reduce dimensionalidad solo si hay muchas features
    Mantiene el % de varianza especificado
    """
    if X.shape[1] <= 15:
        return X, None
    
    try:
        pca = PCA(n_components=varianza_objetivo, random_state=42)
        X_reduced = pca.fit_transform(X)
        print(f"      PCA: {X.shape[1]} → {X_reduced.shape[1]} features (varianza: {pca.explained_variance_ratio_.sum():.3f})")
        return X_reduced, pca
    except:
        return X, None

def crear_features_estacionales_fuertes(df):
    """
    Crea features estacionales más agresivos para capturar patrones temporales
    """
    df_seasonal = df.copy()
    
    if 'mes' in df.columns:
        # Codificación cíclica más compleja
        df_seasonal['mes_sin_2'] = np.sin(4 * np.pi * df['mes'] / 12)
        df_seasonal['mes_cos_2'] = np.cos(4 * np.pi * df['mes'] / 12)
        df_seasonal['mes_sin_3'] = np.sin(6 * np.pi * df['mes'] / 12)
        df_seasonal['mes_cos_3'] = np.cos(6 * np.pi * df['mes'] / 12)
        
        # Indicadores de meses específicos
        df_seasonal['es_enero'] = (df['mes'] == 1).astype(int)
        df_seasonal['es_diciembre'] = (df['mes'] == 12).astype(int)
        df_seasonal['es_junio'] = (df['mes'] == 6).astype(int)
        
        # Cuadrantes del año
        df_seasonal['cuadrante_año'] = pd.cut(df['mes'], bins=[0, 3, 6, 9, 12], labels=[1,2,3,4])
        df_seasonal['cuadrante_año'] = df_seasonal['cuadrante_año'].astype(int)
    
    if 'trimestre' in df.columns:
        # One-hot encoding de trimestre
        for t in range(1, 5):
            df_seasonal[f'trim_{t}'] = (df['trimestre'] == t).astype(int)
    
    return df_seasonal

def detectar_patron_intermitente(y):
    """
    Detecta si la serie tiene patrón intermitente (muchos ceros)
    """
    prop_ceros = (y == 0).sum() / len(y)
    return prop_ceros > 0.3  # Más de 30% ceros = intermitente

def modelo_para_intermitentes(X_train, y_train, X_test, y_test):
    """
    Modelo especializado para series con compras intermitentes
    Usa enfoque de dos etapas: clasificación + regresión
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Etapa 1: Predecir si habrá compra (clasificación)
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train_binary)
    
    prob_compra = clf.predict_proba(X_test)[:, 1]
    
    # Etapa 2: Predecir monto dado que hay compra (regresión)
    mask_train = y_train > 0
    if mask_train.sum() > 5:  # Mínimo 5 muestras con compra
        X_train_nonzero = X_train[mask_train]
        y_train_nonzero = y_train[mask_train]
        
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        reg.fit(X_train_nonzero, y_train_nonzero)
        
        # Predicción final: prob_compra * monto_predicho
        monto_pred = reg.predict(X_test)
        y_pred = prob_compra * monto_pred
    else:
        # Si muy pocas compras, usar solo promedio
        y_pred = prob_compra * y_train[y_train > 0].mean()
    
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'r2': r2, 'mae': mae, 'model': (clf, reg if 'reg' in locals() else None)}

def modelo_stacking_robusto(X_train, y_train, X_test, y_test, modelos_base):
    """
    Crea un stacking ensemble robusto con los mejores modelos base
    """
    from sklearn.linear_model import Ridge
    
    # Filtrar modelos que funcionaron
    estimators = []
    for name, model in modelos_base.items():
        if model is not None and hasattr(model, 'predict'):
            estimators.append((name, model))
    
    if len(estimators) < 2:
        return None
    
    try:
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3
        )
        
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)
        
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {'r2': r2, 'mae': mae, 'model': stacking}
    except:
        return None

# =============================================================================
# FUNCIÓN PRINCIPAL DE RESCATE
# =============================================================================

def rescatar_categoria_problematica(df_categoria, nombre_categoria, modelo_anterior=None):
    """
    Función principal que aplica técnicas de rescate a categorías problemáticas
    """
    print(f"\n MODO RESCATE activado para: {nombre_categoria}")
    print("="*70)
    
    # 1. DIAGNÓSTICO
    es_intermitente = detectar_patron_intermitente(df_categoria['gasto_total'])
    print(f"   Patrón intermitente: {'SÍ' if es_intermitente else 'NO'}")
    print(f"   Datos disponibles: {len(df_categoria)} meses")
    
    # 2. CREAR FEATURES BASE (reutilizar función existente)
    df_features = crear_features_avanzados(df_categoria, nombre_categoria)
    
    # 3. APLICAR TÉCNICAS DE RESCATE
    print("   Aplicando técnicas de rescate...")
    
    # 3.1 Transformación de potencia para normalizar
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['fecha', 'año', 'mes', 'trimestre']]
    df_features = aplicar_transformacion_potencia(df_features, numeric_cols[:10])  # Top 10
    
    # 3.2 Features estacionales fuertes
    df_features = crear_features_estacionales_fuertes(df_features)
    
    # 3.3 Features de interacción
    top_features = [
        'gasto_total_lag_1', 'gasto_total_ma_3', 'num_transacciones_lag_1',
        'gasto_promedio_lag_1', 'mes_sin', 'mes_cos'
    ]
    df_features = crear_features_interaccion(df_features, top_features)
    
    # 4. PREPARAR DATOS
    exclude_cols = [
        'categoria', 'familia', 'fecha', 'año_mes', 'gasto_total',
        'num_transacciones', 'ordenes_unicas', 'centros_costo',
        'solicitantes_unicos', 'gasto_promedio', 'gasto_std', 'año'
    ]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    y_gasto = df_features['gasto_total']
    y_cantidad = df_features['num_transacciones']
    
    # Split temporal
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
    y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
    
    # 5. REDUCCIÓN DE DIMENSIONALIDAD SI ES NECESARIO
    X_train_red, pca = reducir_dimensionalidad_adaptativa(X_train)
    X_test_red = pca.transform(X_test) if pca else X_test
    
    # 6. ENTRENAR MODELOS DE RESCATE
    resultados_rescate = {}
    modelos_rescate = {}
    
    # 6.1 GASTO
    print("   Entrenando modelos de rescate para GASTO...")
    
    if es_intermitente:
        # Usar modelo especializado para intermitentes
        resultado_gasto = modelo_para_intermitentes(
            X_train_red, y_gasto_train, X_test_red, y_gasto_test
        )
        if resultado_gasto and resultado_gasto['r2'] > 0.3:
            modelos_rescate['gasto_intermitente'] = resultado_gasto['model']
            resultados_rescate['gasto'] = {
                'r2': resultado_gasto['r2'],
                'mae': resultado_gasto['mae']
            }
            print(f"      Modelo intermitente: R² = {resultado_gasto['r2']:.3f}")
    
    # Siempre intentar modelos robustos adicionales
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor
    
    modelos_base = {}
    
    # Extra Trees (robusto a outliers)
    try:
        et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train_red, y_gasto_train)
        y_pred = et.predict(X_test_red)
        r2 = r2_score(y_gasto_test, y_pred)
        if r2 > 0.2:
            modelos_base['et_robusto'] = et
            print(f"      Extra Trees robusto: R² = {r2:.3f}")
    except:
        pass
    
    # Huber Regressor (robusto a outliers)
    try:
        huber = HuberRegressor(epsilon=1.5, max_iter=200)
        huber.fit(X_train_red, y_gasto_train)
        y_pred = huber.predict(X_test_red)
        r2 = r2_score(y_gasto_test, y_pred)
        if r2 > 0.2:
            modelos_base['huber'] = huber
            print(f"      Huber Regressor: R² = {r2:.3f}")
    except:
        pass
    
    # Gradient Boosting con parámetros conservadores
    try:
        gb_robust = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        gb_robust.fit(X_train_red, y_gasto_train)
        y_pred = gb_robust.predict(X_test_red)
        r2 = r2_score(y_gasto_test, y_pred)
        if r2 > 0.2:
            modelos_base['gb_robust'] = gb_robust
            print(f"      GB robusto: R² = {r2:.3f}")
    except:
        pass
    
    # Stacking de modelos base
    if len(modelos_base) >= 2:
        resultado_stack = modelo_stacking_robusto(
            X_train_red, y_gasto_train, X_test_red, y_gasto_test, modelos_base
        )
        if resultado_stack and resultado_stack['r2'] > resultados_rescate.get('gasto', {}).get('r2', -999):
            modelos_rescate['gasto_stack'] = resultado_stack['model']
            resultados_rescate['gasto'] = {
                'r2': resultado_stack['r2'],
                'mae': resultado_stack['mae']
            }
            print(f"      Stacking: R² = {resultado_stack['r2']:.3f} ⭐")
    
    # 6.2 CANTIDAD (misma lógica)
    print("   Entrenando modelos de rescate para CANTIDAD...")
    
    if es_intermitente:
        resultado_cantidad = modelo_para_intermitentes(
            X_train_red, y_cantidad_train, X_test_red, y_cantidad_test
        )
        if resultado_cantidad and resultado_cantidad['r2'] > 0.3:
            modelos_rescate['cantidad_intermitente'] = resultado_cantidad['model']
            resultados_rescate['cantidad'] = {
                'r2': resultado_cantidad['r2'],
                'mae': resultado_cantidad['mae']
            }
            print(f"      Modelo intermitente: R² = {resultado_cantidad['r2']:.3f}")
    
    # Modelos adicionales para cantidad
    try:
        et_cant = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        et_cant.fit(X_train_red, y_cantidad_train)
        y_pred = et_cant.predict(X_test_red)
        r2 = r2_score(y_cantidad_test, y_pred)
        if r2 > resultados_rescate.get('cantidad', {}).get('r2', -999):
            modelos_rescate['cantidad_et'] = et_cant
            resultados_rescate['cantidad'] = {'r2': r2, 'mae': mean_absolute_error(y_cantidad_test, y_pred)}
            print(f"      Extra Trees cantidad: R² = {r2:.3f}")
    except:
        pass
    
    # 7. RETORNAR RESULTADO
    if not resultados_rescate:
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
        'datos_usados': len(df_features)
    }

# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

def entrenar_categoria_avanzada_con_rescate(df_categoria, nombre_categoria, verbose=True):
    """
    Versión mejorada que primero intenta entrenamiento normal,
    y si falla o da mal resultado, activa el modo rescate
    """
    # PASO 1: Intentar entrenamiento normal
    resultado_normal = entrenar_categoria_avanzada(df_categoria, nombre_categoria, verbose=False)
    
    # PASO 2: Evaluar si necesita rescate
    necesita_rescate = False
    
    if resultado_normal is None:
        necesita_rescate = True
        razon = "entrenamiento normal falló"
    elif nombre_categoria in CATEGORIAS_PROBLEMATICAS:
        necesita_rescate = True
        razon = "categoría en lista problemática"
    else:
        # Verificar métricas
        r2_gasto = resultado_normal.get('resultados', {}).get('gasto', {}).get('r2', -999)
        r2_cantidad = resultado_normal.get('resultados', {}).get('cantidad', {}).get('r2', -999)
        
        if r2_gasto < UMBRAL_GASTO_MALO or r2_cantidad < UMBRAL_CANTIDAD_MALO:
            necesita_rescate = True
            razon = f"R² bajo (gasto={r2_gasto:.3f}, cantidad={r2_cantidad:.3f})"
    
    # PASO 3: Aplicar rescate si es necesario
    if necesita_rescate:
        if verbose:
            print(f"\n ⚠️  {nombre_categoria} necesita rescate: {razon}")
        
        resultado_rescate = rescatar_categoria_problematica(df_categoria, nombre_categoria, resultado_normal)
        
        if resultado_rescate is not None:
            # Usar resultado de rescate
            return resultado_rescate
        elif resultado_normal is not None:
            # Rescate falló, usar resultado normal aunque sea malo
            if verbose:
                print(f"   Usando resultado normal (rescate no mejoró)")
            return resultado_normal
        else:
            # Todo falló
            return None
    else:
        # No necesita rescate, usar resultado normal
        if verbose:
            print(f" {nombre_categoria}: OK, no necesita rescate")
        return resultado_normal

# =============================================================================
# REEMPLAZAR FUNCIÓN EN FLUJO PRINCIPAL
# =============================================================================

def ejecutar_entrenamiento_avanzado_con_rescate(df, nivel='CATEGORIA'):
    """
    Versión mejorada de ejecutar_entrenamiento_avanzado que incluye rescate
    """
    print(f"\n INICIANDO ENTRENAMIENTO AVANZADO CON RESCATE AUTOMÁTICO")
    print("="*80)
    
    # Crear series temporales (igual que antes)
    df_temporal = crear_series_temporales(df, nivel)
    
    categorias = df_temporal[nivel.lower()].unique()
    print(f" Categorías a procesar: {len(categorias)}")
    print(f" Categorías en lista problemática: {len([c for c in categorias if c in CATEGORIAS_PROBLEMATICAS])}")
    
    modelos_exitosos = {}
    resultados_globales = []
    
    for i, categoria in enumerate(categorias):
        print(f"\n Progreso: {i+1}/{len(categorias)} - {categoria}")
        
        df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
        df_cat = df_cat.sort_values('fecha')
        
        # USAR NUEVA FUNCIÓN CON RESCATE
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

# =============================================================================
# INSTRUCCIONES DE USO
# =============================================================================
"""
CÓMO INTEGRAR ESTE MÓDULO:

1. Agregar este código AL FINAL de tu script original (antes de la sección de ejemplo de uso)

2. REEMPLAZAR la llamada original:
   
   # ANTES:
   modelos, resultados = ejecutar_entrenamiento_avanzado(df)
   
   # DESPUÉS:
   modelos, resultados = ejecutar_entrenamiento_avanzado_con_rescate(df)

3. Todo lo demás permanece igual. El sistema:
   - Detecta automáticamente categorías problemáticas
   - Aplica rescate solo donde se necesita
   - Mantiene intactos los buenos resultados existentes

EJEMPLO COMPLETO:

# Cargar datos
df = cargar_y_procesar_datos_avanzado("tu_archivo.csv")

# Entrenar CON rescate automático
modelos, resultados = ejecutar_entrenamiento_avanzado_con_rescate(df)

# Resto del código igual...
mostrar_resumen_avanzado(resultados)
predicciones = generar_predicciones_avanzadas(modelos, meses_prediccion=1)
"""
