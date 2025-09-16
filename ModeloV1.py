#!/usr/bin/env python3
"""
Modelo de Predicción de Compras - Grupo Salinas
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

# Librerías para Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Librerías para Deep Learning (opcional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow no disponible. Solo se usarán modelos clásicos.")
    TENSORFLOW_AVAILABLE = False

# Librerías para Series Temporales
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️ Prophet no disponible. Solo se usarán modelos clásicos.")
    PROPHET_AVAILABLE = False

# Utilidades
from datetime import datetime, timedelta
import json
import os

class PredictorComprasSalinas:
    """
    Clase principal para predecir compras de Grupo Salinas
    """
    
    def __init__(self, modo_debug=True):
        self.modelos = {}
        self.datos_temporales = {}
        self.categorias = []
        self.familias = []
        self.modo_debug = modo_debug
        self.metricas = {}
        
    def cargar_y_limpiar_datos(self, ruta_excel, sheet_name="Detalle"):
        """
        Carga y limpia los datos de compras
        """
        print("📂 Cargando datos...")
        
        # Cargar datos
        df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
        print(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # Limpiar fechas
        df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
        
        # Filtrar datos válidos
        df = df.dropna(subset=['FECHAPEDIDO', 'TOTALPESOS'])
        df = df[df['TOTALPESOS'] > 0]  # Solo montos positivos
        
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
        
        print(f"🧹 Datos limpios: {len(df)} registros válidos")
        print(f"📅 Período: {df['FECHAPEDIDO'].min()} a {df['FECHAPEDIDO'].max()}")
        print(f"💰 Total gastado: ${df['TOTALPESOS'].sum():,.2f}")
        
        return df
    
    def crear_series_temporales(self, df, nivel='CATEGORIA'):
        """
        Convierte datos transaccionales en series temporales
        """
        print(f"📊 Creando series temporales por {nivel}...")
        
        # Agregación mensual
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
        df_completo = []
        for categoria in df_agregado[nivel.lower()].unique():
            df_cat = df_agregado[df_agregado[nivel.lower()] == categoria].copy()
            
            # Crear rango completo de fechas
            fecha_min = df_cat['fecha'].min()
            fecha_max = df_cat['fecha'].max()
            fechas_completas = pd.date_range(fecha_min, fecha_max, freq='MS')
            
            # Reindexar
            df_cat = df_cat.set_index('fecha').reindex(fechas_completas)
            df_cat[nivel.lower()] = categoria
            df_cat = df_cat.fillna(0)  # Llenar huecos con 0
            df_cat['fecha'] = df_cat.index
            
            df_completo.append(df_cat)
        
        df_final = pd.concat(df_completo, ignore_index=True)
        
        print(f"✅ Series creadas: {len(df_final)} puntos para {df_final[nivel.lower()].nunique()} {nivel.lower()}s")
        return df_final
    
    def crear_features_temporales(self, df_serie):
        """
        Crea features de ingeniería temporal
        """
        df = df_serie.copy()
        
        # Lags (valores pasados)
        for lag in [1, 2, 3, 6, 12]:
            df[f'gasto_lag_{lag}'] = df['gasto_total'].shift(lag)
            df[f'transacciones_lag_{lag}'] = df['num_transacciones'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'gasto_rolling_mean_{window}'] = df['gasto_total'].rolling(window=window).mean()
            df[f'gasto_rolling_std_{window}'] = df['gasto_total'].rolling(window=window).std()
            df[f'transacciones_rolling_mean_{window}'] = df['num_transacciones'].rolling(window=window).mean()
        
        # Features estacionales
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
        df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
        
        # Tendencia
        df['tendencia'] = range(len(df))
        
        # Ratios y cambios
        df['gasto_cambio_mes'] = df['gasto_total'].pct_change()
        df['transacciones_cambio_mes'] = df['num_transacciones'].pct_change()
        
        # Features de momento (para predecir cuándo comprar)
        df['dias_sin_compra'] = (df['num_transacciones'] == 0).astype(int)
        df['dias_sin_compra_acumulados'] = df['dias_sin_compra'].cumsum()
        
        # Eliminar infinitos y NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def entrenar_modelo_individual(self, df_categoria, nombre_categoria, target='gasto_total'):
        """
        Entrena modelo para una categoría específica
        """
        if len(df_categoria) < 12:  # Menos de 1 año de datos
            print(f"⚠️ Pocos datos para {nombre_categoria}: {len(df_categoria)} meses")
            return None
        
        # Crear features
        df_features = self.crear_features_temporales(df_categoria)
        
        # Eliminar filas con muchos NaN (por los lags)
        df_features = df_features.dropna()
        
        if len(df_features) < 6:
            print(f"⚠️ Muy pocos datos después de crear features para {nombre_categoria}")
            return None
        
        # Definir features y targets
        exclude_cols = ['categoria', 'familia', 'fecha', 'año_mes', 'gasto_total', 
                       'num_transacciones', 'ordenes_unicas']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols]
        y_gasto = df_features['gasto_total']
        y_cantidad = df_features['num_transacciones']
        y_momento = df_features['dias_sin_compra']
        
        # Split temporal (80% entrenamiento, 20% prueba)
        split_idx = int(len(X) * 0.8)
        if split_idx < 3:  # Mínimo 3 puntos para entrenar
            print(f"⚠️ Datos insuficientes para split en {nombre_categoria}")
            return None
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_gasto_train, y_gasto_test = y_gasto.iloc[:split_idx], y_gasto.iloc[split_idx:]
        y_cantidad_train, y_cantidad_test = y_cantidad.iloc[:split_idx], y_cantidad.iloc[split_idx:]
        y_momento_train, y_momento_test = y_momento.iloc[:split_idx], y_momento.iloc[split_idx:]
        
        # Modelos
        modelos = {
            'gasto': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'cantidad': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'momento': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        # Entrenar modelos
        modelos['gasto'].fit(X_train, y_gasto_train)
        modelos['cantidad'].fit(X_train, y_cantidad_train) 
        modelos['momento'].fit(X_train, y_momento_train)
        
        # Evaluar
        pred_gasto = modelos['gasto'].predict(X_test)
        pred_cantidad = modelos['cantidad'].predict(X_test)
        pred_momento = modelos['momento'].predict(X_test)
        
        # Métricas
        metricas = {
            'mae_gasto': mean_absolute_error(y_gasto_test, pred_gasto),
            'r2_gasto': r2_score(y_gasto_test, pred_gasto),
            'mae_cantidad': mean_absolute_error(y_cantidad_test, pred_cantidad),
            'r2_cantidad': r2_score(y_cantidad_test, pred_cantidad),
            'mae_momento': mean_absolute_error(y_momento_test, pred_momento),
            'r2_momento': r2_score(y_momento_test, pred_momento)
        }
        
        if self.modo_debug:
            print(f"🎯 {nombre_categoria}:")
            print(f"   💰 Gasto - MAE: ${metricas['mae_gasto']:,.0f}, R²: {metricas['r2_gasto']:.3f}")
            print(f"   📊 Cantidad - MAE: {metricas['mae_cantidad']:.1f}, R²: {metricas['r2_cantidad']:.3f}")
        
        return {
            'modelos': modelos,
            'feature_cols': feature_cols,
            'metricas': metricas,
            'datos_entrenamiento': df_features,
            'split_idx': split_idx
        }
    
    def entrenar_todos_modelos(self, df, nivel='CATEGORIA'):
        """
        Entrena modelos para todas las categorías/familias
        """
        print(f"🚀 Entrenando modelos por {nivel}...")
        
        # Crear series temporales
        df_temporal = self.crear_series_temporales(df, nivel)
        self.datos_temporales = df_temporal
        
        # Obtener categorías únicas
        categorias = df_temporal[nivel.lower()].unique()
        modelos_exitosos = 0
        
        for categoria in categorias:
            df_cat = df_temporal[df_temporal[nivel.lower()] == categoria].copy()
            
            resultado = self.entrenar_modelo_individual(df_cat, categoria)
            if resultado:
                self.modelos[categoria] = resultado
                self.metricas[categoria] = resultado['metricas']
                modelos_exitosos += 1
        
        print(f"✅ Entrenamiento completo: {modelos_exitosos}/{len(categorias)} modelos exitosos")
        
        if nivel == 'CATEGORIA':
            self.categorias = list(self.modelos.keys())
        else:
            self.familias = list(self.modelos.keys())
        
        return modelos_exitosos
    
    def predecir_categoria(self, categoria, meses_adelante=1):
        """
        Predice para una categoría específica
        """
        if categoria not in self.modelos:
            return {"error": f"No hay modelo entrenado para {categoria}"}
        
        modelo_info = self.modelos[categoria]
        
        # Obtener últimos datos
        df_cat = modelo_info['datos_entrenamiento']
        ultima_fila = df_cat.iloc[-1]
        
        # Crear features para predicción (simulando mes siguiente)
        features_pred = []
        for col in modelo_info['feature_cols']:
            if 'lag_1' in col:
                # Lag 1 = valor actual
                if 'gasto' in col:
                    features_pred.append(ultima_fila['gasto_total'])
                else:
                    features_pred.append(ultima_fila['num_transacciones'])
            elif 'lag_' in col:
                # Otros lags
                features_pred.append(ultima_fila[col])
            elif 'rolling' in col:
                # Rolling features
                features_pred.append(ultima_fila[col])
            elif 'tendencia' in col:
                # Tendencia + 1 mes
                features_pred.append(ultima_fila[col] + 1)
            elif col in ['mes_sin', 'mes_cos', 'trimestre_sin', 'trimestre_cos']:
                # Features estacionales (próximo mes)
                proximo_mes = (ultima_fila['mes'] % 12) + 1
                if 'mes_sin' in col:
                    features_pred.append(np.sin(2 * np.pi * proximo_mes / 12))
                elif 'mes_cos' in col:
                    features_pred.append(np.cos(2 * np.pi * proximo_mes / 12))
                else:
                    features_pred.append(ultima_fila[col])
            else:
                # Otros features
                features_pred.append(ultima_fila[col] if col in ultima_fila else 0)
        
        # Convertir a array
        X_pred = np.array(features_pred).reshape(1, -1)
        
        # Hacer predicciones
        pred_gasto = modelo_info['modelos']['gasto'].predict(X_pred)[0]
        pred_cantidad = modelo_info['modelos']['cantidad'].predict(X_pred)[0]
        pred_momento = modelo_info['modelos']['momento'].predict(X_pred)[0]
        
        # Calcular fecha próxima compra
        fecha_ultima = df_cat['fecha'].max()
        fecha_prediccion = fecha_ultima + timedelta(days=30 * meses_adelante)
        
        return {
            "categoria": categoria,
            "gasto_predicho": max(0, round(pred_gasto, 2)),
            "cantidad_predicha": max(0, round(pred_cantidad)),
            "momento_predicho": max(0, round(pred_momento)),
            "fecha_prediccion": fecha_prediccion.strftime("%Y-%m-%d"),
            "confianza_gasto": modelo_info['metricas']['r2_gasto'],
            "confianza_cantidad": modelo_info['metricas']['r2_cantidad'],
            "error_promedio_gasto": modelo_info['metricas']['mae_gasto']
        }
    
    def predecir_todas_categorias(self, meses_adelante=1):
        """
        Predice para todas las categorías
        """
        print(f"🔮 Generando predicciones para {meses_adelante} mes(es)...")
        
        resultados = []
        for categoria in self.modelos.keys():
            pred = self.predecir_categoria(categoria, meses_adelante)
            if "error" not in pred:
                resultados.append(pred)
        
        # Convertir a DataFrame para análisis
        df_pred = pd.DataFrame(resultados)
        
        # Resumen total
        total_gasto = df_pred['gasto_predicho'].sum()
        total_transacciones = df_pred['cantidad_predicha'].sum()
        confianza_promedio = df_pred['confianza_gasto'].mean()
        
        print(f"📊 RESUMEN PREDICCIONES:")
        print(f"   💰 Gasto total predicho: ${total_gasto:,.2f}")
        print(f"   📈 Transacciones totales: {total_transacciones:,.0f}")
        print(f"   🎯 Confianza promedio: {confianza_promedio:.3f}")
        
        return df_pred
    
    def guardar_modelos(self, ruta_base="modelo_salinas"):
        """
        Guarda todos los modelos y metadatos
        """
        print("💾 Guardando modelos...")
        
        # Crear directorio si no existe
        os.makedirs(ruta_base, exist_ok=True)
        
        # Guardar cada modelo
        for categoria, modelo_info in self.modelos.items():
            categoria_clean = categoria.replace(" ", "_").replace("/", "_")
            
            # Guardar modelos individuales
            joblib.dump(modelo_info['modelos']['gasto'], 
                       f"{ruta_base}/{categoria_clean}_gasto.pkl")
            joblib.dump(modelo_info['modelos']['cantidad'], 
                       f"{ruta_base}/{categoria_clean}_cantidad.pkl")
            joblib.dump(modelo_info['modelos']['momento'], 
                       f"{ruta_base}/{categoria_clean}_momento.pkl")
            
            # Guardar features
            joblib.dump(modelo_info['feature_cols'], 
                       f"{ruta_base}/{categoria_clean}_features.pkl")
        
        # Guardar metadatos generales
        metadata = {
            'categorias': list(self.modelos.keys()),
            'fecha_entrenamiento': datetime.now().isoformat(),
            'num_modelos': len(self.modelos),
            'metricas': self.metricas
        }
        
        with open(f"{ruta_base}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modelos guardados en '{ruta_base}/' para {len(self.modelos)} categorías")
    
    def cargar_modelos(self, ruta_base="modelo_salinas"):
        """
        Carga modelos previamente entrenados
        """
        print("📂 Cargando modelos...")
        
        # Cargar metadata
        with open(f"{ruta_base}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Cargar cada modelo
        for categoria in metadata['categorias']:
            categoria_clean = categoria.replace(" ", "_").replace("/", "_")
            
            try:
                modelos = {
                    'gasto': joblib.load(f"{ruta_base}/{categoria_clean}_gasto.pkl"),
                    'cantidad': joblib.load(f"{ruta_base}/{categoria_clean}_cantidad.pkl"),
                    'momento': joblib.load(f"{ruta_base}/{categoria_clean}_momento.pkl")
                }
                
                feature_cols = joblib.load(f"{ruta_base}/{categoria_clean}_features.pkl")
                
                self.modelos[categoria] = {
                    'modelos': modelos,
                    'feature_cols': feature_cols,
                    'metricas': metadata['metricas'].get(categoria, {})
                }
                
            except FileNotFoundError as e:
                print(f"⚠️ No se pudo cargar modelo para {categoria}: {e}")
        
        print(f"✅ Modelos cargados para {len(self.modelos)} categorías")
        return len(self.modelos)
    
    def generar_reporte_completo(self, df_predicciones=None):
        """
        Genera reporte completo de predicciones
        """
        print("\n" + "="*80)
        print("📋 REPORTE COMPLETO DE PREDICCIONES - GRUPO SALINAS")
        print("="*80)
        
        if df_predicciones is None:
            df_predicciones = self.predecir_todas_categorias()
        
        # Top 5 categorías por gasto
        top_gastos = df_predicciones.nlargest(5, 'gasto_predicho')
        print("\n🏆 TOP 5 CATEGORÍAS POR GASTO PREDICHO:")
        for _, row in top_gastos.iterrows():
            print(f"   {row['categoria']}: ${row['gasto_predicho']:,.2f}")
        
        # Top 5 por cantidad
        top_cantidad = df_predicciones.nlargest(5, 'cantidad_predicha')
        print("\n📊 TOP 5 CATEGORÍAS POR CANTIDAD DE TRANSACCIONES:")
        for _, row in top_cantidad.iterrows():
            print(f"   {row['categoria']}: {row['cantidad_predicha']:.0f} transacciones")
        
        # Estadísticas generales
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   Total categorías: {len(df_predicciones)}")
        print(f"   Gasto total predicho: ${df_predicciones['gasto_predicho'].sum():,.2f}")
        print(f"   Transacciones totales: {df_predicciones['cantidad_predicha'].sum():.0f}")
        print(f"   Gasto promedio por categoría: ${df_predicciones['gasto_predicho'].mean():,.2f}")
        print(f"   Confianza promedio: {df_predicciones['confianza_gasto'].mean():.3f}")
        
        return df_predicciones


def main_completo():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("🏢 MODELO DE PREDICCIÓN DE COMPRAS - GRUPO SALINAS")
    print("="*60)
    
    # Configuración
    RUTA_EXCEL = "Compras/Compras Totales Ene - Dic '24 (20).xlsx"  # Cambia por tu ruta
    SHEET_NAME = "Detalle"
    
    # Inicializar modelo
    predictor = PredictorComprasSalinas(modo_debug=True)
    
    try:
        # 1. Cargar y limpiar datos
        df = predictor.cargar_y_limpiar_datos(RUTA_EXCEL, SHEET_NAME)
        
        # 2. Entrenar modelos por categoría
        modelos_exitosos = predictor.entrenar_todos_modelos(df, nivel='CATEGORIA')
        
        if modelos_exitosos == 0:
            print("❌ No se pudo entrenar ningún modelo")
            return None
        
        # 3. Hacer predicciones
        df_predicciones = predictor.predecir_todas_categorias(meses_adelante=1)
        
        # 4. Generar reporte
        predictor.generar_reporte_completo(df_predicciones)
        
        # 5. Guardar modelos
        predictor.guardar_modelos("modelos_salinas_v1")
        
        print("\n✅ PROCESO COMPLETADO EXITOSAMENTE")
        
        return predictor, df_predicciones
        
    except Exception as e:
        print(f"❌ Error en el proceso: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Para ejecutar todo:
# predictor, predicciones = main_completo()

# Para usar modelos guardados:
# predictor = PredictorComprasSalinas()
# predictor.cargar_modelos("modelos_salinas_v1")
# nueva_prediccion = predictor.predecir_categoria("Tecnología")