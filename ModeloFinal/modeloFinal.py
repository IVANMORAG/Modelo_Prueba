"""
PIPELINE PREDICTIVO ROBUSTO V2 - ANÁLISIS DE COMPRAS
Dataset: archivo_listo.csv
Encoding: latin-1
Mejoras: Análisis por ORIGEN/UN, Alertas, Recomendaciones, Features Avanzados
"""

# ============================================================================
# 0. INSTALACIÓN Y CONFIGURACIÓN INICIAL
# ============================================================================
!pip install prophet xgboost scikit-learn matplotlib seaborn plotly joblib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import re
from scipy import stats
import joblib
from collections import defaultdict

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
import xgboost as xgb
from prophet import Prophet

# Visualización
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print(" Librerías cargadas correctamente")


# ============================================================================
# 1. CLASE GESTORA DEL PIPELINE MEJORADA
# ============================================================================
class GestorPredictivoV2:
    """Gestor completo del pipeline de análisis y predicción - Versión Mejorada"""

    def __init__(self, ruta_csv, directorio_salida='/content/modelos'):
        self.ruta_csv = ruta_csv
        self.directorio_salida = directorio_salida
        self.df_raw = None
        self.df_clean = None
        self.df_model = None
        self.modelos = {}
        self.resultados = {}
        self.encoders = {}
        self.analisis_origen = {}
        self.analisis_un = {}
        self.alertas = []
        self.recomendaciones = {}

        # Crear directorio para modelos
        import os
        os.makedirs(directorio_salida, exist_ok=True)

    def cargar_datos(self):
        """Carga robusta del dataset con manejo de errores"""
        print("\n" + "="*70)
        print(" CARGANDO DATOS")
        print("="*70)

        try:
            self.df_raw = pd.read_csv(self.ruta_csv, encoding='latin-1', low_memory=False)
            print(f" Datos cargados: {self.df_raw.shape[0]:,} filas x {self.df_raw.shape[1]} columnas")
            print(f" Tamaño en memoria: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Información básica
            print(f"\n Columnas: {list(self.df_raw.columns)}")
            print(f"\n Tipos de datos:\n{self.df_raw.dtypes.value_counts()}")

        except Exception as e:
            print(f" Error al cargar datos: {e}")
            raise

    def explorar_datos(self):
        """EDA inicial completo"""
        print("\n" + "="*70)
        print(" EXPLORACIÓN INICIAL")
        print("="*70)

        # Info general
        print("\n Resumen del DataFrame:")
        print(self.df_raw.info())

        # Estadísticas
        print("\n Estadísticas Descriptivas (columnas numéricas):")
        print(self.df_raw.describe())

        # Valores nulos
        nulos = self.df_raw.isnull().sum()
        nulos_pct = (nulos / len(self.df_raw) * 100).round(2)
        df_nulos = pd.DataFrame({'Nulos': nulos, '%': nulos_pct}).sort_values('%', ascending=False)
        print("\n Valores Nulos por Columna:")
        print(df_nulos[df_nulos['Nulos'] > 0].head(15))

        # Valores únicos en categóricas
        print("\n  Top 10 Valores Únicos en Columnas Clave:")
        cols_categoricas = ['ORIGEN', 'EMPRESA', 'UN', 'CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO']
        for col in cols_categoricas:
            if col in self.df_raw.columns:
                print(f"\n{col}:")
                print(self.df_raw[col].value_counts().head(10))

    def limpiar_datos(self):
        """Limpieza robusta del dataset"""
        print("\n" + "="*70)
        print(" LIMPIEZA DE DATOS")
        print("="*70)

        self.df_clean = self.df_raw.copy()

        # 1. Filtrar filas con demasiados "Desconocido"
        print("\n1 Filtrando filas con valores 'Desconocido'...")
        cols_importantes = ['DESTINO', 'CATEGORIA', 'CLASE', 'FAMILIA', 'DETALLEFINAL']
        mask = True
        for col in cols_importantes:
            if col in self.df_clean.columns:
                mask &= (self.df_clean[col] != 'Desconocido')

        filas_antes = len(self.df_clean)
        self.df_clean = self.df_clean[mask].copy()
        filas_despues = len(self.df_clean)
        print(f"    Eliminadas {filas_antes - filas_despues:,} filas ({(filas_antes - filas_despues)/filas_antes*100:.2f}%)")
        print(f"    Filas restantes: {filas_despues:,}")

        # 2. Convertir fechas
        print("\n2 Convirtiendo fechas...")
        for col in ['FECHAPEDIDO', 'FECHASOLICITUD']:
            if col in self.df_clean.columns:
                self.df_clean[col] = pd.to_datetime(self.df_clean[col], errors='coerce')
                nulos = self.df_clean[col].isnull().sum()
                print(f"   {col}: {len(self.df_clean) - nulos:,} válidas, {nulos:,} nulas")

        # 3. Limpiar TOTALPESOS
        print("\n3 Limpiando TOTALPESOS (outliers)...")
        if 'TOTALPESOS' in self.df_clean.columns:
            # Eliminar negativos y outliers extremos (>99.5 percentil)
            q995 = self.df_clean['TOTALPESOS'].quantile(0.995)
            mask_validos = (self.df_clean['TOTALPESOS'] > 0) & (self.df_clean['TOTALPESOS'] <= q995)
            filas_antes = len(self.df_clean)
            self.df_clean = self.df_clean[mask_validos].copy()
            print(f"    Eliminados {filas_antes - len(self.df_clean):,} outliers/negativos")
            print(f"    TOTALPESOS - Min: ${self.df_clean['TOTALPESOS'].min():,.2f}, Max: ${self.df_clean['TOTALPESOS'].max():,.2f}, Media: ${self.df_clean['TOTALPESOS'].mean():,.2f}")

        # 4. Extraer cantidad de DETALLEFINAL (MEJORADO)
        print("\n4 Extrayendo cantidades de DETALLEFINAL (regex mejorado)...")
        if 'DETALLEFINAL' in self.df_clean.columns:
            def extraer_cantidad_mejorado(texto):
                if pd.isna(texto):
                    return np.nan
                texto_str = str(texto)
                # Buscar patrones múltiples: "Cant.X", "Cant:X", "Cantidad:X", "Cant X"
                patrones = [
                    r'Cant[.:]?\s*(\d+)',
                    r'Cantidad[.:]?\s*(\d+)',
                    r'Qty[.:]?\s*(\d+)',
                    r'Q[.:]?\s*(\d+)',
                ]
                for patron in patrones:
                    match = re.search(patron, texto_str, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
                # Si no encuentra, asumir 1 (compra unitaria)
                return 1

            self.df_clean['CANTIDAD'] = self.df_clean['DETALLEFINAL'].apply(extraer_cantidad_mejorado)
            cant_validas = self.df_clean['CANTIDAD'].notna().sum()
            print(f"    Extraídas {cant_validas:,} cantidades ({cant_validas/len(self.df_clean)*100:.2f}% del total)")
            if cant_validas > 0:
                print(f"    Distribución de cantidades:")
                print(self.df_clean['CANTIDAD'].value_counts().head(10))

        # 5. Crear variables temporales
        print("\n5 Creando variables temporales...")
        if 'FECHAPEDIDO' in self.df_clean.columns:
            self.df_clean['AÑO'] = self.df_clean['FECHAPEDIDO'].dt.year
            self.df_clean['MES'] = self.df_clean['FECHAPEDIDO'].dt.month
            self.df_clean['TRIMESTRE'] = self.df_clean['FECHAPEDIDO'].dt.quarter
            self.df_clean['DIA_SEMANA'] = self.df_clean['FECHAPEDIDO'].dt.dayofweek
            self.df_clean['DIA_MES'] = self.df_clean['FECHAPEDIDO'].dt.day
            self.df_clean['SEMANA_AÑO'] = self.df_clean['FECHAPEDIDO'].dt.isocalendar().week
            print(f"    Variables temporales creadas: AÑO, MES, TRIMESTRE, DIA_SEMANA, DIA_MES, SEMANA_AÑO")
            print(f"    Rango de fechas: {self.df_clean['FECHAPEDIDO'].min()} a {self.df_clean['FECHAPEDIDO'].max()}")

        print(f"\n Limpieza completada. Dataset final: {len(self.df_clean):,} filas")

    def feature_engineering(self):
        """Crear features avanzados para modelado (MEJORADO)"""
        print("\n" + "="*70)
        print("  FEATURE ENGINEERING AVANZADO")
        print("="*70)

        self.df_model = self.df_clean.copy()

        # 1. Agregaciones por múltiples dimensiones
        print("\n1 Creando features agregados por dimensión...")

        # Por CATEGORIA
        if 'CATEGORIA' in self.df_model.columns and 'TOTALPESOS' in self.df_model.columns:
            gasto_cat = self.df_model.groupby('CATEGORIA')['TOTALPESOS'].agg(['mean', 'median', 'std', 'count']).reset_index()
            gasto_cat.columns = ['CATEGORIA', 'GASTO_PROM_CAT', 'GASTO_MED_CAT', 'GASTO_STD_CAT', 'COUNT_CAT']
            self.df_model = self.df_model.merge(gasto_cat, on='CATEGORIA', how='left')
            print(f"    Agregados por CATEGORIA")

        # Por DESTINO
        if 'DESTINO' in self.df_model.columns:
            gasto_dest = self.df_model.groupby('DESTINO')['TOTALPESOS'].agg(['mean', 'median']).reset_index()
            gasto_dest.columns = ['DESTINO', 'GASTO_PROM_DEST', 'GASTO_MED_DEST']
            self.df_model = self.df_model.merge(gasto_dest, on='DESTINO', how='left')
            print(f"    Agregados por DESTINO")

        # Por ORIGEN
        if 'ORIGEN' in self.df_model.columns:
            gasto_orig = self.df_model.groupby('ORIGEN')['TOTALPESOS'].agg(['mean', 'count']).reset_index()
            gasto_orig.columns = ['ORIGEN', 'GASTO_PROM_ORIG', 'COUNT_ORIG']
            self.df_model = self.df_model.merge(gasto_orig, on='ORIGEN', how='left')
            print(f"    Agregados por ORIGEN")

        # Por UN (Unidad de Negocio)
        if 'UN' in self.df_model.columns:
            gasto_un = self.df_model.groupby('UN')['TOTALPESOS'].agg(['mean', 'count']).reset_index()
            gasto_un.columns = ['UN', 'GASTO_PROM_UN', 'COUNT_UN']
            self.df_model = self.df_model.merge(gasto_un, on='UN', how='left')
            print(f"    Agregados por UN")

        # 2. Ratios y relaciones
        print("\n2 Creando ratios y relaciones...")
        if 'GASTO_PROM_CAT' in self.df_model.columns:
            self.df_model['RATIO_GASTO_CAT'] = self.df_model['TOTALPESOS'] / (self.df_model['GASTO_PROM_CAT'] + 1)
            print(f"    RATIO_GASTO_CAT creado")

        if 'GASTO_PROM_DEST' in self.df_model.columns:
            self.df_model['RATIO_GASTO_DEST'] = self.df_model['TOTALPESOS'] / (self.df_model['GASTO_PROM_DEST'] + 1)
            print(f"    RATIO_GASTO_DEST creado")

        # 3. Flags temporales
        print("\n3 Creando flags temporales...")
        if 'DIA_MES' in self.df_model.columns:
            self.df_model['ES_FIN_MES'] = (self.df_model['DIA_MES'] >= 25).astype(int)
            self.df_model['ES_INICIO_MES'] = (self.df_model['DIA_MES'] <= 5).astype(int)
            print(f"    ES_FIN_MES, ES_INICIO_MES creados")

        if 'MES' in self.df_model.columns:
            self.df_model['ES_FIN_TRIMESTRE'] = self.df_model['MES'].isin([3, 6, 9, 12]).astype(int)
            self.df_model['ES_FIN_AÑO'] = (self.df_model['MES'] == 12).astype(int)
            print(f"    ES_FIN_TRIMESTRE, ES_FIN_AÑO creados")

        # 4. Features de interacción
        print("\n4 Creando features de interacción...")
        if 'CANTIDAD' in self.df_model.columns and 'TOTALPESOS' in self.df_model.columns:
            self.df_model['PRECIO_UNITARIO'] = self.df_model['TOTALPESOS'] / (self.df_model['CANTIDAD'] + 0.01)
            print(f"    PRECIO_UNITARIO creado")

        # 5. Lag features (gasto del mes anterior)
        print("\n5 Creando lag features...")
        if 'FECHAPEDIDO' in self.df_model.columns and 'ORIGEN' in self.df_model.columns:
            self.df_model = self.df_model.sort_values('FECHAPEDIDO')
            self.df_model['GASTO_MES_ANTERIOR'] = self.df_model.groupby('ORIGEN')['TOTALPESOS'].shift(1)
            self.df_model['GASTO_MES_ANTERIOR'].fillna(self.df_model['TOTALPESOS'].median(), inplace=True)
            print(f"    GASTO_MES_ANTERIOR creado")

        # 6. Clasificación de cantidad
        if 'CANTIDAD' in self.df_model.columns:
            self.df_model['CANTIDAD_CLASE'] = pd.cut(
                self.df_model['CANTIDAD'],
                bins=[0, 1, 5, 10, np.inf],
                labels=['Unitaria', 'Baja', 'Media', 'Alta']
            )
            print(f"    CANTIDAD_CLASE creado (Unitaria/Baja/Media/Alta)")

        print(f"\n Feature engineering completado. Total features: {self.df_model.shape[1]}")

    def detectar_anomalias(self):
        """Detectar gastos anómalos usando Isolation Forest"""
        print("\n" + "="*70)
        print(" DETECCIÓN DE ANOMALÍAS EN GASTOS")
        print("="*70)

        if 'TOTALPESOS' not in self.df_clean.columns:
            print(" TOTALPESOS no disponible")
            return

        # Isolation Forest
        print("\n Aplicando Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        self.df_clean['ES_ANOMALIA'] = iso_forest.fit_predict(
            self.df_clean[['TOTALPESOS']].values
        )

        # -1 es anomalía, 1 es normal
        anomalias = self.df_clean[self.df_clean['ES_ANOMALIA'] == -1].copy()
        normales = self.df_clean[self.df_clean['ES_ANOMALIA'] == 1].copy()

        print(f"\n Resumen de Anomalías:")
        print(f"   Total anomalías: {len(anomalias):,} ({len(anomalias)/len(self.df_clean)*100:.2f}%)")
        print(f"   Gasto promedio normal: ${normales['TOTALPESOS'].mean():,.2f}")
        print(f"   Gasto promedio anómalo: ${anomalias['TOTALPESOS'].mean():,.2f}")
        print(f"   Gasto máximo anómalo: ${anomalias['TOTALPESOS'].max():,.2f}")

        # Top 20 anomalías por gasto
        top_anomalias = anomalias.nlargest(20, 'TOTALPESOS')[
            ['FECHAPEDIDO', 'ORIGEN', 'CATEGORIA', 'DESTINO', 'TOTALPESOS', 'DETALLEFINAL']
        ]

        print(f"\n Top 20 Gastos Anómalos:")
        print(top_anomalias.to_string())

        # Guardar alertas
        self.alertas = top_anomalias.to_dict('records')

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histograma
        axes[0].hist(normales['TOTALPESOS'], bins=50, alpha=0.7, label='Normal', color='green')
        axes[0].hist(anomalias['TOTALPESOS'], bins=50, alpha=0.7, label='Anomalía', color='red')
        axes[0].set_xlabel('Gasto (Pesos)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Gastos: Normal vs Anomalía')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Boxplot
        axes[1].boxplot([normales['TOTALPESOS'], anomalias['TOTALPESOS']],
                       labels=['Normal', 'Anomalía'])
        axes[1].set_ylabel('Gasto (Pesos)')
        axes[1].set_title('Comparación de Gastos')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n Detección de anomalías completada")

    def analizar_por_origen(self):
        """Análisis detallado por ORIGEN"""
        print("\n" + "="*70)
        print(" ANÁLISIS POR ORIGEN")
        print("="*70)

        if 'ORIGEN' not in self.df_clean.columns:
            print(" ORIGEN no disponible")
            return

        origenes = self.df_clean['ORIGEN'].unique()

        for origen in origenes:
            print(f"\n{'='*50}")
            print(f" ORIGEN: {origen}")
            print(f"{'='*50}")

            df_origen = self.df_clean[self.df_clean['ORIGEN'] == origen].copy()

            # Estadísticas generales
            print(f"\n Estadísticas Generales:")
            print(f"   Total pedidos: {len(df_origen):,}")
            print(f"   Gasto total: ${df_origen['TOTALPESOS'].sum():,.2f}")
            print(f"   Gasto promedio: ${df_origen['TOTALPESOS'].mean():,.2f}")
            print(f"   Gasto mediano: ${df_origen['TOTALPESOS'].median():,.2f}")

            # Top destinos
            if 'DESTINO' in df_origen.columns:
                top_destinos = df_origen.groupby('DESTINO')['TOTALPESOS'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5)
                print(f"\n Top 5 Destinos:")
                for idx, row in top_destinos.iterrows():
                    print(f"   {idx}: ${row['sum']:,.2f} ({int(row['count'])} pedidos)")

            # Top categorías
            if 'CATEGORIA' in df_origen.columns:
                top_cats = df_origen.groupby('CATEGORIA')['TOTALPESOS'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5)
                print(f"\n Top 5 Categorías Compradas:")
                for idx, row in top_cats.iterrows():
                    print(f"   {idx}: ${row['sum']:,.2f} ({int(row['count'])} pedidos)")

            # Pronóstico mensual
            if 'FECHAPEDIDO' in df_origen.columns:
                df_mensual = df_origen.groupby(pd.Grouper(key='FECHAPEDIDO', freq='M'))['TOTALPESOS'].sum().reset_index()
                df_mensual.columns = ['ds', 'y']
                df_mensual = df_mensual[df_mensual['ds'].notna()]

                if len(df_mensual) >= 24:  # Mínimo 2 años de datos
                    print(f"\n Pronóstico Próximos 12 Meses:")
                    try:
                        modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                        modelo.fit(df_mensual)
                        future = modelo.make_future_dataframe(periods=12, freq='M')
                        forecast = modelo.predict(future)

                        print(forecast[['ds', 'yhat']].tail(12).to_string(index=False))

                        # Guardar resultados
                        self.analisis_origen[origen] = {
                            'estadisticas': {
                                'total_pedidos': len(df_origen),
                                'gasto_total': df_origen['TOTALPESOS'].sum(),
                                'gasto_promedio': df_origen['TOTALPESOS'].mean()
                            },
                            'forecast': forecast
                        }
                    except:
                        print("    No hay suficientes datos para pronóstico")
                else:
                    print(f"    Solo {len(df_mensual)} meses de datos (mínimo 24 para pronóstico)")

        print("\n Análisis por ORIGEN completado")

    def analizar_por_un(self):
        """Análisis detallado por UN (Unidad de Negocio)"""
        print("\n" + "="*70)
        print(" ANÁLISIS POR UNIDAD DE NEGOCIO (UN)")
        print("="*70)

        if 'UN' not in self.df_clean.columns:
            print(" UN no disponible")
            return

        # Filtrar NaN
        df_un = self.df_clean[self.df_clean['UN'].notna()].copy()
        unidades = df_un['UN'].unique()

        for un in unidades[:5]:  # Top 5 para no saturar
            print(f"\n{'='*50}")
            print(f" UNIDAD DE NEGOCIO: {un}")
            print(f"{'='*50}")

            df_unidad = df_un[df_un['UN'] == un].copy()

            # Estadísticas generales
            print(f"\n Estadísticas Generales:")
            print(f"   Total pedidos: {len(df_unidad):,}")
            print(f"   Gasto total: ${df_unidad['TOTALPESOS'].sum():,.2f}")
            print(f"   Gasto promedio: ${df_unidad['TOTALPESOS'].mean():,.2f}")

            # Top categorías
            if 'CATEGORIA' in df_unidad.columns:
                top_cats = df_unidad.groupby('CATEGORIA')['TOTALPESOS'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(3)
                print(f"\n🛒 Top 3 Categorías:")
                for idx, row in top_cats.iterrows():
                    print(f"   {idx}: ${row['sum']:,.2f} ({int(row['count'])} pedidos)")

            # Guardar para recomendaciones
            self.analisis_un[un] = {
                'gasto_total': df_unidad['TOTALPESOS'].sum(),
                'gasto_promedio': df_unidad['TOTALPESOS'].mean(),
                'total_pedidos': len(df_unidad)
            }

        print("\n Análisis por UN completado")

    def sistema_recomendaciones(self):
        """Sistema de recomendaciones por área (ORIGEN/UN)"""
        print("\n" + "="*70)
        print(" SISTEMA DE RECOMENDACIONES")
        print("="*70)

        # Recomendaciones por ORIGEN
        if 'ORIGEN' in self.df_clean.columns:
            print("\n Recomendaciones por ORIGEN:")
            for origen, analisis in self.analisis_origen.items():
                gasto_prom = analisis['estadisticas']['gasto_promedio']
                total_pedidos = analisis['estadisticas']['total_pedidos']

                print(f"\n🔹 {origen}:")
                if gasto_prom > self.df_clean['TOTALPESOS'].quantile(0.75):
                    print(f"    ATENCIÓN: Gasto promedio alto (${gasto_prom:,.2f})")
                    print(f"    Sugerencia: Revisar contratos con proveedores y negociar descuentos por volumen")
                elif gasto_prom < self.df_clean['TOTALPESOS'].quantile(0.25):
                    print(f"    Gasto promedio bajo (${gasto_prom:,.2f})")
                    print(f"    Sugerencia: Proceso eficiente, mantener prácticas actuales")
                else:
                    print(f"    Gasto promedio normal (${gasto_prom:,.2f})")

                if total_pedidos > self.df_clean.groupby('ORIGEN').size().quantile(0.75):
                    print(f"    Alto volumen de pedidos ({total_pedidos:,})")
                    print(f"    Sugerencia: Considerar consolidación de pedidos para reducir costos operativos")

        # Recomendaciones por UN
        if self.analisis_un:
            print("\n Recomendaciones por Unidad de Negocio:")
            gasto_un_df = pd.DataFrame(self.analisis_un).T
            gasto_un_df = gasto_un_df.sort_values('gasto_total', ascending=False)

            for idx, row in gasto_un_df.head(5).iterrows():
                print(f"\n🔹 {idx}:")
                print(f"   Gasto total: ${row['gasto_total']:,.2f}")
                print(f"   Gasto promedio: ${row['gasto_promedio']:,.2f}")
                print(f"   Total pedidos: {int(row['total_pedidos']):,}")

                # Comparar con promedio general
                if row['gasto_promedio'] > self.df_clean['TOTALPESOS'].mean() * 1.5:
                    print(f"    Gasto promedio 50% por encima del general")
                    print(f"    Sugerencia: Revisar procesos de aprobación y buscar alternativas más económicas")
                elif row['total_pedidos'] < 100:
                    print(f"    Bajo volumen de compras")
                    print(f"    Sugerencia: Consolidar compras con otras unidades para negociar mejores precios")

        print("\n Sistema de recomendaciones completado")

    def visualizaciones(self):
        """Visualizaciones clave para entender los datos"""
        print("\n" + "="*70)
        print(" VISUALIZACIONES")
        print("="*70)

        # 1. Distribución de TOTALPESOS
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(self.df_clean['TOTALPESOS'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribución de TOTALPESOS', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Total Pesos')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(alpha=0.3)

        axes[1].hist(np.log1p(self.df_clean['TOTALPESOS']), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_title('Distribución Log(TOTALPESOS)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Log(Total Pesos)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 2. Gasto promedio por categoría
        if 'CATEGORIA' in self.df_clean.columns:
            gasto_cat = self.df_clean.groupby('CATEGORIA')['TOTALPESOS'].mean().sort_values(ascending=False).head(10)

            plt.figure(figsize=(12, 6))
            gasto_cat.plot(kind='barh', color='steelblue', edgecolor='black')
            plt.title('Top 10 Categorías por Gasto Promedio', fontsize=14, fontweight='bold')
            plt.xlabel('Gasto Promedio (Pesos)')
            plt.ylabel('Categoría')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 3. Tendencia temporal (gasto mensual)
        if 'FECHAPEDIDO' in self.df_clean.columns:
            gasto_mensual = self.df_clean.groupby(pd.Grouper(key='FECHAPEDIDO', freq='M'))['TOTALPESOS'].sum()

            plt.figure(figsize=(14, 6))
            plt.plot(gasto_mensual.index, gasto_mensual.values, marker='o', linewidth=2, color='darkblue')
            plt.fill_between(gasto_mensual.index, gasto_mensual.values, alpha=0.3)
            plt.title('Tendencia de Gasto Total por Mes', fontsize=14, fontweight='bold')
            plt.xlabel('Mes')
            plt.ylabel('Gasto Total (Pesos)')
            plt.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # 4. Gasto por ORIGEN
        if 'ORIGEN' in self.df_clean.columns:
            gasto_origen = self.df_clean.groupby('ORIGEN')['TOTALPESOS'].sum().sort_values(ascending=False)

            plt.figure(figsize=(12, 6))
            gasto_origen.plot(kind='bar', color='coral', edgecolor='black')
            plt.title('Gasto Total por ORIGEN', fontsize=14, fontweight='bold')
            plt.xlabel('Origen')
            plt.ylabel('Gasto Total (Pesos)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        print(" Visualizaciones generadas")

    def preparar_datos_modelo(self, target='TOTALPESOS', tipo='regresion'):
        """Preparar datos para modelado (encoding, split) - MEJORADO"""
        print(f"\n Preparando datos para {tipo.upper()} (target: {target})...")

        # Seleccionar features relevantes (AMPLIADO)
        features_base = [
            'CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN',
            'CENTROCOSTO', 'AÑO', 'MES', 'TRIMESTRE', 'DIA_SEMANA',
            'SEMANA_AÑO', 'ES_FIN_MES', 'ES_INICIO_MES', 'ES_FIN_TRIMESTRE',
            'CANTIDAD', 'PRECIO_UNITARIO'
        ]

        if tipo == 'regresion':
            features_base.extend([
                'GASTO_PROM_CAT', 'GASTO_MED_CAT', 'GASTO_STD_CAT',
                'GASTO_PROM_DEST', 'GASTO_MED_DEST',
                'GASTO_PROM_ORIG', 'GASTO_PROM_UN',
                'RATIO_GASTO_CAT', 'RATIO_GASTO_DEST',
                'GASTO_MES_ANTERIOR'
            ])

        # Filtrar features disponibles
        features = [f for f in features_base if f in self.df_model.columns]

        # Dataset completo sin nulos en target y features
        df_trabajo = self.df_model[features + [target]].dropna().copy()
        print(f"   Dataset para modelo: {len(df_trabajo):,} filas, {len(features)} features")

        # Encoding de categóricas
        cat_features = ['CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN']
        for col in cat_features:
            if col in df_trabajo.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_trabajo[col] = self.encoders[col].fit_transform(df_trabajo[col].astype(str))
                else:
                    df_trabajo[col] = self.encoders[col].transform(df_trabajo[col].astype(str))

        X = df_trabajo[features]
        y = df_trabajo[target]

        print(f"   Features finales: {list(X.columns)}")

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def modelo_prediccion_gasto(self):
        """Modelo 1: Predecir TOTALPESOS (Regresión) - MEJORADO"""
        print("\n" + "="*70)
        print(" MODELO 1: PREDICCIÓN DE GASTO (TOTALPESOS) - MEJORADO")
        print("="*70)

        X_train, X_test, y_train, y_test = self.preparar_datos_modelo(target='TOTALPESOS', tipo='regresion')

        # Random Forest Regressor
        print("\n Entrenando Random Forest Regressor...")
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # XGBoost Regressor con mejores hiperparámetros
        print(" Entrenando XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)

        # Gradient Boosting Regressor
        print("Entrenando Gradient Boosting Regressor...")
        gb = GradientBoostingRegressor(
            n_estimators=40,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)

        # Evaluación
        resultados = {
            'Random Forest': {
                'R2': r2_score(y_test, y_pred_rf),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                'MAE': mean_absolute_error(y_test, y_pred_rf)
            },
            'XGBoost': {
                'R2': r2_score(y_test, y_pred_xgb),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
                'MAE': mean_absolute_error(y_test, y_pred_xgb)
            },
            'Gradient Boosting': {
                'R2': r2_score(y_test, y_pred_gb),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
                'MAE': mean_absolute_error(y_test, y_pred_gb)
            }
        }

        print("\n Resultados:")
        df_resultados = pd.DataFrame(resultados).T
        print(df_resultados)

        # Feature importance del mejor modelo
        mejor_r2 = max([r['R2'] for r in resultados.values()])
        if resultados['Random Forest']['R2'] == mejor_r2:
            mejor_modelo = rf
            nombre_mejor = 'Random Forest'
        elif resultados['XGBoost']['R2'] == mejor_r2:
            mejor_modelo = xgb_model
            nombre_mejor = 'XGBoost'
        else:
            mejor_modelo = gb
            nombre_mejor = 'Gradient Boosting'

        print(f"\n Mejor modelo: {nombre_mejor} (R² = {mejor_r2:.4f})")

        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': mejor_modelo.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\n🔝 Top 15 Features Más Importantes:")
        print(importances.head(15))

        # Guardar mejor modelo
        self.modelos['prediccion_gasto'] = mejor_modelo
        self.resultados['prediccion_gasto'] = resultados

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].scatter(y_test, y_pred_rf if nombre_mejor == 'Random Forest' else y_pred_xgb, alpha=0.3)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Real')
        axes[0].set_ylabel('Predicho')
        axes[0].set_title(f'{nombre_mejor}: Real vs Predicho')
        axes[0].grid(alpha=0.3)

        importances.head(15).plot(x='Feature', y='Importance', kind='barh', ax=axes[1], legend=False)
        axes[1].set_xlabel('Importancia')
        axes[1].set_title('Feature Importance (Top 15)')

        plt.tight_layout()
        plt.show()

        print("\n Modelo de predicción de gasto completado")

    def modelo_demanda_temporal(self):
        """Modelo 2: Pronóstico de demanda temporal con Prophet"""
        print("\n" + "="*70)
        print(" MODELO 2: PRONÓSTICO DE DEMANDA TEMPORAL")
        print("="*70)

        if 'FECHAPEDIDO' not in self.df_clean.columns:
            print(" FECHAPEDIDO no disponible para series temporales")
            return

        # Agregar datos mensuales
        df_mensual = self.df_clean.groupby(pd.Grouper(key='FECHAPEDIDO', freq='M')).agg({
            'TOTALPESOS': 'sum',
            'SOLICITUD': 'count'
        }).reset_index()
        df_mensual.columns = ['ds', 'y', 'num_pedidos']
        df_mensual = df_mensual[df_mensual['ds'].notna()].copy()

        print(f"\n Datos mensuales: {len(df_mensual)} meses")
        print(f"   Rango: {df_mensual['ds'].min()} a {df_mensual['ds'].max()}")

        # Entrenar Prophet
        print("\n Entrenando Prophet para gasto mensual...")
        modelo_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        modelo_prophet.fit(df_mensual[['ds', 'y']])

        # Forecast 12 meses
        future = modelo_prophet.make_future_dataframe(periods=12, freq='M')
        forecast = modelo_prophet.predict(future)

        print("\n Pronóstico próximos 12 meses:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_string())

        # Visualización
        fig = modelo_prophet.plot(forecast, figsize=(14, 6))
        plt.title('Pronóstico de Gasto Mensual (Prophet)', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Gasto Total (Pesos)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        fig2 = modelo_prophet.plot_components(forecast, figsize=(14, 8))
        plt.tight_layout()
        plt.show()

        self.modelos['demanda_temporal'] = modelo_prophet
        self.resultados['demanda_temporal'] = forecast

        print("\n Modelo de demanda temporal completado")

    def guardar_modelos(self):
        """Guardar todos los modelos y encoders con joblib"""
        print("\n" + "="*70)
        print(" GUARDANDO MODELOS Y ENCODERS")
        print("="*70)

        try:
            # Guardar modelos
            for nombre, modelo in self.modelos.items():
                ruta = f"{self.directorio_salida}/modelo_{nombre}.pkl"
                joblib.dump(modelo, ruta)
                print(f"    Guardado: modelo_{nombre}.pkl")

            # Guardar encoders
            ruta_encoders = f"{self.directorio_salida}/encoders.pkl"
            joblib.dump(self.encoders, ruta_encoders)
            print(f"    Guardado: encoders.pkl")

            # Guardar resultados
            ruta_resultados = f"{self.directorio_salida}/resultados.pkl"
            joblib.dump(self.resultados, ruta_resultados)
            print(f"    Guardado: resultados.pkl")

            # Guardar análisis
            analisis = {
                'origen': self.analisis_origen,
                'un': self.analisis_un,
                'alertas': self.alertas
            }
            ruta_analisis = f"{self.directorio_salida}/analisis.pkl"
            joblib.dump(analisis, ruta_analisis)
            print(f"    Guardado: analisis.pkl")

            print(f"\n Todos los archivos guardados en: {self.directorio_salida}")

        except Exception as e:
            print(f" Error al guardar modelos: {e}")

    def cargar_modelos(self):
        """Cargar modelos y encoders previamente guardados"""
        print("\n" + "="*70)
        print(" CARGANDO MODELOS Y ENCODERS")
        print("="*70)

        try:
            # Cargar modelos
            archivos_modelos = [f for f in os.listdir(self.directorio_salida) if f.startswith('modelo_')]
            for archivo in archivos_modelos:
                nombre = archivo.replace('modelo_', '').replace('.pkl', '')
                ruta = f"{self.directorio_salida}/{archivo}"
                self.modelos[nombre] = joblib.load(ruta)
                print(f"    Cargado: {archivo}")

            # Cargar encoders
            ruta_encoders = f"{self.directorio_salida}/encoders.pkl"
            self.encoders = joblib.load(ruta_encoders)
            print(f"    Cargado: encoders.pkl")

            print(f"\n Modelos cargados exitosamente")

        except Exception as e:
            print(f" Error al cargar modelos: {e}")

    def predecir_nuevo_pedido(self, datos_pedido):
        """Hacer predicción para un nuevo pedido"""
        print("\n" + "="*70)
        print(" PREDICCIÓN NUEVO PEDIDO")
        print("="*70)

        if 'prediccion_gasto' not in self.modelos:
            print(" Modelo de predicción de gasto no disponible")
            return None

        try:
            # Convertir a DataFrame
            df_pred = pd.DataFrame([datos_pedido])

            # Encoding
            for col in ['CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN']:
                if col in df_pred.columns and col in self.encoders:
                    df_pred[col] = self.encoders[col].transform(df_pred[col].astype(str))

            # Predecir
            modelo = self.modelos['prediccion_gasto']
            prediccion = modelo.predict(df_pred)

            print(f"\n Gasto estimado: ${prediccion[0]:,.2f}")
            print(f"\n Datos del pedido:")
            for key, value in datos_pedido.items():
                print(f"   {key}: {value}")

            return prediccion[0]

        except Exception as e:
            print(f" Error en predicción: {e}")
            return None

    def generar_reporte_ejecutivo(self):
        """Generar reporte ejecutivo con todos los insights"""
        print("\n" + "="*70)
        print(" REPORTE EJECUTIVO")
        print("="*70)

        print(f"\n RESUMEN GENERAL:")
        print(f"   Total registros analizados: {len(self.df_clean):,}")
        print(f"   Gasto total: ${self.df_clean['TOTALPESOS'].sum():,.2f}")
        print(f"   Gasto promedio por pedido: ${self.df_clean['TOTALPESOS'].mean():,.2f}")
        print(f"   Período analizado: {self.df_clean['FECHAPEDIDO'].min().date()} a {self.df_clean['FECHAPEDIDO'].max().date()}")

        if self.resultados.get('prediccion_gasto'):
            mejor_r2 = max([r['R2'] for r in self.resultados['prediccion_gasto'].values()])
            print(f"\n MODELO PREDICTIVO:")
            print(f"   Precisión (R²): {mejor_r2:.2%}")
            print(f"   Capacidad de predicción: {'Excelente' if mejor_r2 > 0.7 else 'Buena' if mejor_r2 > 0.5 else 'Aceptable'}")

        if self.alertas:
            print(f"\n ALERTAS DE GASTOS ANÓMALOS:")
            print(f"   Total anomalías detectadas: {len(self.alertas)}")
            print(f"   Gasto anómalo promedio: ${np.mean([a['TOTALPESOS'] for a in self.alertas]):,.2f}")
            print(f"   Mayor anomalía: ${max([a['TOTALPESOS'] for a in self.alertas]):,.2f}")

        if self.analisis_origen:
            print(f"\n ANÁLISIS POR ORIGEN:")
            for origen, data in list(self.analisis_origen.items())[:3]:
                print(f"   {origen}: ${data['estadisticas']['gasto_total']:,.2f} ({data['estadisticas']['total_pedidos']:,} pedidos)")

        print("\n Reporte ejecutivo completado")

    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo mejorado"""
        print("\n" + "="*70)
        print(" INICIANDO PIPELINE PREDICTIVO COMPLETO V2")
        print("="*70)

        self.cargar_datos()
        self.explorar_datos()
        self.limpiar_datos()
        self.feature_engineering()
        self.detectar_anomalias()
        self.visualizaciones()
        self.modelo_prediccion_gasto()
        self.modelo_demanda_temporal()
        self.analizar_por_origen()
        self.analizar_por_un()
        self.sistema_recomendaciones()
        self.guardar_modelos()
        self.generar_reporte_ejecutivo()

        print("\n" + "="*70)
        print(" PIPELINE COMPLETADO")
        print("="*70)
        print(f"\n Modelos entrenados: {list(self.modelos.keys())}")
        print(f" Resultados guardados: {list(self.resultados.keys())}")
        print(f" Archivos guardados en: {self.directorio_salida}")

        return self

# ============================================================================
# 2. EJECUCIÓN DEL PIPELINE
# ============================================================================
if __name__ == "__main__":
    # Montar Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Ruta del CSV
    RUTA_CSV = '/content/drive/MyDrive/DATASET/archivo_listo.csv'
    
    # Ejecutar pipeline
    gestor = GestorPredictivoV2(RUTA_CSV)
    gestor = gestor.ejecutar_pipeline_completo()
    
    # ============================================================================
    # 3. RE-ENTRENAMIENTO SIN DATA LEAKAGE (PRECIO_UNITARIO)
    # ============================================================================
    print("\n" + "="*70)
    print("🔧 RE-ENTRENANDO MODELO SIN DATA LEAKAGE")
    print("="*70)
    print("Nota: El modelo anterior tenía R²=99.98% porque usaba PRECIO_UNITARIO,")
    print("que básicamente ES el target (TOTALPESOS/CANTIDAD). Ahora entrenamos sin él.")
    
    # Preparar datos SIN precio unitario
    features_limpios = [
        'CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN',
        'CENTROCOSTO', 'AÑO', 'MES', 'TRIMESTRE', 'DIA_SEMANA', 
        'SEMANA_AÑO', 'ES_FIN_MES', 'ES_INICIO_MES', 'ES_FIN_TRIMESTRE',
        'CANTIDAD',
        'GASTO_PROM_CAT', 'GASTO_MED_CAT', 'GASTO_STD_CAT',
        'GASTO_PROM_DEST', 'GASTO_MED_DEST', 
        'GASTO_PROM_ORIG', 'GASTO_PROM_UN',
        'RATIO_GASTO_CAT', 'RATIO_GASTO_DEST',
        'GASTO_MES_ANTERIOR'
    ]
    
    # Filtrar solo features que existan
    features_disponibles = [f for f in features_limpios if f in gestor.df_model.columns]
    df_limpio = gestor.df_model[features_disponibles + ['TOTALPESOS']].dropna().copy()
    
    print(f"\n📊 Dataset: {len(df_limpio):,} filas, {len(features_disponibles)} features")
    print(f"Features usados: {features_disponibles}")
    
    # Encoding (reutilizar encoders ya entrenados)
    for col in ['CATEGORIA', 'CLASE', 'FAMILIA', 'DESTINO', 'ORIGEN', 'UN']:
        if col in df_limpio.columns:
            df_limpio[col] = gestor.encoders[col].transform(df_limpio[col].astype(str))
    
    # Split
    X = df_limpio[features_disponibles]
    y = df_limpio['TOTALPESOS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar XGBoost optimizado
    print("\n🚀 Entrenando XGBoost sin data leakage...")
    modelo_limpio = xgb.XGBRegressor(
        n_estimators=300, 
        max_depth=10, 
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )
    modelo_limpio.fit(X_train, y_train)
    y_pred = modelo_limpio.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS DEL MODELO LIMPIO:")
    print(f"   R²: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAE: ${mae:,.2f}")
    
    if r2 > 0.6:
        print("   ✅ Modelo con buena capacidad predictiva")
    elif r2 > 0.4:
        print("   ⚠️ Modelo con capacidad predictiva aceptable")
    else:
        print("   ❌ Modelo necesita más features o ajuste")
    
    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': modelo_limpio.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🔝 Top 15 Features Más Importantes:")
    print(importances.head(15))
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real vs Predicho
    axes[0].scatter(y_test, y_pred, alpha=0.3)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Gasto Real')
    axes[0].set_ylabel('Gasto Predicho')
    axes[0].set_title(f'Modelo Limpio: Real vs Predicho (R²={r2:.4f})')
    axes[0].grid(alpha=0.3)
    
    # Feature importance
    importances.head(15).plot(x='Feature', y='Importance', kind='barh', ax=axes[1], legend=False)
    axes[1].set_xlabel('Importancia')
    axes[1].set_title('Top 15 Features')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar modelo limpio
    joblib.dump(modelo_limpio, '/content/modelos/modelo_prediccion_limpio.pkl')
    print("\n💾 Modelo limpio guardado: modelo_prediccion_limpio.pkl")
    
    # Actualizar en gestor
    gestor.modelos['prediccion_gasto_limpio'] = modelo_limpio
    gestor.resultados['prediccion_gasto_limpio'] = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }
    
    print("\n✅ Re-entrenamiento completado. Usa 'prediccion_gasto_limpio' para predicciones reales.")
    
    print("\n🎉 ¡Pipeline ejecutado exitosamente!")
    print("\n💡 GUÍA DE USO:")
    print("="*70)
    print("# Acceder a modelos y datos:")
    print("  gestor.modelos                    # Diccionario de modelos")
    print("  gestor.resultados                 # Resultados de evaluación")
    print("  gestor.df_clean                   # Datos limpios")
    print("  gestor.df_model                   # Datos con features")
    print("  gestor.alertas                    # Alertas de gastos anómalos")
    print("  gestor.analisis_origen            # Análisis por ORIGEN")
    print("  gestor.analisis_un                # Análisis por UN")
    print()
    print("# Hacer predicción nueva:")
    print("  nuevo_pedido = {")
    print("      'CATEGORIA': 'Tecnología',")
    print("      'CLASE': 'Scanners',")
    print("      'FAMILIA': 'Scanners',")
    print("      'DESTINO': '4450 - SECTOR MERCADO MASIVO',")
    print("      'ORIGEN': 'ARIBA SAP',")
    print("      'UN': 'EKT',")
    print("      'CENTROCOSTO': 320000,")
    print("      'AÑO': 2025,")
    print("      'MES': 10,")
    print("      'CANTIDAD': 1,")
    print("      'GASTO_PROM_CAT': 25000,  # Usar promedios del análisis")
    print("      'ES_FIN_MES': 0")
    print("  }")
    print("  gestor.predecir_nuevo_pedido(nuevo_pedido)")
    print()
    print("# Cargar modelos previamente guardados:")
    print("  gestor_nuevo = GestorPredictivoV2(RUTA_CSV)")
    print("  gestor_nuevo.cargar_modelos()")
    print("="*70)