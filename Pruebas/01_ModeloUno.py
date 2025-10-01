from google.colab import drive
drive.mount('/content/drive')

!pip install --upgrade numpy pmdarima
!pip install xlsxwriter

"""
SISTEMA DE PREDICCIÓN DE COMPRAS CORPORATIVAS - VERSIÓN CORREGIDA
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Modelos estadísticos
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

try:
    from pmdarima import auto_arima
    PMDARIMA_DISPONIBLE = True
except:
    PMDARIMA_DISPONIBLE = False

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GestorPrediccionCompras:
    """
    Gestor inteligente de predicciones con corrección de errores
    """

    def __init__(self):
        self.modelos = {}
        self.metricas = {}
        self.categorias_info = {}
        self.predicciones = {}

    def cargar_datos(self, ruta_archivo):
        """Carga y prepara el dataset"""
        print("\n" + "="*80)
        print("📂 CARGANDO DATOS")
        print("="*80)

        try:
            df = pd.read_csv(ruta_archivo, encoding='latin1')
            print(f"✅ Dataset cargado: {len(df):,} registros")

            df['FECHAPEDIDO'] = pd.to_datetime(df['FECHAPEDIDO'], errors='coerce')
            df = df[df['FECHAPEDIDO'].notna()].copy()
            df = df[df['TOTALPESOS'] >= 0].copy()
            df['CATEGORIA'] = df['CATEGORIA'].str.strip().str.title()

            print(f"✅ Datos procesados: {len(df):,} registros válidos")
            self.df_original = df
            return df

        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            raise

    def analizar_categorias(self):
        """Analiza características de cada categoría"""
        print("\n" + "="*80)
        print("🔍 ANÁLISIS DE CATEGORÍAS")
        print("="*80)

        df = self.df_original
        resultados = []

        for categoria in df['CATEGORIA'].unique():
            if categoria.lower() == 'desconocido':
                continue

            df_cat = df[df['CATEGORIA'] == categoria].copy()
            df_cat['YearMonth'] = df_cat['FECHAPEDIDO'].dt.to_period('M')
            
            # CORRECCIÓN: Renombrar SOLICITUD a cantidad_ordenes
            serie_mensual = df_cat.groupby('YearMonth').agg({
                'TOTALPESOS': 'sum',
                'SOLICITUD': 'count'
            }).reset_index()
            
            serie_mensual.columns = ['YearMonth', 'TOTALPESOS', 'cantidad_ordenes']
            serie_mensual['YearMonth'] = serie_mensual['YearMonth'].dt.to_timestamp()
            serie_mensual = serie_mensual.sort_values('YearMonth')

            n_meses = len(serie_mensual)
            cv = serie_mensual['TOTALPESOS'].std() / serie_mensual['TOTALPESOS'].mean() if serie_mensual['TOTALPESOS'].mean() > 0 else np.inf

            # Detectar tendencia
            if n_meses >= 12:
                from scipy import stats
                x = np.arange(len(serie_mensual))
                y = serie_mensual['TOTALPESOS'].values
                slope, _, r_value, _, _ = stats.linregress(x, y)
                tendencia = 'creciente' if slope > 0 else 'decreciente'
                fuerza_tendencia = abs(r_value)
            else:
                tendencia = 'insuficiente'
                fuerza_tendencia = 0

            # Detectar estacionalidad
            estacionalidad = False
            if n_meses >= 24:
                from statsmodels.tsa.seasonal import seasonal_decompose
                try:
                    decomposition = seasonal_decompose(
                        serie_mensual.set_index('YearMonth')['TOTALPESOS'],
                        model='additive',
                        period=12,
                        extrapolate_trend='freq'
                    )
                    var_seasonal = np.var(decomposition.seasonal)
                    var_residual = np.var(decomposition.resid.dropna())
                    estacionalidad = var_seasonal > var_residual
                except:
                    estacionalidad = False

            # Clasificar y asignar modelo
            if n_meses < 12:
                tipo = 'DATOS_INSUFICIENTES'
                modelo_recomendado = 'MediaMovil'
            elif n_meses >= 36 and estacionalidad:
                tipo = 'SERIE_COMPLETA_ESTACIONAL'
                modelo_recomendado = 'SARIMA'
            elif n_meses >= 24 and fuerza_tendencia > 0.5:
                tipo = 'SERIE_CON_TENDENCIA'
                modelo_recomendado = 'Prophet'
            elif cv > 1.5:
                tipo = 'ALTA_VOLATILIDAD'
                modelo_recomendado = 'GradientBoosting'
            else:
                tipo = 'SERIE_ESTANDAR'
                modelo_recomendado = 'ExponentialSmoothing'

            info = {
                'categoria': categoria,
                'n_registros': len(df_cat),
                'n_meses': n_meses,
                'gasto_total': df_cat['TOTALPESOS'].sum(),
                'gasto_promedio_mes': serie_mensual['TOTALPESOS'].mean(),
                'cv': cv,
                'tendencia': tendencia,
                'fuerza_tendencia': fuerza_tendencia,
                'estacionalidad': estacionalidad,
                'tipo': tipo,
                'modelo_recomendado': modelo_recomendado,
                'serie_temporal': serie_mensual
            }

            self.categorias_info[categoria] = info
            resultados.append({
                'Categoría': categoria,
                'Registros': f"{len(df_cat):,}",
                'Meses': n_meses,
                'Tipo': tipo,
                'Modelo': modelo_recomendado
            })

        df_resumen = pd.DataFrame(resultados)
        print("\n📊 RESUMEN DE CATEGORÍAS Y MODELOS ASIGNADOS:")
        print(df_resumen.to_string(index=False))

        return self.categorias_info

    def entrenar_modelo_categoria(self, categoria, horizonte=12):
        """Entrena el modelo específico para una categoría - VERSIÓN CORREGIDA"""

        info = self.categorias_info[categoria]
        serie = info['serie_temporal'].copy()
        modelo_tipo = info['modelo_recomendado']

        print(f"\n🎯 Entrenando {modelo_tipo} para: {categoria}")
        print(f"   Datos: {len(serie)} meses | Horizonte: {horizonte} meses")

        # CORRECCIÓN: Guardar cantidad_ordenes antes de establecer índice
        cantidad_promedio_ordenes = serie['cantidad_ordenes'].mean()
        
        serie = serie.set_index('YearMonth')

        # Split train/test
        test_size = min(3, int(len(serie) * 0.2))
        train = serie.iloc[:-test_size] if test_size > 0 else serie
        test = serie.iloc[-test_size:] if test_size > 0 else None

        modelo_entrenado = None
        predicciones_test = None

        try:
            # ================================================================
            # ENTRENAMIENTO SEGÚN TIPO DE MODELO
            # ================================================================

            if modelo_tipo == 'SARIMA':
                if PMDARIMA_DISPONIBLE:
                    modelo = auto_arima(
                        train['TOTALPESOS'],
                        seasonal=True,
                        m=12,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_p=3, max_q=3,
                        max_P=2, max_Q=2,
                        max_d=2, max_D=1,
                        trace=False
                    )
                    modelo_entrenado = modelo
                    if test is not None:
                        predicciones_test = modelo.predict(n_periods=len(test))
                    predicciones_futuro = modelo.predict(n_periods=horizonte)
                else:
                    modelo = SARIMAX(
                        train['TOTALPESOS'],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    modelo_entrenado = modelo
                    if test is not None:
                        predicciones_test = modelo.forecast(steps=len(test))
                    predicciones_futuro = modelo.forecast(steps=horizonte)

            elif modelo_tipo == 'Prophet':
                df_prophet = train.reset_index()
                df_prophet.columns = ['ds', 'y', 'ordenes']
                df_prophet = df_prophet[['ds', 'y']]

                modelo = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    interval_width=0.95
                )
                modelo.fit(df_prophet)
                modelo_entrenado = modelo

                if test is not None:
                    future_test = pd.DataFrame({'ds': test.index})
                    forecast_test = modelo.predict(future_test)
                    predicciones_test = forecast_test['yhat'].values

                future = modelo.make_future_dataframe(periods=horizonte, freq='MS')
                forecast = modelo.predict(future)
                predicciones_futuro = forecast['yhat'].iloc[-horizonte:].values

            elif modelo_tipo == 'ExponentialSmoothing':
                modelo = ExponentialSmoothing(
                    train['TOTALPESOS'],
                    seasonal_periods=12 if len(train) >= 24 else None,
                    trend='add',
                    seasonal='add' if len(train) >= 24 else None,
                    damped_trend=True
                ).fit(optimized=True)
                modelo_entrenado = modelo

                if test is not None:
                    predicciones_test = modelo.forecast(len(test))
                predicciones_futuro = modelo.forecast(horizonte)

            elif modelo_tipo == 'GradientBoosting':
                def crear_features(df, lags=[1, 2, 3, 6, 12]):
                    df = df.copy()
                    for lag in lags:
                        if len(df) > lag:
                            df[f'lag_{lag}'] = df['TOTALPESOS'].shift(lag)
                    df['mes'] = df.index.month
                    df['trimestre'] = df.index.quarter
                    df['rolling_mean_3'] = df['TOTALPESOS'].rolling(3, min_periods=1).mean()
                    df['rolling_std_3'] = df['TOTALPESOS'].rolling(3, min_periods=1).std()
                    return df.fillna(method='bfill').fillna(0)

                train_features = crear_features(train)
                X_train = train_features.drop('TOTALPESOS', axis=1).select_dtypes(include=[np.number])
                y_train = train_features['TOTALPESOS']

                modelo = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                )
                modelo.fit(X_train, y_train)
                modelo_entrenado = modelo

                if test is not None:
                    test_features = crear_features(pd.concat([train, test]))
                    test_features = test_features.iloc[-len(test):]
                    X_test = test_features.drop('TOTALPESOS', axis=1).select_dtypes(include=[np.number])
                    predicciones_test = modelo.predict(X_test)

                # Predicción iterativa
                predicciones_futuro = []
                ultimo_dato = serie.copy()

                for _ in range(horizonte):
                    features = crear_features(ultimo_dato)
                    X = features.iloc[-1:].drop('TOTALPESOS', axis=1).select_dtypes(include=[np.number])
                    pred = modelo.predict(X)[0]
                    predicciones_futuro.append(pred)

                    nueva_fecha = ultimo_dato.index[-1] + pd.DateOffset(months=1)
                    nuevo_registro = pd.DataFrame({
                        'TOTALPESOS': [pred],
                        'cantidad_ordenes': [int(cantidad_promedio_ordenes)]
                    }, index=[nueva_fecha])
                    ultimo_dato = pd.concat([ultimo_dato, nuevo_registro])

                predicciones_futuro = np.array(predicciones_futuro)

            else:  # MediaMovil
                ventana = min(3, len(train))
                media = train['TOTALPESOS'].iloc[-ventana:].mean()
                predicciones_futuro = np.array([media] * horizonte)
                if test is not None:
                    predicciones_test = np.array([media] * len(test))

            # ================================================================
            # EVALUACIÓN Y MÉTRICAS
            # ================================================================

            metricas = {}
            if test is not None and predicciones_test is not None:
                y_true = test['TOTALPESOS'].values
                y_pred = np.maximum(predicciones_test, 0)

                metricas = {
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100,
                    'R2': r2_score(y_true, y_pred)
                }
                print(f"   ✅ MAE: ${metricas['MAE']:,.0f} | MAPE: {metricas['MAPE']:.1f}% | R²: {metricas['R2']:.3f}")

            # ================================================================
            # GENERAR FECHAS Y PREDICCIONES FUTURAS - CORRECCIÓN AQUÍ
            # ================================================================

            ultima_fecha = serie.index[-1]
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=horizonte,
                freq='MS'
            )

            predicciones_futuro = np.maximum(predicciones_futuro, 0)
            
            # CORRECCIÓN: Usar la variable guardada anteriormente
            cantidad_futura = [int(cantidad_promedio_ordenes)] * horizonte

            df_predicciones = pd.DataFrame({
                'fecha': fechas_futuras,
                'gasto_predicho': predicciones_futuro,
                'cantidad_predicha': cantidad_futura,
                'intervalo_inf': predicciones_futuro * 0.85,
                'intervalo_sup': predicciones_futuro * 1.15
            })

            self.modelos[categoria] = modelo_entrenado
            self.metricas[categoria] = metricas
            self.predicciones[categoria] = df_predicciones

            return df_predicciones, metricas

        except Exception as e:
            print(f"   ❌ Error en entrenamiento: {str(e)}")
            # Fallback robusto
            media = serie['TOTALPESOS'].mean()
            ultima_fecha = serie.index[-1]
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=horizonte,
                freq='MS'
            )

            df_predicciones = pd.DataFrame({
                'fecha': fechas_futuras,
                'gasto_predicho': [max(media, 0)] * horizonte,
                'cantidad_predicha': [int(cantidad_promedio_ordenes)] * horizonte,
                'intervalo_inf': [max(media * 0.7, 0)] * horizonte,
                'intervalo_sup': [media * 1.3] * horizonte
            })

            self.predicciones[categoria] = df_predicciones
            return df_predicciones, {}

    def entrenar_todos(self, horizonte=12, categorias_seleccionadas=None):
        """Entrena modelos para todas las categorías válidas"""
        print("\n" + "="*80)
        print("🚀 ENTRENAMIENTO DE MODELOS")
        print("="*80)

        categorias = categorias_seleccionadas if categorias_seleccionadas else list(self.categorias_info.keys())

        resultados = []
        for i, categoria in enumerate(categorias, 1):
            print(f"\n[{i}/{len(categorias)}] {categoria}")
            try:
                _, metricas = self.entrenar_modelo_categoria(categoria, horizonte)
                resultados.append({
                    'Categoría': categoria,
                    'Modelo': self.categorias_info[categoria]['modelo_recomendado'],
                    'Status': '✅ OK'
                })
            except Exception as e:
                print(f"   ⚠️  Error general: {str(e)}")
                resultados.append({
                    'Categoría': categoria,
                    'Modelo': 'Error',
                    'Status': '❌ Fallido'
                })

        print("\n" + "="*80)
        print("📊 RESUMEN DE ENTRENAMIENTO")
        print("="*80)
        df_resultados = pd.DataFrame(resultados)
        print(df_resultados.to_string(index=False))

        exitosos = len([r for r in resultados if r['Status'] == '✅ OK'])
        print(f"\n✅ Modelos entrenados exitosamente: {exitosos}/{len(categorias)}")

    def generar_reporte_completo(self, ruta_salida='reporte_predicciones.xlsx'):
        """Genera reporte Excel con todas las predicciones"""
        print("\n" + "="*80)
        print("📄 GENERANDO REPORTE COMPLETO")
        print("="*80)

        with pd.ExcelWriter(ruta_salida, engine='xlsxwriter') as writer:
            resumen_data = []
            for categoria, pred_df in self.predicciones.items():
                info = self.categorias_info[categoria]
                resumen_data.append({
                    'Categoría': categoria,
                    'Modelo': info['modelo_recomendado'],
                    'Meses_Historicos': info['n_meses'],
                    'Gasto_Total_Anual_Predicho': pred_df['gasto_predicho'].sum(),
                    'Gasto_Promedio_Mensual': pred_df['gasto_predicho'].mean(),
                    'Cantidad_Total_Anual': pred_df['cantidad_predicha'].sum()
                })

            df_resumen = pd.DataFrame(resumen_data)
            df_resumen.to_excel(writer, sheet_name='Resumen_General', index=False)

            for categoria, pred_df in self.predicciones.items():
                sheet_name = categoria[:31]
                pred_df_export = pred_df.copy()
                pred_df_export['fecha'] = pred_df_export['fecha'].dt.strftime('%Y-%m')
                pred_df_export.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ Reporte guardado en: {ruta_salida}")
        return ruta_salida

    def visualizar_predicciones(self, categoria, guardar=True):
        """Visualiza predicciones de una categoría específica"""
        if categoria not in self.predicciones:
            print(f"❌ No hay predicciones para: {categoria}")
            return

        info = self.categorias_info[categoria]
        serie_hist = info['serie_temporal'].set_index('YearMonth')
        pred_df = self.predicciones[categoria]

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        ax1 = axes[0]
        ax1.plot(serie_hist.index, serie_hist['TOTALPESOS'],
                label='Histórico', linewidth=2, marker='o', markersize=4)
        ax1.plot(pred_df['fecha'], pred_df['gasto_predicho'],
                label='Predicción', linewidth=2, marker='s', markersize=4, color='red')
        ax1.fill_between(pred_df['fecha'],
                         pred_df['intervalo_inf'],
                         pred_df['intervalo_sup'],
                         alpha=0.3, color='red', label='Intervalo de confianza')

        ax1.set_title(f'Predicción de Gastos - {categoria}\nModelo: {info["modelo_recomendado"]}',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fecha', fontsize=12)
        ax1.set_ylabel('Gasto Total (Pesos)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        ax2 = axes[1]
        x = np.arange(len(pred_df))
        width = 0.35

        ax2.bar(x - width/2, pred_df['gasto_predicho'], width,
               label='Gasto Predicho', alpha=0.8)
        ax2.bar(x + width/2, pred_df['cantidad_predicha'], width,
               label='Cantidad Predicha', alpha=0.8)

        ax2.set_title('Distribución Mensual - Próximos 12 Meses', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Mes', fontsize=12)
        ax2.set_ylabel('Valores', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pred_df['fecha'].dt.strftime('%Y-%m'), rotation=45)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if guardar:
            nombre_archivo = f'prediccion_{categoria.replace(" ", "_")}.png'
            plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {nombre_archivo}")

        plt.show()


def ejecutar_pipeline_completo(ruta_csv):
    """Ejecuta el pipeline completo de predicción"""
    print("\n" + "="*80)
    print("🎯 SISTEMA DE PREDICCIÓN DE COMPRAS CORPORATIVAS")
    print("="*80)

    gestor = GestorPrediccionCompras()
    df = gestor.cargar_datos(ruta_csv)
    categorias_info = gestor.analizar_categorias()

    categorias_validas = [
        cat for cat, info in categorias_info.items()
        if info['tipo'] != 'DATOS_INSUFICIENTES'
    ]

    print(f"\n🎯 Entrenando modelos para {len(categorias_validas)} categorías válidas")
    gestor.entrenar_todos(horizonte=12, categorias_seleccionadas=categorias_validas)
    gestor.generar_reporte_completo()

    print("\n" + "="*80)
    print("📊 GENERANDO VISUALIZACIONES")
    print("="*80)

    top_categorias = sorted(
        [(cat, info['gasto_total']) for cat, info in categorias_info.items()
         if cat in gestor.predicciones],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for i, (categoria, _) in enumerate(top_categorias, 1):
        print(f"\n[{i}/5] Visualizando: {categoria}")
        gestor.visualizar_predicciones(categoria, guardar=True)

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80)

    return gestor


# PUNTO DE ENTRADA
if __name__ == "__main__":
    
    RUTA_CSV = '/content/drive/MyDrive/DATASET/archivo_listo.csv'
    gestor = ejecutar_pipeline_completo(RUTA_CSV)
    
    print("\n🎉 ¡Sistema listo!")
    print("   - gestor.predicciones: Diccionario con todas las predicciones")
    print("   - gestor.metricas: Métricas de evaluación por categoría")
    print("   - gestor.modelos: Modelos entrenados")
    print("   - reporte_predicciones.xlsx: Reporte completo en Excel")