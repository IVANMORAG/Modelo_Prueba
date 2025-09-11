# Modelo Predictivo - Compras Internas Grupo Salinas
# ================================================

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Librerías para machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("✅ Librerías importadas correctamente")
print("📊 Configuración de visualización establecida")

# =============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================

def cargar_datos(ruta_archivo):
    """
    Carga los datos del Excel y realiza exploración inicial
    """
    try:
        # Cargar el archivo Excel
        df = pd.read_excel(ruta_archivo)
        print(f"✅ Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Información básica del dataset
        print("\n" + "="*50)
        print("📊 INFORMACIÓN GENERAL DEL DATASET")
        print("="*50)
        print(f"Dimensiones: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return None

def exploracion_inicial(df):
    """
    Realiza exploración inicial de los datos
    """
    print("\n" + "="*50)
    print("🔍 EXPLORACIÓN INICIAL")
    print("="*50)
    
    # Información básica
    print("📋 Info del DataFrame:")
    print(df.info())
    
    print("\n📊 Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n🔍 Primeras 5 filas:")
    print(df.head())
    
    print("\n❓ Valores nulos por columna:")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0])
    
    print("\n📈 Tipos de datos:")
    print(df.dtypes)
    
    return df

# =============================================================================
# 2. LIMPIEZA Y PREPROCESAMIENTO
# =============================================================================

def limpiar_datos(df):
    """
    Limpia y preprocesa los datos
    """
    print("\n" + "="*50)
    print("🧹 LIMPIEZA DE DATOS")
    print("="*50)
    
    df_clean = df.copy()
    
    # Identificar columnas numéricas y categóricas
    columnas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    print(f"📊 Columnas numéricas: {columnas_numericas}")
    print(f"📝 Columnas categóricas: {columnas_categoricas}")
    
    # Manejar valores nulos
    print("\n🔧 Manejando valores nulos...")
    
    # Para columnas numéricas: rellenar con mediana
    if columnas_numericas:
        imputer_num = SimpleImputer(strategy='median')
        df_clean[columnas_numericas] = imputer_num.fit_transform(df_clean[columnas_numericas])
    
    # Para columnas categóricas: rellenar con moda
    if columnas_categoricas:
        for col in columnas_categoricas:
            if df_clean[col].isnull().sum() > 0:
                moda = df_clean[col].mode()
                if len(moda) > 0:
                    df_clean[col].fillna(moda[0], inplace=True)
                else:
                    df_clean[col].fillna('Desconocido', inplace=True)
    
    print(f"✅ Valores nulos después de limpieza: {df_clean.isnull().sum().sum()}")
    
    return df_clean, columnas_numericas, columnas_categoricas

# =============================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

def analisis_exploratorio(df, columnas_numericas, columnas_categoricas):
    """
    Realiza análisis exploratorio completo
    """
    print("\n" + "="*50)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*50)
    
    # 1. Distribución de variables numéricas
    if columnas_numericas:
        print("📊 Analizando distribución de variables numéricas...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(columnas_numericas[:4]):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.show()
    
    # 2. Matriz de correlación
    if len(columnas_numericas) > 1:
        print("\n🔗 Matriz de correlación:")
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[columnas_numericas].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
    
    # 3. Análisis de variables categóricas
    if columnas_categoricas:
        print("\n📊 Analizando variables categóricas...")
        for col in columnas_categoricas[:3]:  # Primeras 3 columnas categóricas
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'Top 10 valores en {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.tight_layout()
            plt.show()
    
    return correlation_matrix if len(columnas_numericas) > 1 else None

# =============================================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================

def ingenieria_caracteristicas(df, target_column=None):
    """
    Crea nuevas características y prepara datos para modelado
    """
    print("\n" + "="*50)
    print("⚙️ INGENIERÍA DE CARACTERÍSTICAS")
    print("="*50)
    
    df_features = df.copy()
    
    # Identificar columnas de fecha
    date_columns = []
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            try:
                pd.to_datetime(df_features[col], errors='raise')
                date_columns.append(col)
            except:
                pass
    
    print(f"📅 Columnas de fecha identificadas: {date_columns}")
    
    # Procesar columnas de fecha
    for col in date_columns:
        try:
            df_features[col] = pd.to_datetime(df_features[col])
            df_features[f'{col}_año'] = df_features[col].dt.year
            df_features[f'{col}_mes'] = df_features[col].dt.month
            df_features[f'{col}_día_semana'] = df_features[col].dt.dayofweek
            df_features[f'{col}_trimestre'] = df_features[col].dt.quarter
            print(f"✅ Características temporales creadas para {col}")
        except:
            print(f"⚠️ No se pudo procesar {col} como fecha")
    
    # Encoding de variables categóricas
    le_dict = {}
    for col in df_features.select_dtypes(include=['object']).columns:
        if col not in date_columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
            le_dict[col] = le
            print(f"✅ Encoding aplicado a {col}")
    
    return df_features, le_dict

# =============================================================================
# 5. PREPARACIÓN PARA MODELADO
# =============================================================================

def preparar_modelado(df, target_column, test_size=0.2, random_state=42):
    """
    Prepara los datos para el modelado
    """
    print("\n" + "="*50)
    print("🎯 PREPARACIÓN PARA MODELADO")
    print("="*50)
    
    # Separar características y variable objetivo
    if target_column not in df.columns:
        print(f"❌ La columna objetivo '{target_column}' no existe en el dataset")
        print(f"Columnas disponibles: {list(df.columns)}")
        return None, None, None, None, None
    
    # Seleccionar solo columnas numéricas para el modelo
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    X = df[numeric_columns]
    y = df[target_column]
    
    print(f"✅ Variables predictoras: {len(X.columns)}")
    print(f"✅ Variable objetivo: {target_column}")
    print(f"📊 Shape de X: {X.shape}")
    print(f"📊 Shape de y: {y.shape}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ División de datos: {len(X_train)} entrenamiento, {len(X_test)} prueba")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# =============================================================================
# 6. ENTRENAMIENTO DE MODELOS
# =============================================================================

def entrenar_modelos(X_train, y_train, X_test, y_test):
    """
    Entrena múltiples modelos y compara su rendimiento
    """
    print("\n" + "="*50)
    print("🤖 ENTRENAMIENTO DE MODELOS")
    print("="*50)
    
    # Definir modelos
    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0)
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n🔄 Entrenando {nombre}...")
        
        # Entrenar modelo
        modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        resultados[nombre] = {
            'modelo': modelo,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"✅ MAE Test: {mae_test:.4f}")
        print(f"✅ R² Test: {r2_test:.4f}")
    
    return resultados

# =============================================================================
# 7. EVALUACIÓN Y VISUALIZACIÓN DE RESULTADOS
# =============================================================================

def evaluar_modelos(resultados, y_test):
    """
    Evalúa y visualiza el rendimiento de los modelos
    """
    print("\n" + "="*50)
    print("📈 EVALUACIÓN DE MODELOS")
    print("="*50)
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'MAE_Test': [res['mae_test'] for res in resultados.values()],
        'MSE_Test': [res['mse_test'] for res in resultados.values()],
        'R2_Test': [res['r2_test'] for res in resultados.values()]
    })
    
    # Ordenar por R²
    df_resultados = df_resultados.sort_values('R2_Test', ascending=False)
    
    print("📊 Ranking de modelos por R²:")
    print(df_resultados)
    
    # Visualización de resultados
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE
    axes[0].bar(df_resultados['Modelo'], df_resultados['MAE_Test'])
    axes[0].set_title('Mean Absolute Error (Test)')
    axes[0].set_xlabel('Modelo')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MSE
    axes[1].bar(df_resultados['Modelo'], df_resultados['MSE_Test'])
    axes[1].set_title('Mean Squared Error (Test)')
    axes[1].set_xlabel('Modelo')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[2].bar(df_resultados['Modelo'], df_resultados['R2_Test'])
    axes[2].set_title('R² Score (Test)')
    axes[2].set_xlabel('Modelo')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Mejor modelo
    mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
    mejor_modelo = resultados[mejor_modelo_nombre]
    
    print(f"\n🏆 Mejor modelo: {mejor_modelo_nombre}")
    print(f"📊 R² Test: {mejor_modelo['r2_test']:.4f}")
    print(f"📊 MAE Test: {mejor_modelo['mae_test']:.4f}")
    
    # Gráfico de predicciones vs valores reales
    plt.figure(figsize=(10, 8))
    y_pred = mejor_modelo['y_pred_test']
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicciones vs Valores Reales - {mejor_modelo_nombre}')
    plt.show()
    
    return mejor_modelo, df_resultados

# =============================================================================
# 8. PREDICCIONES Y EXPORTACIÓN
# =============================================================================

def hacer_predicciones(modelo, scaler, X_new):
    """
    Hace predicciones con el modelo entrenado
    """
    X_new_scaled = scaler.transform(X_new)
    predicciones = modelo.predict(X_new_scaled)
    return predicciones

def exportar_resultados(df_resultados, predicciones=None, nombre_archivo='resultados_modelo.xlsx'):
    """
    Exporta los resultados a Excel
    """
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_resultados.to_excel(writer, sheet_name='Comparacion_Modelos', index=False)
        if predicciones is not None:
            pd.DataFrame({'Predicciones': predicciones}).to_excel(
                writer, sheet_name='Predicciones', index=False
            )
    
    print(f"✅ Resultados exportados a {nombre_archivo}")

# =============================================================================
# 9. FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("🚀 INICIANDO ANÁLISIS PREDICTIVO - GRUPO SALINAS")
    print("="*60)
    
    # PASO 1: Cargar datos (MODIFICA LA RUTA DE TU ARCHIVO AQUÍ)
    ruta_archivo = "tu_archivo.xlsx"  # ← CAMBIA ESTA RUTA
    
    print("⚠️  IMPORTANTE: Asegúrate de cambiar la ruta del archivo Excel")
    print(f"📁 Ruta actual: {ruta_archivo}")
    print("\n¿Quieres continuar con datos de ejemplo? (s/n)")
    
    # Por ahora, creamos datos de ejemplo
    print("📊 Creando datos de ejemplo para demostración...")
    
    # Crear datos sintéticos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    df_ejemplo = pd.DataFrame({
        'unidad_negocio': np.random.choice(['Elektra', 'Banco Azteca', 'Salinas y Rocha', 'TV Azteca'], n_samples),
        'fecha_compra': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'monto_compra': np.random.normal(50000, 15000, n_samples),
        'categoria_producto': np.random.choice(['Tecnología', 'Mobiliario', 'Servicios', 'Marketing'], n_samples),
        'proveedor': np.random.choice(['Proveedor_A', 'Proveedor_B', 'Proveedor_C', 'Proveedor_D'], n_samples),
        'cantidad': np.random.randint(1, 100, n_samples),
        'descuento': np.random.uniform(0, 0.3, n_samples),
        'urgencia': np.random.choice(['Alta', 'Media', 'Baja'], n_samples)
    })
    
    # Hacer que monto_compra sea más realista
    df_ejemplo['monto_compra'] = np.abs(df_ejemplo['monto_compra'])
    
    print(f"✅ Datos de ejemplo creados: {df_ejemplo.shape}")
    
    # Ejecutar pipeline completo
    try:
        # PASO 2: Exploración inicial
        df_explorado = exploracion_inicial(df_ejemplo)
        
        # PASO 3: Limpieza
        df_limpio, cols_num, cols_cat = limpiar_datos(df_explorado)
        
        # PASO 4: EDA
        correlation_matrix = analisis_exploratorio(df_limpio, cols_num, cols_cat)
        
        # PASO 5: Ingeniería de características
        df_features, encoders = ingenieria_caracteristicas(df_limpio)
        
        # PASO 6: Preparación para modelado (usando 'monto_compra' como target)
        target = 'monto_compra'
        X_train, X_test, y_train, y_test, scaler = preparar_modelado(df_features, target)
        
        if X_train is not None:
            # PASO 7: Entrenamiento
            resultados = entrenar_modelos(X_train, y_train, X_test, y_test)
            
            # PASO 8: Evaluación
            mejor_modelo, df_res = evaluar_modelos(resultados, y_test)
            
            # PASO 9: Exportar resultados
            exportar_resultados(df_res)
            
            print("\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
            print("="*60)
            print("📋 PRÓXIMOS PASOS:")
            print("1. Cambia la ruta del archivo Excel con tus datos reales")
            print("2. Ajusta la variable objetivo según tus necesidades")
            print("3. Modifica los parámetros de los modelos si es necesario")
            print("4. Ejecuta predicciones con datos nuevos")
            
        else:
            print("❌ Error en la preparación de datos. Verifica la variable objetivo.")
    
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar si es el script principal
if __name__ == "__main__":
    # Ejecutar análisis completo
    main()
    
    print("\n" + "="*60)
    print("🔧 INSTRUCCIONES DE USO:")
    print("="*60)
    print("1. Cambia la variable 'ruta_archivo' por la ruta de tu Excel")
    print("2. Ejecuta: main()")
    print("3. Para usar funciones individuales:")
    print("   - df = cargar_datos('tu_archivo.xlsx')")
    print("   - df_clean, cols_num, cols_cat = limpiar_datos(df)")
    print("   - Y así sucesivamente...")
    print("\n✨ ¡Tu modelo predictivo está listo!")
