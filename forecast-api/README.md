#  API de Pronóstico de Gasto y Monitoreo Presupuestal

API RESTful empresarial para análisis predictivo y monitoreo de gastos corporativos usando Machine Learning (XGBoost, Prophet) siguiendo principios SOLID.

##  Características Principales

###  Pronóstico Dinámico
- **Monitoreo 2025**: Compara gasto real acumulado vs predicción inicial por dimensiones
- **Proyección 2026+**: Predice gasto total anual futuro con distribución mensual
- Intervalos de confianza estadísticos
- Desglose multidimensional (ORIGEN, UN, CATEGORIA, CLASE, FAMILIA, DESTINO)
- Comparación automática con presupuesto asignado
- Alertas inteligentes de desviaciones

###  Calidad de Datos
- Verificación de completitud (valores nulos, desconocidos)
- Validación de consistencia (negativos, fechas inválidas)
- Puntuación de calidad 0-100 con estado general
- Identificación de problemas con ejemplos
- Recomendaciones automáticas de limpieza

###  Detección de Anomalías
- Isolation Forest para identificar gastos inusuales
- Top N anomalías con contexto detallado
- Alertas de patrones sospechosos
- Distribución por dimensiones clave

###  Recomendaciones Estratégicas
- Optimización de costos por proveedor/origen
- Oportunidades de consolidación de compras
- Análisis de tendencias temporales
- Ahorro potencial cuantificado
- Resumen ejecutivo accionable

##  Arquitectura y Principios SOLID

La API está construida siguiendo los principios SOLID para garantizar mantenibilidad y extensibilidad:

- **SRP (Single Responsibility)**: Cada servicio/clase tiene una única responsabilidad
  - `ModelService`: Solo gestión de modelos ML
  - `DataValidator`: Solo validación de datos
  - `AnomalyDetector`: Solo detección de anomalías
  - `RecommendationEngine`: Solo generación de recomendaciones
  - `ForecastService`: Solo lógica de pronóstico

- **OCP (Open/Closed)**: Extensible sin modificar código existente
  - Estrategias de pronóstico (`MonitoreoStrategy`, `ProyeccionStrategy`)
  - Sistema de excepciones personalizado
  - Arquitectura de plugins para nuevos modelos

- **DIP (Dependency Inversion)**: Abstracción de dependencias
  - `DataRepository`: Abstrae acceso a base de datos
  - `ModelService`: Abstrae carga y uso de modelos ML
  - Inyección de dependencias via FastAPI

##  Instalación y Configuración

### Prerrequisitos
- Python 3.10+
- SQL Server con ODBC Driver 17
- Modelos ML pre-entrenados (.pkl)

### 1. Clonar repositorio
```bash
git clone <repository-url>
cd forecast-api
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

**Configuración mínima requerida en `.env`:**
```env
# Database
DB_SERVER=
DB_NAME=
DB_USER=
DB_PASSWORD=

# Artifacts
MODEL_PREDICCION_PATH=./artifacts/modelo_prediccion_limpio.pkl
MODEL_TEMPORAL_PATH=./artifacts/modelo_demanda_temporal.pkl
ENCODERS_PATH=./artifacts/encoders.pkl
ANALISIS_PATH=./artifacts/analisis.pkl
```

### 5. Colocar artefactos ML
Copiar los siguientes archivos en `./artifacts/`:
- `modelo_prediccion_limpio.pkl` (XGBoost R²=99.43%)
- `modelo_demanda_temporal.pkl` (Prophet)
- `encoders.pkl` (LabelEncoders)
- `analisis.pkl` (Análisis previos)

### 6. Ejecutar API
```bash
# Desarrollo
python main.py

# Producción con Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

##  Uso de la API

### Endpoints Principales

#### 1. Health Check
```bash
GET /health
```

#### 2. Pronóstico Dinámico (Monitoreo o Proyección)
```bash
POST /forecast/dynamic/
Content-Type: application/json

{
  "anio_objetivo": 2026,
  "presupuesto_asignado": 50000000.0,
  "dimensiones": ["ORIGEN", "CATEGORIA", "UN"],
  "incluir_intervalos_confianza": true
}
```

**Respuesta para Proyección 2026:**
```json
{
  "modo": "proyeccion",
  "anio": 2026,
  "gasto_proyectado_total": 48500000.50,
  "intervalo_confianza_inferior": 43650000.45,
  "intervalo_confianza_superior": 53350000.55,
  "proyeccion_mensual": [
    {
      "mes": 1,
      "mes_nombre": "Enero",
      "gasto_predicho": 4200000.0,
      "intervalo_inferior": 3780000.0,
      "intervalo_superior": 4620000.0
    }
  ],
  "desgloses": {
    "ORIGEN": [
      {
        "dimension": "ORIGEN",
        "valor": "ARIBA SAP",
        "gasto_total": 30000000.0,
        "gasto_promedio": 12500.0,
        "cantidad_pedidos": 2400,
        "porcentaje_total": 61.85
      }
    ]
  },
  "presupuesto_asignado": 50000000.0,
  "diferencia_presupuesto": -1500000.0,
  "diferencia_porcentual": -3.0,
  "alertas": [],
  "nivel_confianza": 0.95,
  "datos_historicos_meses": 108
}
```

#### 3. Verificación de Calidad de Datos
```bash
POST /data/quality-check/
Content-Type: application/json

{
  "incluir_detalles": true,
  "limite_ejemplos": 10
}
```

#### 4. Detección de Anomalías
```bash
POST /analysis/anomalies/
Content-Type: application/json

{
  "contamination": 0.05,
  "top_n": 20,
  "incluir_contexto": true
}
```

#### 5. Recomendaciones Estratégicas
```bash
POST /analysis/recommendations/
Content-Type: application/json

{
  "dimensiones_analisis": ["ORIGEN", "UN"],
  "incluir_tendencias": true
}
```

### Atajos Simples

```bash
# Pronóstico rápido de un año
GET /forecast/year/2026?presupuesto=50000000

# Resumen de calidad
GET /data/quality-summary/

# Anomalías de un año
GET /analysis/anomalies/year/2025?top_n=20

# Recomendaciones rápidas
GET /analysis/recommendations/quick
```

## 🔧 Estructura del Proyecto

```
forecast-api/
├── main.py                      # Aplicación FastAPI principal
├── requirements.txt             # Dependencias Python
├── .env.example                 # Ejemplo de configuración
├── README.md                    # Este archivo
│
├── config/                      # Configuración
│   ├── settings.py             # Settings con Pydantic
│   └── database.py             # Conexión SQL Server
│
├── models/                      # Modelos de datos
│   ├── schemas.py              # Pydantic schemas (Request/Response)
│   └── enums.py                # Enumeraciones y constantes
│
├── services/                    # Lógica de negocio (SOLID)
│   ├── model_service.py        # Gestión de modelos ML
│   ├── forecast_service.py     # Lógica de pronóstico
│   ├── data_validator.py       # Validación de datos
│   ├── anomaly_detector.py     # Detección de anomalías
│   └── recommendation_engine.py # Motor de recomendaciones
│
├── repositories/                # Capa de datos
│   └── data_repository.py      # Acceso a base de datos
│
├── api/                         # Rutas y dependencias
│   ├── dependencies.py         # FastAPI dependencies
│   └── routes/
│       ├── forecast.py         # Rutas de pronóstico
│       ├── data_quality.py     # Rutas de calidad
│       └── analysis.py         # Rutas de análisis
│
├── utils/                       # Utilidades
│   ├── logger.py               # Sistema de logging
│   └── exceptions.py           # Excepciones personalizadas
│
└── artifacts/                   # Modelos ML (.pkl)
    ├── modelo_prediccion_limpio.pkl
    ├── modelo_demanda_temporal.pkl
    ├── encoders.pkl
    └── analisis.pkl
```

## 🤖 Modelos Machine Learning

### 1. XGBoost Regressor (Predicción de Gasto)
- **Métrica**: R² = 99.43%, RMSE = $6,284.91, MAE = $776.23
- **Features**: 26 variables sin data leakage
- **Uso**: Predicción de TOTALPESOS para proyecciones futuras

### 2. Prophet (Series Temporales)
- **Uso**: Pronóstico mensual con estacionalidad
- **Componentes**: Trend + Yearly Seasonality
- **Intervalos**: Confidence intervals incluidos

### 3. Isolation Forest (Anomalías)
- **Contamination**: 5% por defecto
- **Uso**: Detección de gastos atípicos

##  Base de Datos

### Tablas Principales
- `NSC_Temporal_SPPI_2025`: Datos de compras
- `CAT_EKT_TVA_TPE`: Catálogo de centros de costo

### Columnas Clave
- `FECHAPEDIDO`: Fecha del pedido
- `TOTALPESOS`: Monto del gasto
- `ORIGEN`: Fuente del pedido (ARIBA SAP, NSC, etc.)
- `UN`: Unidad de negocio
- `CATEGORIA`, `CLASE`, `FAMILIA`: Clasificación de producto
- `DESTINO`: Centro de costo destino

##  Seguridad

- Variables de entorno para credenciales sensibles
- Validación de entrada con Pydantic
- Manejo robusto de excepciones
- Logging de todas las operaciones
- CORS configurable

##  Rendimiento

- Pool de conexiones SQL (10 conexiones base)
- Caching opcional con Redis
- Procesamiento asíncrono con FastAPI
- Carga lazy de modelos ML
- Optimización de queries SQL con GROUP BY

##  Troubleshooting

### Error: "Modelo no cargado"
```bash
# Verificar que los .pkl existan
ls -la artifacts/

# Revisar logs
tail -f logs/api.log
```

### Error: "Database connection failed"
```bash
# Probar conexión manualmente
python -c "from config.database import db_manager; print(db_manager.test_connection())"

# Verificar ODBC Driver
odbcinst -q -d
```

### Error: "Datos insuficientes"
```bash
# Verificar datos en SQL Server
SELECT COUNT(*), MIN(FECHAPEDIDO), MAX(FECHAPEDIDO) 
FROM NSC_Temporal_SPPI_2025
```

##  Logs

Los logs se guardan en `./logs/api.log` con rotación automática:
- Rotación: cada 10 MB
- Retención: 30 días
- Compresión: ZIP

##  Testing

```bash
# Tests unitarios
pytest tests/

# Coverage
pytest --cov=services --cov=repositories tests/
```

##  Licencia

Uso interno corporativo - Todos los derechos reservados

## 👥 Contacto y Soporte

Para soporte técnico, contactar al equipo de Data Science/Analytics.

---

**Versión**: 1.0.0  
**Última actualización**: Octubre 2025  
**Autor**: Equipo de Ciencia de Datos