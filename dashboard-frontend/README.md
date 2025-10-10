# 📁 Estructura del Frontend

```
forecast-dashboard/
├── package.json
├── vite.config.js
├── tailwind.config.js
├── index.html
├── .env.example
│
├── public/
│   └── logo.svg
│
└── src/
    ├── main.jsx                    # Entry point
    ├── App.jsx                     # App principal
    │
    ├── config/
    │   └── api.config.js          # Configuración API
    │
    ├── services/                   # Services (SOLID - SRP)
    │   ├── ApiService.js          # Cliente HTTP base
    │   ├── ForecastService.js     # Servicio de pronóstico
    │   ├── AnalysisService.js     # Servicio de análisis
    │   └── DataQualityService.js  # Servicio de calidad
    │
    ├── hooks/                      # Custom Hooks
    │   ├── useForecast.js
    │   ├── useAnomalies.js
    │   ├── useRecommendations.js
    │   └── useDataQuality.js
    │
    ├── components/                 # Componentes (SOLID - SRP)
    │   ├── layout/
    │   │   ├── Sidebar.jsx
    │   │   ├── Header.jsx
    │   │   └── MainLayout.jsx
    │   │
    │   ├── common/
    │   │   ├── Card.jsx
    │   │   ├── StatCard.jsx
    │   │   ├── LoadingSpinner.jsx
    │   │   ├── ErrorBoundary.jsx
    │   │   └── Badge.jsx
    │   │
    │   ├── charts/
    │   │   ├── MonthlyBarChart.jsx
    │   │   ├── TrendLineChart.jsx
    │   │   ├── DonutChart.jsx
    │   │   ├── AreaChart.jsx
    │   │   └── HeatMap.jsx
    │   │
    │   └── sections/
    │       ├── ForecastSection.jsx
    │       ├── AnomaliesSection.jsx
    │       ├── RecommendationsSection.jsx
    │       └── DataQualitySection.jsx
    │
    ├── pages/                      # Páginas principales
    │   ├── Dashboard.jsx          # Dashboard principal
    │   ├── ForecastView.jsx       # Vista de pronóstico
    │   ├── AnalysisView.jsx       # Vista de análisis
    │   └── SettingsView.jsx       # Configuración
    │
    ├── utils/                      # Utilidades
    │   ├── formatters.js          # Formateo de números/fechas
    │   ├── colors.js              # Paleta de colores
    │   └── constants.js           # Constantes
    │
    └── styles/
        └── index.css              # Estilos globales + Tailwind
```

## 🎨 Características del Dashboard

### Visual & UX
- ✨ Diseño corporativo moderno y elegante
- 📊 Gráficas interactivas con Recharts
- 🎭 Animaciones suaves con Framer Motion
- 🌓 Modo oscuro/claro
- 📱 100% Responsive
- ⚡ Carga optimizada con React Query

### Funcional
- 📈 Dashboard principal con métricas clave
- 🔮 Pronóstico visual por año
- 🚨 Detección de anomalías interactiva
- 💡 Recomendaciones accionables
- 🔍 Calidad de datos en tiempo real
- 🔄 Auto-refresh cada 5 minutos

### Técnico (SOLID)
- **SRP**: Cada componente/servicio una responsabilidad
- **OCP**: Componentes extensibles vía props
- **DIP**: Servicios abstraídos con interfaces
- **ISP**: Hooks especializados