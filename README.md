# Sistema de Trading Algorítmico con Aprendizaje por Refuerzo

**Trabajo de Fin de Grado — Ciencia de Datos (UOC)**

## Descripción

Sistema completo de trading algorítmico para el mercado Forex que combina descubrimiento estadístico de señales con modelos de Reinforcement Learning (RL). El proyecto abarca todo el ciclo de vida de un sistema de ML en producción: desde la adquisición y procesamiento de datos hasta el despliegue y monitorización en tiempo real.

## Arquitectura del Sistema

El pipeline consta de 5 fases:

1. **Signal Discovery (Fase 1)**: Evaluación estadística de combinaciones de indicadores técnicos sobre datos de entrenamiento (60%)
2. **Optimización de Hiperparámetros (Fase 3)**: Optuna con 50 trials para encontrar la configuración óptima del agente RL
3. **Entrenamiento Ensemble (Fase 4)**: 30-fold cross-validation con PPO (Proximal Policy Optimization)
4. **Validación en Test (Fase 5)**: Evaluación sobre datos no vistos (20%) — gate de aprobación: PF ≥ 1.5, WR ≥ 40%
5. **Paper Trading**: Ejecución simulada en tiempo real con gestión de riesgo adaptativa (Kelly Criterion)

### Split de datos (cronológico)

```
|-------- 60% Training --------|---- 20% Validation ----|---- 20% Test ----|
         Fases 1, 3, 4                  (interno)              Fase 5
```

## Tecnologías

| Categoría | Tecnología |
|-----------|-----------|
| **ML/RL** | Stable Baselines3, PPO, Gymnasium |
| **Optimización** | Optuna (Bayesian hyperparameter search) |
| **Datos** | PostgreSQL, pandas, CSV |
| **Backend** | Node.js, Express |
| **Frontend** | HTML/CSS/JavaScript (dashboards) |
| **Infraestructura** | Docker, Docker Compose |
| **Lenguajes** | Python 3.11, JavaScript/TypeScript |

## Requisitos

- Docker y Docker Compose
- (Opcional) NVIDIA GPU con CUDA para entrenamiento acelerado
- ~2GB de espacio para datos CSV

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Dixter999/trading-TFG.git
cd trading-TFG
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

### 3. Obtener datos

Los archivos CSV de datos no están incluidos en el repositorio por su tamaño.
Consulta `data/README.md` para instrucciones de cómo obtener los datasets.

### 4. Levantar el stack

```bash
# Arrancar base de datos + backend + frontend
docker compose up -d

# Cargar datos CSV en la base de datos
docker compose run --rm db-loader

# Acceder al frontend
open http://localhost:8080
```

### 5. Ejecutar entrenamiento (opcional)

```bash
# Entrenar un modelo RL para una señal específica
docker compose --profile training run --rm training \
  python scripts/pipelines/run_hybrid_v4_pipeline.py \
  --symbol eurusd --direction long --signal EMA_RSI
```

## Estructura del Proyecto

```
trading-TFG/
├── src/                    # Código fuente Python
│   ├── gym_trading_env/    # Entorno Gymnasium para RL
│   ├── paper_trading/      # Motor de paper trading
│   ├── backtesting/        # Framework de backtesting
│   ├── analysis/           # Herramientas de análisis
│   └── database/           # Capa de acceso a datos
├── scripts/                # Scripts de automatización
│   ├── pipelines/          # Pipeline Hybrid V4
│   ├── discovery/          # Signal discovery
│   └── training/           # Entrenamiento local
├── backend/                # API REST (Node.js)
├── frontend-simple/        # Dashboards HTML
├── notebooks/              # Jupyter notebooks (narrativa TFG)
├── config/                 # Configuración
├── data/                   # Datos CSV (no incluidos en Git)
├── models/                 # Modelos entrenados
└── tests/                  # Tests
```

## Notebooks

Los notebooks Jupyter en `notebooks/` presentan el proyecto de forma narrativa:

1. `01_data_exploration.ipynb` — Análisis exploratorio de datos FOREX
2. `02_signal_discovery.ipynb` — Proceso de descubrimiento de señales
3. `03_rl_training.ipynb` — Entrenamiento RL y optimización
4. `04_backtesting_results.ipynb` — Resultados y métricas
5. `05_paper_trading_analysis.ipynb` — Análisis de paper trading

## Métricas del Proyecto

- **644 issues** gestionados en GitHub
- **912 commits** de desarrollo
- **284K líneas** de código Python
- **208 señales** evaluadas, 159 aprobadas en validación
- **2310 modelos** entrenados (30-fold × múltiples señales)

## Autor

Trabajo de Fin de Grado para el Grado en Ciencia de Datos
Universitat Oberta de Catalunya (UOC)

## Licencia

Este proyecto es parte de un trabajo académico. Todos los derechos reservados.
