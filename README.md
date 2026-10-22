# Faurecast: Plataforma de pronóstico de fútbol y aprendizaje Continuo

**Faurecast** es una plataforma en Python diseñada para pronosticar resultados de partidos de fútbol (goles, marcador exacto y signo $1X2$), con capacidad de extenderse progresivamente hacia la predicción de estadísticas avanzadas (tarjetas, saques de esquina, minutos de gol, etc.).

El proyecto se ejecuta mediante **scripts en Python** y utiliza **almacenamiento en archivos dinámicos** (SQLite/Parquet/JSON) sin necesidad de infraestructura en la nube o servidores desplegados. La automatización de procesos (scraping, evaluación de jornadas y reentrenamiento) se realiza mediante **GitHub Actions**.

---

## Inicio Rápido

Prepara el entorno virtual, instala las dependencias y ejecuta los scripts directamente desde tu consola:

```bash
# 1. Crear el entorno virtual
py -m venv .venv

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate # En Windows (PowerShell)
source .venv/bin/activate # En macOS / Linux (o Git Bash)

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Inicializar y poblar la base de datos
python scripts/01_init_db.py

# 5. Generar pronósticos para la jornada activa
python scripts/02_run_forecast.py

# 6. Evaluar resultados y reentrenar el modelo
python scripts/03_evaluate_and_train.py
```

---

## 1. Misión, alcance y principios del proyecto

1. **Objetivo Principal:** Pronosticar el marcador exacto y resultado ($1X2$) de los partidos. A medida que el sistema se consolide, se añadirán módulos para otras estadísticas.
2. **Ejecución Ligera (Script-First):** El sistema arranca como un conjunto de scripts CLI limpios en Python. Capas adicionales como pruebas unitarias avanzadas, APIs REST o interfaces de usuario se irán evaluando e incorporando solo cuando sean necesarias.
3. **Infraestructura Cero (Serverless / File-Based):** Todos los datos (histórico, predicciones, métricas y estado del modelo) se guardan localmente en la carpeta del proyecto mediante archivos SQLite y/o archivos de datos (Parquet/CSV/JSON).
4. **Automatización Gratuita:** Uso de **GitHub Actions** (`cron workflows`) para automatizar el raspado de datos, la generación semanal de pronósticos y la evaluación post-jornada.
5. **Evaluación Transparente (Walk-Forward):** Las predicciones se registran e inmovilizan antes de cada jornada. Al terminar los partidos, se comparan con los resultados reales y se calcula el rendimiento sin fuga de datos (*data leakage*).

---

## 2. Aprovisionamiento de datos y gestión del dominio

### Ámbito inicial
- **Competición:** Liga Española (LaLiga) en primera instancia, manteniendo la abstracción para añadir otras ligas en el futuro.
- **Ventana Temporal:** 
  * **Temporada 2025/2026:** Datos históricos de calibración y entrenamiento base.
  * **Temporada 2026/2027:** Ejecución activa del bucle de pronóstico y evaluación en vivo.

### Abstracción de datos (`MatchDataProvider`)
Los datos se obtienen mediante librerías de Python como `soccerdata` o archivos locales procesados. La lógica se aísla tras la clase base `MatchDataProvider` para que, si en el futuro se desea implementar un scraper propio (Playwright / BeautifulSoup), no haya que modificar el resto del sistema.

---

## 3. Estrategia (Flexible y gradual)

El vector de características **no está cerrado de forma rígida**; se determinará mediante experimentación empírica. Se busca un equilibrio para no recargar el modelo con variables superfluas que provoquen *overfitting*.

### Atributos Candidatos a Experimentación
- **Elo del Equipo ($\mu$):** Puntuación dinámica de fuerza del equipo.
- **Fiabilidad del Elo ($\sigma$):** Grado de certeza sobre el Elo.
  * *Equipos asentados:* Alta fiabilidad.
  * *Recién ascendidos:* Baja fiabilidad inicial que aumenta a medida que acumulan partidos en la categoría.
- **Goles Promedio (Local/Visitante):** Capacidad goleadora y concesión defensiva reciente.
- **Contexto de Dificultad del Rival (Ajuste por Elo Opponent):** Ponderación del rendimiento según la entidad del rival. No computa igual marcar 3 goles al Real Madrid o Barcelona que a un equipo recién ascendido o de zona baja.
- **Métricas de Forma Reciente:** Promedios ponderados por recencia (EWMA).
- **Días de Descanso / Fatiga:** (Candidato a extensión).
- **Impacto de Bajas / Lesiones:** (Candidato a extensión).

---

## 4. Arquitectura de Machine Learning & Gymnasium

```text
                  [Datos Históricos y Jornada Activa]
                                  │
                                  ▼
                    [Feature Extractor / Builder]
       (Calcula Elo, Fiabilidad, Rendimiento Ajustado por Rival)
                                  │
                                  ▼
               [Motor 1: Red Neuronal (PyTorch / Scikit-Learn)]
                     ├── Predicción Goles Local (y_L)
                     └── Predicción Goles Visitante (y_V)
                                  │
                                  ▼
               [Motor 2: Entorno Gymnasium (SeasonEnv)]
           (Optimización de decisiones y calibración de sesgos)
                                  │
                                  ▼
                [Archivos Dinámicos (SQLite / Parquet)]
                     ├── /data/faurecast.db
                     └── /models/model_v1.pt
```

### Motor 1: Red Neuronal Supervisada
- **Entrada:** Vector reducido de características seleccionadas.
- **Salida:** Regresión doble (Goles esperados Local / Visitante) o distribución probabilística de marcadores.
- **Técnicas de Control:** Regularización ($L_2$, Dropout) para asegurar que el modelo no memorice los datos.

### Motor 2: Entorno Gymnasium (`SeasonEnv`)
- Se utiliza para entrenar un agente de Reinforcement Learning que tome las predicciones crudas de la Red Neuronal y aprenda a realizar ajustes dinámicos de sesgo según el contexto de la jornada, maximizando una función de recompensa basada en el acierto de **Marcador Exacto (+3 pts)** y **Signo 1X2 (+1 pt)**.

---

## 5. Almacenamiento dinámico en archivos (Sin servidor)

Toda la persistencia reside en la carpeta `/data` del repositorio:

- **`faurecast.db` (SQLite):** Archivo de base de datos relacional ligero procesado vía **SQLAlchemy**.
  * `leagues`: Ligas registradas.
  * `teams`: Datos e identidades de equipos.
  * `matches`: Calendario, fechas, estatus y marcadores reales.
  * `predictions`: Registro inmutable de pronósticos (guardados antes de que se juegue el partido).
  * `evaluations`: Histórico de aciertos, MAE de goles y puntuaciones por jornada.
- **Modelos Exportados (`/models/`):** Archivos `.pt` (PyTorch) o `.joblib` con el estado de los pesos reentrenados tras cada jornada.

---

## 6. Automatización mediante GitHub Actions

Se configuran dos workflows `.yml` en `.github/workflows/`:

1. **`weekly_forecast.yml`:** Se ejecuta automáticamente antes del inicio de cada jornada (ej. cada viernes). Descarga los próximos partidos, genera los pronósticos con el modelo actualizado, los guarda en `faurecast.db` y hace un *commit* automático al repositorio.
2. **`weekly_evaluation.yml`:** Se ejecuta al finalizar la jornada (ej. cada lunes). Descarga los resultados reales, evalúa la precisión, ejecuta el proceso de reentrenamiento/ajuste continuo y guarda los nuevos pesos y métricas.

---

## 7. Estructura del Repositorio

```text
faurecast/
├── .github/
│   └── workflows/
│       ├── weekly_forecast.yml
│       └── weekly_evaluation.yml
├── data/
│   ├── faurecast.db              # Base de datos SQLite (archivo local)
│   └── raw/                      # Capturas o cachés temporales de datos
├── models/
│   └── faurecast_nn.pt           # Pesos guardados de la Red Neuronal
├── config.yaml                   # Configuración de variables y parámetros
├── requirements.txt              # Dependencias de Python
├── README.md                     # Documentación principal del proyecto
├── src/
│   ├── database/                 # Tablas SQLAlchemy y conexión SQLite
│   │   ├── connection.py
│   │   └── models.py
│   ├── data_loader/              # Carga de datos vía soccerdata o local
│   │   ├── base.py
│   │   └── provider.py
│   ├── features/                 # Cálculo de Elo, Fiabilidad y Variables
│   │   └── feature_builder.py
│   ├── models/                   # Red Neuronal PyTorch y Gym Env
│   │   ├── neural_network.py
│   │   └── gym_env.py
│   └── services/                 # Servicios de predicción y evaluación
│       ├── predict_service.py
│       └── evaluate_service.py
└── scripts/                      # Scripts CLI de ejecución manual/automatizada
    ├── 01_init_db.py             # Puebla BD con histórico 25/26 y J1-J7 de 26/27
    ├── 02_run_forecast.py        # Genera y registra pronósticos
    └── 03_evaluate_and_train.py  # Evalúa la jornada y reentrena el modelo
```

---

## 8. Guía de desarrollo

> **Instrucciones de implementación para la Fase 1:**
>
> Construir los módulos clave descritos en la estructura inicial con código Python funcional:
> 1. `src/database/models.py`: Modelos SQLAlchemy para SQLite (Ligas, Equipos, Partidos, Predicciones, Evaluaciones).
> 2. `src/database/connection.py`: Conexión a la base de datos local SQLite (`data/faurecast.db`).
> 3. `src/data_loader/provider.py`: Adaptador `MatchDataProvider` con opciones `SoccerDataProvider` y `SimulatedProvider` para cargar datos de la temporada 2025/2026 y la activa 2026/2027.
> 4. `scripts/01_init_db.py`: Script ejecutable CLI que inicialice las tablas y las pueble con los datos correspondientes según los argumentos `--reset` y `--source`.
