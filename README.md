PROYECTO: Predicción de Riesgo de Incumplimiento Crediticio
==========================================================

Autores: 
Felipe Cardenas
Jose Saez
Docente: Matias Ezequiel Arriola Aravena
Curso: Machine Learning
Dataset: Home Credit Default Risk

----------------------------------------------------------
1. CONTEXTO DEL NEGOCIO
----------------------------------------------------------
Una institución financiera necesita mejorar su proceso de evaluación de solicitudes
de crédito. El objetivo es predecir la probabilidad de que un solicitante incumpla
el pago de su préstamo, con el fin de reducir pérdidas y apoyar la toma de decisiones
de aprobación de crédito.

----------------------------------------------------------
2. OBJETIVO DEL PROYECTO
----------------------------------------------------------
Desarrollar un modelo de clasificación binaria que estime la probabilidad de default
(incumplimiento de pago), integrando múltiples fuentes de datos y desplegando el
modelo final como una API REST consumible.

----------------------------------------------------------
3. ESTRUCTURA DEL PROYECTO (CRISP-DM)
----------------------------------------------------------
/CARPETA BASE
│
├── 01_data_understanding/
│   └── Análisis exploratorio de datos (EDA)
│
├── 02_data_preparation/
│   └── Limpieza, agregaciones e ingeniería de características
│
├── 03_modeling/
│   └── Entrenamiento y validación del modelo
│
├── 04_evaluation/
│   └── Evaluación final sobre conjunto de test
│
├── 05_deployment/
│   └── API REST con FastAPI
│
├── artifacts/
│   ├── model_lgbm.joblib
│   ├── model_features.txt
│   ├── test_idx.joblib
│   └── métricas y datasets procesados
│
├── requirements.txt
└── README.txt

----------------------------------------------------------
4. DESCRIPCIÓN DEL DATASET
----------------------------------------------------------
Se utiliza el dataset "Home Credit Default Risk", el cual contiene:
- Una tabla principal de solicitudes de crédito
- Tablas secundarias con información histórica del cliente:
  bureau, bureau_balance, previous_application,
  POS_CASH_balance, credit_card_balance, installments_payments

El dataset es altamente desbalanceado y de alta dimensionalidad.

----------------------------------------------------------
5. FASE 2: DATA PREPARATION
----------------------------------------------------------
En esta fase se realizaron:

- Lectura de múltiples tablas en formato parquet
- Selección exclusiva de variables numéricas
- Agregaciones por cliente usando funciones estadísticas:
  mean, max, min, count (y std en variables donde tiene sentido)
- Unión progresiva de las tablas agregadas a la tabla principal
- Optimización de memoria reduciendo el número de features
- Generación del dataset final consolidado para modelado

El resultado es un dataset de entrenamiento con cientos de features agregadas
a nivel de cliente.

----------------------------------------------------------
6. FASE 3: MODELING
----------------------------------------------------------
Se entrenó un modelo LightGBM para clasificación binaria.

Decisiones técnicas:
- Se utilizó LightGBM por su eficiencia con alta dimensionalidad
- Se seleccionaron solo variables numéricas
- Se imputaron valores nulos con cero
- Se dividieron los datos en:
    70% entrenamiento
    15% validación (early stopping)
    15% test (reservado para evaluación final)

Se guardaron:
- El modelo entrenado (model_lgbm.joblib)
- La lista de features utilizadas (model_features.txt)
- Los índices del conjunto de test

----------------------------------------------------------
7. FASE 4: EVALUATION
----------------------------------------------------------
La evaluación final se realizó exclusivamente sobre el conjunto de test
no visto por el modelo.

Métricas utilizadas:
- AUC-ROC
- Matriz de confusión
- Precision, Recall y F1-score

Resultados observados:
- Buen poder discriminativo (AUC ~ 0.77)
- Bajo recall de la clase minoritaria, consistente con un dataset desbalanceado
- El modelo prioriza minimizar falsos positivos, alineado con riesgo financiero

----------------------------------------------------------
8. FASE 5: DEPLOYMENT (API)
----------------------------------------------------------
El modelo fue desplegado como una API REST usando FastAPI.

Características principales:
- Validación de entrada con Pydantic
- Uso exclusivo de las features del entrenamiento
- Manejo de errores con HTTPException
- Lógica de negocio integrada para toma de decisiones

Endpoint principal:
POST /evaluate_risk
----------------------------------------------------------
9. EJECUCION REAL DE  API
----------------------------------------------------------

-Entrada (JSON):
{
  "data": {
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000
  }
}


-Salida:
{
  "probabilidad_default": 0.2241,
  "decision": "APROBAR"
}


-Reglas de decisión:
- Probabilidad < 0.30 → APROBAR
- 0.30 ≤ Probabilidad < 0.60 → REVISIÓN MANUAL
- Probabilidad ≥ 0.60 → RECHAZAR


----------------------------------------------------------
10. EJECUCIÓN DEL PROYECTO
----------------------------------------------------------
0. crear carpeta de nombre 'data' en carpeta base y dentro colocar dataset Home Credit Default Risk
1. Crear entorno virtual
2. Instalar dependencias:
   pip install -r requirements.txt

3. Ejecutar fases en orden:
   python 02_data_preparation/preprocess.py
   python 03_modeling/train_model.py
   python 04_evaluation/evaluate_model.py

4. Levantar API:
   python -m uvicorn 05_deployment.api:app --reload

5. Probar API en:
   http://127.0.0.1:8000/docs

----------------------------------------------------------
11. CONCLUSIÓN
----------------------------------------------------------
El proyecto implementa un flujo completo de Machine Learning siguiendo
CRISP-DM, desde la exploración de datos hasta el despliegue del modelo.
El resultado es una solución funcional, reproducible y alineada con
un caso de negocio real en el sector financiero.

