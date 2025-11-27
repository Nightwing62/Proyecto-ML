# Análisis No Supervisado — PCA (Principal Component Analysis)

## 1. Introducción 

Este análisis corresponde a la etapa no supervisada del proyecto de scoring crediticio.  
El objetivo es aplicar **PCA (Principal Component Analysis)** sobre el **conjunto de entrenamiento** utilizado por el modelo supervisado, con el fin de:

- Identificar Redundancias entre variables,
- Analizar estructura interna del dataset,
- Visualizar los patrones globales,
- Evaluar si la reducción de dimensionalidad como home credit, donde existen cientos de variables derivadas del historial financiaero de los clientes.

---

## 2. Instruccion de ejecución 

1. Abrir la carpeta notebooks y despues abrir el archivo llamado 'unsupervised_pca.ipynb'
2. Después hay que ir ejecutando todas las celdas en orden 
3. El Notebook utiliza el dataset del proyecto: /artifacts/train_merged.parquet
4. Las variables de entrada corresponden exactamente a las features utilizadas por el modelo LightGBM, almacenadas en:/artifacts/model_features.txt
5. El pipeline realiza:
- carga los datos,
- imputación ('fillna(0)'),
- estandarización ('StandardScaler'),
- graficos de varianza y proyección 2d

## 3. Resultados y análisis 

### 3.1 Varianza Explicada PCA

El analisis muestra que:
- PC1 explica aproximadamente **10–15%** de la varianza.
- Las primeras **10** componentes explican alrededor del **50%**.
- Para alcanzar **80%**, se requieren cerca de **50 componentes**.
- Para llegar al **90%**, es necesario incluir más de **150 componentes**

Interpretación:
La información del dataset está altamente distribuida entre muchas variables.  
No existe una estructura simple de baja dimensionalidad que permita reducir eficientemente la cantidad de features sin perder información relevante.  
Esto es normal en datasets como el de Home Credit, donde el comportamiento crediticio genera cientos de atributos derivados.

---

### 3.2 Interpretación de PC1 (loading matrix)

Las variable con mayor carga en PC1 fueron:

- `INST_NUM_INSTALMENT_VERSION_COUNT`
- `INST_SK_ID_PREV_COUNT`
- `INST_DAYS_INSTALMENT_COUNT`
- `INST_NUM_INSTALMENT_NUMBER_COUNT`
- `INST_SK_ID_CURR_COUNT`
- `INST_AMT_INSTALMENT_COUNT`
- `INST_AMT_PAYMENT_COUNT`
- `INST_DAYS_ENTRY_PAYMENT_COUNT`
- `CCB_MONTHS_BALANCE_MEAN`
- `CCB_CNT_INSTALMENT_MATURE_CUM_COUNT`

Interpretación:
PC1 resume información relacionada con el historial crediticio del cliente, incluyendo:

- el número de cuotas,
- los montos pagados,
- la cantidad de créditos asociados,
- atrasos,
- comportamiento en POS/CASH y tarjetas.

Esto indica que la mayor fuente de variabilidad del dataset proviene del comportamiento financiero previo del cliente, más que de variables demográficas o de ingreso.

---

### 3.3 Proyección PC1 vs PC2 
- Los clientes morosos (TARGET=1) y no morosos (TARGET=0) aparecen mezclados.
- No existe separación clara entre ambos grupos.
- Las primeras dos componentes principales no capturan de manera lineal patrones relacionados con el riesgo de incumplimiento.

Interpretación:
El riesgo crediticio depende de relaciones no lineales y de alta dimensionalidad, por lo que PCA no es adecuado para separar o discriminar entre morosos y no morosos. Esto justifica el uso de modelos como LightGBM, que sí capturan interacciones complejas.

---

## 4. ¿PCA debe incorporarse al proyecto final?
## Razones para no incorporarlo 

Segun los resultados obtenidos:
- la reducción de dimensionalidad no es eficiente.
- Se requieren más de 150 componentes para conservar el 90% de la información.
- Las primeras componentes no están correlacionadas de manera evidente con la variable TARGET.
- PCA reduce interpretabilidad del modelo (las nuevas features no tienen significado financiero directo).
- Modelos como LightGBM manejan de manera natural alta dimensionalidad y relaciones no lineales.

Conclusión:
PCA NO debe usarse como preprocesamiento en el modelo de scoring pues no mejora el desempeño ni la interpretabilidad, y podría incluso degradar o empeorar el modelo.

### 4.2 Valor del análisis PCA dentro del proyecto

Aun cuando PCA no se integrará al pipeline final, el análisis sí aporta valor:

- permite entender qué grupos de variables dominan la estructura del dataset,
- confirma la alta dimensionalidad del problema,
- ayuda a detectar redundancia entre variables derivadas,
- aporta transparencia al análisis exploratorio del proyecto.

---

## 5. Punto final

El análisis PCA permitió comprender mejor la estructura interna del dataset.  
Sin embargo, debido a la complejidad y naturaleza no lineal del riesgo crediticio, PCA no es adecuado como etapa de reducción de dimensionalidad en el modelo supervisado.  
Aun así, constituye un análisis exploratorio relevante para entender el comportamiento financiero subyacente en los datos del proyecto.

---