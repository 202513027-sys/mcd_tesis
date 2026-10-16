# Tesis de Maestría en Ciencia de Datos — Modelo de Riesgo Crediticio

Proyecto reproducible de Machine Learning para predecir el incumplimiento de
pago (*default*) de clientes de tarjeta de crédito, desarrollado como base
experimental de la tesis de Melany Cristina Huarcaya.

---

## 1. Objetivo

Construir y evaluar de forma rigurosa un modelo de clasificación binaria de
riesgo crediticio, comparando un **baseline** contra **modelos propuestos**
de boosting, con:

- prevención explícita de *data leakage*,
- validación cruzada y múltiples corridas independientes para medir
  estabilidad,
- búsqueda de hiperparámetros correctamente aislada del test,
- diagnóstico de fairness (Fairlearn) y explicabilidad (SHAP),
- resultados y modelos reproducibles y documentados.

---

## 2. Estructura del proyecto

```text
thesis-project/
├── data/
│   ├── raw/                 # DATA.csv original, sin modificar
│   └── processed/           # data_prepared.csv (tras limpieza determinística)
├── notebooks/
│   └── 01_eda.ipynb         # análisis exploratorio (no contiene el pipeline)
├── src/
│   ├── utils.py             # funciones reutilizables (métricas, plots, SHAP, Fairlearn...)
│   ├── run_experiment.py    # pipeline experimental completo (punto de entrada único)
│   └── post_analysis.py     # post-análisis estadístico sobre results/ (no reentrena)
├── models/                  # modelos entrenados (.joblib)
├── results/
│   ├── metrics.csv          # métricas tidy de todas las corridas
│   ├── metrics_summary.csv  # mean ± std por modelo y métrica
│   ├── best_hyperparameters.json
│   ├── fairness_results.json
│   ├── shap_feature_selection.csv
│   ├── shap_importance_final_model.csv
│   ├── run_summary.json
│   ├── table5_runs.csv      # Tabla 5: media ± DE de las 10 corridas (sin split oficial)
│   ├── data_quality.json    # duplicados y categorías fuera de rango
│   ├── friedman_kendall_holm.csv, pairwise_wilcoxon_holm.csv,
│   │   assumptions_tests.csv, delong_pairs_holm.csv   # post-análisis (S5)
│   ├── run_log.txt, post_analysis_log.txt             # salida impresa de cada script
│   └── figures/              # 12 figuras (ver sección Resultados)
├── requirements.txt
└── README.md
```

---

## 3. Datos

- **Fuente**: conjunto **público** *Default of Credit Card Clients* (Yeh, 2009),
  UCI Machine Learning Repository, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H),
  licencia CC BY 4.0. Clientes de tarjeta de crédito de un banco de **Taiwán**,
  historial de abril a septiembre de 2005; montos en dólares taiwaneses (NT$).
  Usado también por Yeh y Lien (2009). Detalle, enlace de descarga, checksum y
  correspondencia de columnas en [`data/README.md`](data/README.md).
- **Archivo**: `data/raw/DATA.csv` (30,000 clientes, 25 columnas, separador `;`),
  mismo contenido que el original de UCI con las columnas renombradas al español.
  No se versiona en el repositorio; se descarga del enlace indicado.
- **Validez**: los resultados valen para esa población (Taiwán, 2005). No se
  generalizan a la banca peruana ni a periodos actuales sin validación externa.
- **Target**: `Y` — binaria (`1` = incumplimiento de pago el mes siguiente).
  **Desbalanceada**: 77.9% clase 0 / 22.1% clase 1 (6,636 positivos).
- **Identificador**: `ID` — único para las 30,000 filas, se excluye como
  predictor (no aporta señal, sólo identifica al cliente).
- **Variables numéricas** (29 tras ingeniería): `MONTO_PP`, `EDAD`,
  `PAGO_M1..M6`, `MONTO_FACTU_M1..M6`, `PAGO_EFECT_M1..M6`, y los agregados
  derivados `PAGO_M_{MAX,MIN,STD}`, `MONTO_FACTU_M_{MAX,MIN,STD}`,
  `PAGO_EFECT_M_{MAX,MIN,STD}`.
- **Variables categóricas**: `SEXO` (2 categorías), `EDUCACION` (documentado
  1-4), `ESTADO_CIVIL` (documentado 1-3).

### Preprocessing y limpieza

- **Sin nulos y sin `ID` duplicado.** Contando la columna `ID` no hay filas
  duplicadas; **sin contar `ID` hay 35 filas que repiten exactamente a otra**
  (70 filas involucradas) y 21 pares adicionales con las mismas variables pero
  distinta `Y`. Se conservan (no se eliminan). Conteo impreso por
  `src/post_analysis.py` y guardado en `results/data_quality.json`.
- **Categorías anómalas documentadas, no eliminadas**: `EDUCACION` tiene
  categorías 0, 5 y 6 (345 filas: 14 + 280 + 51) fuera del rango documentado (1-4);
  `ESTADO_CIVIL` tiene categoría 0 (54 filas) fuera de rango (1-3). Se
  agrupan de forma determinística en la categoría más cercana/"otros"
  (`utils.clean_anomalous_categories`), una transformación fila a fila que
  no aprende ningún parámetro de la muestra y por lo tanto es segura de
  aplicar en cualquier momento del pipeline sin generar leakage.
- **Valores negativos en `MONTO_FACTU_Mx`**: representan saldo a favor del
  cliente (créditos/sobrepagos); son válidos, no se tratan como error.
- **Variables excluidas**: únicamente `ID`.

### Variables sensibles / temporales

- **Variable sensible (Fairlearn)**: `SEXO`. Se eligió por ser binaria, sin
  categorías anómalas, y por ser la variable demográfica estándar para este
  tipo de análisis en modelos de riesgo crediticio. `EDUCACION` y
  `ESTADO_CIVIL` se descartaron para este fin por arrastrar categorías fuera
  de rango. *(Decisión confirmada explícitamente con la autora antes de implementar.)*
- **Variable temporal / OOT**: el dataset **no contiene ninguna columna de
  fecha**. Los sufijos `_M1..M6` representan una ventana de 6 meses **por
  cliente**, no una línea de tiempo compartida entre clientes. Por lo tanto
  **no se construyó una partición Out-of-Time (OOT)** — hacerlo habría
  requerido inventar un supuesto temporal que los datos no sustentan. Esta es
  una limitación a tener en cuenta: los resultados no evalúan estabilidad del
  modelo ante *drift* temporal real, solo ante distintas particiones
  aleatorias de la misma muestra.

---

## 4. Metodología

### 4.1 Partición de datos

- Train/test estratificado por `Y`, `test_size=0.30`.
- Validación mediante `StratifiedKFold` (5 folds) dentro del train, usada en
  la búsqueda de hiperparámetros.
- No hay partición OOT (ver limitación arriba).

### 4.2 Prevención de leakage

- Todo el preprocesamiento que aprende parámetros de los datos
  (`StandardScaler`, `OneHotEncoder`) está dentro de un `sklearn.Pipeline` /
  `ColumnTransformer`, ajustado **siempre** solo con el train de cada
  fold/corrida — nunca con el dataset completo.
- La selección de variables por importancia SHAP (sección 4.4) se calculó
  usando **únicamente el train oficial**, a diferencia de una primera versión
  exploratoria del proyecto que también usaba el test en ese paso — esto se
  corrigió explícitamente en `run_experiment.py`.
- Las variables agregadas (MAX/MIN/STD) son operaciones fila a fila sin
  parámetros aprendidos, por lo que no generan leakage sin importar en qué
  momento se calculen.
- El punto de corte de clasificación (Youden's J) se calcula siempre sobre
  **train** y se aplica igual a test, para no usar información de test en la
  elección del umbral.
- Ninguna variable representa directa o indirectamente al target, ni
  contiene información posterior al evento que se predice.

### 4.3 Reproducibilidad

- `utils.set_seed(seed)` fija `PYTHONHASHSEED`, `random.seed` y `np.random.seed`.
- `BASE_SEED = 1234` en `run_experiment.py`; cada corrida usa
  `seed = BASE_SEED + run_id`.
- Semilla explícita en: partición train/test, `StratifiedKFold`,
  `RandomizedSearchCV`, y cada modelo (`random_state`).
- Verificado: dos ejecuciones completas del pipeline con la misma
  configuración producen exactamente las mismas métricas (AUC de test
  idéntico a la 4ª cifra decimal) y el modelo `.joblib` guardado reproduce el
  mismo AUC al cargarse.

### 4.4 Selección de variables (SHAP)

Un modelo LightGBM preliminar (parámetros por defecto) se entrena sobre el
**train oficial**, se calculan valores SHAP sobre ese mismo train, y se
conservan las variables con importancia SHAP relativa > 1% (29 de 38
columnas codificadas). Esta lista de variables queda fija y se aplica dentro
del `Pipeline` (paso `ColumnSelector`) tanto al split oficial como a las 10
corridas de estabilidad.

*Decisión metodológica a revisar*: la selección de variables se calculó una
sola vez (sobre el train oficial) y se reutiliza en las 10 corridas de
estabilidad, en vez de recalcularse en cada corrida con su propio train. Esto
mantiene la arquitectura comparable entre corridas y reduce el tiempo de
cómputo, pero implica que las corridas de estabilidad no son 100%
independientes entre sí en cuanto a la elección de variables.

### 4.5 Baseline vs. modelo propuesto

| Rol | Modelo | Detalle |
|---|---|---|
| Referencia mínima | `DummyClassifier` | `strategy='most_frequent'` |
| Baseline estadístico | Regresión Logística | `class_weight='balanced'` |
| Propuesto 1 | XGBoost | `scale_pos_weight` = razón neg/pos (≈3.52) |
| Propuesto 2 | LightGBM (DART) | `class_weight='balanced'` |

*(Conjunto de modelos confirmado explícitamente con la autora: mantener los
dos candidatos de boosting del notebook original, en vez de reducir a uno solo.)*

Se seleccionaron dos candidatos de boosting (en vez de uno) porque la
tesis original ya los comparaba y el costo adicional de mantenerlos es bajo;
no se agregaron modelos adicionales para no incrementar la complejidad sin
justificación.

### 4.6 Manejo del desbalance de clases

`class_weight='balanced'` para Regresión Logística y LightGBM;
`scale_pos_weight` equivalente para XGBoost (no soporta `class_weight`
directamente). *(Decisión confirmada explícitamente con la autora frente a
SMOTE o no aplicar balanceo, por ser simple y no introducir riesgo de leakage.)*

### 4.7 Búsqueda de hiperparámetros

`RandomizedSearchCV` (30 iteraciones, `scoring='roc_auc'`), **una sola vez**
sobre el train oficial:

- XGBoost: `StratifiedKFold(5)`, espacio sobre `n_estimators`, `max_depth`,
  `learning_rate`, `subsample`, `colsample_bytree`, `gamma`.
- LightGBM-DART: `cv=3`, espacio sobre `n_estimators`, `max_depth`,
  `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`,
  `drop_rate`, `skip_drop`.

Los mejores hiperparámetros encontrados quedan fijos y se reutilizan (con
distinta semilla del modelo) en las 10 corridas de estabilidad, en vez de
volver a buscarse en cada una. *(Decisión confirmada explícitamente con la
autora, por ser mucho más rápido y metodológicamente aceptado para un
análisis de estabilidad: se fija la arquitectura y se mide su variabilidad
ante distintos splits/semillas, en vez de optimizar 10 veces.)* Ver
`results/best_hyperparameters.json` para los valores exactos encontrados.

### 4.8 Corridas múltiples

`N_RUNS = 10` (parametrizable en `run_experiment.py`). Cada corrida usa un
split train/test estratificado distinto (`seed = BASE_SEED + run_id`) y
reentrena los 4 modelos con la configuración de hiperparámetros fija. Para
cada corrida se guardan en `results/metrics.csv` (formato *tidy*, una fila
por métrica): modelo, `run_id`, `seed`, dataset (train/test) y valor.

### 4.9 Métricas

ROC AUC, GINI (`2·AUC−1`), KS, LogLoss, Precision, Recall/Sensitivity,
Specificity, F1 y Accuracy (esta última solo como complemento, dado el
desbalance de clases). El punto de corte usado para las métricas de clase
(F1, Precision, Sensitivity, Specificity) es el óptimo de Youden's J
calculado sobre train.

### 4.10 Contraste de hipótesis (paso 9 de `run_experiment.py`)

Cada hipótesis se contrasta con una **métrica**, un **umbral** fijado antes
de ver los resultados (sección `CONFIG`) y una **prueba estadística**
(α = 0,05). Para HE1 y HE3 se entrenan variantes adicionales de los modelos,
tanto en el split oficial como en cada una de las 10 corridas:

| Variante | Hiperparámetros | Variables | Hipótesis |
|---|---|---|---|
| Principal | optimizados (RL: fijos) | seleccionadas por SHAP | HG, HE2, HE4 |
| Por defecto | por defecto de la librería (mismo manejo del desbalance) | seleccionadas | HE3 |
| Todas las variables | optimizados (RL: fijos) | las 38 variables | HE1 |

| Hipótesis | Métrica | Umbral | Prueba |
|---|---|---|---|
| HG | AUC-ROC en test | AUC ≥ 0,75 y mayor que RL | DeLong unilateral (split oficial); Wilcoxon pareado (corridas) |
| HE1 | ΔAUC (seleccionadas − todas) | pérdida ≤ 0,01 | DeLong unilateral de no inferioridad (margen −0,01) |
| HE2 | AUC, Recall, F1 | α = 0,05 | Friedman + post-hoc de Nemenyi (corridas); DeLong por pares |
| HE3 | ΔAUC (optimizado − por defecto) | mejora ≥ 0,01 | DeLong unilateral; Wilcoxon pareado (corridas) |
| HE4 | Spearman entre rankings SHAP de las corridas; Jaccard top-10 | ρ medio ≥ 0,70; Jaccard ≥ 0,60 | Correlación de Spearman |

La prueba de DeLong (implementación rápida de Sun & Xu, 2014, en
`utils.delong_test`) compara dos AUC calculados sobre las mismas
observaciones de test, teniendo en cuenta su correlación. Friedman/Nemenyi
sigue a Demšar (2006).

Salidas: `results/hypothesis_summary.csv` (tabla para la tesis),
`results/hypothesis_tests.json` (detalle), `results/delong_tests.csv`,
`results/metrics_hypotheses.csv`, `results/shap_importance_by_run.csv` y
`results/figures/13_auc_variantes_hipotesis.png`.

**Limitación**: las 10 corridas usan splits que se solapan entre sí y con el
train oficial (donde se eligieron variables e hiperparámetros), por lo que
las pruebas sobre corridas son algo optimistas; la prueba principal es
DeLong sobre el test oficial, que nunca se usó para ajustar nada.

---

## 5. Resultados

### 5.1 Modelo final (split oficial, `results/run_summary.json`)

| Modelo | AUC (test) | GINI | KS | F1 |
|---|---|---|---|---|
| Dummy | 0.500 | 0.000 | 0.000 | 0.000 |
| Regresión Logística | 0.743 | 0.486 | 0.388 | 0.507 |
| XGBoost | 0.783 | 0.565 | 0.418 | 0.512 |
| **LightGBM-DART (final)** | **0.783** | **0.567** | **0.426** | **0.525** |

El modelo final se selecciona automáticamente como el de mayor AUC en test
entre los dos candidatos de boosting — en esta ejecución, **LightGBM-DART**
(diferencia con XGBoost muy pequeña, ~0.0006 AUC).

### 5.2 Estabilidad entre las 10 corridas (`results/table5_runs.csv`, test)

| Modelo | AUC (mean ± std) | GINI | KS | F1 |
|---|---|---|---|---|
| Dummy | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Regresión Logística | 0.7400 ± 0.0056 | 0.4800 ± 0.0113 | 0.3848 ± 0.0086 | 0.5058 ± 0.0075 |
| XGBoost | 0.7861 ± 0.0046 | 0.5722 ± 0.0093 | 0.4329 ± 0.0105 | 0.5289 ± 0.0088 |
| LightGBM-DART | 0.7868 ± 0.0045 | 0.5735 ± 0.0091 | 0.4343 ± 0.0088 | 0.5353 ± 0.0086 |

Media ± DE de las **10 corridas** (`run_id` 0-9, semillas 1234-1243). No incluye
el split oficial (`run_id = -1`), que sí estaba promediado en
`metrics_summary.csv` (11 filas). La corrida 0 usa la misma semilla (1234) y
por tanto la misma partición que el split oficial.

Ambos modelos de boosting son consistentemente superiores al baseline
logístico (~4-5 puntos de AUC), con desviaciones estándar bajas (< 0.005 en
AUC) a través de las 10 corridas — el resultado es estable y no depende de
una partición particular "afortunada".

### 5.3 Fairness (Fairlearn, variable sensible `SEXO`, modelo final)

- Demographic parity difference: **0.037** | ratio: **0.897**
- Equalized odds difference: **0.022** | ratio: **0.917**

| Grupo | Selection rate | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| SEXO=1 | 0.363 | 0.720 | 0.460 | 0.666 | 0.544 |
| SEXO=2 | 0.326 | 0.741 | 0.412 | 0.667 | 0.509 |

**Interpretación**: las diferencias entre grupos son moderadas (todas < 0.04
en diferencia absoluta), sin una brecha grande de *selection rate* o
*recall* entre sexos. Esto **no constituye una certificación de que el
modelo es "justo"** — es un diagnóstico complementario al desempeño
predictivo, que debería contrastarse con criterio de negocio/regulatorio
antes de cualquier despliegue. Ver `results/fairness_results.json` y
`results/figures/11_fairness_by_group.png`.

**Limitación**: solo se evaluó `SEXO` como variable sensible (decisión
confirmada con la autora); no se evaluó fairness respecto a `EDUCACION` ni
`ESTADO_CIVIL` por las categorías anómalas ya documentadas.

### 5.4 Explicabilidad (SHAP)

Variables más importantes para el modelo final (`results/shap_importance_final_model.csv`,
figuras `09_shap_summary.png` y `10_shap_importance.png`): `PAGO_M1` (estado
de pago del mes más reciente) domina claramente la importancia, seguido de
`MONTO_PP` (línea de crédito) y `PAGO_M_MAX`.

**Importante — SHAP no implica causalidad**: los valores SHAP cuantifican
cuánto contribuye cada variable a la *predicción* de este modelo en
particular; no establecen una relación causal entre, por ejemplo, un atraso
de pago y el riesgo real de default. La dirección de la asociación (ver
`09_shap_summary.png` y `12_shap_dependence_top1.png`) es descriptiva del
comportamiento del modelo, no evidencia causal.

### 5.5 Figuras generadas (`results/figures/`)

| Archivo | Contenido |
|---|---|
| `01_target_distribution.png` | Distribución de `Y` |
| `02_train_test_distribution.png` | Comparación de distribuciones train vs. test |
| `03_metric_stability.png` | Estabilidad de métricas entre las 10 corridas |
| `04_learning_curve.png` | Learning curve del modelo final |
| `05_train_validation_comparison.png` | Train vs. test por modelo (diagnóstico de overfitting) |
| `06_confusion_matrix.png` | Matriz de confusión del modelo final (test oficial) |
| `07_roc_curve.png` | Curvas ROC de los 4 modelos |
| `08_precision_recall_curve.png` | Curvas Precision-Recall de los 4 modelos |
| `09_shap_summary.png` | SHAP summary/beeswarm plot del modelo final |
| `10_shap_importance.png` | Top variables por importancia SHAP |
| `11_fairness_by_group.png` | Métricas de desempeño por grupo (SEXO) |
| `12_shap_dependence_top1.png` | Dependence plot de la variable más importante |

No se generó un gráfico de PSI/drift porque, al no existir una partición
temporal (OOT), no hay dos ventanas de tiempo reales que comparar — se
evaluó en su lugar la similitud de distribuciones train/test (figura 02).

### 5.6 Diagnóstico de overfitting

La learning curve (`04_learning_curve.png`) muestra una brecha train-validación
moderada que se **reduce** a medida que aumenta el tamaño de muestra (de
~0.20 de AUC con poca data a ~0.05 con el train completo), sin señales de
overfitting severo. La comparación train vs. test (`05_train_validation_comparison.png`)
es consistente con esto: los modelos de boosting generalizan razonablemente
bien, sin caída abrupta de test respecto a train.

### 5.7 Contraste de hipótesis (`results/hypothesis_summary.csv`)

| Hipótesis | Resultado principal (test oficial) | Prueba | Veredicto |
|---|---|---|---|
| HG | LightGBM-DART AUC = 0,783 vs. RL 0,743 (+0,040) | DeLong p < 0,001; Wilcoxon p = 0,001 (gana 10/10) | Se confirma |
| HE1 | Con 29 de 38 variables la pérdida de AUC es ≤ 0,001 en los tres modelos | DeLong de no inferioridad p < 0,001 | Se confirma |
| HE2 | Boosting > RL; XGBoost ≈ LightGBM (dif. 0,0006, DeLong p = 0,51) | Friedman AUC p < 0,001; Nemenyi: RL difiere de ambos | Se confirma |
| HE3 | XGBoost +0,031 (p < 0,001); LightGBM-DART +0,002 (p = 0,086) | DeLong unilateral; Wilcoxon | Parcial |
| HE4 | Spearman medio 0,84 entre corridas; Jaccard top-10 = 0,67 | Spearman p < 0,001 | Se confirma |

---

## 6. Reproducibilidad — cómo ejecutar

```bash
pip install -r requirements.txt
python src/run_experiment.py
```

Esto genera (o regenera, de forma reproducible):

- `data/processed/data_prepared.csv`
- `results/metrics.csv`, `results/metrics_summary.csv`
- `results/*.json`, `results/*.csv` (hiperparámetros, fairness, SHAP)
- `results/hypothesis_summary.csv`, `results/hypothesis_tests.json`, `results/delong_tests.csv` (contraste de hipótesis)
- `results/figures/*.png` (13 figuras)
- `results/run_log.txt` (todo lo que imprime la corrida)
- Post-análisis (`src/post_analysis.py`, se llama al final automáticamente y
  también se puede correr solo con `python src/post_analysis.py`, sin reentrenar):
  `results/data_quality.json`, `results/table5_runs.csv`,
  `results/friedman_kendall_holm.csv` (Friedman + W de Kendall + Holm),
  `results/assumptions_tests.csv` (Shapiro-Wilk y Levene),
  `results/pairwise_wilcoxon_holm.csv`, `results/delong_pairs_holm.csv`
  (IC 95 % de DeLong, Holm, equivalencia ±0,01 y morosos adicionales) y
  `results/post_analysis_log.txt`
- `models/*.joblib` (los 4 modelos oficiales, listos para cargar con `joblib.load`)

Tiempo de ejecución aproximado: **6-12 minutos** en una máquina de 2 núcleos.
Todos los parámetros relevantes (número de corridas, folds, semilla,
iteraciones de búsqueda, umbral de selección de variables) están
centralizados en la sección `CONFIG` al inicio de `src/run_experiment.py` —
no hay rutas absolutas hardcodeadas (se usa `pathlib` relativo a la
ubicación del script).

El notebook `notebooks/01_eda.ipynb` es solo exploratorio (no reproduce el
experimento) y reutiliza las funciones de `src/utils.py` — no duplica lógica.

---

## 7. Decisiones metodológicas a revisar antes de usar estos resultados en la tesis

1. **No hay partición OOT** — el dataset no tiene variable temporal real
   entre clientes; solo se usó train/CV/test sobre una muestra transversal.
2. **Selección de variables SHAP fija** — se calculó una vez sobre el train
   oficial y se reutilizó en las 10 corridas, en vez de recalcularse por
   corrida.
3. **Hiperparámetros fijos entre corridas** — se buscaron una sola vez y se
   reutilizan con distinta semilla en las 10 corridas de estabilidad (en vez
   de una búsqueda independiente por corrida).
4. **Fairness evaluado solo en `SEXO`** — no se evaluó `EDUCACION` ni
   `ESTADO_CIVIL` por sus categorías anómalas.
5. **Modelo final elegido automáticamente** por mayor AUC en el split
   oficial (XGBoost y LightGBM-DART quedaron prácticamente empatados,
   diferencia ~0.0006 AUC) — vale la pena revisar si conviene privilegiar
   otro criterio (interpretabilidad, tiempo de inferencia, KS) para la
   tesis.
6. **Corrección respecto al notebook exploratorio original**: la selección
   de variables por SHAP en el notebook de `Ejecuciones/Tesis.ipynb` usaba
   también el test set para calcular los valores SHAP; en este pipeline se
   corrigió para usar únicamente el train.
