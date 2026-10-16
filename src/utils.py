"""
utils.py
========
Funciones reutilizables para el pipeline experimental de la tesis:
- Reproducibilidad (set_seed)
- Carga y tipado de datos
- Ingeniería de variables (agregados por fila, sin leakage)
- Métricas de clasificación (GINI, AUC, KS, LogLoss, Accuracy, F1, etc.)
- Evaluación de modelos y punto de corte óptimo (Youden's J)
- Visualizaciones (distribución, estabilidad, learning curves, SHAP, etc.)
- Guardado/carga de resultados y modelos

Todas las funciones son puras/reutilizables: ni notebooks ni run_experiment.py
deben reimplementar esta lógica.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

# Nota: no forzamos matplotlib.use("Agg") aquí a propósito. Al ejecutar
# run_experiment.py como script headless, matplotlib ya usa 'agg' por
# defecto (no hay DISPLAY) y savefig funciona igual con cualquier backend.
# Forzarlo aquí rompería el modo inline (%matplotlib inline) de los
# notebooks que hacen `import utils`.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn import metrics as mt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija la semilla en todos los generadores de números aleatorios relevantes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Carga y tipado de datos
# ---------------------------------------------------------------------------

# Columnas conocidas del dataset (ver README para el detalle de cada una)
ID_COL = "ID"
TARGET_COL = "Y"
SENSITIVE_COL = "SEXO"  # variable sensible usada en el análisis de Fairlearn

RAW_CATEGORICAL = ["SEXO", "EDUCACION", "ESTADO_CIVIL"]

# Categorías válidas fijadas explícitamente (tras `clean_anomalous_categories`)
# para que el ColumnTransformer produzca siempre las mismas columnas sin
# importar qué subconjunto de datos (corrida/fold) se use para ajustarlo.
CATEGORY_VALUES = {
    "SEXO": [1, 2],
    "EDUCACION": [1, 2, 3, 4],
    "ESTADO_CIVIL": [1, 2, 3],
}


def load_data(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """Carga el CSV crudo y aplica únicamente limpieza que NO depende de ningún split
    (no imputa, no escala, no aprende parámetros de los datos)."""
    df = pd.read_csv(path, sep=sep)
    return df


def clean_anomalous_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa categorías no documentadas de EDUCACION y ESTADO_CIVIL en 'otros'.

    EDUCACION: valores documentados 1-4; se observan además 0, 5 y 6 (345 filas: 14 + 280 + 51).
    ESTADO_CIVIL: valores documentados 1-3; se observa además 0 (54 filas).

    Esta limpieza es determinística (no aprende nada de los datos) por lo que es
    segura de aplicar antes o después del split train/test sin generar leakage.
    """
    df = df.copy()
    df["EDUCACION"] = df["EDUCACION"].apply(lambda x: x if x in (1, 2, 3, 4) else 4)
    df["ESTADO_CIVIL"] = df["ESTADO_CIVIL"].apply(lambda x: x if x in (1, 2, 3) else 3)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables derivadas (MAX/MIN/STD) por fila sobre los tres bloques
    temporales (PAGO_M, MONTO_FACTU_M, PAGO_EFECT_M).

    Estas operaciones son fila a fila: no aprenden ningún parámetro de la muestra
    completa (a diferencia de un scaler o un encoder), por lo que no generan
    data leakage sin importar en qué momento del pipeline se calculen.
    """
    df = df.copy()

    pago_cols = [c for c in df.columns if c.startswith("PAGO_M") and c != "PAGO_M1"]
    factu_cols = [c for c in df.columns if c.startswith("MONTO_FACTU_M")]
    efect_cols = [c for c in df.columns if c.startswith("PAGO_EFECT_M")]

    df["PAGO_M_MAX"] = df[pago_cols].max(axis=1)
    df["PAGO_M_MIN"] = df[pago_cols].min(axis=1)
    df["PAGO_M_STD"] = df[pago_cols].std(axis=1)

    df["MONTO_FACTU_M_MAX"] = df[factu_cols].max(axis=1)
    df["MONTO_FACTU_M_MIN"] = df[factu_cols].min(axis=1)
    df["MONTO_FACTU_M_STD"] = df[factu_cols].std(axis=1)

    df["PAGO_EFECT_M_MAX"] = df[efect_cols].max(axis=1)
    df["PAGO_EFECT_M_MIN"] = df[efect_cols].min(axis=1)
    df["PAGO_EFECT_M_STD"] = df[efect_cols].std(axis=1)

    return df


def get_feature_types(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    id_col: str = ID_COL,
    categorical: Iterable[str] = RAW_CATEGORICAL,
) -> tuple[list[str], list[str]]:
    """Devuelve (numeric_features, categorical_features) excluyendo target e ID."""
    drop_cols = {target, id_col}
    categorical = [c for c in categorical if c in df.columns and c not in drop_cols]
    numeric = [c for c in df.columns if c not in drop_cols and c not in categorical]
    return numeric, categorical


def clean_feature_name(name: str) -> str:
    """Quita el prefijo técnico (`num__`/`cat__`) que agrega el ColumnTransformer,
    solo para fines de visualización/reportes (no se usa para indexar columnas)."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def clean_feature_names(columns) -> list[str]:
    return [clean_feature_name(c) for c in columns]


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    """ColumnTransformer con categorías fijas explícitas (ver CATEGORY_VALUES),
    de forma que sin importar qué subconjunto de filas se use para ajustarlo
    (una corrida, un fold de CV), siempre produce las mismas columnas de
    salida en el mismo orden. Esto es lo que permite reutilizar una lista fija
    de variables seleccionadas (`ColumnSelector`) a través de distintas
    corridas/folds sin desalinear columnas.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categories = [CATEGORY_VALUES[c] for c in categorical_features]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def compute_scale_pos_weight(y) -> float:
    """Razón negativos/positivos, usada como `scale_pos_weight` en XGBoost
    (equivalente allí a `class_weight='balanced'`)."""
    y = np.asarray(y).ravel()
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer estilo scikit-learn que selecciona, por nombre, un
    subconjunto fijo de columnas de la salida (ya codificada) del
    ColumnTransformer. Se usa para aplicar la selección de variables por
    SHAP dentro del Pipeline (y por lo tanto dentro de cada fold/corrida),
    en vez de recortar el DataFrame antes del split."""

    def __init__(self, selected_features: list[str], all_features: list[str]):
        # sklearn's clone() requires constructor params to be stored unchanged
        # (no copying/casting) so that clone(est).get_params() == est.get_params()
        self.selected_features = selected_features
        self.all_features = all_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_features = list(self.all_features)
        idx = [all_features.index(f) for f in self.selected_features]
        return X[:, idx]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline determinístico de preparación: limpia categorías anómalas,
    agrega variables derivadas y elimina el identificador. No aprende nada de
    los datos, por lo que es seguro aplicarlo antes del split train/test."""
    df = clean_anomalous_categories(df)
    df = add_engineered_features(df)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estadístico Kolmogorov-Smirnov entre las distribuciones de score de
    positivos (Y=1) y negativos (Y=0)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    return ks_2samp(y_score[y_true == 1], y_score[y_true != 1]).statistic


def optimal_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Punto de corte que maximiza el estadístico J de Youden (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def calcular_indicadores(y_true: np.ndarray, y_score: np.ndarray, threshold: Optional[float] = None) -> dict:
    """Calcula el conjunto completo de métricas para un vector de scores.

    Si `threshold` es None, se calcula el óptimo vía Youden's J sobre el
    mismo conjunto (usar el threshold de TRAIN al evaluar TEST/OOT para no
    filtrar información del test en la elección del corte).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    logloss = mt.log_loss(y_true, y_score)
    ks = ks_stat(y_true, y_score)

    if threshold is None:
        threshold = optimal_threshold_youden(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    cm = mt.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = mt.accuracy_score(y_true, y_pred)
    f1 = mt.f1_score(y_true, y_pred, zero_division=0)
    precision = mt.precision_score(y_true, y_pred, zero_division=0)
    sensitivity = mt.recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "AUC": auc,
        "GINI": gini,
        "LogLoss": logloss,
        "KS": ks,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def metrics_rows_from_dict(
    indicadores: dict,
    model: str,
    run_id: int,
    seed: int,
    dataset: str,
    fold: Optional[int] = None,
) -> list[dict]:
    """Convierte un dict de `calcular_indicadores` en filas 'tidy' (una fila por
    métrica) listas para acumular en results/metrics.csv."""
    rows = []
    for metric, value in indicadores.items():
        if metric in ("TN", "FP", "FN", "TP", "Threshold"):
            continue
        rows.append(
            {
                "model": model,
                "run_id": run_id,
                "seed": seed,
                "fold": fold,
                "dataset": dataset,
                "metric": metric,
                "value": value,
            }
        )
    return rows


def save_metrics_csv(rows: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def summarize_metrics(metrics_df: pd.DataFrame, dataset: str = "test") -> pd.DataFrame:
    """Tabla resumen mean ± std / min / max por modelo y métrica, para un dataset dado."""
    sub = metrics_df[metrics_df["dataset"] == dataset]
    summary = (
        sub.groupby(["model", "metric"])["value"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

FIGURES_DEFAULT_DPI = 150


def _savefig(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURES_DEFAULT_DPI, bbox_inches="tight")
    plt.close()


def plot_target_distribution(y: pd.Series, path: str | Path) -> None:
    plt.figure(figsize=(6, 5))
    counts = y.value_counts().sort_index()
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/counts.sum():.1%})", ha="center", va="bottom")
    plt.title("Distribución de la variable objetivo (Y)")
    plt.xlabel("Y (0 = no default, 1 = default)")
    plt.ylabel("Cantidad de clientes")
    _savefig(path)


def plot_train_test_distribution(
    train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str], path: str | Path, max_vars: int = 6
) -> None:
    """Compara histogramas/KDE de las variables numéricas más relevantes entre train y test."""
    vars_to_plot = numeric_features[:max_vars]
    n = len(vars_to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(vars_to_plot):
        ax = axes[i]
        sns.kdeplot(train[col], label="train", fill=True, alpha=0.3, ax=ax, warn_singular=False)
        sns.kdeplot(test[col], label="test", fill=True, alpha=0.3, ax=ax, warn_singular=False)
        ax.set_title(col)
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Comparación de distribuciones: Train vs Test", y=1.02)
    _savefig(path)


def plot_metric_stability(metrics_df: pd.DataFrame, path: str | Path, dataset: str = "test") -> None:
    """Boxplot/stripplot de cada métrica a través de las corridas, por modelo."""
    sub = metrics_df[
        (metrics_df["dataset"] == dataset)
        & (metrics_df["metric"].isin(["AUC", "GINI", "KS", "F1", "Sensitivity", "Precision"]))
    ]
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=sub, x="metric", y="value", hue="model")
    sns.stripplot(data=sub, x="metric", y="value", hue="model", dodge=True, alpha=0.5, legend=False)
    plt.title(f"Estabilidad de métricas entre corridas ({dataset})")
    plt.ylabel("Valor")
    _savefig(path)


def plot_learning_curve(estimator, X, y, path: str | Path, cv=5, scoring="roc_auc", train_sizes=None, n_jobs=1) -> None:
    """n_jobs=1 por defecto a propósito: el propio estimador (XGBoost/LightGBM)
    ya paraleliza internamente, así que paralelizar también aquí sobreexpone
    los núcleos disponibles y puede volver la ejecución muchísimo más lenta
    (o parecer colgada) en máquinas con pocos cores."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=n_jobs
    )
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Train")
    plt.fill_between(
        sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    plt.plot(sizes, val_scores.mean(axis=1), "o-", label="Validation (CV)")
    plt.fill_between(
        sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    plt.xlabel("Tamaño de muestra de entrenamiento")
    plt.ylabel(scoring)
    plt.title("Learning Curve")
    plt.legend()
    _savefig(path)


def plot_train_validation_comparison(records: list[dict], path: str | Path) -> None:
    """records: lista de {'model':.., 'metric':.., 'train':.., 'test':..} para graficar
    barras agrupadas train vs test por modelo/métrica (diagnóstico de overfitting)."""
    df = pd.DataFrame(records)
    df_melt = df.melt(id_vars=["model", "metric"], value_vars=["train", "test"], var_name="split", value_name="value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="metric", y="value", hue="split")
    plt.title("Comparación Train vs Test por métrica y modelo")
    _savefig(path)


def plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str = "Matriz de Confusión") -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(title)
    _savefig(path)


def plot_roc_curve(y_true, y_score_dict: dict, path: str | Path) -> None:
    """y_score_dict: {model_name: y_score} para graficar varias curvas ROC juntas."""
    plt.figure(figsize=(7, 6))
    for name, y_score in y_score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    _savefig(path)


def plot_precision_recall_curve(y_true, y_score_dict: dict, path: str | Path) -> None:
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(7, 6))
    for name, y_score in y_score_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    _savefig(path)


# ---------------------------------------------------------------------------
# I/O de modelos
# ---------------------------------------------------------------------------

def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Fairlearn
# ---------------------------------------------------------------------------

def evaluate_fairness(y_true, y_pred, sensitive_features, group_names: Optional[dict] = None) -> dict:
    """Calcula métricas de fairness (demographic parity y equalized odds,
    en diferencia y en ratio) para una variable sensible binaria/categórica,
    además de las métricas de desempeño por grupo.

    No concluye si el modelo es "justo" o no: solo reporta las métricas para
    que se interpreten junto con el resto del diagnóstico del modelo.
    """
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        selection_rate,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    sensitive_features = np.asarray(sensitive_features).ravel()

    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "accuracy": mt.accuracy_score,
            "precision": lambda yt, yp: mt.precision_score(yt, yp, zero_division=0),
            "recall": lambda yt, yp: mt.recall_score(yt, yp, zero_division=0),
            "f1": lambda yt, yp: mt.f1_score(yt, yp, zero_division=0),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    by_group = mf.by_group.reset_index()
    if group_names:
        by_group[by_group.columns[0]] = by_group[by_group.columns[0]].map(group_names).fillna(
            by_group[by_group.columns[0]]
        )

    return {
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
        "equalized_odds_ratio": eo_ratio,
        "by_group": by_group.to_dict(orient="records"),
    }


def plot_fairness_by_group(fairness_result: dict, path: str | Path, title: str = "Métricas por grupo (SEXO)") -> None:
    by_group = pd.DataFrame(fairness_result["by_group"])
    group_col = by_group.columns[0]
    metric_cols = [c for c in by_group.columns if c != group_col]
    df_melt = by_group.melt(id_vars=group_col, value_vars=metric_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_melt, x="metric", y="value", hue=group_col)
    plt.title(title)
    plt.xticks(rotation=20)
    _savefig(path)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_feature_importance(model, X: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
    """Importancia media absoluta de SHAP por variable (en % del total)."""
    import shap

    X_sample = X.sample(min(max_samples, len(X)), random_state=1234) if len(X) > max_samples else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": importance})
    df_imp["Importancia_Shap_%"] = 100 * df_imp["mean_abs_shap"] / df_imp["mean_abs_shap"].sum()
    df_imp = df_imp.sort_values("Importancia_Shap_%", ascending=False).reset_index(drop=True)
    return df_imp, shap_values, X_sample


def plot_shap_importance(df_imp: pd.DataFrame, path: str | Path, top_n: int = 20, title: str = "Top variables por importancia SHAP") -> None:
    top = df_imp.head(top_n).sort_values("Importancia_Shap_%", ascending=True)
    plt.figure(figsize=(11, 8))
    bars = plt.barh(top["feature"], top["Importancia_Shap_%"])
    for bar in bars:
        v = bar.get_width()
        plt.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9)
    plt.xlabel("Importancia SHAP (%)")
    plt.title(title)
    _savefig(path)


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, path: str | Path) -> None:
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    _savefig(path)


def plot_shap_dependence(shap_values, X_sample: pd.DataFrame, feature: str, path: str | Path) -> None:
    import shap

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    _savefig(path)


# ---------------------------------------------------------------------------
# Pruebas estadísticas para el contraste de hipótesis
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Rangos medios (con empates) — auxiliar del algoritmo rápido de DeLong."""
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    ranks_sorted = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong(predictions_sorted: np.ndarray, n_pos: int):
    """Algoritmo rápido de DeLong (Sun & Xu, 2014).
    `predictions_sorted`: matriz (k modelos x n) con los positivos primero.
    Devuelve (aucs, matriz de covarianza de los aucs)."""
    m = n_pos
    n = predictions_sorted.shape[1] - m
    pos = predictions_sorted[:, :m]
    neg = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.array([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([_compute_midrank(predictions_sorted[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true, score_a, score_b, alternative: str = "two-sided", margin: float = 0.0) -> dict:
    """Prueba de DeLong para comparar el AUC de dos modelos evaluados sobre
    las MISMAS observaciones (curvas ROC correlacionadas).

    H0: AUC_a - AUC_b = margin   (por defecto margin = 0)
    alternative:
      - "two-sided": H1: AUC_a != AUC_b
      - "greater":   H1: AUC_a - AUC_b > margin  (superioridad, o
                     no-inferioridad si margin < 0)
      - "less":      H1: AUC_a - AUC_b < margin

    Devuelve AUCs, diferencia, error estándar, z, p-valor e IC 95 % de la diferencia.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true).ravel().astype(int)
    order = np.argsort(-y_true, kind="mergesort")  # positivos primero
    n_pos = int(y_true.sum())
    preds = np.vstack([np.asarray(score_a).ravel(), np.asarray(score_b).ravel()])[:, order]

    aucs, cov = _fast_delong(preds, n_pos)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-16)))
    z = (diff - margin) / se

    if alternative == "two-sided":
        p = 2 * norm.sf(abs(z))
    elif alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(diff),
        "se": se,
        "z": float(z),
        "p_value": float(p),
        "ci95_low": float(diff - 1.959964 * se),
        "ci95_high": float(diff + 1.959964 * se),
        "alternative": alternative,
        "margin": margin,
    }


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Prueba de Friedman (+ post-hoc de Nemenyi) sobre una matriz
    (filas = corridas/bloques, columnas = modelos) de una métrica
    donde MAYOR es mejor. Nemenyi según Demšar (2006): dos modelos difieren si
    la diferencia de sus rangos medios supera la diferencia crítica (CD)."""
    from scipy.stats import friedmanchisquare, rankdata

    # valores críticos q_alpha (alpha = 0.05) de Demšar (2006), tabla 5a
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    n, k = scores.shape
    stat, p = friedmanchisquare(*[scores[c].values for c in scores.columns])
    ranks = np.array([rankdata(-row) for row in scores.values])  # rango 1 = mejor
    mean_ranks = pd.Series(ranks.mean(axis=0), index=scores.columns)
    cd = q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))

    pairs = []
    cols = list(scores.columns)
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(mean_ranks[cols[i]] - mean_ranks[cols[j]])
            pairs.append({"model_a": cols[i], "model_b": cols[j], "rank_diff": float(d),
                          "critical_difference": float(cd), "significant": bool(d > cd)})
    return {"statistic": float(stat), "p_value": float(p), "n_blocks": int(n),
            "mean_ranks": mean_ranks.to_dict(), "critical_difference": float(cd), "nemenyi": pairs}


def wilcoxon_paired(a, b, alternative: str = "greater") -> dict:
    """Prueba de Wilcoxon de rangos con signo pareada (a vs b, por corrida)."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = wilcoxon(a, b, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p), "mean_diff": float(np.mean(a - b)),
            "n_wins": int((a > b).sum()), "n": int(len(a)), "alternative": alternative}


def jaccard_top_k(features_a: Iterable[str], features_b: Iterable[str], k: int = 10) -> float:
    """Índice de Jaccard entre las k primeras variables de dos rankings."""
    a, b = set(list(features_a)[:k]), set(list(features_b)[:k])
    return len(a & b) / len(a | b) if (a | b) else np.nan
