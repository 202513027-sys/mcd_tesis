"""
run_experiment.py
==================
Punto de entrada único del experimento de la tesis.

Ejecuta de punta a punta:
  1. Carga y preparación de datos (sin leakage).
  2. Split train/test oficial (estratificado).
  3. Selección de variables por importancia SHAP (usando solo TRAIN).
  4. Búsqueda de hiperparámetros (una sola vez) para XGBoost y LightGBM-DART.
  5. Ajuste de los modelos "oficiales" (baseline + propuestos) sobre el split oficial.
  6. Métricas, figuras, SHAP y Fairlearn para los modelos oficiales.
  7. N_RUNS corridas independientes (splits y semillas distintas) con la
     configuración de hiperparámetros fija, para el análisis de estabilidad.
  8. Guardado de resultados (results/), figuras (results/figures/) y modelos (models/).
  9. Contraste de hipótesis (HG, HE1–HE4) con métrica, umbral y prueba estadística:
       - HE1: modelos con TODAS las variables vs. variables seleccionadas por SHAP.
       - HE3: modelos con hiperparámetros POR DEFECTO vs. optimizados.
       - Pruebas: DeLong (split oficial), Friedman + Nemenyi y Wilcoxon (corridas),
         Spearman / Jaccard (estabilidad SHAP).

Uso:
    python src/run_experiment.py

Todos los parámetros relevantes están centralizados en la sección CONFIG.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))
import utils  # noqa: E402

# ===========================================================================
# CONFIG — parámetros del experimento (modificar aquí, no en el código)
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "DATA.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

BASE_SEED = 1234
N_RUNS = 10                 # corridas independientes para el análisis de estabilidad
TEST_SIZE = 0.30
N_FOLDS_CV = 5               # folds de StratifiedKFold en la búsqueda de hiperparámetros
SEARCH_N_ITER = 30           # iteraciones de RandomizedSearchCV (por modelo)
SHAP_IMPORTANCE_THRESHOLD = 1.0  # % mínimo de importancia SHAP para seleccionar una variable
SENSITIVE_COL = utils.SENSITIVE_COL  # "SEXO" (decisión confirmada con la usuaria)

# --- Contraste de hipótesis (umbrales fijados ANTES de ver los resultados) ---
HYP_ALPHA = 0.05            # nivel de significancia de todas las pruebas
HG_MIN_AUC = 0.75           # HG: AUC-ROC mínimo del mejor modelo en test
HE1_MAX_AUC_LOSS = 0.01     # HE1: pérdida máxima de AUC tolerada al usar solo las variables seleccionadas
HE3_MIN_AUC_GAIN = 0.01     # HE3: mejora mínima de AUC por la optimización de hiperparámetros
HE4_MIN_SPEARMAN = 0.70     # HE4: correlación de Spearman mínima entre rankings SHAP de distintas corridas
HE4_MIN_JACCARD = 0.60      # HE4: Jaccard mínimo entre top-k SHAP (selección vs. modelo final)
HE4_TOP_K = 10
BASE_MODELS = ["LogisticRegression", "XGBoost", "LightGBM_DART"]

for d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def make_pipeline(model, numeric_features, categorical_features, selected_features, all_features) -> Pipeline:
    """Pipeline único: preprocesamiento -> selección de variables SHAP -> modelo.
    Todo el preprocesamiento se ajusta SIEMPRE dentro del pipeline (nunca antes
    del split), por lo que es seguro reutilizar este pipeline en cada fold o
    corrida sin generar leakage."""
    return Pipeline(
        steps=[
            ("preprocess", utils.build_preprocessor(numeric_features, categorical_features)),
            ("select", utils.ColumnSelector(selected_features, all_features)),
            ("model", model),
        ]
    )


def build_models(seed, scale_pos_weight, hp="optimizado", xgb_params=None, lgbm_params=None) -> dict:
    """Modelos RL / XGBoost / LightGBM-DART.
    hp="optimizado": hiperparámetros encontrados por RandomizedSearchCV.
    hp="default":    hiperparámetros por defecto de cada librería (configuración inicial, HE3).
    En ambos casos se mantiene el MISMO tratamiento del desbalance (class_weight /
    scale_pos_weight) para que la única diferencia sea la optimización.
    La Regresión Logística no se optimiza, por lo que es igual en ambos casos."""
    xgb_kw = dict(xgb_params or {}) if hp == "optimizado" else {}
    lgbm_kw = dict(lgbm_params or {}) if hp == "optimizado" else {}
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", random_state=seed,
            tree_method="hist", n_jobs=-1, scale_pos_weight=scale_pos_weight, **xgb_kw,
        ),
        "LightGBM_DART": lgb.LGBMClassifier(
            boosting_type="dart", objective="binary", random_state=seed, n_jobs=-1,
            verbosity=-1, class_weight="balanced", **lgbm_kw,
        ),
    }


# Variantes adicionales para contrastar HE1 y HE3: (hiperparámetros, conjunto de variables, modelos)
HYP_VARIANTS = [
    ("default", "seleccion", ["XGBoost", "LightGBM_DART"]),                      # HE3
    ("optimizado", "todas", ["LogisticRegression", "XGBoost", "LightGBM_DART"]),  # HE1
]


def hp_label(model_name: str, hp: str) -> str:
    return "fijo" if model_name == "LogisticRegression" else hp


def tag_rows(rows: list[dict], base_model: str, hp: str, features: str) -> list[dict]:
    for r in rows:
        r["base_model"] = base_model
        r["hp"] = hp_label(base_model, hp)
        r["features"] = features
        r["model"] = f"{base_model}|{r['hp']}|{features}"
    return rows


def fit_eval(pipeline, X_train, y_train, X_test, y_test, threshold_from="train"):
    """Ajusta el pipeline y devuelve indicadores de train y test.
    El punto de corte (Youden's J) se calcula SIEMPRE sobre train y se aplica
    también a test, para no usar información de test en la elección del corte."""
    pipeline.fit(X_train, y_train)
    score_train = pipeline.predict_proba(X_train)[:, 1]
    score_test = pipeline.predict_proba(X_test)[:, 1]

    threshold = utils.optimal_threshold_youden(y_train, score_train)
    ind_train = utils.calcular_indicadores(y_train, score_train, threshold=threshold)
    ind_test = utils.calcular_indicadores(y_test, score_test, threshold=threshold)
    return pipeline, ind_train, ind_test, score_train, score_test


def evaluate_hypotheses(y_test, variant_scores, hyp_df, shap_runs_df, df_imp_selection, df_imp_final,
                        final_model_name, n_selected, n_total):
    """Contrasta HG y HE1–HE4 con métrica, umbral y prueba estadística.
    - Pruebas de DeLong: sobre el split oficial (mismas observaciones de test).
    - Friedman/Nemenyi y Wilcoxon: sobre las N_RUNS corridas (una observación por corrida).
    Guarda results/hypothesis_tests.json (detalle), results/hypothesis_summary.csv
    (tabla para la tesis), results/delong_tests.csv y la figura 13."""
    from itertools import combinations
    from scipy.stats import spearmanr

    a = HYP_ALPHA
    summary, detail, delong_rows = [], {}, []

    def auc_runs(model, hp, feats, metric="AUC"):
        sub = hyp_df[(hyp_df["base_model"] == model) & (hyp_df["hp"] == hp)
                     & (hyp_df["features"] == feats) & (hyp_df["metric"] == metric)]
        return sub.sort_values("run_id")["value"].values

    def delong(key_a, key_b, alternative, margin=0.0, hyp=""):
        r = utils.delong_test(y_test, variant_scores[key_a], variant_scores[key_b], alternative, margin)
        delong_rows.append({"hipotesis": hyp, "modelo_a": "|".join(key_a), "modelo_b": "|".join(key_b), **r})
        return r

    opt = lambda m: (m, hp_label(m, "optimizado"), "seleccion")  # noqa: E731

    # ---------------- HG ----------------
    r = delong(opt(final_model_name), opt("LogisticRegression"), "greater", hyp="HG")
    w = utils.wilcoxon_paired(auc_runs(final_model_name, "optimizado", "seleccion"),
                              auc_runs("LogisticRegression", "fijo", "seleccion"), "greater")
    ok = (r["auc_a"] >= HG_MIN_AUC) and (r["p_value"] < a)
    detail["HG"] = {"delong_vs_RL": r, "wilcoxon_runs_vs_RL": w}
    summary.append({
        "Hipotesis": "HG", "Metrica": "AUC-ROC (test)",
        "Umbral": f"AUC >= {HG_MIN_AUC} y AUC > RL",
        "Prueba": "DeLong unilateral (split oficial); Wilcoxon pareado (corridas)",
        "Resultado": (f"{final_model_name}: AUC={r['auc_a']:.4f} vs RL={r['auc_b']:.4f} "
                      f"(dif={r['diff']:+.4f}); corridas: dif media={w['mean_diff']:+.4f}, gana {w['n_wins']}/{w['n']}"),
        "p_valor": f"DeLong p={r['p_value']:.2e}; Wilcoxon p={w['p_value']:.4f}",
        "Veredicto": "Se confirma" if ok else "No se confirma",
    })

    # ---------------- HE1 ----------------
    he1, oks = {}, []
    for m in BASE_MODELS:
        hp = hp_label(m, "optimizado")
        r = delong((m, hp, "seleccion"), (m, hp, "todas"), "greater", margin=-HE1_MAX_AUC_LOSS, hyp="HE1")
        runs_diff = auc_runs(m, hp, "seleccion") - auc_runs(m, hp, "todas")
        he1[m] = {"delong_no_inferioridad": r, "dif_media_corridas": float(runs_diff.mean())}
        oks.append(r["p_value"] < a)
    detail["HE1"] = {"n_seleccionadas": n_selected, "n_total": n_total, "por_modelo": he1}
    summary.append({
        "Hipotesis": "HE1", "Metrica": "Dif. AUC (seleccionadas - todas)",
        "Umbral": f"pérdida <= {HE1_MAX_AUC_LOSS} (no inferioridad)",
        "Prueba": "DeLong unilateral de no inferioridad (margen -{:.2f})".format(HE1_MAX_AUC_LOSS),
        "Resultado": f"{n_selected} de {n_total} variables; " + "; ".join(
            f"{m}: dif={v['delong_no_inferioridad']['diff']:+.4f}" for m, v in he1.items()),
        "p_valor": "; ".join(f"{m}: p={v['delong_no_inferioridad']['p_value']:.2e}" for m, v in he1.items()),
        "Veredicto": "Se confirma" if all(oks) else ("Parcial" if any(oks) else "No se confirma"),
    })

    # ---------------- HE2 ----------------
    he2 = {}
    for metric in ["AUC", "Sensitivity", "F1"]:
        mat = pd.DataFrame({m: auc_runs(m, hp_label(m, "optimizado"), "seleccion", metric) for m in BASE_MODELS})
        he2[metric] = utils.friedman_nemenyi(mat, alpha=a)
    for m1, m2 in combinations(BASE_MODELS, 2):
        delong(opt(m1), opt(m2), "two-sided", hyp="HE2")
    detail["HE2"] = he2
    sig_pairs = [f"{p['model_a']} vs {p['model_b']}" for p in he2["AUC"]["nemenyi"] if p["significant"]]
    summary.append({
        "Hipotesis": "HE2", "Metrica": "AUC-ROC, Recall (Sensitivity), F1",
        "Umbral": f"alfa = {a}",
        "Prueba": "Friedman + post-hoc Nemenyi (corridas); DeLong por pares (split oficial)",
        "Resultado": "Rangos medios AUC: " + ", ".join(f"{k}={v:.2f}" for k, v in he2["AUC"]["mean_ranks"].items())
                     + f"; Nemenyi significativos (AUC): {', '.join(sig_pairs) or 'ninguno'}",
        "p_valor": "; ".join(f"Friedman {k}: p={v['p_value']:.2e}" for k, v in he2.items()),
        "Veredicto": "Se confirma" if he2["AUC"]["p_value"] < a else "No se confirma",
    })

    # ---------------- HE3 ----------------
    he3, oks = {}, []
    for m in ["XGBoost", "LightGBM_DART"]:
        r = delong((m, "optimizado", "seleccion"), (m, "default", "seleccion"), "greater", hyp="HE3")
        w = utils.wilcoxon_paired(auc_runs(m, "optimizado", "seleccion"), auc_runs(m, "default", "seleccion"), "greater")
        ok = (r["diff"] >= HE3_MIN_AUC_GAIN) and (r["p_value"] < a)
        he3[m] = {"delong": r, "wilcoxon_runs": w, "cumple": bool(ok)}
        oks.append(ok)
    detail["HE3"] = he3
    summary.append({
        "Hipotesis": "HE3", "Metrica": "Dif. AUC (optimizado - por defecto)",
        "Umbral": f"mejora >= {HE3_MIN_AUC_GAIN}",
        "Prueba": "DeLong unilateral (split oficial); Wilcoxon pareado (corridas)",
        "Resultado": "; ".join(f"{m}: dif={v['delong']['diff']:+.4f} (corridas {v['wilcoxon_runs']['mean_diff']:+.4f}, "
                               f"gana {v['wilcoxon_runs']['n_wins']}/{v['wilcoxon_runs']['n']})" for m, v in he3.items()),
        "p_valor": "; ".join(f"{m}: DeLong p={v['delong']['p_value']:.2e}, Wilcoxon p={v['wilcoxon_runs']['p_value']:.4f}"
                             for m, v in he3.items()),
        "Veredicto": "Se confirma" if all(oks) else ("Parcial" if any(oks) else "No se confirma"),
    })

    # ---------------- HE4 ----------------
    piv = shap_runs_df.pivot_table(index="feature", columns="run_id", values="mean_abs_shap")
    rhos, pvals = [], []
    for c1, c2 in combinations(piv.columns, 2):
        rho, p = spearmanr(piv[c1], piv[c2])
        rhos.append(rho)
        pvals.append(p)
    jac = utils.jaccard_top_k(df_imp_selection["feature"], df_imp_final["feature"], HE4_TOP_K)
    jac_runs = [utils.jaccard_top_k(piv[c1].sort_values(ascending=False).index, piv[c2].sort_values(ascending=False).index, HE4_TOP_K)
                for c1, c2 in combinations(piv.columns, 2)]
    ok = (np.mean(rhos) >= HE4_MIN_SPEARMAN) and (max(pvals) < a) and (jac >= HE4_MIN_JACCARD)
    detail["HE4"] = {"spearman_media": float(np.mean(rhos)), "spearman_min": float(np.min(rhos)),
                     "spearman_p_max": float(np.max(pvals)), "n_pares": len(rhos),
                     "jaccard_top_k_seleccion_vs_final": float(jac), "jaccard_top_k_entre_corridas_media": float(np.mean(jac_runs)),
                     "top_k": HE4_TOP_K}
    summary.append({
        "Hipotesis": "HE4", "Metrica": f"Spearman entre rankings SHAP; Jaccard top-{HE4_TOP_K}",
        "Umbral": f"Spearman medio >= {HE4_MIN_SPEARMAN}; Jaccard >= {HE4_MIN_JACCARD}",
        "Prueba": f"Correlación de Spearman ({len(rhos)} pares de corridas)",
        "Resultado": (f"Spearman medio={np.mean(rhos):.3f} (mín={np.min(rhos):.3f}); Jaccard selección vs final={jac:.2f}; "
                      f"Jaccard medio entre corridas={np.mean(jac_runs):.2f}"),
        "p_valor": f"p máximo={np.max(pvals):.2e}",
        "Veredicto": "Se confirma" if ok else "No se confirma",
    })

    # ---------------- guardado ----------------
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(delong_rows).to_csv(RESULTS_DIR / "delong_tests.csv", index=False)
    utils.save_json(json.loads(json.dumps(detail, default=float)), RESULTS_DIR / "hypothesis_tests.json")
    plot_hypothesis_variants(hyp_df, FIGURES_DIR / "13_auc_variantes_hipotesis.png")

    print("\n      RESUMEN DEL CONTRASTE DE HIPÓTESIS")
    for _, row in summary_df.iterrows():
        print(f"      {row['Hipotesis']:4s} -> {row['Veredicto']:15s} | {row['Resultado']}")
        print(f"                              {row['p_valor']}")
    return summary_df


def plot_hypothesis_variants(hyp_df: pd.DataFrame, path) -> None:
    """Boxplot del AUC en test (corridas) por modelo y variante (HE1 y HE3)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = hyp_df[hyp_df["metric"] == "AUC"].copy()
    labels = {"seleccion": "Principal (optimizado + variables SHAP)", "todas": "Todas las variables (HE1)",
              "default": "Hiperparámetros por defecto (HE3)"}
    sub["Variante"] = np.where(sub["hp"] == "default", labels["default"], sub["features"].map(labels))
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=sub, x="base_model", y="value", hue="Variante",
                hue_order=[labels["seleccion"], labels["default"], labels["todas"]])
    plt.xlabel("")
    plt.ylabel("AUC-ROC (test, 10 corridas)")
    plt.title("AUC por variante: optimización de hiperparámetros (HE3) y selección de variables (HE1)")
    plt.legend(title="Variante", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    t0 = time.time()
    print("=" * 78)
    print("PIPELINE EXPERIMENTAL — TESIS DE MAESTRÍA EN CIENCIA DE DATOS")
    print("=" * 78)

    # -----------------------------------------------------------------
    # 1) CARGA Y PREPARACIÓN
    # -----------------------------------------------------------------
    print("\n[1/9] Cargando y preparando datos...")
    df_raw = utils.load_data(DATA_PATH)
    print(f"      Filas: {df_raw.shape[0]} | Columnas: {df_raw.shape[1]}")

    df = utils.prepare_features(df_raw)
    numeric_features, categorical_features = utils.get_feature_types(df)
    print(f"      Numéricas: {len(numeric_features)} | Categóricas: {len(categorical_features)}")

    df.to_csv(PROCESSED_DIR / "data_prepared.csv", index=False)

    X = df.drop(columns=[utils.TARGET_COL])
    y = df[utils.TARGET_COL]

    utils.plot_target_distribution(y, FIGURES_DIR / "01_target_distribution.png")

    # -----------------------------------------------------------------
    # 2) SPLIT OFICIAL (para búsqueda de hiperparámetros y modelos "oficiales")
    # -----------------------------------------------------------------
    print("\n[2/9] Partición train/test oficial (estratificada, seed={})...".format(BASE_SEED))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=BASE_SEED
    )
    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    utils.plot_train_test_distribution(
        pd.concat([X_train, y_train], axis=1),
        pd.concat([X_test, y_test], axis=1),
        numeric_features,
        FIGURES_DIR / "02_train_test_distribution.png",
    )

    # -----------------------------------------------------------------
    # 3) SELECCIÓN DE VARIABLES POR IMPORTANCIA SHAP (solo con TRAIN oficial)
    # -----------------------------------------------------------------
    print("\n[3/9] Selección de variables por importancia SHAP (LightGBM preliminar sobre TRAIN)...")
    preprocessor_fs = utils.build_preprocessor(numeric_features, categorical_features)
    X_train_enc = preprocessor_fs.fit_transform(X_train)
    all_features_enc = list(preprocessor_fs.get_feature_names_out())
    X_train_enc_df = pd.DataFrame(X_train_enc, columns=all_features_enc, index=X_train.index)

    lgbm_fs = lgb.LGBMClassifier(random_state=BASE_SEED, verbosity=-1)
    lgbm_fs.fit(X_train_enc_df, y_train)

    df_imp, _, _ = utils.shap_feature_importance(lgbm_fs, X_train_enc_df)
    df_imp_display = df_imp.copy()
    df_imp_display["feature"] = utils.clean_feature_names(df_imp_display["feature"])
    df_imp_display.to_csv(RESULTS_DIR / "shap_feature_selection.csv", index=False)

    selected_features = df_imp[df_imp["Importancia_Shap_%"] > SHAP_IMPORTANCE_THRESHOLD]["feature"].tolist()
    if not selected_features:  # salvaguarda: si el umbral deja 0 variables, usar todas
        selected_features = all_features_enc
    print(f"      Variables seleccionadas: {len(selected_features)} de {len(all_features_enc)} "
          f"(umbral > {SHAP_IMPORTANCE_THRESHOLD}%)")

    # -----------------------------------------------------------------
    # 4) BÚSQUEDA DE HIPERPARÁMETROS — UNA SOLA VEZ, sobre el train oficial
    # -----------------------------------------------------------------
    print("\n[4/9] Búsqueda de hiperparámetros (RandomizedSearchCV, una sola vez sobre TRAIN)...")
    cv = StratifiedKFold(n_splits=N_FOLDS_CV, shuffle=True, random_state=BASE_SEED)
    scale_pos_weight = utils.compute_scale_pos_weight(y_train)
    print(f"      scale_pos_weight (XGB) = {scale_pos_weight:.3f}  |  class_weight='balanced' (LogReg/LightGBM)")

    # --- XGBoost ---
    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=BASE_SEED,
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    xgb_pipe = make_pipeline(xgb_base, numeric_features, categorical_features, selected_features, all_features_enc)
    xgb_param_dist = {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": uniform(0.01, 0.19),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
        "model__gamma": uniform(0, 0.3),
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_param_dist, n_iter=SEARCH_N_ITER, cv=cv, scoring="roc_auc",
        random_state=BASE_SEED, n_jobs=-1, verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best_params = {k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()}
    print(f"      XGBoost — mejor ROC-AUC (CV): {xgb_search.best_score_:.4f}")
    print(f"      XGBoost — mejores params: {xgb_best_params}")

    # --- LightGBM DART ---
    lgbm_base = lgb.LGBMClassifier(
        boosting_type="dart", objective="binary", random_state=BASE_SEED,
        n_jobs=1, verbosity=-1, class_weight="balanced",
    )
    lgbm_pipe = make_pipeline(lgbm_base, numeric_features, categorical_features, selected_features, all_features_enc)
    lgbm_param_dist = {
        "model__n_estimators": randint(100, 300),
        "model__max_depth": [-1, 5, 10],
        "model__learning_rate": uniform(0.01, 0.19),
        "model__num_leaves": randint(15, 63),
        "model__subsample": uniform(0.8, 0.2),
        "model__colsample_bytree": uniform(0.8, 0.2),
        "model__drop_rate": uniform(0.1, 0.1),
        "model__skip_drop": uniform(0.5, 0.2),
    }
    lgbm_search = RandomizedSearchCV(
        lgbm_pipe, lgbm_param_dist, n_iter=SEARCH_N_ITER, cv=3, scoring="roc_auc",
        random_state=BASE_SEED, n_jobs=-1, verbose=1,
    )
    lgbm_search.fit(X_train, y_train)
    lgbm_best_params = {k.replace("model__", ""): v for k, v in lgbm_search.best_params_.items()}
    print(f"      LightGBM-DART — mejor ROC-AUC (CV): {lgbm_search.best_score_:.4f}")
    print(f"      LightGBM-DART — mejores params: {lgbm_best_params}")

    utils.save_json(
        {"xgboost": xgb_best_params, "lightgbm_dart": lgbm_best_params, "scale_pos_weight": scale_pos_weight},
        RESULTS_DIR / "best_hyperparameters.json",
    )

    # -----------------------------------------------------------------
    # 5) MODELOS OFICIALES (baseline + propuestos) sobre el split oficial
    # -----------------------------------------------------------------
    print("\n[5/9] Ajustando modelos oficiales sobre el split oficial...")

    models_official = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=BASE_SEED),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", random_state=BASE_SEED,
            tree_method="hist", n_jobs=-1, scale_pos_weight=scale_pos_weight, **xgb_best_params,
        ),
        "LightGBM_DART": lgb.LGBMClassifier(
            boosting_type="dart", objective="binary", random_state=BASE_SEED, n_jobs=-1,
            verbosity=-1, class_weight="balanced", **lgbm_best_params,
        ),
    }

    official_pipelines, official_scores_test, official_metrics_rows = {}, {}, []
    train_val_records = []

    for name, model in models_official.items():
        pipe = make_pipeline(model, numeric_features, categorical_features, selected_features, all_features_enc)
        pipe, ind_train, ind_test, score_train, score_test = fit_eval(pipe, X_train, y_train, X_test, y_test)
        official_pipelines[name] = pipe
        official_scores_test[name] = score_test

        official_metrics_rows += utils.metrics_rows_from_dict(ind_train, name, run_id=-1, seed=BASE_SEED, dataset="train")
        official_metrics_rows += utils.metrics_rows_from_dict(ind_test, name, run_id=-1, seed=BASE_SEED, dataset="test")

        for metric in ["AUC", "GINI", "KS", "F1"]:
            train_val_records.append({"model": name, "metric": metric, "train": ind_train[metric], "test": ind_test[metric]})

        print(f"      {name:20s} | Test AUC={ind_test['AUC']:.4f}  GINI={ind_test['GINI']:.4f}  KS={ind_test['KS']:.4f}  F1={ind_test['F1']:.4f}")

    # Modelo final = el mejor de los dos propuestos (XGBoost / LightGBM-DART) por AUC en test oficial
    candidate_models = {k: v for k, v in official_scores_test.items() if k in ("XGBoost", "LightGBM_DART")}
    final_model_name = max(candidate_models, key=lambda k: roc_auc_score(y_test, candidate_models[k]))
    print(f"\n      >> Modelo final seleccionado (mayor AUC en test oficial): {final_model_name}")

    utils.plot_train_validation_comparison(train_val_records, FIGURES_DIR / "05_train_validation_comparison.png")
    utils.plot_roc_curve(y_test, official_scores_test, FIGURES_DIR / "07_roc_curve.png")
    utils.plot_precision_recall_curve(y_test, official_scores_test, FIGURES_DIR / "08_precision_recall_curve.png")

    final_pipe = official_pipelines[final_model_name]
    final_score_test = official_scores_test[final_model_name]
    final_threshold = utils.optimal_threshold_youden(
        y_train, final_pipe.predict_proba(X_train)[:, 1]
    )
    final_pred_test = (final_score_test >= final_threshold).astype(int)
    from sklearn.metrics import confusion_matrix

    utils.plot_confusion_matrix(
        confusion_matrix(y_test, final_pred_test), FIGURES_DIR / "06_confusion_matrix.png",
        title=f"Matriz de Confusión — {final_model_name} (test oficial)",
    )

    # -----------------------------------------------------------------
    # 6) SHAP y FAIRLEARN sobre el modelo final (split oficial)
    # -----------------------------------------------------------------
    print("\n[6/9] Explicabilidad (SHAP) y fairness (Fairlearn) del modelo final...")
    X_test_transformed = final_pipe.named_steps["select"].transform(
        final_pipe.named_steps["preprocess"].transform(X_test)
    )
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=selected_features, index=X_test.index)

    df_imp_final, shap_values_final, X_shap_sample = utils.shap_feature_importance(
        final_pipe.named_steps["model"], X_test_transformed_df
    )
    # Copias solo para visualización, con nombres de variable legibles (sin
    # prefijos num__/cat__ del ColumnTransformer); la selección de variables y
    # el modelo ya están entrenados y no se ven afectados por este renombrado.
    df_imp_final_display = df_imp_final.copy()
    df_imp_final_display["feature"] = utils.clean_feature_names(df_imp_final_display["feature"])
    df_imp_final_display.to_csv(RESULTS_DIR / "shap_importance_final_model.csv", index=False)
    utils.plot_shap_importance(df_imp_final_display, FIGURES_DIR / "10_shap_importance.png",
                                title=f"{final_model_name} — Top variables por importancia SHAP")

    X_shap_sample_display = X_shap_sample.copy()
    X_shap_sample_display.columns = utils.clean_feature_names(X_shap_sample_display.columns)
    utils.plot_shap_summary(shap_values_final, X_shap_sample_display, FIGURES_DIR / "09_shap_summary.png")

    top1_feature_display = df_imp_final_display.iloc[0]["feature"]
    try:
        utils.plot_shap_dependence(
            shap_values_final, X_shap_sample_display, top1_feature_display, FIGURES_DIR / "12_shap_dependence_top1.png"
        )
    except Exception as e:  # el dependence_plot puede fallar para ciertas variables one-hot; no es crítico
        print(f"      (aviso) no se pudo generar el dependence plot para {top1_feature_display}: {e}")

    sensitive_test = X_test[SENSITIVE_COL].values
    fairness_result = utils.evaluate_fairness(
        y_test, final_pred_test, sensitive_test, group_names={1: "SEXO=1", 2: "SEXO=2"}
    )
    utils.save_json(fairness_result, RESULTS_DIR / "fairness_results.json")
    utils.plot_fairness_by_group(fairness_result, FIGURES_DIR / "11_fairness_by_group.png",
                                  title=f"{final_model_name} — Métricas por grupo (SEXO)")
    print(f"      Demographic parity difference: {fairness_result['demographic_parity_difference']:.4f}")
    print(f"      Equalized odds difference:     {fairness_result['equalized_odds_difference']:.4f}")

    # -----------------------------------------------------------------
    # 7) LEARNING CURVE del modelo final
    # -----------------------------------------------------------------
    print("\n[7/9] Learning curve del modelo final...")
    final_model_class = models_official[final_model_name].__class__
    final_model_params = models_official[final_model_name].get_params()
    lc_pipe = make_pipeline(
        final_model_class(**final_model_params), numeric_features, categorical_features, selected_features, all_features_enc
    )
    utils.plot_learning_curve(lc_pipe, X_train, y_train, FIGURES_DIR / "04_learning_curve.png", cv=3, scoring="roc_auc")

    # -----------------------------------------------------------------
    # 8) CORRIDAS INDEPENDIENTES (análisis de estabilidad) — config fija
    # -----------------------------------------------------------------
    print(f"\n[8/9] {N_RUNS} corridas independientes (splits/semillas distintas, hiperparámetros fijos)...")
    all_metrics_rows = list(official_metrics_rows)
    hyp_rows, shap_runs = [], []  # insumos del contraste de hipótesis (paso 9)

    for run_id in range(N_RUNS):
        seed = BASE_SEED + run_id
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
        )

        run_models = {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            "XGBoost": xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", random_state=seed, tree_method="hist",
                n_jobs=-1, scale_pos_weight=utils.compute_scale_pos_weight(yr_train), **xgb_best_params,
            ),
            "LightGBM_DART": lgb.LGBMClassifier(
                boosting_type="dart", objective="binary", random_state=seed, n_jobs=-1,
                verbosity=-1, class_weight="balanced", **lgbm_best_params,
            ),
        }

        run_summary = []
        for name, model in run_models.items():
            pipe = make_pipeline(model, numeric_features, categorical_features, selected_features, all_features_enc)
            fitted_pipe, ind_train, ind_test, _, _ = fit_eval(pipe, Xr_train, yr_train, Xr_test, yr_test)
            all_metrics_rows += utils.metrics_rows_from_dict(ind_train, name, run_id=run_id, seed=seed, dataset="train")
            all_metrics_rows += utils.metrics_rows_from_dict(ind_test, name, run_id=run_id, seed=seed, dataset="test")
            run_summary.append(f"{name}: AUC={ind_test['AUC']:.3f}")

            if name in BASE_MODELS:
                hyp_rows += tag_rows(
                    utils.metrics_rows_from_dict(ind_test, name, run_id=run_id, seed=seed, dataset="test"),
                    name, "optimizado", "seleccion",
                )
            if name == final_model_name:  # HE4: importancia SHAP del modelo final en cada corrida
                Xr_test_t = pd.DataFrame(
                    fitted_pipe.named_steps["select"].transform(fitted_pipe.named_steps["preprocess"].transform(Xr_test)),
                    columns=selected_features, index=Xr_test.index,
                )
                df_imp_run, _sv, _xs = utils.shap_feature_importance(fitted_pipe.named_steps["model"], Xr_test_t)
                df_imp_run["run_id"] = run_id
                shap_runs.append(df_imp_run)

        # --- Variantes para el contraste de hipótesis (HE1: todas las variables, HE3: hiperparámetros por defecto)
        spw_r = utils.compute_scale_pos_weight(yr_train)
        for hp, feats, names in HYP_VARIANTS:
            variant_models = build_models(seed, spw_r, hp, xgb_best_params, lgbm_best_params)
            feat_list = selected_features if feats == "seleccion" else all_features_enc
            for name in names:
                pipe = make_pipeline(variant_models[name], numeric_features, categorical_features, feat_list, all_features_enc)
                _, _itr, ind_test, _, _ = fit_eval(pipe, Xr_train, yr_train, Xr_test, yr_test)
                hyp_rows += tag_rows(
                    utils.metrics_rows_from_dict(ind_test, name, run_id=run_id, seed=seed, dataset="test"),
                    name, hp, feats,
                )

        print(f"      Run {run_id + 1:2d}/{N_RUNS} (seed={seed}) -> " + " | ".join(run_summary))

    metrics_df = utils.save_metrics_csv(all_metrics_rows, RESULTS_DIR / "metrics.csv")
    summary_df = utils.summarize_metrics(metrics_df, dataset="test")
    summary_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    utils.plot_metric_stability(metrics_df[metrics_df["run_id"] >= 0], FIGURES_DIR / "03_metric_stability.png")

    print("\n      Resumen (mean ± std) de métricas en TEST a través de las corridas:")
    for _, row in summary_df.iterrows():
        print(f"      {row['model']:20s} {row['metric']:12s} {row['mean']:.4f} ± {row['std']:.4f}")

    # -----------------------------------------------------------------
    # GUARDADO DE MODELOS
    # -----------------------------------------------------------------
    print("\nGuardando modelos oficiales en models/ ...")
    for name, pipe in official_pipelines.items():
        joblib.dump(pipe, MODELS_DIR / f"{name.lower()}_pipeline.joblib")
    utils.save_json(
        {"final_model": final_model_name, "final_threshold": float(final_threshold),
         "selected_features": selected_features, "sensitive_col": SENSITIVE_COL},
        RESULTS_DIR / "run_summary.json",
    )

    # -----------------------------------------------------------------
    # 9) CONTRASTE DE HIPÓTESIS
    # -----------------------------------------------------------------
    print("\n[9/9] Contraste de hipótesis (HG, HE1–HE4)...")
    # 9.1 Variantes sobre el split oficial (para DeLong)
    variant_scores = {(n, hp_label(n, "optimizado"), "seleccion"): official_scores_test[n] for n in BASE_MODELS}
    for hp, feats, names in HYP_VARIANTS:
        variant_models = build_models(BASE_SEED, scale_pos_weight, hp, xgb_best_params, lgbm_best_params)
        feat_list = selected_features if feats == "seleccion" else all_features_enc
        for name in names:
            pipe = make_pipeline(variant_models[name], numeric_features, categorical_features, feat_list, all_features_enc)
            _, _itr, ind_test, _, score_test = fit_eval(pipe, X_train, y_train, X_test, y_test)
            variant_scores[(name, hp_label(name, hp), feats)] = score_test
            print(f"      {name:20s} | hp={hp_label(name, hp):10s} | variables={feats:9s} | Test AUC={ind_test['AUC']:.4f}")

    hyp_df = pd.DataFrame(hyp_rows)
    hyp_df.to_csv(RESULTS_DIR / "metrics_hypotheses.csv", index=False)
    shap_runs_df = pd.concat(shap_runs, ignore_index=True)
    shap_runs_df["feature"] = utils.clean_feature_names(shap_runs_df["feature"])
    shap_runs_df.to_csv(RESULTS_DIR / "shap_importance_by_run.csv", index=False)

    evaluate_hypotheses(
        y_test=y_test, variant_scores=variant_scores, hyp_df=hyp_df, shap_runs_df=shap_runs_df,
        df_imp_selection=df_imp, df_imp_final=df_imp_final, final_model_name=final_model_name,
        n_selected=len(selected_features), n_total=len(all_features_enc),
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"EXPERIMENTO COMPLETO EN {elapsed/60:.1f} min")
    print(f"Modelo final: {final_model_name}  |  Resultados en: {RESULTS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    # Todo lo que imprime la corrida queda también en results/run_log.txt
    import post_analysis

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _tee = post_analysis.Tee(RESULTS_DIR / "run_log.txt")
    sys.stdout = _tee
    try:
        main()
        # Post-análisis: calidad de datos, Tabla 5 (10 corridas), W de Kendall,
        # Shapiro-Wilk/Levene, Holm y DeLong con IC y morosos (ver post_analysis.py)
        post_analysis.main()
    finally:
        sys.stdout = _tee.stdout
        _tee.close()
