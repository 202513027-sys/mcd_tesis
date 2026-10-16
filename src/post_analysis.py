"""
post_analysis.py
================
Análisis estadístico complementario sobre las salidas YA GUARDADAS en results/
(no reentrena ningún modelo, así que no cambia ninguna cifra existente).

Responde a la retroalimentación de la S5 (ficha del 13/10/2026, puntos 4.3 y 4.4):
  1. Calidad de datos: duplicados (con y sin ID) y categorías fuera de rango.
  2. Tabla 5: media ± DE de las 10 corridas (run_id >= 0, SIN el split oficial).
  3. W de Kendall para cada prueba de Friedman (AUC, Sensibilidad, F1).
  4. Supuestos: Shapiro-Wilk sobre las diferencias pareadas (10 corridas) y Levene.
  5. Corrección de Holm:
       (a) familia de las 3 pruebas de Friedman (AUC, Sensibilidad, F1);
       (b) Wilcoxon pareado por pares de modelos (corridas), Holm dentro de cada métrica;
       (c) DeLong por pares (split oficial, HE2), Holm sobre los 3 pares.
  6. IC 95 % de DeLong por par, chequeo de equivalencia (IC dentro de ±0,01)
     y traducción de la diferencia de Recall a morosos detectados.

Salidas en results/:
  data_quality.json, table5_runs.csv, assumptions_tests.csv, friedman_kendall_holm.csv,
  pairwise_wilcoxon_holm.csv, delong_pairs_holm.csv, post_analysis_log.txt

Uso:
    python src/post_analysis.py
(run_experiment.py también lo llama al final de cada corrida completa.)
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, levene, shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.append(str(Path(__file__).resolve().parent))
import utils  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG (mismos valores que la sección CONFIG de run_experiment.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "DATA.csv"
RESULTS_DIR = BASE_DIR / "results"

ALPHA = 0.05
EQUIV_MARGIN = 0.01          # diferencia de AUC considerada prácticamente relevante
BASE_SEED = 1234
MODELS = ["LogisticRegression", "XGBoost", "LightGBM_DART"]
FRIEDMAN_METRICS = ["AUC", "Sensitivity", "F1"]
TABLE5_METRICS = ["AUC", "GINI", "KS", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "LogLoss"]


class Tee:
    """Escribe a la vez en consola y en un archivo de log."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def hp_label(model: str) -> str:
    return "fijo" if model == "LogisticRegression" else "optimizado"


def runs_matrix(hyp: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Matriz corridas x modelos de la variante principal (optimizado + variables SHAP)."""
    cols = {}
    for m in MODELS:
        sub = hyp[(hyp["base_model"] == m) & (hyp["hp"] == hp_label(m))
                  & (hyp["features"] == "seleccion") & (hyp["metric"] == metric)]
        cols[m] = sub.sort_values("run_id")["value"].values
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# 1. Calidad de datos
# ---------------------------------------------------------------------------
def data_quality() -> dict:
    print("\n[1] CALIDAD DE DATOS (archivo crudo)")
    df = utils.load_data(DATA_PATH)
    feats_y = [c for c in df.columns if c != utils.ID_COL]
    feats_x = [c for c in feats_y if c != utils.TARGET_COL]

    edu_out = df.loc[~df["EDUCACION"].isin([1, 2, 3, 4]), "EDUCACION"].value_counts().sort_index()
    civ_out = df.loc[~df["ESTADO_CIVIL"].isin([1, 2, 3]), "ESTADO_CIVIL"].value_counts().sort_index()
    q = {
        "n_filas": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "n_positivos_Y1": int(df[utils.TARGET_COL].sum()),
        "tasa_Y1": float(df[utils.TARGET_COL].mean()),
        "nulos_total": int(df.isna().sum().sum()),
        "ID_duplicados": int(df[utils.ID_COL].duplicated().sum()),
        "filas_duplicadas_con_ID": int(df.duplicated().sum()),
        "filas_duplicadas_sin_ID": int(df.duplicated(subset=feats_y).sum()),
        "filas_involucradas_en_duplicados_sin_ID": int(df.duplicated(subset=feats_y, keep=False).sum()),
        "filas_mismas_X_sin_ID_ni_Y": int(df.duplicated(subset=feats_x).sum()),
        "EDUCACION_fuera_rango_1a4": {str(k): int(v) for k, v in edu_out.items()},
        "EDUCACION_fuera_rango_total": int(edu_out.sum()),
        "ESTADO_CIVIL_fuera_rango_1a3": {str(k): int(v) for k, v in civ_out.items()},
        "ESTADO_CIVIL_fuera_rango_total": int(civ_out.sum()),
    }
    q["filas_mismas_X_distinta_Y"] = q["filas_mismas_X_sin_ID_ni_Y"] - q["filas_duplicadas_sin_ID"]
    print(f"    Filas: {q['n_filas']} | Columnas: {q['n_columnas']} | Nulos: {q['nulos_total']} | ID duplicados: {q['ID_duplicados']}")
    print(f"    Morosos (Y=1): {q['n_positivos_Y1']} ({q['tasa_Y1']:.2%})")
    print(f"    Filas duplicadas contando ID: {q['filas_duplicadas_con_ID']}")
    print(f"    Filas duplicadas SIN contar ID (copias adicionales): {q['filas_duplicadas_sin_ID']} "
          f"({q['filas_involucradas_en_duplicados_sin_ID']} filas involucradas)")
    print(f"    Filas con las mismas X (sin ID ni Y): {q['filas_mismas_X_sin_ID_ni_Y']} "
          f"-> de ellas con Y distinta: {q['filas_mismas_X_distinta_Y']}")
    print(f"    EDUCACION fuera de 1-4: {q['EDUCACION_fuera_rango_1a4']} = {q['EDUCACION_fuera_rango_total']} filas")
    print(f"    ESTADO_CIVIL fuera de 1-3: {q['ESTADO_CIVIL_fuera_rango_1a3']} = {q['ESTADO_CIVIL_fuera_rango_total']} filas")
    utils.save_json(q, RESULTS_DIR / "data_quality.json")
    return q


# ---------------------------------------------------------------------------
# 2. Tabla 5 (solo las 10 corridas)
# ---------------------------------------------------------------------------
def table5(metrics: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] TABLA 5 — media ± DE de las 10 corridas (run_id >= 0, sin el split oficial)")
    test = metrics[metrics["dataset"] == "test"]
    runs = test[test["run_id"] >= 0]
    seeds = sorted(runs["seed"].unique())
    print(f"    Corridas: {runs['run_id'].nunique()} | semillas {seeds[0]}..{seeds[-1]}")
    if BASE_SEED in seeds:
        print(f"    Nota: la corrida con semilla {BASE_SEED} usa la misma partición que el split oficial.")
    agg = runs.groupby(["model", "metric"])["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    off = test[test["run_id"] == -1].set_index(["model", "metric"])["value"].rename("split_oficial")
    agg = agg.join(off, on=["model", "metric"])
    agg = agg[agg["metric"].isin(TABLE5_METRICS)]
    agg["model"] = pd.Categorical(agg["model"], ["Dummy"] + MODELS, ordered=True)
    agg["metric"] = pd.Categorical(agg["metric"], TABLE5_METRICS, ordered=True)
    agg = agg.sort_values(["model", "metric"]).reset_index(drop=True)
    agg.to_csv(RESULTS_DIR / "table5_runs.csv", index=False)
    for m in MODELS:
        row = agg[(agg["model"] == m) & (agg["metric"] == "AUC")].iloc[0]
        print(f"    {m:20s} AUC = {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])}) | split oficial {row['split_oficial']:.4f}")
    return agg


# ---------------------------------------------------------------------------
# 3-5. Friedman + W de Kendall + Holm; supuestos; Wilcoxon por pares + Holm
# ---------------------------------------------------------------------------
def friedman_block(hyp: pd.DataFrame):
    print("\n[3] FRIEDMAN + W DE KENDALL + HOLM (familia: 3 métricas)")
    rows = []
    for metric in FRIEDMAN_METRICS:
        mat = runs_matrix(hyp, metric)
        n, k = mat.shape
        stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
        w = stat / (n * (k - 1))   # W de Kendall = chi2_Friedman / (n (k - 1))
        rows.append({"metric": metric, "n_corridas": n, "k_modelos": k, "chi2_friedman": stat,
                     "p_value": p, "kendall_W": w})
    df = pd.DataFrame(rows)
    rej, p_holm, _, _ = multipletests(df["p_value"], alpha=ALPHA, method="holm")
    df["p_holm"] = p_holm
    df["rechaza_H0_holm"] = rej
    df.to_csv(RESULTS_DIR / "friedman_kendall_holm.csv", index=False)
    for _, r in df.iterrows():
        print(f"    {r['metric']:12s} chi2={r['chi2_friedman']:.2f}  p={r['p_value']:.2e}  "
              f"p_Holm={r['p_holm']:.2e}  W de Kendall={r['kendall_W']:.3f}  -> "
              f"{'rechaza H0' if r['rechaza_H0_holm'] else 'no rechaza H0'}")
    return df


def assumptions_block(hyp: pd.DataFrame):
    print("\n[4] SUPUESTOS — Shapiro-Wilk sobre diferencias pareadas (10 corridas) y Levene")
    rows = []
    for metric in FRIEDMAN_METRICS:
        mat = runs_matrix(hyp, metric)
        for a, b in combinations(MODELS, 2):
            d = mat[a].values - mat[b].values
            w, p = shapiro(d)
            rows.append({"prueba": "Shapiro-Wilk", "metric": metric, "grupo": f"{a} - {b}",
                         "estadistico": w, "p_value": p,
                         "decision": "no se rechaza normalidad" if p >= ALPHA else "se rechaza normalidad"})
        stat, p = levene(*[mat[c].values for c in MODELS], center="median")
        rows.append({"prueba": "Levene (mediana)", "metric": metric, "grupo": " / ".join(MODELS),
                     "estadistico": stat, "p_value": p,
                     "decision": "varianzas homogéneas" if p >= ALPHA else "varianzas distintas"})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "assumptions_tests.csv", index=False)
    for _, r in df.iterrows():
        print(f"    {r['prueba']:16s} {r['metric']:12s} {r['grupo']:55s} est={r['estadistico']:.3f}  "
              f"p={r['p_value']:.3f} -> {r['decision']}")
    sw = df[df["prueba"] == "Shapiro-Wilk"]
    print("    Decisión: " + ("Shapiro-Wilk no rechaza normalidad en ningún par; "
                              if (sw["p_value"] >= ALPHA).all() else
                              "Shapiro-Wilk rechaza normalidad en al menos un par; ")
          + "se mantiene Friedman (no paramétrico), que no exige normalidad y es robusto con n = 10.")
    return df


def pairwise_block(hyp: pd.DataFrame):
    print("\n[5] WILCOXON PAREADO POR PARES (corridas) + HOLM dentro de cada métrica (3 pares)")
    rows = []
    for metric in FRIEDMAN_METRICS:
        mat = runs_matrix(hyp, metric)
        block = []
        for a, b in combinations(MODELS, 2):
            stat, p = wilcoxon(mat[a].values, mat[b].values, alternative="two-sided")
            d = mat[a].values - mat[b].values
            block.append({"metric": metric, "modelo_a": a, "modelo_b": b, "dif_media": d.mean(),
                          "a_gana": int((d > 0).sum()), "n": len(d), "wilcoxon_W": stat, "p_value": p})
        rej, p_holm, _, _ = multipletests([r["p_value"] for r in block], alpha=ALPHA, method="holm")
        for r, ph, rj in zip(block, p_holm, rej):
            r["p_holm"] = ph
            r["significativo_holm"] = bool(rj)
        rows += block
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "pairwise_wilcoxon_holm.csv", index=False)
    for _, r in df.iterrows():
        print(f"    {r['metric']:12s} {r['modelo_a']:18s} vs {r['modelo_b']:14s} dif={r['dif_media']:+.4f} "
              f"(gana {r['a_gana']}/{r['n']})  p={r['p_value']:.4f}  p_Holm={r['p_holm']:.4f}")
    return df


# ---------------------------------------------------------------------------
# 6. DeLong por pares (split oficial) + Holm + equivalencia + morosos
# ---------------------------------------------------------------------------
def delong_block(metrics: pd.DataFrame, n_pos_test: int):
    print("\n[6] DeLONG POR PARES (split oficial, HE2) + HOLM + IC 95 % + morosos")
    dl = pd.read_csv(RESULTS_DIR / "delong_tests.csv")
    he2 = dl[dl["hipotesis"] == "HE2"].copy()
    rej, p_holm, _, _ = multipletests(he2["p_value"], alpha=ALPHA, method="holm")
    he2["p_holm"] = p_holm
    he2["significativo_holm"] = rej
    he2["IC_dentro_de_±0.01"] = (he2["ci95_low"] > -EQUIV_MARGIN) & (he2["ci95_high"] < EQUIV_MARGIN)

    test = metrics[metrics["dataset"] == "test"]
    rec_off = test[(test["run_id"] == -1) & (test["metric"] == "Sensitivity")].set_index("model")["value"]
    rec_runs = test[(test["run_id"] >= 0) & (test["metric"] == "Sensitivity")].groupby("model")["value"].mean()
    auc_runs = test[(test["run_id"] >= 0) & (test["metric"] == "AUC")].groupby("model")["value"].mean()

    out = []
    for _, r in he2.iterrows():
        a, b = r["modelo_a"].split("|")[0], r["modelo_b"].split("|")[0]
        out.append({
            "modelo_a": a, "modelo_b": b,
            "auc_a_oficial": r["auc_a"], "auc_b_oficial": r["auc_b"], "dif_auc_oficial": r["diff"],
            "ic95_low": r["ci95_low"], "ic95_high": r["ci95_high"], "p_value": r["p_value"],
            "p_holm": r["p_holm"], "significativo_holm": bool(r["significativo_holm"]),
            "ic_dentro_de_pm_0.01": bool(r["IC_dentro_de_±0.01"]),
            "dif_auc_media_corridas": auc_runs[a] - auc_runs[b],
            "dif_recall_oficial": rec_off[a] - rec_off[b],
            "morosos_adicionales_oficial": (rec_off[a] - rec_off[b]) * n_pos_test,
            "dif_recall_media_corridas": rec_runs[a] - rec_runs[b],
            "morosos_adicionales_corridas": (rec_runs[a] - rec_runs[b]) * n_pos_test,
        })
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_DIR / "delong_pairs_holm.csv", index=False)
    print(f"    Morosos en el test oficial: {n_pos_test} (1 punto de Recall = {n_pos_test / 100:.1f} morosos)")
    for _, r in df.iterrows():
        print(f"    {r['modelo_a']:18s} vs {r['modelo_b']:14s} dif AUC={r['dif_auc_oficial']:+.4f} "
              f"IC95 [{r['ic95_low']:+.4f}, {r['ic95_high']:+.4f}]  p={r['p_value']:.2e}  p_Holm={r['p_holm']:.2e}  "
              f"IC dentro de ±{EQUIV_MARGIN}: {'sí' if r['ic_dentro_de_pm_0.01'] else 'no'}")
        print(f"        dif AUC media corridas={r['dif_auc_media_corridas']:+.5f} | dif Recall oficial={r['dif_recall_oficial']:+.4f} "
              f"(≈ {r['morosos_adicionales_oficial']:+.1f} morosos) | media corridas={r['dif_recall_media_corridas']:+.4f} "
              f"(≈ {r['morosos_adicionales_corridas']:+.1f} morosos)")
    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(RESULTS_DIR / "post_analysis_log.txt")
    sys.stdout = tee
    try:
        import scipy
        import statsmodels
        print("=" * 78)
        print("POST-ANÁLISIS ESTADÍSTICO (sobre results/, sin reentrenar)")
        print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M} | Python {platform.python_version()} | "
              f"pandas {pd.__version__} | numpy {np.__version__} | scipy {scipy.__version__} | "
              f"statsmodels {statsmodels.__version__}")
        print("=" * 78)

        q = data_quality()
        metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
        hyp = pd.read_csv(RESULTS_DIR / "metrics_hypotheses.csv")
        n_pos_test = int(round(q["n_positivos_Y1"] * 0.30))  # partición estratificada 70/30

        table5(metrics)
        friedman_block(hyp)
        assumptions_block(hyp)
        pairwise_block(hyp)
        delong_block(metrics, n_pos_test)

        print("\nNota: las 10 corridas son un hold-out repetido con particiones que se solapan; "
              "esto vuelve optimistas a Friedman/Wilcoxon. La cifra principal es el AUC del split "
              "oficial con su IC 95 % de DeLong.")
        print("Archivos guardados en results/: data_quality.json, table5_runs.csv, friedman_kendall_holm.csv, "
              "assumptions_tests.csv, pairwise_wilcoxon_holm.csv, delong_pairs_holm.csv, post_analysis_log.txt")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
