"""
Utilidades compartidas por todos los experimentos.
Carga de datos, control de semillas, y metricas ordinales (QWK, MAE, kappa).
"""
from __future__ import annotations
import os, random, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, mean_absolute_error, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "limpieza_final" / "etiquetado_humano_unificado.csv"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "experimentos" / "resultados"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 61298  # la semilla original del proyecto


def set_all_seeds(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


def load_gold(text_col="comentario"):
    """Devuelve el gold set (581): comentario, label 0..4, categoria."""
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]
    df = df[df["etiquetado_humano"].notna()].copy()
    df["label"] = df["etiquetado_humano"].astype(int) - 1  # 1..5 -> 0..4
    df["text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    return df[["text", "label", "categoria"]]


def load_full_corpus():
    """Devuelve el corpus limpio completo (2578) con o sin etiqueta."""
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]
    df["text"] = df["comentario"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label"] = df["etiquetado_humano"]  # NaN donde no hay etiqueta
    return df[["text", "label", "categoria"]]


def ordinal_metrics(y_true, y_pred) -> dict:
    """Todas las metricas relevantes para una escala ordinal 0..4 (o 1..5)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_lineal": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n=1000, seed=SEED):
    """IC 95% bootstrap para una metrica (metric_fn(y_true,y_pred)->float)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    N = len(y_true); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric_fn, n=1000, seed=SEED):
    """IC de la diferencia metric(A)-metric(B) sobre el MISMO test (pareado)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    N = len(y_true); diffs = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            diffs.append(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi), float((np.array(diffs) > 0).mean())


def qwk_fn(yt, yp):
    return cohen_kappa_score(yt, yp, weights="quadratic")


def f1m_fn(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


def save_result(name: str, obj: dict):
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return p


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    set_all_seeds()
    g = load_gold()
    print("gold:", len(g), "dist:", g["label"].value_counts().sort_index().to_dict())
    full = load_full_corpus()
    print("corpus:", len(full), "con etiqueta:", full["label"].notna().sum())
