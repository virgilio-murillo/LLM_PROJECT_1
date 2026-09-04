"""
Clasifica los 581 del gold set con los 4 LLMs que funcionan on-demand.
Guarda una matriz (581 x 4) de predicciones 1-5 (o NaN si abstencion) + la etiqueta humana.
Alimenta E1 (triple comparacion) y E2 (agregacion). Paraleliza con hilos (I/O bound).
"""
import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import bedrock_llm as B

OUT = C.RESULTS / "llm_preds_581.csv"
KEYS = B.ENSEMBLE_KEYS


def classify_one(idx, text):
    return idx, {k: B.classify(text, k) for k in KEYS}


def main():
    df = C.load_gold().reset_index(drop=True)
    texts = df["text"].tolist()
    preds = {k: [None] * len(texts) for k in KEYS}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(classify_one, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futs):
            i, row = fut.result()
            for k in KEYS: preds[k][i] = row[k]
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(texts)}  ({time.time()-t0:.0f}s)", flush=True)
    out = df[["text", "label", "categoria"]].copy()
    out["human"] = out["label"] + 1  # de vuelta a 1-5
    for k in KEYS:
        out[f"llm_{k}"] = preds[k]
    out.to_csv(OUT, index=False)
    # reporte de abstenciones
    for k in KEYS:
        na = out[f"llm_{k}"].isna().sum()
        print(f"  {k}: abstenciones {na}/{len(out)}", flush=True)
    print(f"GUARDADO {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
