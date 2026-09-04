"""
E3 (parte Bedrock): clasifica el pool SIN etiqueta (~1997) con los 4 LLMs.
Guarda predicciones para: (a) auto-etiquetas por concordancia (donde >=3 LLMs coinciden),
(b) cola de active learning (baja concordancia / desacuerdo).
Paralelizado con hilos. No usa GPU (no compite con el entrenamiento local).
"""
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import bedrock_llm as B

OUT = C.RESULTS / "llm_preds_pool.csv"
KEYS = B.ENSEMBLE_KEYS


def main():
    full = C.load_full_corpus()
    pool = full[full["label"].isna()].reset_index(drop=True)  # sin etiqueta
    texts = pool["text"].tolist()
    print(f"pool sin etiqueta: {len(texts)}", flush=True)
    preds = {k: [None] * len(texts) for k in KEYS}
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(lambda i, t: (i, {k: B.classify(t, k) for k in KEYS}), i, t): i
                for i, t in enumerate(texts)}
        for fut in as_completed(futs):
            i, row = fut.result()
            for k in KEYS: preds[k][i] = row[k]
            done += 1
            if done % 100 == 0: print(f"  {done}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    out = pool[["text", "categoria"]].copy()
    for k in KEYS: out[f"llm_{k}"] = preds[k]
    # concordancia: cuantos LLMs coinciden en la moda
    M = out[[f"llm_{k}" for k in KEYS]].to_numpy(dtype=float)
    def agree(r):
        v = r[~np.isnan(r)]
        if len(v) == 0: return 0, np.nan
        u, c = np.unique(v, return_counts=True)
        return int(c.max()), int(u[np.argmax(c)])
    ag = [agree(r) for r in M]
    out["n_agree"] = [a[0] for a in ag]
    out["consensus"] = [a[1] for a in ag]
    out.to_csv(OUT, index=False)
    hi = (out["n_agree"] >= 3).sum()
    print(f"alta concordancia (>=3 coinciden): {hi}/{len(out)}  -> auto-etiquetas candidatas", flush=True)
    print(f"GUARDADO {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
