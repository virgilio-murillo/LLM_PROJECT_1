"""
Robustez multi-semilla del baseline P0. Responde a la critica "una sola corrida de CV".
Corre CV estratificada con varias semillas y reporta media +/- sd de QWK/F1/MAE entre semillas.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import runner as R

SEEDS = [61298, 7, 2024]


def run():
    df = C.load_gold(); y = df["label"].to_numpy(); texts = df["text"].tolist()
    per_seed = []
    for s in SEEDS:
        C.SEED = s  # runner usa C.SEED para el split y el trainer
        C.set_all_seeds(s)
        out = R.cv_train(texts, y, k=5, epochs=7, tag=f"seed{s}")
        g = out["oof_global"]
        per_seed.append({"seed": s, "qwk": g["qwk"], "f1_macro": g["f1_macro"], "mae": g["mae"]})
        print(f"  seed {s}: QWK={g['qwk']:.3f} F1={g['f1_macro']:.3f} MAE={g['mae']:.3f}", flush=True)
    qwks = np.array([p["qwk"] for p in per_seed]); f1s = np.array([p["f1_macro"] for p in per_seed]); maes = np.array([p["mae"] for p in per_seed])
    res = {"seeds": SEEDS, "per_seed": per_seed,
           "qwk_mean": float(qwks.mean()), "qwk_sd": float(qwks.std(ddof=1)),
           "f1_mean": float(f1s.mean()), "f1_sd": float(f1s.std(ddof=1)),
           "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=1))}
    C.save_result("p0_multiseed", res)
    print(f"\n[MULTI-SEED] QWK {res['qwk_mean']:.3f} +/- {res['qwk_sd']:.3f}  F1 {res['f1_mean']:.3f} +/- {res['f1_sd']:.3f}", flush=True)


if __name__ == "__main__":
    run(); print("DONE multiseed", flush=True)
