"""
Corre E7 (limpieza A/B), E6 (backbone) y E4 (cabeza ordinal) en secuencia.
Cada uno varia UN factor, todo lo demas fijo (StratifiedKFold, mismas semillas).
Guarda resultados JSON. Diseñado para tmux (log a archivo).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import runner as R
import limpieza as L


def e7_limpieza():
    print("\n===== E7: limpieza A/B (agresiva vs minima) =====", flush=True)
    df = L.cargar_raw_con_labels()
    y = df["label"].to_numpy()
    res = {}
    for variante in ["text_agresiva", "text_minima"]:
        texts = df[variante].tolist()
        out = R.cv_train(texts, y, tag=f"e7_{variante}", k=10, epochs=7)
        res[variante] = {"oof_global": out["oof_global"], "fold_mean": out["fold_mean"],
                         "fold_sd": out["fold_sd"], "oof_pred": out["oof_pred"]}
        print(f"  {variante}: QWK={out['oof_global']['qwk']:.3f} F1={out['oof_global']['f1_macro']:.3f}", flush=True)
    # bootstrap pareado minima - agresiva (mismo y, mismo orden)
    pa = np.array(res["text_agresiva"]["oof_pred"]); pm = np.array(res["text_minima"]["oof_pred"])
    m, lo, hi, pgt = C.paired_bootstrap_diff(y, pm, pa, C.qwk_fn, n=2000)
    res["delta_qwk_minima_minus_agresiva"] = {"mean": m, "ci_low": lo, "ci_high": hi, "p_gt": pgt}
    res["n"] = len(y)
    C.save_result("e7_limpieza", res)
    print(f"  Delta QWK (minima - agresiva) = {m:+.3f} IC[{lo:+.3f},{hi:+.3f}] P={pgt:.2f}", flush=True)


def e6_backbone():
    print("\n===== E6: backbone espanol social =====", flush=True)
    df = C.load_gold(); texts = df["text"].tolist(); y = df["label"].to_numpy()
    backbones = {
        "nlptown-baseline": "nlptown/bert-base-multilingual-uncased-sentiment",
        "beto": "dccuchile/bert-base-spanish-wwm-uncased",
        "robertuito": "pysentimiento/robertuito-base-uncased",
    }
    res = {}
    for name, mid in backbones.items():
        try:
            out = R.cv_train(texts, y, model_name=mid, tag=f"e6_{name}", k=10, epochs=7)
            res[name] = {"model": mid, "oof_global": out["oof_global"], "fold_sd": out["fold_sd"]}
            print(f"  {name}: QWK={out['oof_global']['qwk']:.3f} F1={out['oof_global']['f1_macro']:.3f}", flush=True)
        except Exception as e:
            res[name] = {"model": mid, "error": str(e)[:200]}
            print(f"  {name}: ERROR {str(e)[:120]}", flush=True)
    C.save_result("e6_backbone", res)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e7"): e7_limpieza()
    if which in ("all", "e6"): e6_backbone()
    print("\nDONE driver", flush=True)
