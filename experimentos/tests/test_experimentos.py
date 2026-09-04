"""
Tests ligeros de validacion (pytest). Suben la credibilidad a bajo costo:
- invariantes del gold set (N=581, distribucion de clases exacta),
- QWK de vectores identicos == 1,
- esquema de los JSON de resultados.
Correr: python -m pytest experimentos/tests/ -q  (o: python experimentos/tests/test_experimentos.py)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import common as C


def test_gold_invariantes():
    g = C.load_gold()
    assert len(g) == 581, f"gold debe ser 581, es {len(g)}"
    dist = g["label"].value_counts().sort_index().to_dict()
    assert dist == {0: 173, 1: 159, 2: 204, 3: 30, 4: 15}, f"distribucion inesperada: {dist}"


def test_corpus_completo():
    full = C.load_full_corpus()
    assert len(full) == 2578, f"corpus limpio debe ser 2578, es {len(full)}"
    assert full["label"].notna().sum() == 581


def test_qwk_identico():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    assert abs(C.qwk_fn(y, y) - 1.0) < 1e-9, "QWK de vectores identicos debe ser 1"


def test_metricas_ordinales_keys():
    m = C.ordinal_metrics([0, 1, 2], [0, 1, 2])
    for k in ["accuracy", "qwk", "mae", "f1_macro", "balanced_accuracy"]:
        assert k in m


def test_esquema_resultados():
    esperados = {
        "p0_stratified.json": ["oof_global", "fold_sd"],
        "e1_triple.json": ["bert", "llm_consensus", "delta_qwk_llm_minus_bert"],
        "e4_corn.json": ["corn_metrics", "delta_mae_neg_corn_minus_base"],
        "e8_augmentation.json": ["metrics", "delta_qwk_vs_p0", "f1_clases45_aug"],
    }
    for f, keys in esperados.items():
        p = C.RESULTS / f
        if p.exists():
            d = json.load(open(p))
            for k in keys:
                assert k in d, f"{f} debe tener la clave {k}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("Todos los tests pasaron.")
