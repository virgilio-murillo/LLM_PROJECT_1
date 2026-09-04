"""
Limpieza de texto: dos variantes para el experimento A/B (E7).
- agresiva: replica el pipeline actual del proyecto (mapea+borra emojis, sustituye mexicanismos,
  quita todo lo no alfanumerico, minusculas).
- minima: solo repara encoding (ftfy) y normaliza espacios; CONSERVA emojis, puntuacion, mayusculas.
El gold set del proyecto ya viene limpio con la version agresiva; para E7 partimos del RAW.
"""
from __future__ import annotations
import re, unicodedata
from pathlib import Path
import pandas as pd

try:
    import ftfy
except ImportError:
    ftfy = None

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

MEXICANISMOS = {
    r"\bnmms\b": "no me gusta", r"\bno mames\b": "sorprendente", r"\bno manches\b": "sorprendente",
    r"\bchido\b": "me gusta", r"\bpadre\b": "me gusta", r"\bgenial\b": "me gusta",
    r"\bculero\b": "malo", r"\bgacho\b": "malo", r"\bpinche\b": "", r"\bverga\b": "",
    r"\bchingon\b": "muy bueno", r"\bchingona\b": "muy bueno", r"\bchingada\b": "muy malo",
}


def _repair(t):
    if not isinstance(t, str): return ""
    try: t = t.encode("latin1").decode("utf-8")
    except Exception: pass
    if ftfy: t = ftfy.fix_text(t)
    return t


def limpiar_agresiva(t):
    """Replica el pipeline actual: destruye emojis/puntuacion, sustituye slang."""
    t = _repair(t)
    t = unicodedata.normalize("NFKC", t.lower())
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    for pat, rep in MEXICANISMOS.items():
        t = re.sub(pat, f" {rep} ", t, flags=re.IGNORECASE)
    t = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", t)   # borra emojis y puntuacion
    return re.sub(r"\s+", " ", t).strip()


def limpiar_minima(t):
    """Transformer-friendly: conserva emojis, puntuacion, mayusculas. Solo encoding + espacios + URLs/@."""
    t = _repair(t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)      # quita URLs
    t = re.sub(r"@\w+", " ", t)                          # quita menciones
    t = re.sub(r"\s+", " ", t).strip()                  # normaliza espacios
    return t


def cargar_raw_con_labels():
    """
    Carga el RAW y le pega la etiqueta humana emparejando por el comentario ya limpio.
    Devuelve DataFrame con: text_raw, text_agresiva, text_minima, label(0..4), categoria.
    Solo las filas con etiqueta humana (el gold 581).
    """
    import sys
    sys.path.insert(0, str(ROOT / "experimentos" / "src"))
    import common as C
    gold = C.load_gold()  # texto ya limpio (agresivo) + label + categoria
    # cargar raw por eje
    raws = []
    for cat, fn in [("infraestructura", "raw_infraestructura.csv"),
                    ("seguridad", "raw_seguridad.csv"), ("turismo", "raw_turismo.csv")]:
        p = RAW / fn
        d = pd.read_csv(p, encoding="utf-8-sig")
        col = "text" if "text" in d.columns else d.columns[0]
        d = d.rename(columns={col: "text_raw"})
        d["categoria"] = cat
        raws.append(d[["text_raw", "categoria"]])
    raw = pd.concat(raws, ignore_index=True)
    # emparejar por clave NORMALIZADA robusta: minusculas + solo alfanumerico (colapsa emojis/puntuacion)
    def norm_key(s):
        s = _repair(str(s)).lower()
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)
        s = re.sub(r"[^a-záéíóúüñ0-9]", "", s)  # quita TODO lo no alfanumerico incl espacios
        return s
    raw["_key"] = raw["text_raw"].map(norm_key)
    gold = gold.rename(columns={"text": "text_agresiva"})
    gold["_key"] = gold["text_agresiva"].map(norm_key)
    merged = gold.merge(raw.drop_duplicates("_key")[["text_raw", "_key"]], on="_key", how="left")
    matched = merged[merged["text_raw"].notna()].copy()
    matched["text_minima"] = matched["text_raw"].map(limpiar_minima)
    return matched[["text_raw", "text_agresiva", "text_minima", "label", "categoria"]]


if __name__ == "__main__":
    df = cargar_raw_con_labels()
    print("emparejados con raw:", len(df), "de 581")
    ej = df.iloc[0]
    print("RAW      :", ej["text_raw"][:80])
    print("AGRESIVA :", ej["text_agresiva"][:80])
    print("MINIMA   :", ej["text_minima"][:80])
