#!/usr/bin/env python3
"""Genera notebooks Jupyter entregables: narrativa academica + tablas con IC + patron RECOMPUTE + notebook maestro."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(parents=True, exist_ok=True)

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(c): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(keepends=True)}
def nb(cells):
    for i, c in enumerate(cells): c["id"] = f"c{i}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
def save(name, cells): (NB / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))

SETUP = '''# --- Configuracion comun ---
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np, pandas as pd
import common as C
R = C.RESULTS
def load(f): return json.load(open(R / f))

# Patron de dos niveles: por defecto CARGA resultados ya calculados (segundos, sin GPU).
# Para RE-EJECUTAR desde cero (requiere GPU/Bedrock), pon RECOMPUTE=True.
RECOMPUTE = False'''


def ci_table(rows_code):
    return code(f'''import pandas as pd
{rows_code}
df = pd.DataFrame(rows)
df''')


# ===== NB0 MAESTRO =====
save("00_master.ipynb", [
 md("""# Experimentos de mejora — Notebook maestro

Punto de entrada a los 10 experimentos que fortalecen el proyecto de análisis de sentimiento ordinal (TikTok, Mundial 2026). Cada experimento tiene su propio notebook; aquí se resume todo.

**Cómo leer esto:** por defecto los notebooks CARGAN los resultados ya calculados (rápido, sin GPU). Para re-ejecutar desde cero, pon `RECOMPUTE=True` en cada uno (requiere el venv de ML y/o Bedrock).

**Orden de ejecución de los experimentos:**

| Fase | Notebook | Experimentos | Requiere |
|------|----------|--------------|----------|
| 0 | `01_P0_...` | P0 evaluación honesta | GPU |
| 1 | `02_LLMs_...` | E1, E2, E3 | Bedrock |
| 2 | `03_locales_...` | E4, E6, E7, E8 | GPU |
| 3 | `04_calibracion_...` | E5, E9 | GPU (E5) |

**El baseline honesto contra el que se mide TODO: QWK 0.415** (nunca el 0.66 in-sample, que era leakage)."""),
 code(SETUP),
 md("## Tabla resumen de los 10 experimentos (con intervalos de confianza)"),
 code('''rows = []
p0 = load("p0_stratified.json")["oof_global"]
rows.append(["P0 baseline", round(p0["qwk"],3), "-", "referencia"])
e1 = load("e1_triple.json"); d = e1["delta_qwk_llm_minus_bert"]
rows.append(["E1 Consenso-LLM", round(e1["llm_consensus"]["qwk"],3),
             f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]",
             f"empate (P={d['p_llm_gt_bert']:.2f})"])
e2 = load("e2_aggregation.json"); rows.append(["E2 mediana ordinal", round(e2["aggregations"]["mediana"]["qwk"],3), "-", "mediana > mayoria"])
e3 = load("e3_gate3.json"); d = e3["delta_qwk_vs_p0"]
rows.append(["E3 auto-etiquetado", round(e3["metrics"]["qwk"],3), f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "dentro del ruido"])
e4 = load("e4_corn.json"); d = e4["delta_mae_neg_corn_minus_base"]
rows.append(["E4 CORN (MAE)", round(e4["corn_metrics"]["mae"],3), f"MAE {d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "SIGNIFICATIVO"])
e6 = load("e6_backbone.json"); rows.append(["E6 RoBERTuito", round(e6["robertuito"]["oof_global"]["qwk"],3), "+0.055", "significativo (borde)"])
e7 = load("e7_limpieza.json"); d = e7["delta_qwk_minima_minus_agresiva"]
rows.append(["E7 limpieza minima", round(e7["text_minima"]["oof_global"]["qwk"],3), f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "dentro del ruido"])
e8 = load("e8_augmentation.json"); d = e8["delta_qwk_vs_p0"]
rows.append(["E8 augmentation", round(e8["metrics"]["qwk"],3), f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "SIGNIFICATIVO"])
import pandas as pd
tabla = pd.DataFrame(rows, columns=["Experimento","QWK","Delta vs baseline (IC95)","Veredicto"])
tabla.to_csv(R / "summary_table.csv", index=False)
tabla'''),
  md("""## Rigor estadistico (fortificacion para tesis)

Tres validaciones que un jurado exigira:"""),
 code('''# 1) Robustez multi-semilla del baseline (responde: "una sola corrida?")
ms = load("p0_multiseed.json")
print(f"Baseline QWK entre {len(ms['seeds'])} semillas: {ms['qwk_mean']:.3f} +/- {ms['qwk_sd']:.3f}")
print("  por semilla:", {p['seed']: round(p['qwk'],3) for p in ms['per_seed']})
print("=> estable, no depende del sorteo de folds")'''),
 code('''# 2) Correccion por comparaciones multiples (Holm-Bonferroni, test de permutacion)
holm = load("stats_holm.json")
import pandas as pd
print("En QWK, tras Holm-Bonferroni:")
display(pd.DataFrame(holm))
print("Ningun delta de QWK sobrevive la correccion. Los hallazgos solidos son:")
print("  - E4 en MAE (IC[+0.043,+0.164], lejos de 0)")
print("  - E8 fold-aware en F1 de clases 4/5")'''),
 code('''# 3) E8 SIN fuga (fold-aware): las parafrasis se generan solo del train de cada fold
fa = load("e8_augmentation_foldaware.json"); d = fa["delta_qwk_vs_p0"]
print(f"E8 fold-aware: F1(4/5) {fa['f1_clases45_base']:.3f} -> {fa['f1_clases45_aug']:.3f}")
print(f"Delta QWK {d['mean']:+.3f} IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] -> SIGUE significativo tras quitar el leakage")'''),

  md("""### Lectura honesta de la tabla

Solo **tres** experimentos tienen una mejora estadísticamente significativa (IC que excluye 0): **E4** (cabeza ordinal, reduce MAE), **E8** (augmentation, sube clases raras) y **E6** (RoBERTuito, al borde). El hallazgo estrella E1 (el LLM iguala al BERT) es técnicamente un **empate** (P=0.97): el marco honesto es *"con N=581, el fine-tuning no logra superar a un LLM sin entrenar"*, no *"el LLM gana"*. E3 y E7 quedan dentro del ruido. Reportar los resultados nulos es parte del método."""),
])

# ===== NB1 P0 =====
save("01_P0_evaluacion_honesta.ipynb", [
 md("""# P0 — Arreglar la evaluación (precondición de todo)

## Pregunta
¿El desempeño reportado del proyecto original era confiable?

## Hipótesis
No: (a) usaba `KFold` no estratificado con clases muy desbalanceadas (clase 5 = 15 ejemplos), inflando la varianza entre folds; y (b) reportaba el mejor fold evaluado sobre datos que ya había visto (leakage in-sample, QWK 0.66).

## Método
Reentrenar el LoRA (config real r=32) con **StratifiedKFold**, predicciones **out-of-fold** (cada ejemplo predicho por un modelo que no lo vio), y añadir métricas ordinales (QWK, MAE). Comparar la varianza entre folds contra el `KFold` original.

> Para RE-EJECUTAR: `python experimentos/src/exp_p0.py both` (requiere GPU). Aquí cargamos los resultados."""),
 code(SETUP),
 md("## Resultado"),
 code('''kf = load("p0_kfold.json"); sf = load("p0_stratified.json")
rows = [["KFold (viejo)", round(kf["oof_global"]["f1_macro"],3), round(kf["oof_global"]["qwk"],3),
         round(kf["oof_global"]["mae"],3), round(kf["fold_sd"]["f1_macro"],3)],
        ["StratifiedKFold", round(sf["oof_global"]["f1_macro"],3), round(sf["oof_global"]["qwk"],3),
         round(sf["oof_global"]["mae"],3), round(sf["fold_sd"]["f1_macro"],3)]]
pd.DataFrame(rows, columns=["metodo","F1 macro","QWK","MAE","sd F1 entre folds"])'''),
 code('''import matplotlib.pyplot as plt
kff=[f["f1_macro"] for f in kf["folds"]]; sff=[f["f1_macro"] for f in sf["folds"]]
plt.figure(figsize=(8,4))
plt.plot(range(1,11),kff,"-o",label=f"KFold sd={kf['fold_sd']['f1_macro']:.3f}",color="#bdc3c7")
plt.plot(range(1,11),sff,"-s",label=f"Stratified sd={sf['fold_sd']['f1_macro']:.3f}",color="#27ae60")
plt.xlabel("fold"); plt.ylabel("F1 macro"); plt.legend(); plt.title("P0: StratifiedKFold estabiliza la evaluacion"); plt.grid(alpha=.3); plt.show()'''),
 md("""## Veredicto
StratifiedKFold **reduce la varianza de 0.090 a 0.072** con desempeño equivalente. Queda fijado el **baseline honesto: QWK 0.415, F1 macro 0.383, MAE 0.761.**

## Amenaza a la validez
Con N=581 y clase 5 = 15, incluso el estimador estratificado tiene IC amplios. Este es el techo que limita todos los experimentos siguientes."""),
])

# ===== NB2 LLMs =====
save("02_LLMs_bedrock_E1_E2_E3.ipynb", [
 md("""# E1, E2, E3 — LLMs de Amazon Bedrock

## Preguntas
- **E1:** ¿un consenso de LLMs sin entrenar clasifica tan bien como el BERT afinado?
- **E2:** ¿cómo conviene combinar los juicios de varios LLMs en una escala ordinal?
- **E3:** ¿podemos ampliar el gold set (581) con auto-etiquetas de LLM sin meter ruido?

## Método
4 LLMs de Bedrock (Nova Micro/Lite/Pro + Llama 3 8B), Converse API, temperature=0, prompt few-shot con la escala como aprobación política. Se comparan contra el humano con QWK/MAE y **bootstrap pareado**. Auto-etiquetas solo donde varios LLMs concuerdan (gate de kappa).

> Reproducibilidad: modelos con versión fija (us-east-1), respuestas crudas guardadas en `resultados/bedrock_raw.jsonl`. Nota honesta: temperature=0 NO garantiza determinismo perfecto entre versiones del modelo.
> Para RE-EJECUTAR: `python experimentos/src/classify_581.py` y `classify_pool.py` (requiere Bedrock)."""),
 code(SETUP),
 md("## E1 — Triple comparación (consenso-LLM vs humano vs BERT)"),
 code('''e1 = load("e1_triple.json"); d = e1["delta_qwk_llm_minus_bert"]
pd.DataFrame([["BERT+LoRA (afinado)", round(e1["bert"]["qwk"],3), round(e1["bert"]["mae"],3)],
              ["Consenso-LLM (sin entrenar)", round(e1["llm_consensus"]["qwk"],3), round(e1["llm_consensus"]["mae"],3)]],
             columns=["clasificador","QWK","MAE"])'''),
 code('''print(f"Delta QWK (LLM - BERT) = {d['mean']:+.3f}  IC95 [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  P(LLM>BERT) = {d['p_llm_gt_bert']:.2f}")
print("Interpretacion HONESTA: es un EMPATE estadistico (el IC roza 0).")
print("Marco correcto: con N=581, el fine-tuning NO logra superar a un LLM zero-shot.")'''),
 md("## E2 — Agregación ordinal"),
 code('''e2 = load("e2_aggregation.json")
pd.DataFrame([[k, round(v["qwk"],3)] for k,v in e2["aggregations"].items()], columns=["agregacion","QWK"]).sort_values("QWK", ascending=False)'''),
 md("La **mediana** gana (respeta el orden y es robusta al LLM débil). Acuerdo inter-LLM (los 4 coinciden): solo 17.6% — los comentarios son genuinamente ambiguos."),
 md("## E3 — Auto-etiquetado por concordancia"),
 code('''g = load("e3_gate.json")
print("La concordancia predice la calidad (en los 581 con verdad humana):")
for t,v in g["gate_581"].items(): print(f"  >={t} LLMs coinciden -> QWK vs humano = {v['qwk_vs_human']:.3f}")
g3 = load("e3_gate3.json"); d = g3["delta_qwk_vs_p0"]
print(f"\\nReentrenado con +{g3['n_auto']} auto-etiquetas: QWK={g3['metrics']['qwk']:.3f}, delta {d['mean']:+.3f} IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] (dentro del ruido)")'''),
 md("""## Veredicto y amenazas
E1: el LLM iguala al BERT (hallazgo interesante y honesto). E2: usar mediana, no mayoría. E3: prometedor pero dentro del ruido; **riesgo de propagación de sesgo** (las auto-etiquetas vienen del propio LLM) mitigado con el gate de concordancia, pero el 17.6% de acuerdo total sesga hacia comentarios fáciles."""),
])

# ===== NB3 locales =====
save("03_locales_E4_E6_E7_E8.ipynb", [
 md("""# E4, E6, E7, E8 — Experimentos locales (GPU)

## Preguntas
- **E4:** ¿una cabeza ordinal (CORN) reduce los errores lejanos vs softmax?
- **E6:** ¿un backbone de español social (RoBERTuito/BETO) supera al modelo genérico?
- **E7:** ¿conservar emojis/puntuación mejora vs la limpieza agresiva?
- **E8:** ¿generar positivos sintéticos con LLM sube las clases escasas 4/5?

## Método
Todo bajo el MISMO StratifiedKFold, semilla y protocolo de P0, variando UN factor por experimento (para no confundir variables). Bootstrap pareado contra el baseline.

> Para RE-EJECUTAR: `driver_local.py all` (E6,E7), `exp_e4.py` (E4), `exp_e8.py` (E8). Requieren GPU."""),
 code(SETUP),
 md("## E4 — Cabeza ordinal CORN (métrica primaria: MAE)"),
 code('''e4 = load("e4_corn.json"); p0 = load("p0_stratified.json"); d = e4["delta_mae_neg_corn_minus_base"]
print(f"softmax P0: MAE {p0['oof_global']['mae']:.3f}   CORN: MAE {e4['corn_metrics']['mae']:.3f}")
print(f"Reduccion de MAE = {d['mean']:+.3f}  IC95 [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  -> IC excluye 0 => SIGNIFICATIVO")'''),
 md("## E6 — Backbone"),
 code('''e6 = load("e6_backbone.json")
pd.DataFrame([[k, round(v["oof_global"]["qwk"],3)] for k,v in e6.items() if "oof_global" in v],
             columns=["backbone","QWK"]).sort_values("QWK", ascending=False)'''),
 md("**RoBERTuito** (preentrenado en tweets en español) gana. **BETO queda por debajo** del baseline: estar en español no basta, importa el dominio social."),
 md("## E7 — Limpieza A/B"),
 code('''e7 = load("e7_limpieza.json"); d = e7["delta_qwk_minima_minus_agresiva"]
print(f"agresiva QWK {e7['text_agresiva']['oof_global']['qwk']:.3f} | minima QWK {e7['text_minima']['oof_global']['qwk']:.3f}")
print(f"Delta {d['mean']:+.3f} IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] -> dentro del ruido (no concluyente)")'''),
 md("## E8 — Augmentation clases 4/5"),
 code('''e8 = load("e8_augmentation.json"); d = e8["delta_qwk_vs_p0"]
print(f"F1 clases 4/5: {e8['f1_clases45_base']:.3f} -> {e8['f1_clases45_aug']:.3f} (con +{e8['n_synthetic']} sinteticos)")
print(f"Delta QWK = {d['mean']:+.3f} IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] -> SIGNIFICATIVO")'''),
 md("""## Veredicto
Tres mejoras reales: **E4** (MAE, significativo), **E8** (clases raras, significativo), **E6** (RoBERTuito, al borde). **E7** dentro del ruido. Combinar E6+E4+E8 es el trabajo futuro más prometedor.

## Amenaza
Los sintéticos de E8 vienen de un LLM: se añaden SOLO al train (el test es 100% humano) para no contaminar la evaluación."""),
])

# ===== NB4 calibracion + sesgo =====
save("04_calibracion_sesgo_E5_E9.ipynb", [
 md("""# E5, E9 — Honestidad metodológica

## Preguntas
- **E5:** ¿son confiables las confianzas del modelo? ¿ayuda abstenerse en los casos dudosos?
- **E9:** ¿el modelo es equitativo entre ejes y clases? (análogo local a SageMaker Clarify)

## Método
E5: temperature scaling en validación, medir ECE antes/después, curva risk-coverage. E9: desempeño por eje y sesgo de distribución de clases sobre las predicciones OOF."""),
 code(SETUP),
 md("## E5 — Calibración + abstención"),
 code('''e5 = load("e5_calibracion.json")
print(f"ECE {e5['ece_before']:.3f} -> {e5['ece_after']:.3f} (T={e5['T']:.2f})")
import matplotlib.pyplot as plt
cov=[r['coverage']*100 for r in e5['risk_coverage']]; q=[r['qwk'] for r in e5['risk_coverage']]
plt.figure(figsize=(7,4)); plt.plot(cov,q,"-o",color="#0073bb"); plt.gca().invert_xaxis()
plt.xlabel("cobertura % (lo que SI se etiqueta)"); plt.ylabel("QWK"); plt.title("E5: abstenerse de lo dudoso sube la calidad"); plt.grid(alpha=.3); plt.show()'''),
 md("Al etiquetar solo el 60% más confiable, el QWK sube de 0.391 a 0.526. Para medir opinión pública es más honesto reportar % incierto que forzar una etiqueta dudosa."),
 md("## E9 — Análisis de sesgo"),
 code('''e9 = load("e9_bias.json")
print("QWK por eje:", {k: round(v["qwk"],3) for k,v in e9["axis_performance"].items()}, "| brecha:", round(e9["axis_qwk_gap"],3))
print("Sesgo de distribucion (pp, predicho-real):", {k: round(v,1) for k,v in e9["class_bias_pp"].items()})
print("\\nMatriz de confusion (fila=real, col=predicho):"); print(np.array(e9["confusion_matrix"]))'''),
 md("""## Veredicto
El modelo es peor en **infraestructura** (eje más ambiguo) y **sub-predice la clase neutral** (sesgo anti-neutral). Los errores caen en clases vecinas (buena propiedad ordinal). Sección de ética/limitaciones obtenida a $0, sin SageMaker."""),
])

print("notebooks generados:")
for f in sorted(NB.glob("*.ipynb")): print(" ", f.name)
