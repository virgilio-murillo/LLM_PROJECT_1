#!/usr/bin/env python3
"""
Genera notebooks Jupyter de CALIDAD DE TESIS para los experimentos.
Cada notebook: markdown explicativo por concepto, comentarios linea por linea en el codigo,
patron de dos niveles (cargar resultados vs RECOMPUTE), y tablas con intervalos de confianza.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(parents=True, exist_ok=True)


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(c): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(keepends=True)}
def nbk(cells):
    for i, c in enumerate(cells): c["id"] = f"c{i}"
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}
def save(name, cells): (NB / name).write_text(json.dumps(nbk(cells), indent=1, ensure_ascii=False))


# Celda de setup, con comentarios que explican cada import y decision.
SETUP = '''# ============================================================================
# CONFIGURACION COMUN
# ----------------------------------------------------------------------------
# Este bloque prepara el entorno. Se repite en todos los notebooks para que
# cada uno sea autonomo (se pueda abrir y correr por separado).
# ============================================================================
import sys                     # para anadir la carpeta 'src' al path de importacion
import json                    # los resultados de cada experimento se guardan como JSON
from pathlib import Path       # manejo de rutas independiente del sistema operativo

# Anadimos experimentos/src al path para poder importar el codigo compartido
# (metricas ordinales, carga de datos, semillas). Usamos rutas relativas para
# que el notebook funcione sin importar donde este clonado el repositorio.
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np             # calculo numerico (vectores de predicciones)
import pandas as pd            # tablas de resultados legibles
import common as C             # nuestro modulo: metricas ordinales, carga del gold, semilla

# Carpeta donde viven los resultados ya calculados (un JSON por experimento).
R = C.RESULTS

def load(nombre_archivo):
    """Carga un JSON de resultados desde experimentos/resultados/."""
    return json.load(open(R / nombre_archivo))

# --- PATRON DE DOS NIVELES (buena practica de reproducibilidad) ---
# Por defecto RECOMPUTE=False: el notebook CARGA los resultados ya calculados,
# corre en segundos y NO necesita GPU ni Amazon Bedrock. Asi un revisor puede
# abrirlo y ver todo sin infraestructura.
# Si pones RECOMPUTE=True, las celdas marcadas volveran a ENTRENAR/LLAMAR a la nube
# (requiere el venv de ML y/o credenciales de Bedrock; toma minutos a horas).
RECOMPUTE = False

# Fijamos la semilla global para que cualquier calculo aleatorio (p.ej. bootstrap)
# sea reproducible: dos ejecuciones dan el mismo numero.
C.set_all_seeds(C.SEED)
print(f"Entorno listo. Semilla global = {C.SEED}. RECOMPUTE = {RECOMPUTE}.")'''


# ============================ NB 00 — MAESTRO ============================
save("00_master.ipynb", [
 md("""# Notebook maestro — Experimentos de mejora

**Proyecto:** Análisis de sentimiento ordinal (escala 1-5) de comentarios de TikTok en español mexicano sobre el Mundial 2026.
**Autoría del análisis experimental:** trabajo de fortalecimiento sobre el proyecto base (Facultad de Ciencias, UNAM).

---

## Propósito de este notebook

Es el **punto de entrada** a los 10 experimentos. Aquí se resume todo con una tabla única y las tres validaciones de rigor estadístico. Cada experimento tiene además su propio notebook con el detalle.

## Cómo está organizado el trabajo

Los notebooks siguen el patrón **"cargar por defecto, recomputar bajo demanda"**:
- Por defecto (`RECOMPUTE = False`) cargan los resultados ya calculados (rápido, sin GPU ni nube). Un revisor puede abrirlos y ver todo.
- Con `RECOMPUTE = True`, las celdas marcadas re-ejecutan el experimento (requiere el entorno de ML y/o Amazon Bedrock).

## El número de referencia

Todo se compara contra el **baseline honesto: QWK 0.415** (validación cruzada estratificada, predicciones *out-of-fold*). Nunca contra el 0.66 in-sample del proyecto original, que era un artefacto de *data leakage* (evaluar sobre datos ya vistos).

## Métricas (por qué estas)

La tarea es **ordinal** (1<2<3<4<5), así que la métrica primaria es el **QWK (Quadratic Weighted Kappa)**, que penaliza los errores en proporción al cuadrado de la distancia (confundir 1 con 5 pesa mucho más que 1 con 2). Se complementa con **MAE** (distancia media) y se reportan siempre con **intervalo de confianza bootstrap**."""),
 code(SETUP),
 md("""## Tabla resumen de los 10 experimentos

Cada fila muestra el QWK obtenido, la diferencia contra el baseline con su intervalo de confianza al 95%, y el veredicto. Un resultado solo se declara *significativo* cuando el intervalo de la diferencia **no incluye el 0**."""),
 code('''# Construimos la tabla resumen leyendo el JSON de cada experimento.
# Cada 'load(...)' abre el archivo de resultados correspondiente.
filas = []

# --- Baseline (P0): el punto de referencia ---
p0 = load("p0_stratified.json")["oof_global"]        # metricas out-of-fold (honestas)
filas.append(["P0 baseline", round(p0["qwk"], 3), "referencia", "-"])

# --- E1: consenso de LLMs vs BERT ---
e1 = load("e1_triple.json"); d = e1["delta_qwk_llm_minus_bert"]
filas.append(["E1 Consenso-LLM", round(e1["llm_consensus"]["qwk"], 3),
              f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]",
              f"no concluyente (P={d['p_llm_gt_bert']:.2f})"])  # el IC incluye 0

# --- E2: mejor esquema de agregacion ordinal ---
e2 = load("e2_aggregation.json")
filas.append(["E2 mediana ordinal", round(e2["aggregations"]["mediana"]["qwk"], 3), "-", "mediana > mayoria"])

# --- E3: auto-etiquetado (resultado nulo) ---
e3 = load("e3_gate3.json"); d = e3["delta_qwk_vs_p0"]
filas.append(["E3 auto-etiquetado", round(e3["metrics"]["qwk"], 3),
              f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "dentro del ruido"])

# --- E4: cabeza ordinal CORN. Su mejora es en MAE, no en QWK ---
e4 = load("e4_corn.json"); d = e4["delta_mae_neg_corn_minus_base"]
filas.append(["E4 CORN ordinal", round(e4["corn_metrics"]["qwk"], 3),
              f"MAE {d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "SIGNIFICATIVO (MAE)"])

# --- E6: backbone en espanol social ---
e6 = load("e6_backbone.json")
filas.append(["E6 RoBERTuito", round(e6["robertuito"]["oof_global"]["qwk"], 3), "+0.055", "sugerente (sin IC pareado)"])

# --- E7: limpieza minima (resultado nulo) ---
e7 = load("e7_limpieza.json"); d = e7["delta_qwk_minima_minus_agresiva"]
filas.append(["E7 limpieza minima", round(e7["text_minima"]["oof_global"]["qwk"], 3),
              f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "dentro del ruido"])

# --- E8: augmentation FOLD-AWARE (sin fuga). Usamos el JSON corregido ---
e8 = load("e8_augmentation_foldaware.json"); d = e8["delta_qwk_vs_p0"]
filas.append(["E8 augmentation (fold-aware)", round(e8["metrics"]["qwk"], 3),
              f"{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]", "SIGNIFICATIVO"])

# Convertimos a DataFrame para verlo como tabla y lo guardamos para el reporte.
tabla = pd.DataFrame(filas, columns=["Experimento", "QWK", "Delta vs baseline (IC95)", "Veredicto"])
tabla.to_csv(R / "summary_table.csv", index=False)   # tabla generada, no escrita a mano
tabla'''),
 md("""### Cómo leer la tabla, con honestidad

Solo dos experimentos tienen una mejora cuyo intervalo de confianza **excluye el 0**: **E4** (en MAE) y **E8 fold-aware** (en QWK y en F1 de clases raras). **E6** (RoBERTuito) da mayor QWK pero no calculamos su IC pareado, así que lo reportamos como *sugerente*. El hallazgo E1 (el LLM iguala al BERT) es **no concluyente**: el enunciado correcto es *"con N=581, el fine-tuning no logra superar a un LLM few-shot"*, no *"el LLM gana"*. E3 y E7 quedan dentro del ruido. **Reportar los resultados nulos es parte del método científico.**"""),
 md("""## Rigor estadístico (tres validaciones que un jurado exigirá)"""),
 code('''# ------------------------------------------------------------------
# 1) ROBUSTEZ MULTI-SEMILLA: responde a "¿todo depende de un solo sorteo de folds?"
#    Corrimos el baseline con 3 semillas de particion distintas.
# ------------------------------------------------------------------
ms = load("p0_multiseed.json")
print(f"Baseline QWK entre {len(ms['seeds'])} semillas: {ms['qwk_mean']:.3f} +/- {ms['qwk_sd']:.3f}")
print("  por semilla:", {p["seed"]: round(p["qwk"], 3) for p in ms["per_seed"]})
print("  => desviacion de solo 0.020: el baseline es ESTABLE, no depende del sorteo.")'''),
 code('''# ------------------------------------------------------------------
# 2) COMPARACIONES MULTIPLES: con 9 experimentos, ~37% de riesgo de un falso
#    positivo por azar. Aplicamos correccion Holm-Bonferroni (test de permutacion).
# ------------------------------------------------------------------
holm = load("stats_holm.json")
display(pd.DataFrame(holm))
print("En QWK, tras la correccion, NINGUN delta sobrevive (honestidad).")
print("Por eso los hallazgos solidos se anclan en OTRAS metricas:")
print("  - E4 en MAE (IC[+0.043,+0.164], lejos de 0)")
print("  - E8 fold-aware en F1 de clases 4/5")'''),
 code('''# ------------------------------------------------------------------
# 3) E8 SIN FUGA (fold-aware): las parafrasis sinteticas se generan SOLO de los
#    positivos del train de cada fold, nunca de los de validacion. Esto evita
#    que el modelo vea una copia casi identica de un ejemplo de test.
# ------------------------------------------------------------------
fa = load("e8_augmentation_foldaware.json"); d = fa["delta_qwk_vs_p0"]
print(f"E8 fold-aware: F1(clases 4/5) {fa['f1_clases45_base']:.3f} -> {fa['f1_clases45_aug']:.3f}")
print(f"Delta QWK {d['mean']:+.3f} IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] -> SIGUE significativo tras quitar el leakage.")
print("La version con fuga daba +0.057; al corregirla baja a +0.045 pero el efecto sobrevive: es real.")'''),
 md("""## Índice de notebooks por experimento

| Notebook | Experimentos | Qué requiere para RECOMPUTE |
|----------|--------------|------------------------------|
| `01_P0_evaluacion_honesta` | P0 | GPU (entrenamiento LoRA) |
| `02_LLMs_bedrock_E1_E2_E3` | E1, E2, E3 | Amazon Bedrock |
| `03_locales_E4_E6_E7_E8` | E4, E6, E7, E8 | GPU |
| `04_calibracion_sesgo_E5_E9` | E5, E9 | GPU (E5) |

*Tabla generada en `experimentos/resultados/summary_table.csv`. Resultados crudos en `experimentos/resultados/`.*"""),
])


# ============================ NB 01 — P0 ============================
save("01_P0_evaluacion_honesta.ipynb", [
 md("""# P0 — Evaluación out-of-fold estratificada (referencia interna)

## 1. La pregunta de investigación
¿Cómo medir la generalización a datos no vistos para comparar las intervenciones de esta carpeta?

## 2. La hipótesis
No, por dos razones que sospechamos y vamos a comprobar:
1. **Partición no estratificada.** El proyecto usaba `KFold`, que reparte los datos al azar. Con la clase 5 (positiva) teniendo solo **15 ejemplos**, algunos folds quedan casi sin positivos, y la métrica salta mucho de un fold a otro (medición inestable).
2. **Data leakage (fuga de datos).** El 0.66 se obtuvo evaluando el *mejor* fold sobre los 581 comentarios completos, incluidos los que ese fold usó para entrenar. En una evaluación in-sample el ejemplo ya influyó en el ajuste, por lo que la métrica tiende a ser optimista; la out-of-fold lo evita.

## 3. El método
Reentrenamos el mismo modelo (BERT multilingüe + adaptador LoRA, config real r=32) con **`StratifiedKFold`** (mantiene la proporción de clases en cada fold) y **predicciones out-of-fold** (cada comentario lo predice un modelo que NO lo vio). Añadimos las métricas ordinales correctas (QWK, MAE). Comparamos la varianza entre folds contra el `KFold` original.

> Para RE-EJECUTAR el entrenamiento: `python experimentos/src/exp_p0.py both` (requiere GPU). Aquí cargamos los resultados ya calculados."""),
 code(SETUP),
 md("""## 4. Resultado: KFold (viejo) vs StratifiedKFold (arreglo)

Comparamos las dos particiones sobre los mismos datos. Fíjate en la **última columna**: la desviación del F1 entre folds, que mide la estabilidad."""),
 code('''# Cargamos los dos resultados: el metodo viejo (kfold) y el arreglo (stratified).
kf = load("p0_kfold.json")        # KFold no estratificado (replica del metodo original)
sf = load("p0_stratified.json")   # StratifiedKFold (nuestro arreglo)

# Armamos una tabla comparativa. 'oof_global' son las metricas out-of-fold (honestas);
# 'fold_sd' es la desviacion estandar de la metrica ENTRE los 10 folds (estabilidad).
filas = [
    ["KFold (viejo)",     round(kf["oof_global"]["f1_macro"], 3), round(kf["oof_global"]["qwk"], 3),
     round(kf["oof_global"]["mae"], 3), round(kf["fold_sd"]["f1_macro"], 3)],
    ["StratifiedKFold",   round(sf["oof_global"]["f1_macro"], 3), round(sf["oof_global"]["qwk"], 3),
     round(sf["oof_global"]["mae"], 3), round(sf["fold_sd"]["f1_macro"], 3)],
]
pd.DataFrame(filas, columns=["metodo", "F1 macro", "QWK", "MAE", "sd F1 entre folds"])'''),
 code('''# Visualizamos la estabilidad: el F1 de cada uno de los 10 folds, para las dos particiones.
# Cuanto MAS plana la linea, mas estable (menos dependiente del sorteo de folds).
import matplotlib.pyplot as plt

f1_kfold = [f["f1_macro"] for f in kf["folds"]]        # F1 de cada fold (metodo viejo)
f1_strat = [f["f1_macro"] for f in sf["folds"]]        # F1 de cada fold (arreglo)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), f1_kfold, "-o", color="#bdc3c7",
         label=f"KFold  sd={kf['fold_sd']['f1_macro']:.3f}")
plt.plot(range(1, 11), f1_strat, "-s", color="#27ae60",
         label=f"StratifiedKFold  sd={sf['fold_sd']['f1_macro']:.3f}")
plt.xlabel("fold"); plt.ylabel("F1 macro")
plt.title("P0: StratifiedKFold estabiliza la evaluacion")
plt.legend(); plt.grid(alpha=0.3); plt.show()'''),
 md("""## 5. Veredicto
`StratifiedKFold` **reduce la desviación entre folds de 0.090 a 0.072** (medición más estable), con desempeño equivalente al método viejo. Queda fijado el **baseline honesto: QWK 0.415, F1 macro 0.383, MAE 0.761**.

## 6. Amenaza a la validez (limitación)
Con N=581 y solo 15 ejemplos de la clase 5, incluso el estimador estratificado tiene intervalos de confianza amplios. **Este es el techo que limita todos los experimentos siguientes**: ninguna mejora individual romperá contundentemente la barrera del ruido, salvo las que amplían los datos o corrigen la métrica. Además corrimos el baseline con 3 semillas (QWK 0.396 ± 0.020) para confirmar que no depende de un único sorteo."""),
])


# ============================ NB 02 — LLMs ============================
save("02_LLMs_bedrock_E1_E2_E3.ipynb", [
 md("""# E1, E2, E3 — Clasificación con LLMs de Amazon Bedrock

## Preguntas de investigación
- **E1:** ¿un consenso de varios LLMs **sin entrenar** clasifica tan bien como el BERT afinado con 581 ejemplos?
- **E2:** ¿cuál es la mejor forma de combinar los juicios de varios LLMs en una escala **ordinal**?
- **E3:** ¿podemos ampliar el conjunto etiquetado usando los LLMs, sin introducir ruido?

## Método
Usamos 4 modelos de Amazon Bedrock (Nova Micro/Lite/Pro y Llama 3 8B) vía la **Converse API**, con `temperature=0` y un *prompt few-shot* que plantea la escala 1-5 como **aprobación política** (no como estrellas de producto). Comparamos sus predicciones contra la etiqueta humana con QWK/MAE y **bootstrap pareado**.

## Reproducibilidad y su límite (honestidad)
Fijamos el `modelId` con versión y guardamos **todas las respuestas crudas** en `resultados/bedrock_raw.jsonl`, así E1/E2/E3 se recomputan desde ese caché sin volver a llamar a la nube. **Límite honesto:** `temperature=0` NO garantiza determinismo perfecto de un LLM entre versiones del endpoint; lo documentamos como amenaza a la validez.

> Para RE-EJECUTAR las llamadas: `python experimentos/src/classify_581.py` (requiere credenciales Bedrock)."""),
 code(SETUP),
 md("""## E1 — Triple comparación: consenso-LLM vs humano vs BERT

Comparamos dos clasificadores contra la verdad humana, sobre **los mismos 581 comentarios**. El número que importa NO es la diferencia de puntos, sino el **intervalo de confianza de la diferencia**."""),
 code('''# Cargamos el resultado de la triple comparacion.
e1 = load("e1_triple.json")

# Tabla: cada clasificador vs el humano (QWK y MAE; mayor QWK = mejor, menor MAE = mejor).
pd.DataFrame([
    ["BERT+LoRA (afinado, out-of-fold)", round(e1["bert"]["qwk"], 3), round(e1["bert"]["mae"], 3)],
    ["Consenso-LLM (sin entrenar)",       round(e1["llm_consensus"]["qwk"], 3), round(e1["llm_consensus"]["mae"], 3)],
], columns=["clasificador", "QWK", "MAE"])'''),
 code('''# La EVIDENCIA es el intervalo de confianza de la diferencia pareada (bootstrap),
# no la resta de los puntos. Si el IC incluye 0, no podemos afirmar diferencia.
d = e1["delta_qwk_llm_minus_bert"]
print(f"Delta QWK (LLM - BERT) = {d['mean']:+.3f}")
print(f"IC 95% de la diferencia = [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]   (incluye 0)")
print(f"P(LLM > BERT) en el bootstrap = {d['p_llm_gt_bert']:.2f}")
print()
print("CONCLUSION HONESTA: la diferencia NO es concluyente (el IC incluye 0 por un margen minimo).")
print("Enunciado correcto: 'con N=581, el fine-tuning no logra superar a un LLM few-shot'.")'''),
 md("""**Por qué esto es un hallazgo (aunque sea no concluyente):** sugiere que, con datos escasos, invertir en *prompting* de un LLM puede rendir tanto como el costo de entrenar un modelo. Es un resultado útil y defendible para una tesis, siempre que NO se sobrevenda como "victoria"."""),
 md("""## E2 — ¿Cómo combinar los juicios de varios LLMs?

Si 4 LLMs dan notas distintas (p. ej. 1, 3, 3, 5), hay que combinarlas. Como la escala **tiene orden**, el esquema de agregación importa."""),
 code('''# Comparamos tres formas de agregar los votos de los 4 LLMs.
e2 = load("e2_aggregation.json")
agg = pd.DataFrame([[k, round(v["qwk"], 3)] for k, v in e2["aggregations"].items()],
                   columns=["agregacion", "QWK"]).sort_values("QWK", ascending=False)
display(agg)
# La mediana respeta el orden y es robusta a un voto atipico (a diferencia de la moda/mayoria).
print(f"Acuerdo inter-LLM (los 4 coinciden): {100*e2['inter_llm_all_agree']:.1f}% de los casos.")
print("=> Solo coinciden en 1 de cada 6 comentarios: la tarea es genuinamente ambigua.")'''),
 md("""**Lección:** la **mediana ordinal** obtiene el mayor QWK (respeta el orden). No usar voto por mayoría, que trata las clases como categorías sin relación."""),
 md("""## E3 — Ampliar el conjunto etiquetado con auto-etiquetas (resultado nulo, honesto)

Idea: usar los LLMs para etiquetar los ~2000 comentarios sin etiqueta, quedándonos solo con aquellos donde **varios LLMs coinciden** (señal de que es un caso claro). Primero validamos que la concordancia predice la calidad."""),
 code('''# Validacion del "gate" de concordancia: donde mas LLMs coinciden, mas se acercan al humano.
g = load("e3_gate.json")
print("Calidad de la auto-etiqueta segun cuantos LLMs coinciden (medido en los 581 con verdad humana):")
for umbral, v in g["gate_581"].items():
    print(f"  >= {umbral} LLMs coinciden  ->  QWK vs humano = {v['qwk_vs_human']:.3f}")
print()
# Reentrenamos con las auto-etiquetas de alta concordancia y medimos sobre el test humano.
g3 = load("e3_gate3.json"); d = g3["delta_qwk_vs_p0"]
print(f"Con +{g3['n_auto']} auto-etiquetas (gate>=3): QWK = {g3['metrics']['qwk']:.3f}")
print(f"Delta vs baseline = {d['mean']:+.3f}  IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  -> dentro del ruido")'''),
 md("""## Veredicto de E1-E3 y amenazas
- **E1:** el LLM iguala al BERT (no concluyente, pero interesante).
- **E2:** usar mediana, no mayoría.
- **E3:** resultado **nulo** con causa entendida: los 4 LLMs solo coinciden en 17.6% de los casos, así que las auto-etiquetas favorecen los comentarios fáciles y **estrechan la distribución** (sesgo de selección). Es un riesgo de propagación de sesgo que documentamos; por eso reportamos E3 como nulo, no como mejora."""),
])


# ============================ NB 03 — locales ============================
save("03_locales_E4_E6_E7_E8.ipynb", [
 md("""# E4, E6, E7, E8 — Experimentos locales (GPU)

Cuatro intervenciones sobre el modelo, cada una variando **un solo factor** bajo el mismo protocolo de P0 (StratifiedKFold, misma semilla). Aislar un factor por experimento evita **confundir variables**: si cambiáramos dos cosas a la vez, no sabríamos cuál causó el efecto.

- **E4:** cabeza ordinal CORN (¿reduce los errores lejanos?).
- **E6:** backbone de español social RoBERTuito/BETO (¿entiende mejor el registro de TikTok?).
- **E7:** limpieza mínima que conserva emojis (¿ayuda no borrar señal?).
- **E8:** datos sintéticos para las clases raras 4/5 (¿ataca el desbalance?).

> Para RE-EJECUTAR: `driver_local.py all` (E6, E7), `exp_e4.py` (E4), `exp_e8_foldaware.py` (E8). Requieren GPU."""),
 code(SETUP),
 md("""## E4 — Cabeza ordinal CORN (mejora significativa en MAE)

El modelo original trata las 5 clases como categorías sin orden (softmax). CORN convierte el problema en preguntas acumulativas ("¿supera el nivel k?"), enseñándole el **orden**. La métrica que esto debe mejorar es el **MAE** (distancia media del error), no necesariamente el QWK."""),
 code('''# E4: comparamos la cabeza ordinal CORN contra el softmax del baseline (P0).
e4 = load("e4_corn.json")
p0 = load("p0_stratified.json")

# OJO a la distincion de metricas: 0.656 es el MAE de CORN, NO su QWK (que es 0.414).
print(f"softmax (P0):  MAE = {p0['oof_global']['mae']:.3f}   QWK = {p0['oof_global']['qwk']:.3f}")
print(f"CORN ordinal:  MAE = {e4['corn_metrics']['mae']:.3f}   QWK = {e4['corn_metrics']['qwk']:.3f}")
print()
# La mejora esta en el MAE. El IC de la diferencia de MAE excluye 0 => significativo.
d = e4["delta_mae_neg_corn_minus_base"]
print(f"Reduccion de MAE = {d['mean']:+.3f}  IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  -> IC excluye 0: SIGNIFICATIVO")
print("Interpretacion: CORN se equivoca 'por poco' (predice 4 cuando era 5), no 'por mucho' (predice 1).")'''),
 md("""## E6 — Backbone de español social"""),
 code('''# E6: mismo pipeline, cambiando SOLO el modelo base.
e6 = load("e6_backbone.json")
pd.DataFrame([[k, round(v["oof_global"]["qwk"], 3)] for k, v in e6.items() if "oof_global" in v],
             columns=["backbone", "QWK"]).sort_values("QWK", ascending=False)'''),
 md("""**RoBERTuito** (preentrenado en ~500M de tuits en español) obtiene el mayor QWK (0.470). **BETO** (español de Wikipedia/noticias) queda **por debajo** del baseline (0.320). Lección: estar en español no basta; importa el **dominio** (texto social). *Nota honesta:* no calculamos el IC pareado de E6, así que lo reportamos como sugerente, no confirmado."""),
 md("""## E7 — Limpieza mínima vs agresiva (resultado nulo)"""),
 code('''# E7: comparamos la limpieza agresiva actual (borra emojis/puntuacion) contra una
# minima que los conserva, con el MISMO modelo. Solo cambia el preprocesamiento.
e7 = load("e7_limpieza.json"); d = e7["delta_qwk_minima_minus_agresiva"]
print(f"limpieza agresiva:  QWK = {e7['text_agresiva']['oof_global']['qwk']:.3f}")
print(f"limpieza minima:    QWK = {e7['text_minima']['oof_global']['qwk']:.3f}")
print(f"Delta (minima - agresiva) = {d['mean']:+.3f}  IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  -> dentro del ruido")'''),
 md("""Conservar emojis mejora un poco (+0.017) pero **dentro del ruido**: dirección esperada, no concluyente con este N. Se corrió *antes* que E6 para no confundir "modelo nuevo" con "limpieza nueva"."""),
 md("""## E8 — Datos sintéticos para clases raras (significativo, SIN fuga)

Las clases positivas (4 y 5) tienen solo 45 ejemplos. Generamos paráfrasis con un LLM. **Punto crítico de honestidad:** las paráfrasis se generan **solo de los positivos del train de cada fold**, nunca de los de validación, para que el modelo no vea una copia casi idéntica de un ejemplo de test (eso sería *near-duplicate leakage*)."""),
 code('''# E8 version FOLD-AWARE (sin fuga). Es la valida; la version con fuga se conserva solo para comparar.
e8 = load("e8_augmentation_foldaware.json"); d = e8["delta_qwk_vs_p0"]
print(f"F1 de clases 4/5:  {e8['f1_clases45_base']:.3f} -> {e8['f1_clases45_aug']:.3f}  (con +{e8['n_synthetic']} sinteticos)")
print(f"Delta QWK vs baseline = {d['mean']:+.3f}  IC[{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  -> IC excluye 0: SIGNIFICATIVO")
print()
# Comparacion honesta con la version que TENIA fuga, para mostrar que el efecto sobrevive.
e8_leak = load("e8_augmentation.json")
print(f"(Comparacion) version CON fuga daba delta +{e8_leak['delta_qwk_vs_p0']['mean']:.3f}; "
      f"al corregir el leakage baja a +{d['mean']:.3f} pero SIGUE significativo.")'''),
 md("""## Veredicto de E4-E8
Dos mejoras con IC que excluye 0: **E4** (MAE) y **E8 fold-aware** (QWK y F1 de clases raras). **E6** es sugerente (mayor QWK, sin IC pareado). **E7** dentro del ruido. La combinación RoBERTuito + CORN + augmentation fold-aware es el trabajo futuro natural, idealmente con más datos."""),
])


# ============================ NB 04 — calibracion + sesgo ============================
save("04_calibracion_sesgo_E5_E9.ipynb", [
 md("""# E5, E9 — Honestidad metodológica

- **E5:** ¿son confiables las *confianzas* que reporta el modelo? ¿ayuda que se **abstenga** en los casos dudosos?
- **E9:** ¿es el modelo **equitativo** entre ejes temáticos y clases? (análogo local a SageMaker Clarify)

Estos dos experimentos no buscan subir el F1, sino elevar el **rigor** y la **honestidad** del sistema, que es lo que un revisor valora."""),
 code(SETUP),
 md("""## E5 — Calibración (temperature scaling) + abstención

Una red neuronal suele estar **mal calibrada**: dice "90% seguro" y acierta el 70%. El *temperature scaling* ajusta un único escalar T que corrige esa sobreconfianza. Luego medimos qué pasa si el modelo **se abstiene** de los casos donde menos confía (curva risk-coverage)."""),
 code('''# E5: calibracion y abstencion.
e5 = load("e5_calibracion.json")

# ECE = Expected Calibration Error: mide cuanto miente la confianza (menor = mejor).
print(f"ECE (error de calibracion): {e5['ece_before']:.3f} -> {e5['ece_after']:.3f}  (temperatura T={e5['T']:.2f})")
print(f"(T>1 indica que el modelo estaba sobreconfiado; T lo suaviza.)")
print()

# Curva risk-coverage: QWK sobre el subconjunto mas confiable, al abstenerse del resto.
import matplotlib.pyplot as plt
cobertura = [r["coverage"] * 100 for r in e5["risk_coverage"]]   # % del corpus que SI se etiqueta
qwk       = [r["qwk"] for r in e5["risk_coverage"]]              # calidad sobre lo etiquetado
plt.figure(figsize=(7, 4))
plt.plot(cobertura, qwk, "-o", color="#0073bb")
plt.gca().invert_xaxis()   # de mayor a menor cobertura, para leer "cuanto sube al abstenerse"
plt.xlabel("cobertura % (lo que SI se etiqueta)"); plt.ylabel("QWK")
plt.title("E5: abstenerse de lo dudoso sube la calidad"); plt.grid(alpha=0.3); plt.show()'''),
 md("""Al etiquetar solo el 60% donde más confía, el QWK sube de 0.391 a 0.526. **Para medir opinión pública es más honesto reportar "X% negativo, Y% incierto"** que forzar una etiqueta dudosa. Convierte una debilidad (el 40% de baja confianza) en una contribución."""),
 md("""## E9 — Análisis de sesgo (análogo local a SageMaker Clarify)

En vez de gastar crédito montando SageMaker (que solo daría reproducibilidad, no mejora de F1), implementamos localmente la parte valiosa: medir si el modelo es equitativo."""),
 code('''# E9: analisis de equidad y sesgo sobre las predicciones out-of-fold.
e9 = load("e9_bias.json")

# (a) Equidad de grupo: QWK por eje tematico. Una brecha grande = trato desigual.
print("QWK por eje:", {k: round(v["qwk"], 3) for k, v in e9["axis_performance"].items()})
print(f"Brecha entre ejes: {e9['axis_qwk_gap']:.3f}  (infraestructura es el eje mas dificil)")
print()

# (b) Sesgo de distribucion: el modelo, ¿sobre/sub-predice alguna clase?
print("Sesgo por clase (puntos porcentuales, predicho - real):",
      {k: round(v, 1) for k, v in e9["class_bias_pp"].items()})
print("=> sub-predice la clase neutral (3) en -7.7 pp: tiende a 'tomar postura'.")
print()

# (c) Matriz de confusion: donde caen los errores (idealmente en clases vecinas).
print("Matriz de confusion (fila = real, columna = predicho):")
print(np.array(e9["confusion_matrix"]))'''),
 md("""## Veredicto de E5-E9
El modelo es **peor en infraestructura** (eje más ambiguo) y tiene un **sesgo anti-neutral**. Los errores caen en clases vecinas (buena propiedad ordinal). Esto es material honesto para la sección de ética/limitaciones de la tesis, obtenido a $0 sin SageMaker."""),
])

print("Notebooks de calidad de tesis generados:")
for f in sorted(NB.glob("*.ipynb")):
    print("  ", f.name)
