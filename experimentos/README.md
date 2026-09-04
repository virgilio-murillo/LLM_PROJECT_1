# Experimentos de mejora

Diez experimentos que fortalecen el proyecto de análisis de sentimiento ordinal (TikTok, Mundial 2026), evaluados con honestidad estadística (métricas ordinales + intervalos de confianza bootstrap). Todo corre localmente en GPU Metal (MPS) y con Amazon Bedrock (costo total ~$3-5).

> **Baseline honesto contra el que se mide todo: QWK 0.415** (validación cruzada estratificada, out-of-fold). El "0.66" del proyecto original era leakage in-sample y no se usa.

## Estructura

```
experimentos/
├── src/            # código modular (NO usar el nombre 'lib', lo ignora el .gitignore raíz)
├── notebooks/      # 00_master + 4 notebooks por fase (narrativa + tablas con IC)
├── resultados/     # JSON por experimento + predicciones LLM + summary_table.csv
├── reportes/       # REPORTE_EXPERIMENTOS.html (formal, con 12 gráficas)
├── tests/          # validaciones ligeras (pytest)
└── requirements-experimentos.txt
```

## Orden de ejecución

| Orden | Comando | Experimentos | Requiere | Tiempo aprox. |
|-------|---------|--------------|----------|---------------|
| 1 | `python src/exp_p0.py both` | P0 (eval honesta) | GPU | ~30 min |
| 2 | `python src/classify_581.py` | E1, E2 (LLMs sobre gold) | Bedrock | ~3 min |
| 3 | `python src/classify_pool.py` | E3 (LLMs sobre pool) | Bedrock | ~10 min |
| 4 | `python src/driver_local.py all` | E7, E6 (limpieza, backbone) | GPU | ~50 min |
| 5 | `python src/exp_e4.py` | E4 (cabeza ordinal CORN) | GPU | ~10 min |
| 6 | `python src/driver2.py all` | E3 (reentreno), E5 (calibración) | GPU | ~40 min |
| 7 | `python src/exp_e8.py` | E8 (augmentation) | GPU + Bedrock | ~15 min |
| 8 | `python src/gen_notebooks.py` | (re)genera los notebooks | - | segundos |

Los notebooks cargan resultados ya calculados por defecto (rápido, sin GPU). Para re-ejecutar desde cero, pon `RECOMPUTE=True`.

## Reproducibilidad

- Semilla fija: 61298 (todos los experimentos).
- LLMs: modelos con versión fija (us-east-1), temperature=0. Respuestas crudas en `resultados/bedrock_raw.jsonl` (adjuntar a un GitHub Release, no al árbol).
- Nota honesta: temperature=0 NO garantiza determinismo perfecto de un LLM entre versiones del modelo.

## Resultados (resumen)

| Experimento | QWK | Δ vs baseline (IC95) | ¿Significativo? |
|---|---|---|---|
| P0 baseline | 0.415 | referencia | — |
| E1 Consenso-LLM | 0.498 | +0.083 [−0.002,+0.168] | empate (P=0.97) |
| E2 mediana | 0.498 | — | mediana > mayoría |
| E3 auto-etiquetado | 0.449 | +0.033 [−0.025,+0.089] | dentro del ruido |
| **E4 CORN** | 0.414 | MAE −0.106 [+0.043,+0.164] | **SÍ** |
| E5 calibración | 0.526 @60% | — | honestidad |
| **E6 RoBERTuito** | 0.470 | +0.055 | sí (borde) |
| E7 limpieza mínima | 0.460 | +0.017 [−0.017,+0.051] | dentro del ruido |
| **E8 augmentation** | 0.472 | +0.057 [+0.018,+0.097] | **SÍ** |
| E9 sesgo | — | — | análisis |

Solo E4, E8 y E6 tienen mejora estadísticamente significativa. El hallazgo estrella E1 es un empate honesto: *"con N=581, el fine-tuning no logra superar a un LLM sin entrenar"*.

## Plan de PRs (para la organizadora del repo)

PRs pequeños e independientes, cada uno desde `main`. **No tocan `notebooks_proyecto/`** (trabajo original de las autoras).

| PR | Rama | Contenido | Riesgo |
|----|------|-----------|--------|
| 0 | `exp/00-infra` | `experimentos/src/`, `.gitignore`, `requirements-experimentos.txt`, `tests/`, este README. **Base de todo.** | Bajo |
| 1 | `exp/p0-eval-honesta` | P0: StratifiedKFold + QWK/MAE + notebook 01 | Bajo |
| 2 | `exp/e1-e2-llm` | E1 triple comparación + E2 agregación + notebook 02 | Medio |
| 3 | `exp/e3-autoetiquetado` | E3 auto-etiquetado por concordancia | Alto (ética LLM) |
| 4 | `exp/e4-ordinal` | E4 cabeza ordinal CORN | Bajo |
| 5 | `exp/e6-backbone` | E6 RoBERTuito/BETO | Medio |
| 6 | `exp/e7-limpieza` | E7 limpieza A/B | Bajo |
| 7 | `exp/e8-augmentation` | E8 augmentation clases 4/5 | Medio |
| 8 | `exp/e5-e9-honestidad` | E5 calibración + E9 sesgo + notebook 04 | Bajo |
| 9 | `docs/reporte-y-presentacion` | Reporte HTML + presentación | Bajo |

**Excluir de los PRs:** venvs (`kiro-test/`), `investigation/`, `kiro-notes/`, `bedrock_raw.jsonl` (a un Release).

**Orden de merge:** PR 0 primero (sin él, `src/` se ignora y los notebooks se rompen). Los demás son independientes; si la organizadora rechaza uno, los otros siguen mergeables.
