# Guión de defensa — Preguntas del jurado y respuestas

Las 16 preguntas más duras y probables de un jurado de tesis en NLP/ML, con la respuesta basada en la evidencia REAL de los experimentos. Objetivo: que no te sorprendan. Todas las respuestas se apoyan en resultados que están en `experimentos/resultados/`.

> **Encuadre general que debes repetir:** este es un **estudio piloto metodológico** con N=581, no una medición electoral definitiva. La contribución es un pipeline honesto de evaluación ordinal, no un número de aprobación cerrado.

---

**Q1. "Tu hallazgo estrella (E1) es un empate, no una victoria. ¿Por qué lo presentas?"**
Es precisamente el hallazgo. Con N=581, un consenso de 4 LLMs SIN entrenamiento iguala a un BERT afinado (QWK 0.498 vs 0.415, IC de la diferencia [−0.002, +0.168] que incluye 0). El enunciado honesto no es "el LLM gana", sino *"el fine-tuning no logra superar a un LLM few-shot con esta cantidad de datos"*. Es útil: con datos escasos, invertir en prompting puede rendir tanto como entrenar.

**Q2. "P=0.97 suena a que el LLM sí gana. ¿No te contradices?"**
No. P(diff>0)=0.97 es la fracción de remuestreos donde el LLM quedó arriba; pero el IC 95% de la diferencia incluye el 0, así que no rechazo la igualdad al 5%. El enunciado preciso: *"la evidencia es débilmente favorable al LLM pero no concluyente"*. La comparación de puntos (0.415 vs 0.498) NO es la evidencia; la evidencia es el IC de la diferencia pareada.

**Q3. "Usaste una sola partición de CV. ¿Cómo sé que no es suerte?"**
Corrí el baseline con validación cruzada en **3 semillas distintas**: QWK 0.396 ± 0.020 (0.379, 0.392, 0.418). El baseline es **estable entre particiones** (desviación de solo 0.020). Los hallazgos no dependen de un sorteo afortunado.

**Q4. "¿Por qué usar LLMs si admites que no son reproducibles?"**
Dos razones. Son el baseline realista (cualquiera clasifica hoy con un LLM sin etiquetar). Y mitigué la irreproducibilidad: temperature=0, modelId con versión fija, y guardé TODAS las respuestas crudas en `bedrock_raw.jsonl`, así que E1/E2/E3 se recomputan desde ese cache sin volver a llamar a Bedrock. La irreproducibilidad residual (updates del endpoint) la documento como limitación.

**Q5. "La comparación LLM vs BERT es injusta: el LLM vio 6 ejemplos en el prompt."**
Correcto, y por eso NO la enmarco como "igualdad de condiciones" sino como *"few-shot sin entrenamiento vs fine-tuning supervisado"*, dos paradigmas distintos. Verifiqué que los 6 ejemplos few-shot son manuales y NO están en el gold set 581 (0 coincidencias), así que no hay fuga. El punto se fortalece: incluso dándole al LLM la ventaja del few-shot, solo empata.

**Q6. "En E8 generas datos sintéticos de tus positivos. ¿No se filtran al test?"**
Lo detecté y lo arreglé. Reejecuté E8 **fold-aware**: genero paráfrasis solo de los positivos del fold de entrenamiento, nunca del de validación (con `origen_idx` trazable). El efecto **sobrevive**: QWK +0.045 IC[+0.005, +0.087] (sigue excluyendo 0). La versión con el leakage daba +0.057; al corregirlo bajó a +0.045 pero sigue siendo significativo. Eso demuestra que el hallazgo es real, no un artefacto.

**Q7. "N=581 con 15 casos de clase 5. ¿Puedes concluir algo sobre aprobación positiva?"**
No de forma definitiva, y lo digo explícitamente. Las conclusiones sobre clases 4 y 5 son tentativas por diseño; sus IC son muy amplios. El valor es metodológico y como estudio piloto, no como medición electoral.

**Q8. "El auto-etiquetado (E3) suena a fuga circular."**
Por eso E3 es un resultado NEGATIVO que reporto con honestidad: +0.033 QWK con IC[−0.025, +0.089] que incluye 0. El consenso NO ayudó. Lo esperado: los 4 LLMs solo coinciden en 17.6% de los casos, así que las auto-etiquetas favorecen los fáciles y estrechan la distribución. Documentar que no funciona es parte del método.

**Q9. "¿Por qué BETO fue peor que un modelo multilingüe genérico?"**
BETO (español de España, Wikipedia/noticias) dio QWK 0.320, peor que nlptown (0.415) y RoBERTuito (0.470). RoBERTuito está preentrenado en tuits en español (jerga, emojis, informalidad), que casa con TikTok. Lección: el ajuste de **dominio** (redes sociales) importa más que el idioma nominal.

**Q10. "¿Qué es QWK y por qué en vez de accuracy?"**
Quadratic Weighted Kappa mide acuerdo con el humano corrigiendo por azar, y penaliza los errores en proporción al CUADRADO de la distancia ordinal. En una escala 1-5, confundir "muy negativo" con "neutral" (error de 2) debe pesar más que con "parcialmente negativo" (error de 1). Accuracy trata todos los errores igual e ignora el orden. Uso QWK y MAE como métricas ordinales primarias.

**Q11. "Corriste 9 experimentos. ¿No hay inflación de falsos positivos?"**
Apliqué **Holm-Bonferroni** con test de permutación. Con la corrección, ningún delta de **QWK** sobrevive (E1 p=0.061). Por eso soy precisa: mis hallazgos significativos son en OTRAS métricas: E4 en **MAE** (IC[+0.043, +0.164], lejos de 0) y E8 en **F1 de clases 4/5**. No sobrevendo mejoras de QWK que no resisten la corrección.

**Q12. "¿Por qué no combinaste todas las mejoras en un modelo final?"**
Decisión metodológica: cada experimento es un **análisis ablativo** (aísla el efecto de un factor). Combinarlas confundiría las contribuciones. Con N=581 y una sola muestra, un modelo combinado sobre-optimizado arriesga sobreajuste. La combinación (RoBERTuito+CORN+augmentation fold-aware) es el trabajo futuro natural, con más datos.

**Q13. "¿El costo de la nube (Bedrock) se justifica?"**
El costo total fue ~$3-5 USD (581×4 modelos + paráfrasis). El BERT/LoRA corre local en GPU Metal gratis. Bedrock se usó solo donde aportaba: comparar LLMs de frontera sin infraestructura y generar paráfrasis. El empate de E1 justifica el gasto: mostró que un LLM de pago sin entrenar iguala al modelo local entrenado, un dato de decisión por $5.

**Q14. "En E4 (CORN) la balanced accuracy cae aunque baja el MAE. ¿No es un retroceso?"**
Es el trade-off esperado de una cabeza ordinal. CORN optimiza la coherencia del orden (MAE de 0.761 a 0.656, IC[+0.043, +0.164] significativo) prediciendo de forma más central/conservadora, lo que reduce aciertos en clases raras. El hallazgo de E4 es específicamente sobre MAE, no sobre balanced accuracy; lo reporto así, sin sobrevender.

**Q15. "¿Cómo garantizas que el gold de 581 está bien etiquetado? ¿Acuerdo inter-anotador?"**
Punto débil real que reconozco: el gold se etiquetó manualmente (200 por eje). Si no hubo doble anotación con kappa inter-anotador, no conozco el techo humano de la tarea. Como referencia indirecta, donde 4 LLMs coinciden el QWK vs humano es 0.79, lo que sugiere ambigüedad intrínseca. Trabajo futuro: doble anotación de una submuestra para estimar el kappa humano-humano.

**Q16. "El sesgo anti-neutral (E9): ¿invalida tu medición?"**
No la invalida pero la matiza, y lo reporto. El modelo sub-predice la clase neutral en 7.7 pp, empujando neutrales hacia los extremos. Al interpretar la aprobación aplico esta salvedad. Es exactamente el tipo de análisis (análogo a SageMaker Clarify) que un sistema responsable incluye, por eso lo hice explícito.

---

## Las tres frases con las que abres y cierras

1. **Apertura:** "Este es un estudio piloto metodológico: cómo medir aprobación política en escala ordinal con honestidad estadística, no una encuesta electoral."
2. **Hallazgo central:** "Con solo 581 ejemplos, un LLM sin entrenar iguala a nuestro modelo afinado; y las tres mejoras que sí resisten el escrutinio son la cabeza ordinal, el aumento de datos y el modelo en español social."
3. **Cierre:** "Reportamos con intervalos de confianza y corrección por comparaciones múltiples; lo que no supera el ruido lo declaramos nulo. Hicimos ciencia honesta."
