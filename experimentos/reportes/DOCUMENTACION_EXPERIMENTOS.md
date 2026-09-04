# Documentación de los experimentos (explicación clara)

Cada experimento en lenguaje sencillo: qué pregunta responde, cómo se hizo, qué salió, y qué significa. Pensado para que cualquier lector del proyecto (incluido el jurado) lo entienda sin ser experto.

---

## P0 — Arreglar la forma de medir

**La pregunta.** ¿El número que reportaba el proyecto original (0.66) era confiable?

**El problema.** No. Dos fallas: (1) partía los datos al azar (`KFold`), y como la clase "positiva" tiene solo 15 ejemplos, algunos pedazos quedaban casi sin positivos, haciendo que la medición saltara mucho de un pedazo a otro. (2) Reportaba el mejor pedazo evaluándolo sobre datos que el modelo ya había estudiado (hacer trampa en el examen).

**La solución.** Partir los datos manteniendo la proporción de clases (`StratifiedKFold`) y evaluar siempre sobre datos no vistos. Además, añadir métricas correctas para una escala con orden (QWK, MAE).

**El resultado.** La medición se volvió estable (la variación entre pedazos bajó de 0.090 a 0.072). El número honesto es **QWK 0.415** (no 0.66). Todo lo demás se compara contra este.

---

## E1 — ¿Un LLM sin entrenar le gana a nuestro modelo entrenado?

**La pregunta.** Tenemos un BERT afinado con 581 ejemplos. ¿Un modelo de lenguaje grande (LLM) de Amazon Bedrock, sin ningún entrenamiento, clasifica igual de bien?

**Cómo.** Le pedimos a 4 LLMs (Nova Micro/Lite/Pro y Llama) que clasifiquen los mismos 581 comentarios, y comparamos su consenso contra el humano, igual que al BERT.

**El resultado.** El LLM (QWK **0.498**) obtiene más que el BERT afinado (0.415), pero la diferencia **no es concluyente**: el IC de la diferencia [-0.002,+0.168] incluye 0 (evidencia débilmente favorable al LLM, P≈0.97).

**Qué significa.** El marco honesto: *con solo 581 ejemplos, entrenar un modelo no logra superar a un LLM bien preguntado*. Es un hallazgo interesante y defendible para la tesis.

---

## E2 — ¿Cómo combinar la opinión de varios LLMs?

**La pregunta.** Si 4 LLMs dan notas distintas (ej. 1, 3, 3, 5), ¿cuál es "la" nota del equipo?

**Cómo.** Comparamos tres formas de combinar: voto por mayoría, promedio, y mediana (el valor de en medio).

**El resultado.** La **mediana obtiene el mayor QWK** (0.498 vs 0.488 de mayoría vs 0.452 de promedio).

**Qué significa.** Como la escala tiene orden, la mediana lo respeta y no se deja arrastrar por un LLM atípico. Lección práctica: usar mediana, no voto por mayoría.

---

## E3 — Ampliar los datos con etiquetas automáticas

**La pregunta.** Tenemos solo 581 comentarios etiquetados a mano. ¿Podemos usar los LLMs para etiquetar los otros ~2000 sin meter basura?

**Cómo.** Solo aceptamos las etiquetas de los comentarios donde varios LLMs coinciden (señal de que es un caso claro). Verificamos primero: donde los 4 LLMs coinciden, aciertan al humano con QWK 0.79 (casi calidad humana).

**El resultado.** Añadiendo 1210 auto-etiquetas, el modelo sube a QWK 0.449 (+0.033). Mejora, pero aún dentro del margen de ruido.

**Qué significa.** Es la vía más prometedora para romper el techo de datos, aunque hay que cuidar el sesgo: las auto-etiquetas favorecen los casos fáciles.

---

## E4 — Enseñarle al modelo que las notas tienen orden

**La pregunta.** El modelo trata las 5 clases como categorías sueltas. ¿Mejora si le enseñamos que 4 está cerca de 5 y lejos de 1?

**Cómo.** Cambiamos la "cabeza" del modelo por una ordinal (CORN), que predice la nota respetando el orden.

**El resultado.** El error promedio (MAE) bajó de 0.761 a **0.656**, y esta mejora **sí es estadísticamente significativa** (el margen no cruza el cero).

**Qué significa.** El modelo ahora se equivoca "por poco" (predice 4 cuando era 5) en vez de "por mucho" (predice 1). Para una escala de aprobación, eso es exactamente lo deseable. Es de los tres resultados sólidos.

---

## E5 — Que el modelo diga "no estoy seguro"

**La pregunta.** El 40% de las clasificaciones del modelo tienen baja confianza. ¿Es más honesto dejar que se abstenga en esos casos?

**Cómo.** Calibramos la confianza (temperature scaling) y medimos qué pasa si solo etiquetamos los comentarios más confiables.

**El resultado.** Si el modelo etiqueta solo el 60% donde más confía, su calidad sube de QWK 0.391 a **0.526**.

**Qué significa.** Para medir opinión pública es más honesto reportar "35% negativo, 40% incierto" que forzar una etiqueta dudosa. Convierte una debilidad en una contribución de rigor.

---

## E6 — Usar un modelo que ya sabe español de redes

**La pregunta.** Nuestro modelo base fue entrenado para reseñas de productos. ¿Uno entrenado en tuits en español entiende mejor a TikTok?

**Cómo.** Comparamos tres modelos base: el actual, BETO (español), y RoBERTuito (500M de tuits en español).

**El resultado.** **RoBERTuito gana** (QWK 0.470, +0.055). BETO, curiosamente, queda por debajo del baseline.

**Qué significa.** Estar en español no basta; importa el dominio (texto social). RoBERTuito, entrenado en tuits, entiende emojis, slang y el registro coloquial. Es de los tres resultados sólidos.

---

## E7 — No borrar los emojis

**La pregunta.** La limpieza actual borra emojis, signos y mayúsculas. Pero un 😡 o un "!!!" dicen mucho del sentimiento. ¿Conservarlos ayuda?

**Cómo.** Comparamos la limpieza agresiva actual contra una mínima que conserva emojis y puntuación, con el mismo modelo.

**El resultado.** La mínima mejora un poco (+0.017 QWK) pero está dentro del ruido: no es concluyente.

**Qué significa.** Un resultado honesto: la dirección es la esperada, pero con este tamaño de datos no se puede afirmar. Con RoBERTuito (que sí usa emojis) el efecto podría ser mayor (trabajo futuro).

---

## E8 — Inventar ejemplos de las clases raras

**La pregunta.** Las clases positivas (4 y 5) tienen solo 45 ejemplos entre las dos. ¿Ayuda generar más con un LLM?

**Cómo.** Le pedimos a un LLM que reescribiera (parafraseara) los 45 positivos, generando 135 ejemplos sintéticos (paráfrasis solo del train de cada fold, sin fuga). Los añadimos solo al entrenamiento (nunca al examen).

**El resultado.** El F1 de las clases 4/5 subió de 0.22 a **0.26**, y el QWK global +0.045 (IC[+0.005,+0.087]). **Estadísticamente significativo, versión fold-aware sin fuga de datos.**

**Qué significa.** Atacar directamente el desbalance funciona. Es de los tres resultados sólidos.

---

## E9 — ¿Es justo el modelo?

**La pregunta.** ¿El modelo funciona igual de bien en los tres ejes (infraestructura, seguridad, turismo) y en todas las clases?

**Cómo.** Análisis de sesgo (análogo local a SageMaker Clarify) sobre las predicciones.

**El resultado.** No: es peor en **infraestructura** (el eje más ambiguo) y tiende a **evitar la clase neutral** (sub-predice "neutral" en 7.7 puntos).

**Qué significa.** El modelo tiene sesgos que hay que declarar. Es una sección de ética/limitaciones honesta para la tesis, obtenida sin costo.

---

## En una frase

De los 10 experimentos, **tres dan mejoras significativas** (E4 en MAE, E8 fold-aware en F1 de clases raras, E6 en QWK al borde). Con corrección por comparaciones múltiples (Holm-Bonferroni) ningún delta de QWK sobrevive; los hallazgos sólidos son en MAE (E4) y F1 (E8); el hallazgo más llamativo (E1: el LLM iguala al modelo entrenado) es un empate honesto; y el resto aporta rigor y honestidad metodológica. El techo lo pone el tamaño del gold set (581 ejemplos): la mejora de fondo será ampliarlo.
