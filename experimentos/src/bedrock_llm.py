"""
Cliente de Bedrock para clasificacion de sentimiento ordinal (1-5) con multiples LLMs.
Sigue las recomendaciones de la investigacion 11f7e23e:
- Converse API unificada (Nova + Llama), temperature=0, modelId con version (reproducible).
- Prompt few-shot balanceado, escala redactada como APROBACION POLITICA (no estrellas).
- Parsing robusto: 'RESPUESTA: N' -> ultimo digito 1-5 -> reintento -> abstencion (NaN, no default 3).
- Backoff exponencial en ThrottlingException.
- Guarda respuestas CRUDAS (json) para reproducibilidad de tesis.
"""
import re, time, json, threading
from pathlib import Path
import boto3
from botocore.config import Config

REGION = "us-east-1"
RAW_LOG = Path(__file__).resolve().parents[1] / "resultados" / "bedrock_raw.jsonl"
_lock = threading.Lock()

MODELS = {
    "nova-micro": "amazon.nova-micro-v1:0",
    "nova-lite":  "amazon.nova-lite-v1:0",
    "nova-pro":   "amazon.nova-pro-v1:0",
    "llama3-8b":  "meta.llama3-8b-instruct-v1:0",
    "llama31-8b": "meta.llama3-1-8b-instruct-v1:0",
    "llama33-70b":"meta.llama3-3-70b-instruct-v1:0",
}

_client = boto3.client("bedrock-runtime", region_name=REGION,
                       config=Config(retries={"max_attempts": 6, "mode": "adaptive"}))

# Prompt: escala como aprobacion politica, few-shot balanceado, sarcasmo explicito.
SYSTEM = (
    "Eres un analista de opinion publica. Clasificas comentarios de TikTok en espanol mexicano "
    "sobre la gestion del gobierno ante el Mundial 2026. Usa una escala ORDINAL de APROBACION del 1 al 5:\n"
    "1 = muy negativo / fuerte desaprobacion del gobierno\n"
    "2 = parcialmente negativo\n"
    "3 = neutral (pregunta, dato, o ambiguo)\n"
    "4 = parcialmente positivo\n"
    "5 = muy positivo / fuerte aprobacion\n"
    "Regla: el sarcasmo o la ironia que critica al gobierno es NEGATIVO (1 o 2), no neutral.\n"
    "Responde con un razonamiento de UNA linea y termina EXACTAMENTE con: RESPUESTA: N (N es 1-5)."
)
FEWSHOT = [
    ("que verguenza la inseguridad que hay en el pais", 1),
    ("todavia faltan muchas mejoras para que funcione bien", 2),
    ("cual sera la sede mas cercana?", 3),
    ("parece una buena oportunidad para atraer visitantes", 4),
    ("sera excelente para el turismo y la economia", 5),
    ("si claro, gastan millones en el mundial y no arreglan los baches", 1),
]

ENSEMBLE_KEYS = ["nova-micro","nova-lite","nova-pro","llama3-8b"]

_RESP = re.compile(r"RESPUESTA:\s*([1-5])")
_DIGIT = re.compile(r"[1-5]")


def _build_messages(text):
    msgs = []
    for t, lab in FEWSHOT:
        msgs.append({"role": "user", "content": [{"text": f'Comentario: "{t}"'}]})
        msgs.append({"role": "assistant", "content": [{"text": f"Ejemplo. RESPUESTA: {lab}"}]})
    msgs.append({"role": "user", "content": [{"text": f'Comentario: "{text}"'}]})
    return msgs


def _parse(txt):
    m = _RESP.search(txt)
    if m: return int(m.group(1))
    # fallback: ultimo digito 1-5 en el texto
    ds = _DIGIT.findall(txt)
    if ds: return int(ds[-1])
    return None


def classify(text, model_key, max_retries=3):
    model_id = MODELS[model_key]
    for attempt in range(max_retries):
        try:
            r = _client.converse(
                modelId=model_id,
                system=[{"text": SYSTEM}],
                messages=_build_messages(text),
                inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 64},
            )
            out = r["output"]["message"]["content"][0]["text"]
            label = _parse(out)
            with _lock:
                RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(RAW_LOG, "a") as f:
                    f.write(json.dumps({"model": model_id, "text": text, "raw": out,
                                        "parsed": label, "ts": time.time()}, ensure_ascii=False) + "\n")
            if label is not None:
                return label
        except Exception as e:
            if "Throttl" in str(e) or "TooManyRequests" in str(e):
                time.sleep(2 ** attempt)
                continue
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None  # abstencion (no default a 3)


if __name__ == "__main__":
    tests = ["que verguenza la corrupcion", "que padre que mejoren los estadios",
             "si claro, todo perfecto mientras la gente no tiene agua"]
    for t in tests:
        row = {k: classify(t, k) for k in ["nova-micro", "nova-lite", "llama31-8b"]}
        print(f"{t[:45]:45s} -> {row}")
