"""
E8 (opcional): data augmentation para las clases escasas 4 y 5.
Genera parafrasis con un LLM de Bedrock de los positivos del gold, las anade SOLO al train,
y mide OOF sobre el test humano. Salvaguarda: el held-out nunca recibe sinteticos.
"""
import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import boto3
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import runner as R

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_brt = boto3.client("bedrock-runtime", region_name="us-east-1")


def paraphrase(text, n=3):
    """Genera n parafrasis en espanol mexicano conservando el sentimiento."""
    prompt = (f'Reescribe el siguiente comentario de TikTok en espanol mexicano de {n} formas distintas, '
              f'conservando EXACTAMENTE el mismo sentimiento y tono. Una por linea, sin numerar.\n'
              f'Comentario: "{text}"')
    try:
        r = _brt.converse(modelId="amazon.nova-lite-v1:0",
                          messages=[{"role": "user", "content": [{"text": prompt}]}],
                          inferenceConfig={"temperature": 0.7, "maxTokens": 300})
        out = r["output"]["message"]["content"][0]["text"]
        lines = [l.strip(" -*0123456789.") for l in out.split("\n") if len(l.strip()) > 8]
        return lines[:n]
    except Exception:
        return []


def run():
    C.set_all_seeds()
    gold = C.load_gold(); y = gold["label"].to_numpy(); texts = gold["text"].tolist()
    # generar sinteticos para clases 4 y 5 (labels 3 y 4)
    pos_idx = [i for i in range(len(y)) if y[i] in (3, 4)]
    print(f"generando parafrasis para {len(pos_idx)} positivos (clases 4/5)...", flush=True)
    syn_text, syn_lab = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(paraphrase, texts[i], 3): i for i in pos_idx}
        for fut in as_completed(futs):
            i = futs[fut]
            for p in fut.result():
                syn_text.append(p); syn_lab.append(int(y[i]))
    print(f"generados {len(syn_text)} ejemplos sinteticos", flush=True)
    # guardar sinteticos para reproducibilidad
    import pandas as pd
    pd.DataFrame({"text": syn_text, "label": syn_lab}).to_csv(C.RESULTS / "e8_synthetic.csv", index=False)

    tok = AutoTokenizer.from_pretrained(MODEL)
    def tok_fn(b): return tok(b["text"], truncation=True, max_length=128)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=C.SEED)
    oof = np.full(len(y), -1)
    for i, (tr, va) in enumerate(skf.split(texts, y)):
        tr_text = [texts[j] for j in tr] + syn_text
        tr_lab = y[tr].tolist() + syn_lab
        ds_tr = Dataset.from_dict({"text": tr_text, "labels": tr_lab}).map(tok_fn, batched=True)
        ds_va = Dataset.from_dict({"text": [texts[j] for j in va], "labels": y[va].tolist()}).map(tok_fn, batched=True)
        base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.2, bias="lora_only",
                         target_modules=["key","query","value","dense"], modules_to_save=["classifier","score"])
        model = get_peft_model(base, cfg)
        args = TrainingArguments(output_dir=f"/tmp/e8_{i}", num_train_epochs=7, per_device_train_batch_size=16,
                                 learning_rate=1e-4, report_to=[], save_strategy="no", seed=C.SEED, disable_tqdm=True)
        tr_obj = R.WeightedTrainer(class_w=R._class_weights(np.array(tr_lab)), model=model, args=args,
                                   train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
        tr_obj.train()
        oof[va] = np.argmax(tr_obj.predict(ds_va).predictions, axis=-1)
        del model, base, tr_obj
        if DEVICE == "mps": torch.mps.empty_cache()
    m = C.ordinal_metrics(y, oof)
    base_oof = np.load(C.RESULTS / "p0_oof_pred_stratified.npy")
    from sklearn.metrics import f1_score
    f1_45_base = f1_score(y, base_oof, labels=[3,4], average="macro", zero_division=0)
    f1_45_aug = f1_score(y, oof, labels=[3,4], average="macro", zero_division=0)
    dq = C.paired_bootstrap_diff(y, oof, base_oof, C.qwk_fn, n=2000)
    res = {"metrics": m, "n_synthetic": len(syn_text),
           "f1_clases45_base": float(f1_45_base), "f1_clases45_aug": float(f1_45_aug),
           "delta_qwk_vs_p0": {"mean": dq[0], "ci_low": dq[1], "ci_high": dq[2], "p_gt": dq[3]}}
    C.save_result("e8_augmentation", res)
    print(f"\n[E8] QWK={m['qwk']:.3f} F1={m['f1_macro']:.3f}  F1(4/5) {f1_45_base:.3f}->{f1_45_aug:.3f}", flush=True)
    print(f"  Delta QWK vs P0 = {dq[0]:+.3f} IC[{dq[1]:+.3f},{dq[2]:+.3f}]", flush=True)


if __name__ == "__main__":
    run(); print("DONE E8", flush=True)
