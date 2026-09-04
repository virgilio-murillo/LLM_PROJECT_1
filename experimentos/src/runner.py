"""
Runner reutilizable de CV estratificada con LoRA. Permite variar UN factor:
- model_name (E6: backbone)
- textos (E7: limpieza)  -> se pasan texts/labels directamente
- head: 'softmax' (default) | 'ordinal' (E4, CORN) 
Mantiene todo lo demas fijo para no confundir variables.
Devuelve predicciones OOF + metricas por fold + resumen. Corre en MPS.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.unique(labels); w = compute_class_weight("balanced", classes=cls, y=labels)
    full = np.ones(5, dtype=np.float32)
    for c, wt in zip(cls, w): full[c] = wt
    return torch.tensor(full, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_w=None, **kw):
        super().__init__(**kw); self.class_w = class_w
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.get("labels")
        out = model(**inputs); logits = out.get("logits")
        lf = torch.nn.CrossEntropyLoss(weight=self.class_w.to(labels.device) if self.class_w is not None else None)
        loss = lf(logits.view(-1, 5), labels.view(-1))
        return (loss, out) if return_outputs else loss


def _lora_targets(model_name):
    # BERT-like: key/query/value/dense ; RoBERTa-like: query/key/value + output/intermediate dense
    return ["query", "key", "value", "dense"]


def cv_train(texts, y, model_name="nlptown/bert-base-multilingual-uncased-sentiment",
             k=10, epochs=7, maxlen=128, lr=1e-4, r=32, alpha=64, dropout=0.2, tag="run"):
    C.set_all_seeds()
    y = np.asarray(y)
    tok = AutoTokenizer.from_pretrained(model_name)
    def tok_fn(b): return tok(b["text"], truncation=True, max_length=maxlen)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=C.SEED)
    oof = np.full(len(y), -1); fold_m = []; t0 = time.time()
    for i, (tr, va) in enumerate(skf.split(texts, y)):
        ds_tr = Dataset.from_dict({"text": [texts[j] for j in tr], "labels": y[tr].tolist()}).map(tok_fn, batched=True)
        ds_va = Dataset.from_dict({"text": [texts[j] for j in va], "labels": y[va].tolist()}).map(tok_fn, batched=True)
        base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout,
                         bias="lora_only", target_modules=_lora_targets(model_name),
                         modules_to_save=["classifier", "score"])
        model = get_peft_model(base, cfg)
        args = TrainingArguments(output_dir=f"/tmp/{tag}_{i}", num_train_epochs=epochs,
                                 per_device_train_batch_size=16, per_device_eval_batch_size=32,
                                 learning_rate=lr, logging_steps=200, report_to=[], save_strategy="no",
                                 eval_strategy="no", seed=C.SEED, disable_tqdm=True)
        tr_obj = WeightedTrainer(class_w=_class_weights(y[tr]), model=model, args=args,
                                 train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
        tr_obj.train()
        pred = np.argmax(tr_obj.predict(ds_va).predictions, axis=-1)
        oof[va] = pred
        m = C.ordinal_metrics(y[va], pred); m["fold"] = i + 1; fold_m.append(m)
        print(f"  [{tag}] fold {i+1}/{k} f1={m['f1_macro']:.3f} qwk={m['qwk']:.3f} mae={m['mae']:.3f}", flush=True)
        del model, base, tr_obj
        if DEVICE == "mps": torch.mps.empty_cache()
    glob = C.ordinal_metrics(y, oof)
    fm = np.array([[m["f1_macro"], m["qwk"], m["mae"], m["balanced_accuracy"]] for m in fold_m])
    return {
        "tag": tag, "model": model_name, "n": len(y), "seconds": time.time() - t0,
        "oof_global": glob,
        "fold_mean": {"f1_macro": float(fm[:,0].mean()), "qwk": float(fm[:,1].mean()),
                      "mae": float(fm[:,2].mean()), "balanced_accuracy": float(fm[:,3].mean())},
        "fold_sd": {"f1_macro": float(fm[:,0].std(ddof=1)), "qwk": float(fm[:,1].std(ddof=1)),
                    "mae": float(fm[:,2].std(ddof=1))},
        "folds": fold_m, "oof_pred": oof.tolist(),
    }
