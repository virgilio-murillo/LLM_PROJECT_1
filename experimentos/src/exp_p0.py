"""
P0 - Evaluacion out-of-fold estratificada (referencia interna para comparar experimentos).
- StratifiedKFold (no KFold) sobre el gold set 581.
- LoRA con la config REAL del adapter (r=32, alpha=64, dropout=0.2, targets key/query/value/dense).
- Predicciones OUT-OF-FOLD (cada ejemplo predicho por un modelo que NO lo vio) => baseline honesto.
- Metricas ordinales (QWK, MAE, kappa) ademas de F1/accuracy.
- Guarda las predicciones OOF para reusar en E1 (triple comparacion) y tests estadisticos.
Corre en MPS (GPU Metal del Mac) si esta disponible.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
K = 10
EPOCHS = 7
MAXLEN = 128  # el corpus tiene p99=73 tokens; 128 basta y acelera (el proyecto base usa 580; 128 acelera sin truncar (p99=73))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_lora():
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.2,
                     bias="lora_only", target_modules=["key", "query", "value", "dense"],
                     modules_to_save=["classifier", "score"])
    return get_peft_model(m, cfg)


def class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.unique(labels)
    w = compute_class_weight("balanced", classes=cls, y=labels)
    full = np.ones(5, dtype=np.float32)
    for c, wt in zip(cls, w):
        full[c] = wt
    return torch.tensor(full, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_w=None, **kw):
        super().__init__(**kw); self.class_w = class_w
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.get("labels")
        out = model(**inputs)
        logits = out.get("logits")
        lf = torch.nn.CrossEntropyLoss(weight=self.class_w.to(labels.device) if self.class_w is not None else None)
        loss = lf(logits.view(-1, 5), labels.view(-1))
        return (loss, out) if return_outputs else loss


def run(splitter_name="stratified"):
    C.set_all_seeds()
    tok = AutoTokenizer.from_pretrained(MODEL)
    df = C.load_gold()
    texts = df["text"].tolist(); y = df["label"].to_numpy()

    def tok_fn(b): return tok(b["text"], truncation=True, max_length=MAXLEN)

    if splitter_name == "stratified":
        splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=C.SEED)
        split_iter = splitter.split(texts, y)
    else:
        splitter = KFold(n_splits=K, shuffle=True, random_state=C.SEED)
        split_iter = splitter.split(texts)

    oof_pred = np.full(len(y), -1)
    fold_metrics = []
    t0 = time.time()
    for i, (tr, va) in enumerate(split_iter):
        print(f"[{splitter_name}] fold {i+1}/{K}  train={len(tr)} val={len(va)} class5_in_val={(y[va]==4).sum()}", flush=True)
        ds_tr = Dataset.from_dict({"text": [texts[j] for j in tr], "labels": y[tr].tolist()}).map(tok_fn, batched=True)
        ds_va = Dataset.from_dict({"text": [texts[j] for j in va], "labels": y[va].tolist()}).map(tok_fn, batched=True)
        model = build_lora()
        args = TrainingArguments(
            output_dir=f"/tmp/p0_{splitter_name}_{i}", num_train_epochs=EPOCHS,
            per_device_train_batch_size=16, per_device_eval_batch_size=32,
            learning_rate=1e-4, logging_steps=50, report_to=[],
            save_strategy="no", eval_strategy="no", seed=C.SEED,
        )
        trainer = WeightedTrainer(class_w=class_weights(y[tr]), model=model, args=args,
                                  train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
        trainer.train()
        logits = trainer.predict(ds_va).predictions
        pred = np.argmax(logits, axis=-1)
        oof_pred[va] = pred
        m = C.ordinal_metrics(y[va], pred); m["fold"] = i + 1; fold_metrics.append(m)
        print(f"   fold {i+1}: f1_macro={m['f1_macro']:.3f} qwk={m['qwk']:.3f} mae={m['mae']:.3f}", flush=True)

    # metricas OOF globales (honestas: cada ejemplo predicho por modelo que no lo vio)
    glob = C.ordinal_metrics(y, oof_pred)
    fm = np.array([[m["f1_macro"], m["balanced_accuracy"], m["qwk"], m["mae"]] for m in fold_metrics])
    summary = {
        "splitter": splitter_name, "k": K, "device": DEVICE, "seconds": time.time() - t0,
        "oof_global": glob,
        "fold_mean": {"f1_macro": float(fm[:,0].mean()), "balanced_accuracy": float(fm[:,1].mean()),
                      "qwk": float(fm[:,2].mean()), "mae": float(fm[:,3].mean())},
        "fold_sd": {"f1_macro": float(fm[:,0].std(ddof=1)), "balanced_accuracy": float(fm[:,1].std(ddof=1)),
                    "qwk": float(fm[:,2].std(ddof=1)), "mae": float(fm[:,3].std(ddof=1))},
        "folds": fold_metrics,
    }
    # guardar OOF preds para E1
    np.save(C.RESULTS / f"p0_oof_pred_{splitter_name}.npy", oof_pred)
    np.save(C.RESULTS / "p0_y_true.npy", y)
    C.save_result(f"p0_{splitter_name}", summary)
    print(f"\n[{splitter_name}] OOF f1_macro={glob['f1_macro']:.3f} qwk={glob['qwk']:.3f} "
          f"mae={glob['mae']:.3f} bal_acc={glob['balanced_accuracy']:.3f}", flush=True)
    print(f"   fold sd f1_macro={summary['fold_sd']['f1_macro']:.3f}", flush=True)
    return summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "kfold"):
        run("kfold")       # replica el metodo viejo (no estratificado) para comparar
    if which in ("both", "stratified"):
        run("stratified")  # el arreglo
