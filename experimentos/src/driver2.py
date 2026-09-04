"""
Driver 2: E3 (auto-etiquetado por concordancia), E4 (cabeza ordinal via soft labels),
E5 (calibracion + abstencion). Corre tras el driver local (no competir por GPU).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
import runner as R
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def e3_autolabel():
    """Reentrena con gold(581) + auto-etiquetas de alta concordancia; evalua OOF sobre el gold humano."""
    print("\n===== E3: auto-etiquetado por concordancia (gate >=4 LLMs) =====", flush=True)
    gold = C.load_gold(); yg = gold["label"].to_numpy()
    pool = pd.read_csv(C.RESULTS / "llm_preds_pool.csv")
    for gate in [4, 3]:
        auto = pool[pool["n_agree"] >= gate].copy()
        auto_text = auto["text"].tolist()
        auto_lab = (auto["consensus"].astype(int) - 1).tolist()  # 0..4
        print(f"  gate>={gate}: {len(auto_text)} auto-etiquetas anadidas al train", flush=True)
        # CV estratificada sobre el GOLD; en cada fold, train = gold_train + TODAS las auto-etiquetas.
        # El test es SIEMPRE gold humano out-of-fold (nunca auto-etiquetas).
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=C.SEED)
        tok = AutoTokenizer.from_pretrained(MODEL)
        def tok_fn(b): return tok(b["text"], truncation=True, max_length=128)
        oof = np.full(len(yg), -1)
        texts_g = gold["text"].tolist()
        for i, (tr, va) in enumerate(skf.split(texts_g, yg)):
            tr_text = [texts_g[j] for j in tr] + auto_text
            tr_lab = yg[tr].tolist() + auto_lab
            ds_tr = Dataset.from_dict({"text": tr_text, "labels": tr_lab}).map(tok_fn, batched=True)
            ds_va = Dataset.from_dict({"text": [texts_g[j] for j in va], "labels": yg[va].tolist()}).map(tok_fn, batched=True)
            base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
            cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.2,
                             bias="lora_only", target_modules=["key","query","value","dense"],
                             modules_to_save=["classifier","score"])
            model = get_peft_model(base, cfg)
            args = TrainingArguments(output_dir=f"/tmp/e3_{gate}_{i}", num_train_epochs=7,
                                     per_device_train_batch_size=16, learning_rate=1e-4,
                                     report_to=[], save_strategy="no", seed=C.SEED, disable_tqdm=True)
            tr_obj = R.WeightedTrainer(class_w=R._class_weights(np.array(tr_lab)), model=model, args=args,
                                       train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
            tr_obj.train()
            oof[va] = np.argmax(tr_obj.predict(ds_va).predictions, axis=-1)
            del model, base, tr_obj
            if DEVICE == "mps": torch.mps.empty_cache()
        m = C.ordinal_metrics(yg, oof)
        print(f"  gate>={gate}: OOF QWK={m['qwk']:.3f} F1={m['f1_macro']:.3f} MAE={m['mae']:.3f}", flush=True)
        # comparar contra baseline P0 (pareado)
        base_oof = np.load(C.RESULTS / "p0_oof_pred_stratified.npy")
        dq = C.paired_bootstrap_diff(yg, oof, base_oof, C.qwk_fn, n=2000)
        C.save_result(f"e3_gate{gate}", {"metrics": m, "n_auto": len(auto_text),
                      "delta_qwk_vs_p0": {"mean": dq[0], "ci_low": dq[1], "ci_high": dq[2], "p_gt": dq[3]},
                      "oof_pred": oof.tolist()})
        print(f"  gate>={gate}: Delta QWK vs P0 = {dq[0]:+.3f} IC[{dq[1]:+.3f},{dq[2]:+.3f}] P={dq[3]:.2f}", flush=True)


def e5_calibracion():
    """Temperature scaling + abstencion. Entrena en un split 80/20, calibra en val, mide ECE y risk-coverage."""
    print("\n===== E5: calibracion + abstencion =====", flush=True)
    gold = C.load_gold(); y = gold["label"].to_numpy(); texts = gold["text"].tolist()
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=C.SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    def tok_fn(b): return tok(b["text"], truncation=True, max_length=128)
    ds_tr = Dataset.from_dict({"text":[texts[j] for j in tr],"labels":y[tr].tolist()}).map(tok_fn,batched=True)
    ds_va = Dataset.from_dict({"text":[texts[j] for j in va],"labels":y[va].tolist()}).map(tok_fn,batched=True)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.2, bias="lora_only",
                     target_modules=["key","query","value","dense"], modules_to_save=["classifier","score"])
    model = get_peft_model(base, cfg)
    args = TrainingArguments(output_dir="/tmp/e5", num_train_epochs=7, per_device_train_batch_size=16,
                             learning_rate=1e-4, report_to=[], save_strategy="no", seed=C.SEED, disable_tqdm=True)
    tr_obj = R.WeightedTrainer(class_w=R._class_weights(y[tr]), model=model, args=args,
                               train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
    tr_obj.train()
    logits = tr_obj.predict(ds_va).predictions
    yv = y[va]
    logits_t = torch.tensor(logits)
    def ece(probs, yt, bins=10):
        conf = probs.max(1); pred = probs.argmax(1); acc = (pred == yt)
        e = 0.0
        for b in range(bins):
            lo, hi = b/bins, (b+1)/bins
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0: e += m.mean() * abs(acc[m].mean() - conf[m].mean())
        return float(e)
    import torch.nn.functional as F
    p_before = F.softmax(logits_t, dim=1).numpy()
    # ajustar T
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)
    yt = torch.tensor(yv)
    def clo():
        opt.zero_grad(); loss = F.cross_entropy(logits_t / T, yt); loss.backward(); return loss
    opt.step(clo)
    p_after = F.softmax(logits_t / T.detach(), dim=1).numpy()
    ece_b, ece_a = ece(p_before, yv), ece(p_after, yv)
    # risk-coverage con la confianza calibrada
    conf = p_after.max(1); pred = p_after.argmax(1)
    order = np.argsort(-conf)
    rc = []
    for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = int(cov * len(order)); idx = order[:k]
        rc.append({"coverage": cov, "qwk": float(C.qwk_fn(yv[idx], pred[idx]))})
    res = {"ece_before": ece_b, "ece_after": ece_a, "T": float(T.detach().item()),
           "risk_coverage": rc, "n_val": len(va)}
    C.save_result("e5_calibracion", res)
    print(f"  ECE {ece_b:.3f} -> {ece_a:.3f} (T={T.detach().item():.2f})", flush=True)
    print(f"  risk-coverage: {[(r['coverage'], round(r['qwk'],3)) for r in rc]}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e3"): e3_autolabel()
    if which in ("all", "e5"): e5_calibracion()
    print("\nDONE driver2", flush=True)
