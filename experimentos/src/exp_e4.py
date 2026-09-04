"""
E4: cabeza ordinal CORN vs softmax. La cabeza produce K-1=4 logits (no 5).
Usa corn_loss (coral-pytorch). Compara MAE (primaria) y QWK contra el baseline softmax P0.
Mismo StratifiedKFold, mismo backbone, misma semilla: solo cambia la cabeza/perdida.
"""
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class CornModel(torch.nn.Module):
    def __init__(self, base_name, num_classes=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, num_classes - 1)  # K-1 logits
        self.num_classes = num_classes
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # [CLS]
        logits = self.head(self.drop(pooled))
        loss = None
        if labels is not None:
            loss = corn_loss(logits, labels, num_classes=self.num_classes)
        return {"loss": loss, "logits": logits}


def run():
    C.set_all_seeds()
    df = C.load_gold(); y = df["label"].to_numpy(); texts = df["text"].tolist()
    tok = AutoTokenizer.from_pretrained(MODEL)
    def tok_fn(b): return tok(b["text"], truncation=True, max_length=128)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=C.SEED)
    oof = np.full(len(y), -1)
    for i, (tr, va) in enumerate(skf.split(texts, y)):
        ds_tr = Dataset.from_dict({"text":[texts[j] for j in tr],"labels":y[tr].tolist()}).map(tok_fn,batched=True,remove_columns=["text"])
        ds_va = Dataset.from_dict({"text":[texts[j] for j in va],"labels":y[va].tolist()}).map(tok_fn,batched=True,remove_columns=["text"])
        model = CornModel(MODEL)
        cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=32, lora_alpha=64, lora_dropout=0.2,
                         bias="lora_only", target_modules=["key","query","value","dense"],
                         modules_to_save=["head"])
        model = get_peft_model(model, cfg)
        args = TrainingArguments(output_dir=f"/tmp/e4_{i}", num_train_epochs=7, per_device_train_batch_size=16,
                                 learning_rate=1e-4, report_to=[], save_strategy="no", seed=C.SEED,
                                 disable_tqdm=True, remove_unused_columns=False, label_names=["labels"])
        tr_obj = Trainer(model=model, args=args, train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
        tr_obj.train()
        logits = tr_obj.predict(ds_va).predictions
        if isinstance(logits, tuple): logits = logits[0]
        pred = corn_label_from_logits(torch.tensor(logits)).numpy()
        oof[va] = pred
        m = C.ordinal_metrics(y[va], pred)
        print(f"  [e4-corn] fold {i+1}/10 mae={m['mae']:.3f} qwk={m['qwk']:.3f} f1={m['f1_macro']:.3f}", flush=True)
        del model, tr_obj
        if DEVICE == "mps": torch.mps.empty_cache()
    m = C.ordinal_metrics(y, oof)
    base = np.load(C.RESULTS / "p0_oof_pred_stratified.npy")
    # MAE es la metrica primaria de una cabeza ordinal
    def mae_fn(yt, yp): return -np.mean(np.abs(yt - yp))  # negado para "mayor=mejor" en el bootstrap
    dmae = C.paired_bootstrap_diff(y, oof, base, mae_fn, n=2000)  # mean>0 => CORN mejor (menos MAE)
    dqwk = C.paired_bootstrap_diff(y, oof, base, C.qwk_fn, n=2000)
    res = {"corn_metrics": m, "delta_mae_neg_corn_minus_base": {"mean": dmae[0], "ci_low": dmae[1], "ci_high": dmae[2]},
           "delta_qwk_corn_minus_base": {"mean": dqwk[0], "ci_low": dqwk[1], "ci_high": dqwk[2], "p_gt": dqwk[3]},
           "oof_pred": oof.tolist()}
    C.save_result("e4_corn", res)
    print(f"\n[E4-CORN] QWK={m['qwk']:.3f} MAE={m['mae']:.3f} F1={m['f1_macro']:.3f}", flush=True)
    print(f"  vs P0(softmax): dMAE(neg)={dmae[0]:+.3f} IC[{dmae[1]:+.3f},{dmae[2]:+.3f}]  dQWK={dqwk[0]:+.3f} IC[{dqwk[1]:+.3f},{dqwk[2]:+.3f}]", flush=True)


if __name__ == "__main__":
    run(); print("DONE E4", flush=True)
