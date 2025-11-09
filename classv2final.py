# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install ucimlrepo


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
# data (as pandas dataframes) 
X = online_shoppers_purchasing_intention_dataset.data.features 
y = online_shoppers_purchasing_intention_dataset.data.targets 
  
# metadata 
print(online_shoppers_purchasing_intention_dataset.metadata) 
  
# variable information 
print(online_shoppers_purchasing_intention_dataset.variables) 



# --- imports
import os, json, math, random, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# optional (used later)
try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

# --- folders
for d in ["models", "figures", "web"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# --- reproducibility + device
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- small helpers
CLASS_NAMES = ["NoPurchase", "Purchase"]

def show_confmat(cm, class_names=CLASS_NAMES, title="Confusion Matrix"):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names); plt.yticks(ticks, class_names)
    thresh = cm.max()/2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.show()

def evaluate_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn+1e-9); tnr = tn/(tn+fp+1e-9)
    bal_acc = 0.5*(tpr+tnr)
    # Matthews CC
    mcc = ((tp*tn - fp*fn) /
           math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-9))
    roc = roc_auc_score(y_true, scores)
    pr  = average_precision_score(y_true, scores)
    return dict(threshold=float(thr), F1_pos=f1_pos, F1_macro=f1_macro,
                BalancedAcc=bal_acc, MCC=mcc, ROC_AUC=roc, PR_AUC=pr, cm=cm)

def tune_threshold_for_f1(y_true, scores):
    ps = np.linspace(0.1, 0.9, 81)  # 0.10 .. 0.90 (step 0.01)
    best_thr, best_f1 = 0.5, -1
    for t in ps:
        f1p = f1_score(y_true, (scores >= t).astype(int), pos_label=1)
        if f1p > best_f1:
            best_f1, best_thr = f1p, float(t)
    return best_thr, best_f1

def pretty_print_metrics(name, m):
    print(f"=== {name} ===")
    print(f"threshold  : {m['threshold']:.3f}")
    print(f"F1 (positive) : {m['F1_pos']:.3f}")
    print(f"F1 (macro)    : {m['F1_macro']:.3f}")
    print(f"Balanced Acc  : {m['BalancedAcc']:.3f}")
    print(f"MCC           : {m['MCC']:.3f}")
    print(f"ROC-AUC       : {m['ROC_AUC']:.3f}")
    print(f"PR-AUC        : {m['PR_AUC']:.3f}")
    print("Confusion matrix [rows: true 0/1, cols: pred 0/1]:")
    print(m["cm"])
    print()




# Try ucimlrepo (nice API); fall back to direct CSV if unavailable.
X, y = None, None
try:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=468)  # OSPI
    X = ds.data.features.copy()
    y = ds.data.targets.copy().iloc[:,0]  # 'Revenue'
    print("Loaded via ucimlrepo.")
except Exception as e:
    print("ucimlrepo failed, trying direct CSV:", e)
    url = "https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/online_shoppers_intention.csv"
    df = pd.read_csv(url)
    y = df['Revenue'].astype(int)
    X = df.drop(columns=['Revenue'])

print(X.shape, y.shape)
print("Columns:", list(X.columns))
print("Positive rate overall:", y.mean().round(3))




# 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
val_size = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=SEED
)

# Identify numeric & categorical
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ]
)
# fit on train only
preprocessor.fit(X_train)

# transform
X_train_p = preprocessor.transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(X_test)

# stats
print("Shapes (preprocessed):")
print("  X_train:", X_train_p.shape, "| y_train:", y_train.shape)
print("  X_val  :", X_val_p.shape,   "| y_val  :", y_val.shape)
print("  X_test :", X_test_p.shape,  "| y_test :", y_test.shape)
print("Positive rate (train/val/test):",
      round(y_train.mean(),3), round(y_val.mean(),3), round(y_test.mean(),3))

# class-imbalance weight for BCEWithLogitsLoss
n_pos = int(y_train.sum()); n_neg = int(len(y_train) - n_pos)
pos_weight_value = n_neg / max(n_pos, 1)
print("pos_weight (for PyTorch):", round(pos_weight_value,3))

# persist preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Saved -> models/preprocessor.pkl")




# Dummy "always majority" — for reference
maj = 1 if y_train.mean()>0.5 else 0
scores_dummy = np.full_like(y_val.values, fill_value=maj, dtype=float)
thr_dummy = 0.5
metrics_dummy = evaluate_at_threshold(y_val.values, scores_dummy, thr_dummy)
pretty_print_metrics("Dummy (most_frequent, VAL)", metrics_dummy)
show_confmat(metrics_dummy["cm"], title="Dummy (VAL)")

# Logistic Regression (with class weights)
logreg = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
logreg.fit(X_train_p, y_train.values)
val_scores_lr = logreg.predict_proba(X_val_p)[:,1]
thr_lr, _ = tune_threshold_for_f1(y_val.values, val_scores_lr)
metrics_lr_val = evaluate_at_threshold(y_val.values, val_scores_lr, thr_lr)
pretty_print_metrics("Logistic Regression (VAL, tuned thr)", metrics_lr_val)
show_confmat(metrics_lr_val["cm"], title=f"Logistic (VAL) @ thr={thr_lr:.2f}")





# Tensor helpers
def to_tensor(x): return torch.from_numpy(x).float()

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

def train_mlp(X_tr, y_tr, X_va, y_va, hidden, dropout=0.3, lr=1e-3,
              batch_size=256, max_epochs=80, patience=12):
    in_dim = X_tr.shape[1]
    model  = MLP(in_dim, hidden, dropout).to(device)
    pos_w  = torch.tensor([float(pos_weight_value)], device=device)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    ds_tr = TensorDataset(to_tensor(X_tr), torch.from_numpy(y_tr.values).float())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    best = {"f1":-1, "epoch":0, "state":None, "thr":0.5}
    no_imp = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        running = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(ds_tr)

        # eval on val
        model.eval()
        with torch.no_grad():
            val_logits = model(to_tensor(X_va).to(device)).cpu().numpy()
        val_scores = 1/(1+np.exp(-val_logits))
        thr, f1p = tune_threshold_for_f1(y_va.values, val_scores)

        if f1p > best["f1"]:
            best.update({"f1":f1p, "epoch":epoch, "state":model.state_dict(), "thr":thr})
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val F1+ {f1p:.3f} @ thr {thr:.2f}")

        if no_imp >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement {patience} epochs).")
            break

    # restore best
    model.load_state_dict(best["state"])
    model.eval()
    return model, best

# Train two configs and pick best on VAL
cfgs = {"MLP-A":[64,32], "MLP-B":[128,64,32]}
results = {}
for name, hidden in cfgs.items():
    print(f"\nTraining {name}...")
    model, best = train_mlp(X_train_p, y_train, X_val_p, y_val,
                            hidden=hidden, dropout=0.3 if name=='MLP-B' else 0.2)
    # store
    with torch.no_grad():
        val_logits = model(to_tensor(X_val_p).to(device)).cpu().numpy()
    val_scores  = 1/(1+np.exp(-val_logits))
    metrics = evaluate_at_threshold(y_val.values, val_scores, best["thr"])
    pretty_print_metrics(f"{name} (VAL, tuned thr)", metrics)
    show_confmat(metrics["cm"], title=f"{name} (VAL) @ thr={best['thr']:.2f}")
    results[name] = {"model":model, "best":best, "metrics":metrics}

# pick best by F1 positive on VAL
best_name = max(results.keys(), key=lambda n: results[n]["metrics"]["F1_pos"])
best_model = results[best_name]["model"]
best_info  = results[best_name]["best"]
print(f">>> Selected best on VAL: {best_name} (F1+={results[best_name]['metrics']['F1_pos']:.3f}, thr={best_info['thr']:.2f}, epoch={best_info['epoch']})")

# save best model checkpoint (with arch + thr)
arch = {"in_dim": X_train_p.shape[1], "hidden": cfgs[best_name], "dropout": 0.3 if best_name=='MLP-B' else 0.2}
torch.save({"state_dict": best_model.state_dict(), "arch": arch, "threshold": best_info["thr"]},
           "models/mlp_best.pt")
print("Saved -> models/mlp_best.pt")




# MLP test
with torch.no_grad():
    logits_test = best_model(to_tensor(X_test_p).to(device)).cpu().numpy()
scores_test_mlp = 1/(1+np.exp(-logits_test))
metrics_mlp_test = evaluate_at_threshold(y_test.values, scores_test_mlp, best_info["thr"])
pretty_print_metrics("MLP (TEST, thr from VAL)", metrics_mlp_test)
show_confmat(metrics_mlp_test["cm"], title=f"{best_name} (TEST) @ thr={best_info['thr']:.2f}")

# Logistic test
scores_test_lr = logreg.predict_proba(X_test_p)[:,1]
metrics_lr_test = evaluate_at_threshold(y_test.values, scores_test_lr, thr_lr)
pretty_print_metrics("Logistic Regression (TEST, thr from VAL)", metrics_lr_test)
show_confmat(metrics_lr_test["cm"], title=f"Logistic (TEST) @ thr={thr_lr:.2f}")




# ROC & PR
def plot_roc_pr(name_scores_pairs, y_true, out_prefix="figures"):
    # ROC
    plt.figure(figsize=(5,5))
    for name, s in name_scores_pairs:
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = roc_auc_score(y_true, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--",alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC — Test")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}/roc_curve.png", dpi=150); plt.show()

    # PR
    plt.figure(figsize=(5,5))
    for name, s in name_scores_pairs:
        prec, rec, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall — Test")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}/pr_curve.png", dpi=150); plt.show()

plot_roc_pr([
    (best_name, scores_test_mlp),
    ("LogReg", scores_test_lr)
], y_test.values)

print("Saved curves to figures/roc_curve.png and figures/pr_curve.png")




# Group-wise permutation importance on TEST, using PR-AUC drop, grouped by original raw columns.
# (We permute raw columns, re-transform, then score with the best MLP)
base_pr = average_precision_score(y_test.values, scores_test_mlp)

feature_names = preprocessor.get_feature_names_out().tolist()
# Map from raw feature -> indices in transformed matrix
groups = {}
for i, fname in enumerate(feature_names):
    if fname in num_cols:  # numeric passes through with same name
        base = fname
    else:
        base = fname.split('_')[0]  # e.g., "Month_Aug" -> "Month"
    groups.setdefault(base, []).append(i)

# Permute by raw columns: shuffle those raw columns, re-transform, score
drops = []
rng = np.random.default_rng(SEED)
for col in X_test.columns:
    Xp = X_test.copy()
    Xp[col] = Xp[col].sample(frac=1.0, random_state=SEED).values  # permute column
    Xp_p = preprocessor.transform(Xp)
    with torch.no_grad():
        logits = best_model(to_tensor(Xp_p).to(device)).cpu().numpy()
    sc = 1/(1+np.exp(-logits))
    pr = average_precision_score(y_test.values, sc)
    drops.append((col, base_pr - pr))

# sort & plot
drops = sorted(drops, key=lambda t: t[1], reverse=True)[:12]
labels = [d[0] for d in drops]
vals   = [d[1] for d in drops]

plt.figure(figsize=(7,4))
plt.barh(labels[::-1], vals[::-1])
plt.xlabel("Δ PR-AUC when permuted (importance)")
plt.title("Permutation importance — grouped by raw feature (TEST)")
plt.tight_layout(); plt.savefig("figures/permutation_importance.png", dpi=150); plt.show()

print("Top features by PR-AUC drop:")
for k,v in drops:
    print(f"{k:25s} {v:.4f}")




if XGB_OK:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=float(pos_weight_value), n_jobs=-1, tree_method="hist",
        random_state=SEED
    )
    xgb_clf.fit(X_train_p, y_train.values, eval_set=[(X_val_p, y_val.values)],
                verbose=False, early_stopping_rounds=50)
    # tune thr on VAL
    s_val = xgb_clf.predict_proba(X_val_p)[:,1]
    thr_xgb, _ = tune_threshold_for_f1(y_val.values, s_val)
    m_val = evaluate_at_threshold(y_val.values, s_val, thr_xgb)
    pretty_print_metrics("XGBoost (VAL, tuned thr)", m_val)
    show_confmat(m_val["cm"], title=f"XGB (VAL) @ thr={thr_xgb:.2f}")

    # TEST
    s_test = xgb_clf.predict_proba(X_test_p)[:,1]
    metrics_xgb_test = evaluate_at_threshold(y_test.values, s_test, thr_xgb)
    pretty_print_metrics("XGBoost (TEST, thr from VAL)", metrics_xgb_test)
    show_confmat(metrics_xgb_test["cm"], title=f"XGB (TEST) @ thr={thr_xgb:.2f}")
else:
    print("xgboost not available in this environment.")




# Rebuild model on CPU and export ONNX
cp = torch.load("models/mlp_best.pt", map_location="cpu")
arch = cp["arch"]; thr = float(cp["threshold"])

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

mdl = _MLP(arch["in_dim"], arch["hidden"], arch["dropout"]).eval()
mdl.load_state_dict(cp["state_dict"])

dummy = torch.zeros(1, arch["in_dim"], dtype=torch.float32)
torch.onnx.export(
    mdl, dummy, "models/ospi_best.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)
print("Wrote -> models/ospi_best.onnx")

# Build preprocess metadata for web
scaler = preprocessor.named_transformers_["num"]
ohe    = preprocessor.named_transformers_["cat"]

feature_order = preprocessor.get_feature_names_out().tolist()
cat_categories = {}
for col, cats in zip([*preprocessor.transformers_[1][2]], ohe.categories_):
    levels = []
    for x in cats.tolist():
        if isinstance(x, (np.bool_, bool)): levels.append("True" if x else "False")
        elif x is None: levels.append("None")
        else: levels.append(str(x))
    cat_categories[col] = levels

pre_meta = {
    "numeric_cols": list(preprocessor.transformers_[0][2]),
    "numeric_means": [float(x) for x in scaler.mean_],
    "numeric_scales": [float(x) for x in scaler.scale_],
    "categorical_cols": list(preprocessor.transformers_[1][2]),
    "categorical_levels": cat_categories,
    "feature_order": feature_order,
    "threshold": thr,
    "numeric_defaults": {c: float(m) for c,m in zip(preprocessor.transformers_[0][2], scaler.mean_)},
    "categorical_defaults": {c: (cat_categories[c][0] if cat_categories.get(c) else "") for c in preprocessor.transformers_[1][2]}
}
with open("web/preprocess.json","w") as f: json.dump(pre_meta, f, indent=2)
with open("web/threshold.txt","w") as f: f.write(str(thr))

# copy ONNX to web
import shutil; shutil.copyfile("models/ospi_best.onnx", "web/ospi_best.onnx")
print("Wrote web/: ospi_best.onnx, preprocess.json, threshold.txt")




# Build Kaggle-proof web app: app.js embeds model+json; index.html loads UMD ORT
from pathlib import Path
import base64, json

meta = json.load(open("web/preprocess.json","r"))
thr  = float(open("web/threshold.txt","r").read().strip())
onnx_b64 = base64.b64encode(open("web/ospi_best.onnx","rb").read()).decode("ascii")
meta_json = json.dumps(meta, separators=(",",":"))

index_html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSPI — Purchase Intent Predictor (Kaggle)</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<style>
  body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:24px; background:#0b0e13; color:#e9eef5; }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin-bottom: 8px; }
  p.sub { color:#aab3c2; margin-top:0; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap:16px; }
  .card { background:#121826; border:1px solid #1f2a3a; border-radius:16px; padding:16px; }
  label { display:block; font-size:.9rem; margin-bottom:6px; color:#cdd7e5; }
  input[type=number],select { width:100%; padding:10px 12px; border-radius:10px; border:1px solid #2a3445; background:#0f1522; color:#e9eef5; }
  .actions { display:flex; gap:12px; margin-top:10px; align-items:center; }
  button { background:#2e6cff; color:#fff; border:none; border-radius:12px; padding:10px 16px; cursor:pointer; }
  .muted { color:#9badc1; font-size:.85rem; }
  .result { padding:16px; border-radius:14px; background:#0f1522; border:1px solid #2a3445; }
  .prob { font-size:1.4rem; }
  #err { color:#ff9c9c; margin-top:10px; white-space:pre-wrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Online Shoppers Purchasing Intention — Live Demo</h1>
  <p class="sub">Runs a PyTorch MLP (ONNX) with onnxruntime-web (UMD). Kaggle-safe (no fetch for model/JSON).</p>

  <div class="grid" id="inputs"></div>

  <div class="card">
    <div class="actions">
      <button id="predictBtn">Predict</button>
      <div class="muted">Threshold: <span id="thr">–</span></div>
    </div>
    <div class="result" style="margin-top:14px;">
      <div class="prob">Probability of Purchase: <b id="prob">–</b></div>
      <div>Predicted Class: <b id="cls">–</b></div>
      <div id="err"></div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"></script>
<script src="./app.js"></script>
</body>
</html>
"""

app_js = r"""
(function(){
  const META = __META_JSON__;
  const THRESH = __THRESH__;
  const MODEL_B64 = "__MODEL_B64__";

  const errBox = document.getElementById('err');
  document.getElementById('thr').textContent = Number.isFinite(THRESH) ? THRESH.toFixed(2) : '—';
  const showErr = (e)=>{ const m=(e&&e.message)?e.message:String(e); errBox.textContent=m; console.error(e); };

  const grid = document.getElementById('inputs');
  const el = (t,a={},ch=[])=>{ const x=document.createElement(t); for(const[k,v]of Object.entries(a)){k==='text'?x.textContent=v:x.setAttribute(k,v)} ch.forEach(c=>x.appendChild(c)); return x; };
  for (const n of META.numeric_cols) {
    const defv=(META.numeric_defaults||{})[n] ?? 0;
    grid.appendChild(el('div',{class:'card'},[
      el('label',{for:n,text:n}),
      el('input',{type:'number',id:n,value:String(defv),step:'any'})
    ]));
  }
  for (const c of META.categorical_cols) {
    const wrap=el('div',{class:'card'});
    wrap.appendChild(el('label',{for:c,text:c}));
    const sel=el('select',{id:c});
    const levels=META.categorical_levels[c], defc=(META.categorical_defaults||{})[c];
    for (const o of levels) {
      const opt=el('option',{value:String(o),text:String(o)});
      if(String(o)===String(defc)) opt.selected=true;
      sel.appendChild(opt);
    }
    wrap.appendChild(sel); grid.appendChild(wrap);
  }

  const b64ToU8 = (b64)=>{ const bin=atob(b64); const u8=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i); return u8; };

  if (!window.ort) { showErr('onnxruntime-web not loaded. In Kaggle, turn Internet ON.'); return; }
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

  let session;
  (async()=>{ try {
      session = await ort.InferenceSession.create(b64ToU8(MODEL_B64), { executionProviders: ['wasm'] });
    } catch(e) { showErr(e); }
  })();

  const buildVec = ()=>{
    const vec=new Float32Array(META.feature_order.length).fill(0);
    const mean=Object.fromEntries(META.numeric_cols.map((k,i)=>[k,META.numeric_means[i]]));
    const scale=Object.fromEntries(META.numeric_cols.map((k,i)=>[k,META.numeric_scales[i]]));
    for(let i=0;i<META.feature_order.length;i++) {
      const name=META.feature_order[i];
      if (META.numeric_cols.includes(name)) {
        const raw=parseFloat(document.getElementById(name).value||'0');
        const z=(raw-(mean[name]||0))/(scale[name]||1);
        vec[i]=Number.isFinite(z)?z:0;
      } else {
        const base=name.split('_')[0], want=name.substring(base.length+1);
        const cur=String(document.getElementById(base).value);
        vec[i]=(cur===want)?1:0;
      }
    }
    return vec;
  };

  async function predict(){
    errBox.textContent='';
    try {
      if(!session) { showErr('Model is loading… click Predict again in a second.'); return; }
      const x=buildVec();
      const out=await session.run({ input:new ort.Tensor('float32', x, [1,x.length]) });
      const logit=out.logits.data[0];
      const prob=1/(1+Math.exp(-logit));
      document.getElementById('prob').textContent=prob.toFixed(3);
      document.getElementById('cls').textContent=(prob>=THRESH)?'Purchase':'NoPurchase';
    } catch(e) { showErr(e); }
  }
  document.getElementById('predictBtn').addEventListener('click', predict);
})();
"""

# write files
Path("web/index.html").write_text(index_html, encoding="utf-8")
Path("web/app.js").write_text(
    app_js.replace("__META_JSON__", meta_json)
          .replace("__THRESH__", f"{thr:.6f}")
          .replace("__MODEL_B64__", onnx_b64),
    encoding="utf-8"
)
print("Wrote Kaggle-proof web/index.html + app.js")





# Install onnxruntime (CPU) if missing, then import
import sys, subprocess, pkgutil
if pkgutil.find_loader("onnxruntime") is None:
    print("Installing onnxruntime==1.19.2 ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.19.2"])
import onnxruntime as ort
print("onnxruntime version:", ort.__version__)



import numpy as np

# Load ONNX model and threshold written earlier into /web
sess = ort.InferenceSession("web/ospi_best.onnx", providers=["CPUExecutionProvider"])
THR = float(open("web/threshold.txt").read().strip())

def onnx_probs(Xp):
    logits = sess.run(["logits"], {"input": Xp.astype(np.float32)})[0].ravel()
    return 1.0 / (1.0 + np.exp(-logits))

# two examples (one 0, one 1) from TEST
neg_idx = int(np.where(y_test.values==0)[0][0])
pos_idx = int(np.where(y_test.values==1)[0][0])

for idx in [neg_idx, pos_idx]:
    Xp = X_test_p[idx:idx+1]
    p  = float(onnx_probs(Xp)[0])
    print(f"Row {idx} | prob={p:.3f} | pred={int(p>=THR)} | true={int(y_test.iloc[idx])}")




import torch, numpy as np

with torch.no_grad():
    logits = best_model(torch.from_numpy(X_test_p).float().to(device)).cpu().numpy()
probs = 1.0 / (1.0 + np.exp(-logits))
THR = float(open("web/threshold.txt").read().strip())

neg_idx = int(np.where(y_test.values==0)[0][0])
pos_idx = int(np.where(y_test.values==1)[0][0])
for idx in [neg_idx, pos_idx]:
    p = float(probs[idx])
    print(f"[PyTorch] Row {idx} | prob={p:.3f} | pred={int(p>=THR)} | true={int(y_test.iloc[idx])}")




# README (short)
with open("README.md","w") as f:
    f.write(f"""# Online Shoppers Purchasing Intention — Classification

**Data:** UCI OSPI (id=468).  
**Split:** 70/15/15 stratified.  
**Models:** Logistic Regression (baseline), PyTorch {best_name} (best), XGBoost (optional).  
**Imbalance:** pos_weight={pos_weight_value:.3f} + threshold tuning on VAL.

## Test (selected)
- {best_name}: F1⁺={metrics_mlp_test['F1_pos']:.3f}, Macro-F1={metrics_mlp_test['F1_macro']:.3f}, ROC-AUC={metrics_mlp_test['ROC_AUC']:.3f}, PR-AUC={metrics_mlp_test['PR_AUC']:.3f}
- Logistic: F1⁺={metrics_lr_test['F1_pos']:.3f}, Macro-F1={metrics_lr_test['F1_macro']:.3f}
{"- XGBoost: (see notebook)" if not XGB_OK else ""}

Curves and importance in `/figures`. Deployed demo in `/web` (Kaggle-proof).
""")
print("Wrote README.md")

# Zip web
import shutil, os
if os.path.exists("web.zip"): os.remove("web.zip")
shutil.make_archive("web", "zip", "web")
print("Created web.zip")
