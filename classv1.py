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



# STEP 1: setup + load dataset
!pip -q install ucimlrepo

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.max_columns", 120)

# fetch dataset 468 (Online Shoppers Purchasing Intention)
data = fetch_ucirepo(id=468)
df = pd.concat([data.data.features, data.data.targets], axis=1)

# ensure the target is binary int (0/1)
df["Revenue"] = df["Revenue"].astype(int)

print("Shape:", df.shape)
display(df.head(3))
display(df.dtypes)

# basic audit
num_cols = [
    'Administrative','Administrative_Duration','Informational','Informational_Duration',
    'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues'
]
cat_cols = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','SpecialDay']

# keep only columns we’ll use (drop anything unexpected)
keep = num_cols + cat_cols + ["Revenue"]
df = df[keep].copy()

print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))

print("\nTarget balance:")
print(df["Revenue"].value_counts().rename(index={0:"no purchase",1:"purchase"}))

print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)



# STEP 2: split + preprocess (one-hot + standardize)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

SEED = 42

# from Step 1
num_cols = [
    'Administrative','Administrative_Duration','Informational','Informational_Duration',
    'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues'
]
cat_cols = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','SpecialDay']

X = df[num_cols + cat_cols].copy()
y = df['Revenue'].astype(int).values

# stratified 70/15/15
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val,   X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
)

print("train/val/test:", X_train.shape, X_val.shape, X_test.shape)
print("class balance (train):", np.bincount(y_train))
print("class balance (val)  :", np.bincount(y_val))
print("class balance (test) :", np.bincount(y_test))

# preprocessors
num_proc = Pipeline(steps=[("scaler", StandardScaler())])
cat_proc = Pipeline(steps=[("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

ct = ColumnTransformer(
    transformers=[
        ("num", num_proc, num_cols),
        ("cat", cat_proc, cat_cols),
    ],
    remainder="drop"
)

# fit on TRAIN only
ct.fit(X_train)

# transform splits to numpy arrays
X_train_enc = ct.transform(X_train)
X_val_enc   = ct.transform(X_val)
X_test_enc  = ct.transform(X_test)

# meta info
oh: OneHotEncoder = ct.named_transformers_["cat"].named_steps["oh"]
onehot_cols = oh.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + onehot_cols

print("\nfinal feature count:", X_train_enc.shape[1])

# quick numeric slice stats (after scaling) on TRAIN
num_count = len(num_cols)
num_slice = X_train_enc[:, :num_count]
num_df = pd.DataFrame(num_slice, columns=num_cols)
summary = pd.DataFrame({
    "mean": num_df.mean(),
    "std": num_df.std(ddof=0),
    "min": num_df.min(),
    "max": num_df.max()
})
print("\nScaled numeric columns (train) summary:")
display(summary.round(3).head(9))

# peek at one-hot part size
print("one-hot columns:", len(onehot_cols))
print("first 10 one-hot names:", onehot_cols[:10])





# STEP 3 — PyTorch MLP training with early stopping on ROC-AUC
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to tensors
Xtr = torch.from_numpy(X_train_enc.astype(np.float32))
Xva = torch.from_numpy(X_val_enc.astype(np.float32))
Xte = torch.from_numpy(X_test_enc.astype(np.float32))
ytr = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
yva = torch.from_numpy(y_val.astype(np.float32)).view(-1,1)
yte = torch.from_numpy(y_test.astype(np.float32)).view(-1,1)

BATCH = 512
trL = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
vaL = DataLoader(TensorDataset(Xva, yva), batch_size=4096, shuffle=False)
teL = DataLoader(TensorDataset(Xte, yte), batch_size=4096, shuffle=False)

# simple MLP with BatchNorm + Dropout
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 1)  # logits
        )
    def forward(self, x): return self.net(x)

def eval_auc(model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()
            all_p.append(probs); all_y.append(yb.numpy().ravel())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    return roc_auc_score(y, p), average_precision_score(y, p), (y, p)

# class imbalance weight: pos / neg
pos = y_train.sum(); neg = len(y_train) - pos
pos_weight = torch.tensor([(neg / (pos + 1e-9))], dtype=torch.float32, device=DEVICE)

model = MLP(X_train_enc.shape[1]).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)

best_auc, best_state, patience, noimp = -1.0, None, 10, 0

for epoch in range(1, 201):
    model.train()
    for xb, yb in trL:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

    va_auc, va_ap, _ = eval_auc(model, vaL)
    print(f"Epoch {epoch:03d} | val ROC-AUC: {va_auc:.4f} | PR-AUC: {va_ap:.4f}")
    if va_auc > best_auc + 1e-5:
        best_auc, best_state, noimp = va_auc, {k:v.cpu() for k,v in model.state_dict().items()}, 0
    else:
        noimp += 1
    if noimp >= patience:
        print("Early stop."); break

# restore best
if best_state: model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

# ----- final metrics on VAL (and pick threshold by best F1) -----
val_auc, val_ap, (y_true_val, y_prob_val) = eval_auc(model, vaL)

ths = np.linspace(0.05, 0.95, 181)
best_f1, best_t, best_pr = -1, 0.5, (0,0,0)
for t in ths:
    yhat = (y_prob_val >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true_val, yhat, average="binary", zero_division=0)
    if f1 > best_f1:
        best_f1, best_t, best_pr = f1, t, (p, r, f1)

print(f"\nVAL  ROC-AUC: {val_auc:.4f} | PR-AUC: {val_ap:.4f}")
print(f"Best threshold (F1): {best_t:.3f} | P={best_pr[0]:.3f} R={best_pr[1]:.3f} F1={best_pr[2]:.3f}")

# ----- TEST at val-optimal threshold -----
test_auc, test_ap, (y_true_test, y_prob_test) = eval_auc(model, teL)
yhat_test = (y_prob_test >= best_t).astype(int)
p, r, f1, _ = precision_recall_fscore_support(y_true_test, yhat_test, average="binary", zero_division=0)

print(f"\nTEST ROC-AUC: {test_auc:.4f} | PR-AUC: {test_ap:.4f}")
print(f"TEST @t={best_t:.3f}: P={p:.3f} R={r:.3f} F1={f1:.3f}")




import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def probs(model, X, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs].astype(np.float32)).to(DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy().ravel())
    return np.concatenate(out)

# get probs
p_val  = probs(model, X_val_enc)
p_test = probs(model, X_test_enc)

# curves
fpr_v, tpr_v, _ = roc_curve(y_val,  p_val)
fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
pr_v, rc_v, _   = precision_recall_curve(y_val,  p_val)
pr_t, rc_t, _   = precision_recall_curve(y_test, p_test)

plt.figure(figsize=(5,4))
plt.plot(fpr_v, tpr_v, label='VAL')
plt.plot(fpr_t, tpr_t, label='TEST')
plt.plot([0,1],[0,1],'--',lw=1,color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(); plt.grid(alpha=.3)
plt.show()

plt.figure(figsize=(5,4))
plt.plot(rc_v, pr_v, label='VAL')
plt.plot(rc_t, pr_t, label='TEST')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall'); plt.legend(); plt.grid(alpha=.3)
plt.show()

# confusion matrix at the threshold we picked from validation
t = 0.650
yhat_test = (p_test >= t).astype(int)
cm = confusion_matrix(y_test, yhat_test)
cm



import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def sweep_threshold(y, p, steps=np.linspace(0.05,0.95,19)):
    rows=[]
    for t in steps:
        yhat = (p>=t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
        rows.append({'t':round(t,3),'precision':prec,'recall':rec,'f1':f1})
    return pd.DataFrame(rows)

s_val  = sweep_threshold(y_val,  p_val)
s_test = sweep_threshold(y_test, p_test)
print("VAL sweep:"); display(s_val)
print("TEST sweep:"); display(s_test)





# --- Device-safe permutation importance (PyTorch path) ---

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd

# 0) Make sure model is on a device and in eval mode
DEVICE = next(model.parameters()).device  # infer device from model
model.to(DEVICE).eval()

# 1) Probabilities helper: move tensor to the SAME device as the model
def get_proba(x_nd: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xb = torch.from_numpy(x_nd.astype(np.float32)).to(DEVICE)  # <- device fix
        logits = model(xb)                                         # forward on DEVICE
        p = torch.sigmoid(logits).detach().cpu().numpy().ravel()   # back to CPU -> NumPy
        return p

# 2) Permutation-importance measured by ROC-AUC drop
def perm_importance(X_val_enc, y_val, n_repeats=5, random_state=42, base_auc=None):
    rng = np.random.default_rng(random_state)
    if base_auc is None:
        base_auc = roc_auc_score(y_val, get_proba(X_val_enc))
    drops = []
    Xwork = X_val_enc.copy()
    for j in range(Xwork.shape[1]):
        col_drops = []
        for _ in range(n_repeats):
            saved = Xwork[:, j].copy()
            rng.shuffle(Xwork[:, j])               # permute one column
            auc = roc_auc_score(y_val, get_proba(Xwork))
            col_drops.append(base_auc - auc)       # AUC drop
            Xwork[:, j] = saved                    # restore column
        drops.append((np.mean(col_drops), np.std(col_drops)))
    return np.array(drops), base_auc

# 3) Recover feature names in your pipeline
#    If you used ColumnTransformer:
# feature_names = ct.get_feature_names_out().tolist()

#    If you built arrays manually with OneHotEncoder:
# cat_names = ohe.get_feature_names_out(cat_cols).tolist()
# feature_names = list(num_cols) + cat_names

# --- RUN ---
imp, base_auc = perm_importance(X_val_enc, y_val, n_repeats=5)
order = np.argsort(-imp[:,0])
top = 25
imp_df = pd.DataFrame({
    "feature": np.array(feature_names)[order][:top],
    "auc_drop_mean": imp[order,0][:top],
    "auc_drop_std":  imp[order,1][:top],
})
display(imp_df)
print(f"Baseline ROC-AUC (val): {base_auc:.4f}")
