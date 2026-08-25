import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
CKPT_PATH  = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures/checkpoints/tft_best.ckpt"
MCS_PATH   = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"

# ─────────────────────────────────────────────
# HYPERPARAMETERS (identical to TFT_Model_Fit.py)
# ─────────────────────────────────────────────
BEST_ENC              = 5
BEST_HID              = 64
BEST_HEADS            = 2
BEST_DROP             = 0.1
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE            = 64

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────
# DATA  (identical preprocessing to TFT_Model_Fit.py)
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET   = "Global_TVL_USD_log_return"
FEATURES = [c for c in df.columns if c not in ["date", TARGET]]

df_model = df[["date", TARGET] + FEATURES].dropna().reset_index(drop=True)
df_model["time_idx"] = np.arange(len(df_model))
df_model["group"]    = "TVL"

split_idx  = int(len(df_model) * 0.80)
df_train   = df_model.iloc[:split_idx].copy()
df_test    = df_model.iloc[split_idx:].copy()
y_train    = df_model[TARGET].values[:split_idx]
y_test     = df_model[TARGET].values[split_idx:]

# Scaling — fit on train only (no leakage)
scaler_X     = StandardScaler().fit(df_train[FEATURES].values)
y_train_mean = df_train[TARGET].mean()
y_train_std  = df_train[TARGET].std()

df_model_sc           = df_model.copy()
df_model_sc[FEATURES] = scaler_X.transform(df_model[FEATURES].values)
df_model_sc[TARGET]   = (df_model[TARGET] - y_train_mean) / y_train_std

df_train_sc = df_model_sc.iloc[:split_idx].copy()
df_test_sc  = df_model_sc.iloc[split_idx:].copy()

# Inner train/val split (identical to TFT_Model_Fit.py)
es_split       = int(len(df_train_sc) * 0.90)
df_inner_train = df_train_sc.iloc[:es_split].copy()

df_full_sc             = pd.concat([df_train_sc, df_test_sc], ignore_index=True)
df_full_sc["time_idx"] = np.arange(len(df_full_sc))
inner_train_cutoff     = int(df_inner_train["time_idx"].max())
training_cutoff        = int(df_train_sc["time_idx"].max())

# ─────────────────────────────────────────────
# REBUILD training_ds (needed to load checkpoint correctly)
# ─────────────────────────────────────────────
training_ds = TimeSeriesDataSet(
    df_full_sc[df_full_sc["time_idx"] <= inner_train_cutoff],
    time_idx                   = "time_idx",
    target                     = TARGET,
    group_ids                  = ["group"],
    max_encoder_length         = BEST_ENC,
    max_prediction_length      = MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals = [TARGET],
    time_varying_known_reals   = FEATURES,
    target_normalizer          = None,
)

# ─────────────────────────────────────────────
# LOAD CHECKPOINT — no training
# ─────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
best_tft.eval()
print("✓ Checkpoint loaded.")

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _split_pred(pred_obj):
    out = pred_obj.output if hasattr(pred_obj, "output") else pred_obj[0]
    idx = pred_obj.index  if hasattr(pred_obj, "index")  else pred_obj[1]
    out = out.cpu().numpy().flatten() if hasattr(out, "cpu") else np.asarray(out).flatten()
    return out, idx

# ─────────────────────────────────────────────
# TEST SET PREDICTIONS (rolling-origin, one-step-ahead)
# Identical to TFT_Model_Fit.py Stage 3
# ─────────────────────────────────────────────
test_ds = TimeSeriesDataSet.from_dataset(
    training_ds,
    df_full_sc[df_full_sc["time_idx"] > training_cutoff - BEST_ENC],
    predict=False, stop_randomization=True,
)
test_loader = test_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

preds_test_sc, idx_test = _split_pred(
    best_tft.predict(test_loader, mode="prediction", return_index=True)
)
t_idx_test = idx_test["time_idx"].values
mask_test  = t_idx_test > training_cutoff
sel_t_test = t_idx_test[mask_test]

y_pred_test = preds_test_sc[mask_test] * y_train_std + y_train_mean
y_test_al   = df_model[TARGET].values[sel_t_test]

assert len(y_pred_test) == len(y_test_al), \
    f"Mismatch: {len(y_pred_test)} preds vs {len(y_test_al)} test obs"

print(f"✓ Predictions generated. n_test = {len(y_pred_test)}")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
import os
os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_TFT.npy", y_pred_test)
np.save(f"{MCS_PATH}/y_test_TFT.npy", y_test_al)
print("✓ MCS forecasts saved (TFT).")
print(f"  → {MCS_PATH}/y_pred_TFT.npy")
print(f"  → {MCS_PATH}/y_test_TFT.npy")