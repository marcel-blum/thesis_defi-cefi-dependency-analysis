import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
CKPT_DIR   = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures/checkpoints"

# ─────────────────────────────────────────────
# BEST HYPERPARAMETERS  ← UPDATE FROM TFT_Train_CV.py OUTPUT
# Schema changed (known vs unknown reals) → re-run CV before using these.
# ─────────────────────────────────────────────
BEST_ENC     = 5
BEST_HID     = 64
BEST_HEADS   = 2
BEST_DROP    = 0.1
CV_RMSE_BEST = 0.003592   # placeholder — replace with new CV result

MAX_PREDICTION_LENGTH = 1
LEARNING_RATE         = 3e-4   # lower than CV's 1e-3 for stable convergence
BATCH_SIZE            = 64

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET   = "Global_TVL_USD_log_return"
FEATURES = [c for c in df.columns if c not in ["date", TARGET]]

df_model = df[["date", TARGET] + FEATURES].dropna().reset_index(drop=True)
df_model["time_idx"] = np.arange(len(df_model))
df_model["group"]    = "TVL"

split_idx = int(len(df_model) * 0.80)
df_train  = df_model.iloc[:split_idx].copy()
df_test   = df_model.iloc[split_idx:].copy()
y_train   = df_model[TARGET].values[:split_idx]
y_test    = df_model[TARGET].values[split_idx:]

# ─────────────────────────────────────────────
# SCALING
# StandardScaler fit EXCLUSIVELY on training rows (no leakage from test).
# Applied to features and target.
# ─────────────────────────────────────────────
scaler_X = StandardScaler().fit(df_train[FEATURES].values)
y_train_mean = df_train[TARGET].mean()
y_train_std  = df_train[TARGET].std()

df_model_sc = df_model.copy()
df_model_sc[FEATURES] = scaler_X.transform(df_model[FEATURES].values)
df_model_sc[TARGET]   = (df_model[TARGET] - y_train_mean) / y_train_std

df_train_sc = df_model_sc.iloc[:split_idx].copy()
df_test_sc  = df_model_sc.iloc[split_idx:].copy()

# ─────────────────────────────────────────────
# INTERNAL TRAIN / VAL SPLIT FOR EARLY STOPPING
# Last 10% of TRAIN used as ES validation (test set never touched).
# ─────────────────────────────────────────────
es_split = int(len(df_train_sc) * 0.90)
df_inner_train = df_train_sc.iloc[:es_split].copy()
df_inner_val   = df_train_sc.iloc[es_split:].copy()

# ─────────────────────────────────────────────
# BUILD DATASETS
# 38 features → time_varying_known_reals (X_t available at forecast time t),
# target → time_varying_unknown_reals (lagged target only, via encoder).
# This mirrors the contemporaneous-X regression setup of MLR/ENet/GAM/
# XGBoost/SVR/ARX, ensuring identical information sets for DM comparison.
# ─────────────────────────────────────────────
df_full_sc = pd.concat([df_train_sc, df_test_sc], ignore_index=True)
df_full_sc["time_idx"] = np.arange(len(df_full_sc))
inner_train_cutoff = int(df_inner_train["time_idx"].max())
training_cutoff    = int(df_train_sc["time_idx"].max())

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

# Inner val for early stopping (rows from inner_train_cutoff - enc + 1 onwards,
# up to training_cutoff). predict=False → one window per step.
val_ds = TimeSeriesDataSet.from_dataset(
    training_ds,
    df_full_sc[(df_full_sc["time_idx"] >  inner_train_cutoff - BEST_ENC) &
               (df_full_sc["time_idx"] <= training_cutoff)],
    predict=False, stop_randomization=True,
)

train_loader = training_ds.to_dataloader(
    train=True,  batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
val_loader   = val_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# ─────────────────────────────────────────────
# BUILD FINAL MODEL
# MAE loss: identical to CV-stage loss; avoids the median collapse
# observed with QuantileLoss on noisy financial returns.
# gradient_clip_val: prevents exploding gradients.
# ─────────────────────────────────────────────
tft_final = TemporalFusionTransformer.from_dataset(
    training_ds,
    learning_rate              = LEARNING_RATE,
    hidden_size                = BEST_HID,
    attention_head_size        = BEST_HEADS,
    dropout                    = BEST_DROP,
    hidden_continuous_size     = max(4, BEST_HID // 2),
    loss                       = MAE(),
    log_interval               = -1,
    reduce_on_plateau_patience = 3,
)

checkpoint_cb = ModelCheckpoint(
    monitor    = "val_loss",
    mode       = "min",
    save_top_k = 1,
    filename   = "tft_best",
    dirpath    = CKPT_DIR,
)
early_stop_cb = EarlyStopping(
    monitor  = "val_loss",
    patience = 15,
    mode     = "min",
    verbose  = True,
)

trainer_final = pl.Trainer(
    max_epochs           = 150,
    gradient_clip_val    = 0.1,
    enable_progress_bar  = True,
    enable_model_summary = True,
    logger               = False,
    callbacks            = [early_stop_cb, checkpoint_cb],
    accelerator          = "auto",
)

print("Fitting final TFT on inner-train set (last 10% of train held out for ES) ...")
trainer_final.fit(
    tft_final,
    train_dataloaders = train_loader,
    val_dataloaders   = val_loader,
)

print(f"\n─── TRAINING DONE ───")
print(f"Best checkpoint : {checkpoint_cb.best_model_path}")
print(f"Best val_loss   : {checkpoint_cb.best_model_score:.6f}")

# ─────────────────────────────────────────────
# LOAD BEST CHECKPOINT
# ─────────────────────────────────────────────
best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_cb.best_model_path)

# ─────────────────────────────────────────────
# HELPER: SPLIT PREDICTION OUTPUT / INDEX
# ─────────────────────────────────────────────
def _split_pred(pred_obj):
    out = pred_obj.output if hasattr(pred_obj, "output") else pred_obj[0]
    idx = pred_obj.index  if hasattr(pred_obj, "index")  else pred_obj[1]
    out = out.cpu().numpy().flatten() if hasattr(out, "cpu") else np.asarray(out).flatten()
    return out, idx

# ─────────────────────────────────────────────
# IN-SAMPLE PREDICTIONS (full train)
# predict=False → one prediction per training time step.
# ─────────────────────────────────────────────
insample_ds = TimeSeriesDataSet.from_dataset(
    training_ds,
    df_full_sc[df_full_sc["time_idx"] <= training_cutoff],
    predict=False, stop_randomization=True,
)
insample_loader = insample_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

preds_train_sc, idx_train = _split_pred(
    best_tft.predict(insample_loader, mode="prediction", return_index=True)
)
t_idx_train  = idx_train["time_idx"].values
y_pred_train = preds_train_sc * y_train_std + y_train_mean
y_train_al   = df_model[TARGET].values[t_idx_train]

# ─────────────────────────────────────────────
# STAGE 3 — TEST SET PREDICTIONS (rolling-origin, one-step-ahead)
# predict=False → one window per test time step (NOT just the last).
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

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# ─────────────────────────────────────────────
r2_train = r2_score(y_train_al, y_pred_train)
oos_r2   = 1 - np.sum((y_test_al - y_pred_test)**2) / \
               np.sum((y_test_al - np.mean(y_train))**2)

metrics = {
    "train": {
        "R²":   r2_train,
        "RMSE": np.sqrt(mean_squared_error(y_train_al, y_pred_train)),
        "MAE":  mean_absolute_error(y_train_al, y_pred_train),
    },
    "test": {
        "OOS R²": oos_r2,
        "RMSE":   np.sqrt(mean_squared_error(y_test_al, y_pred_test)),
        "MAE":    mean_absolute_error(y_test_al, y_pred_test),
    }
}

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── METRICS ───")
print(f"N Train (used)     : {len(y_train_al)}")
print(f"N Test  (predicted): {len(y_pred_test)}  (full test: {len(df_test)})")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV RMSE (best)     : {CV_RMSE_BEST:.6f}")


# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_TFT.npy",  y_pred_test)
np.save(f"{MCS_PATH}/y_test_TFT.npy",  y_test_al)   # consistency check
print("✓ MCS forecasts saved (TFT).")