import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH  = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures"

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

# ─────────────────────────────────────────────
# TFT REQUIRES:
#   (1) integer time index
#   (2) a group identifier (single series → constant group)
#   (3) covariate role declaration. All 38 exogenous features are declared
#       as time_varying_KNOWN_reals: their contemporaneous value X_t is
#       available at the moment y_t is forecast, mirroring the
#       contemporaneous-regressor setup of the MLR / Elastic Net / GAM /
#       XGBoost / SVR / ARX models. Only the target is unknown in the
#       decoder (lagged target enters via the encoder → nonlinear ARX-type
#       structure). This keeps the information set identical across all
#       models, so the concluding Diebold-Mariano comparison is fair.
# ─────────────────────────────────────────────
df_model["time_idx"] = np.arange(len(df_model))
df_model["group"]    = "TVL"

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT  (80 / 20, chronological)
# ─────────────────────────────────────────────
split_idx = int(len(df_model) * 0.80)

df_train = df_model.iloc[:split_idx].copy()   # UNSCALED — scaling happens per fold
df_test  = df_model.iloc[split_idx:].copy()

y_train   = df_model[TARGET].values[:split_idx]
n_train_n = len(df_train)
n_test_n  = len(df_test)

# ─────────────────────────────────────────────
# SCALING NOTE
# Scaling is performed INSIDE each CV fold (StandardScaler fit on the
# fold-training block only), identical to the Elastic Net / SVR / GAM
# scripts. This prevents leakage across blocks; no global scaler is fit
# on the full training set here.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# HELPER: BLOCKED CV INDICES
# ─────────────────────────────────────────────
N_SPLITS = 5

def blocked_cv_indices(n, n_splits):
    """Yield (train_idx, val_idx) pairs for blocked time-series CV."""
    block_size = n // n_splits
    for i in range(n_splits):
        val_start = i * block_size
        val_end   = val_start + block_size if i < n_splits - 1 else n
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([
            np.arange(0, val_start),
            np.arange(val_end, n)
        ])
        yield train_idx, val_idx

# ─────────────────────────────────────────────
# HELPER: SPLIT PREDICTION OUTPUT / INDEX
# ─────────────────────────────────────────────
def _split_pred(pred_obj):
    """Return (flat_predictions, index_df) from a pytorch-forecasting predict() call."""
    out = pred_obj.output if hasattr(pred_obj, "output") else pred_obj[0]
    idx = pred_obj.index  if hasattr(pred_obj, "index")  else pred_obj[1]
    out = out.cpu().numpy().flatten() if hasattr(out, "cpu") else np.asarray(out).flatten()
    return out, idx

# ─────────────────────────────────────────────
# STAGE 2 — HYPERPARAMETER TUNING
# 5-fold blocked CV on training set.
# Hyperparameters tuned:
#   max_encoder_length : lookback window (temporal context for TFT)
#   hidden_size        : dimensionality of hidden state / embeddings
#   attention_head_size: number of multi-head attention heads
#   dropout            : dropout rate for regularization
# max_prediction_length is fixed at 1 (one-step-ahead forecasting).
# learning_rate and batch_size are fixed to reduce search space.
# Loss is MAE — identical to the final model (Stage 2 selection and the
# final fit therefore optimise the same objective).
# ─────────────────────────────────────────────
encoder_lengths = [5, 10, 15]
hidden_sizes    = [16, 32]
attention_heads = [1, 2]
dropout_rates   = [0.1, 0.2]

MAX_PREDICTION_LENGTH = 1
LEARNING_RATE         = 1e-3
BATCH_SIZE            = 64
MAX_EPOCHS_CV         = 15      # kept low for CV speed; full model uses more

best_rmse      = np.inf
best_params    = {}
cv_results_all = []

print("Running 5-fold blocked CV for TFT hyperparameter tuning ...")

for enc_len in encoder_lengths:
    for hid in hidden_sizes:
        for heads in attention_heads:
            for drop in dropout_rates:

                fold_rmses = []

                for fold_i, (tr_idx, val_idx) in enumerate(blocked_cv_indices(n_train_n, N_SPLITS)):

                    # Val block must hold encoder context + at least one prediction point
                    if len(val_idx) <= enc_len + MAX_PREDICTION_LENGTH:
                        continue

                    # ── PER-FOLD SCALING (fit on fold-train only) ──
                    df_tr_raw  = df_train.iloc[tr_idx].copy().reset_index(drop=True)
                    df_val_raw = df_train.iloc[val_idx].copy().reset_index(drop=True)

                    sc_fold = StandardScaler().fit(df_tr_raw[FEATURES].values)
                    y_mu = df_tr_raw[TARGET].mean()
                    y_sd = df_tr_raw[TARGET].std()

                    df_tr  = df_tr_raw.copy()
                    df_val = df_val_raw.copy()
                    df_tr[FEATURES]  = sc_fold.transform(df_tr_raw[FEATURES].values)
                    df_val[FEATURES] = sc_fold.transform(df_val_raw[FEATURES].values)
                    df_tr[TARGET]  = (df_tr_raw[TARGET]  - y_mu) / y_sd
                    df_val[TARGET] = (df_val_raw[TARGET] - y_mu) / y_sd

                    # Re-index time_idx within fold to be contiguous
                    df_tr["group"]     = "TVL"
                    df_val["group"]    = "TVL"
                    df_tr["time_idx"]  = np.arange(len(df_tr))
                    df_val["time_idx"] = np.arange(len(df_tr), len(df_tr) + len(df_val))

                    df_fold   = pd.concat([df_tr, df_val], ignore_index=True)
                    val_start = len(df_tr)   # first validation time_idx

                    try:
                        training_ds = TimeSeriesDataSet(
                            df_fold[df_fold["time_idx"] <= df_tr["time_idx"].max()],
                            time_idx                  = "time_idx",
                            target                    = TARGET,
                            group_ids                 = ["group"],
                            max_encoder_length        = enc_len,
                            max_prediction_length     = MAX_PREDICTION_LENGTH,
                            time_varying_unknown_reals= [TARGET],     # only the target
                            time_varying_known_reals  = FEATURES,     # 38 IVs contemporaneous
                            target_normalizer         = None,         # pre-scaled
                        )

                        # predict=False → one rolling window per time step (not just the last)
                        validation_ds = TimeSeriesDataSet.from_dataset(
                            training_ds, df_fold, predict=False, stop_randomization=True
                        )

                        train_loader = training_ds.to_dataloader(
                            train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=False
                        )
                        val_loader = validation_ds.to_dataloader(
                            train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False
                        )

                        tft = TemporalFusionTransformer.from_dataset(
                            training_ds,
                            learning_rate          = LEARNING_RATE,
                            hidden_size            = hid,
                            attention_head_size    = heads,
                            dropout                = drop,
                            hidden_continuous_size = max(4, hid // 2),
                            loss                   = MAE(),
                            log_interval           = -1,
                            reduce_on_plateau_patience = 3,
                        )

                        trainer = pl.Trainer(
                            max_epochs          = MAX_EPOCHS_CV,
                            enable_progress_bar = False,
                            enable_model_summary= False,
                            logger              = False,
                            callbacks           = [EarlyStopping(monitor="val_loss", patience=5,
                                                                 mode="min", verbose=False)],
                            accelerator         = "auto",
                        )
                        trainer.fit(tft, train_dataloaders=train_loader,
                                    val_dataloaders=val_loader)

                        # All windows in df_fold are predicted; keep only the val block
                        preds_sc, pred_idx = _split_pred(
                            tft.predict(val_loader, mode="prediction", return_index=True)
                        )
                        m       = pred_idx["time_idx"].values >= val_start
                        sel_t   = pred_idx["time_idx"].values[m]

                        preds_orig = preds_sc[m] * y_sd + y_mu
                        actual_sc  = df_fold.set_index("time_idx")[TARGET].loc[sel_t].values
                        actuals    = actual_sc * y_sd + y_mu

                        fold_rmses.append(np.sqrt(mean_squared_error(actuals, preds_orig)))

                    except Exception as e:
                        print(f"  Fold {fold_i} failed: {e}")
                        continue

                if len(fold_rmses) == 0:
                    continue

                mean_rmse = np.mean(fold_rmses)
                params    = {"enc_len": enc_len, "hidden": hid,
                             "heads": heads, "dropout": drop}
                cv_results_all.append({**params, "cv_rmse": mean_rmse})

                print(f"  enc={enc_len}, hid={hid}, heads={heads}, drop={drop} "
                      f"→ CV RMSE={mean_rmse:.6f}")

                if mean_rmse < best_rmse:
                    best_rmse   = mean_rmse
                    best_params = params.copy()

print(f"\nBest parameters : {best_params}")
print(f"CV RMSE (best)  : {best_rmse:.6f}")

BEST_ENC   = best_params["enc_len"]
BEST_HID   = best_params["hidden"]
BEST_HEADS = best_params["heads"]
BEST_DROP  = best_params["dropout"]

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── CV DONE ───")
print(f"Best enc_len        : {BEST_ENC}")
print(f"Best hidden_size    : {BEST_HID}")
print(f"Best attention_heads: {BEST_HEADS}")
print(f"Best dropout        : {BEST_DROP}")
print(f"CV RMSE (best)      : {best_rmse:.6f}")
print(f"\n>>> Copy these into TFT_Model_Fit.py and TFT_Importance_Plots.py:")
print(f"    BEST_ENC     = {BEST_ENC}")
print(f"    BEST_HID     = {BEST_HID}")
print(f"    BEST_HEADS   = {BEST_HEADS}")
print(f"    BEST_DROP    = {BEST_DROP}")
print(f"    CV_RMSE_BEST = {best_rmse:.6f}")
print(f"\nAll CV results (sorted by CV RMSE):")
for r in sorted(cv_results_all, key=lambda x: x["cv_rmse"]):
    print(f"  enc={r['enc_len']}, hid={r['hidden']}, heads={r['heads']}, "
          f"drop={r['dropout']} → CV RMSE={r['cv_rmse']:.6f}")