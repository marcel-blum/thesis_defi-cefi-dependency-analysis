import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH  = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures"
CKPT_PATH   = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures/checkpoints/tft_best.ckpt"

# ─────────────────────────────────────────────
# BEST HYPERPARAMETERS  ← UPDATE FROM TFT_Train_CV.py OUTPUT
# Must match the values used in TFT_Model_Fit.py.
# ─────────────────────────────────────────────
BEST_ENC     = 5
BEST_HID     = 64
BEST_HEADS   = 2
BEST_DROP    = 0.1
CV_RMSE_BEST = 0.003592   # placeholder — replace with new CV result

MAX_PREDICTION_LENGTH = 1
LEARNING_RATE         = 3e-4   # must match TFT_Model_Fit.py
BATCH_SIZE            = 64

pl.seed_everything(42, workers=True)

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
n_train_n = len(df_train)
n_test_n  = len(df_test)

# ─────────────────────────────────────────────
# SCALING (train-only fit, no leakage)
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
# BUILD DATASETS
# Schema must match the one used during training (known vs unknown reals).
# ─────────────────────────────────────────────
df_full_sc = pd.concat([df_train_sc, df_test_sc], ignore_index=True)
df_full_sc["time_idx"] = np.arange(len(df_full_sc))
training_cutoff = int(df_train_sc["time_idx"].max())

training_ds = TimeSeriesDataSet(
    df_full_sc[df_full_sc["time_idx"] <= training_cutoff],
    time_idx                   = "time_idx",
    target                     = TARGET,
    group_ids                  = ["group"],
    max_encoder_length         = BEST_ENC,
    max_prediction_length      = MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals = [TARGET],
    time_varying_known_reals   = FEATURES,
    target_normalizer          = None,
)

# Test dataset — predict=False → one window per test time step
test_ds = TimeSeriesDataSet.from_dataset(
    training_ds,
    df_full_sc[df_full_sc["time_idx"] > training_cutoff - BEST_ENC],
    predict=False, stop_randomization=True,
)
# In-sample train dataset
train_insample_ds = TimeSeriesDataSet.from_dataset(
    training_ds,
    df_full_sc[df_full_sc["time_idx"] <= training_cutoff],
    predict=False, stop_randomization=True,
)
# Full dataset for attention extraction
full_ds = TimeSeriesDataSet.from_dataset(
    training_ds, df_full_sc, predict=False, stop_randomization=True,
)

test_loader     = test_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
insample_loader = train_insample_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
full_loader     = full_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# ─────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
print("✓ Checkpoint loaded.")

# ─────────────────────────────────────────────
# HELPER: SPLIT PREDICTION OUTPUT / INDEX
# ─────────────────────────────────────────────
def _split_pred(pred_obj):
    out = pred_obj.output if hasattr(pred_obj, "output") else pred_obj[0]
    idx = pred_obj.index  if hasattr(pred_obj, "index")  else pred_obj[1]
    out = out.cpu().numpy().flatten() if hasattr(out, "cpu") else np.asarray(out).flatten()
    return out, idx

# ─────────────────────────────────────────────
# IN-SAMPLE PREDICTIONS (train)
# ─────────────────────────────────────────────
preds_train_sc, idx_train = _split_pred(
    best_tft.predict(insample_loader, mode="prediction", return_index=True)
)
t_idx_train  = idx_train["time_idx"].values
y_pred_train = preds_train_sc * y_train_std + y_train_mean
y_train_al   = df_model[TARGET].values[t_idx_train]

# ─────────────────────────────────────────────
# STAGE 3 — TEST SET PREDICTIONS (rolling-origin, one-step-ahead)
# ─────────────────────────────────────────────
preds_test_sc, idx_test = _split_pred(
    best_tft.predict(test_loader, mode="prediction", return_index=True)
)
t_idx_test = idx_test["time_idx"].values
mask_test  = t_idx_test > training_cutoff
sel_t_test = t_idx_test[mask_test]

y_pred_test = preds_test_sc[mask_test] * y_train_std + y_train_mean
y_test_al   = df_model[TARGET].values[sel_t_test]
dates_al    = df_model["date"].values[sel_t_test]

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

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

# ─────────────────────────────────────────────
# TABLE: MODEL FIT
# ─────────────────────────────────────────────
sep = {"Metric": "", "Value": ""}

fit_rows = [
    {"Metric": "N (Train)",              "Value": str(n_train_n)},
    {"Metric": "N (Test)",               "Value": str(n_test_n)},
    {"Metric": "Features",               "Value": str(len(FEATURES))},
    {"Metric": "max\\_encoder\\_length", "Value": str(BEST_ENC)},
    {"Metric": "hidden\\_size",          "Value": str(BEST_HID)},
    {"Metric": "attention\\_head\\_size","Value": str(BEST_HEADS)},
    {"Metric": "dropout",                "Value": str(BEST_DROP)},
    {"Metric": "learning\\_rate",        "Value": str(LEARNING_RATE)},
    {"Metric": "batch\\_size",           "Value": str(BATCH_SIZE)},
    sep,
    {"Metric": "R² (Train)",             "Value": fmt4(metrics['train']['R²'])},
    {"Metric": "RMSE (Train)",           "Value": fmt4(metrics['train']['RMSE'])},
    {"Metric": "MAE (Train)",            "Value": fmt4(metrics['train']['MAE'])},
    {"Metric": "OOS R² (Test)",          "Value": fmt4(metrics['test']['OOS R²'])},
    {"Metric": "RMSE (Test)",            "Value": fmt4(metrics['test']['RMSE'])},
    {"Metric": "MAE (Test)",             "Value": fmt4(metrics['test']['MAE'])},
    {"Metric": "CV RMSE (best)",         "Value": fmt4(CV_RMSE_BEST)},
]

fit_df  = pd.DataFrame(fit_rows)
fit_tex = to_apa_latex_table(
    fit_df,
    note=(
        r"\textit{Note}: Temporal Fusion Transformer (PyTorch Forecasting), MAE loss. "
        r"The 38 exogenous features are declared as \texttt{time\_varying\_known\_reals} "
        r"(contemporaneous $X_t$ available at the forecast horizon), the target as "
        r"\texttt{time\_varying\_unknown\_reals} (lagged target via the encoder only). "
        r"This mirrors the contemporaneous-regressor information set used by MLR, "
        r"Elastic Net, GAM, XGBoost, SVR, and ARX, ensuring identical information sets "
        r"across models for the concluding Diebold-Mariano comparison. "
        r"Features and target standardized prior to estimation; scaler fitted exclusively on "
        r"the training set. "
        r"Hyperparameters selected via 5-fold blocked cross-validation on the training set "
        r"(enc\_len $\in \{5, 10, 15\}$, hidden\_size $\in \{16, 32\}$, "
        r"attention\_heads $\in \{1, 2\}$, dropout $\in \{0.1, 0.2\}$; 24 combinations) "
        r"under MAE loss, identical to the final-model objective. "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead) using "
        r"the full test window (one prediction per test observation). "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
)
with open(f"{OUTPUT_PATH}/tft_fit.tex", "w") as f:
    f.write(fit_tex)
print("✓ Fit table saved.")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# With time_varying_known_reals, the Variable Selection Network operates
# at TWO points in the architecture: (i) the encoder, processing past
# observations, and (ii) the decoder, processing the known covariates at
# the forecast horizon. Reporting only encoder weights would systematically
# understate the contribution of contemporaneous X_t; both networks are
# therefore extracted and combined.
# Variable names are taken from best_tft.encoder_variables / decoder_variables
# (robust against TFT's internal reordering of inputs).
# ─────────────────────────────────────────────
try:
    raw_result = best_tft.predict(full_loader, mode="raw", return_x=True)
    output     = raw_result.output if hasattr(raw_result, "output") else raw_result[0]
    interp     = best_tft.interpret_output(output, reduction="sum")

    enc_w   = interp["encoder_variables"].cpu().numpy()
    dec_w   = interp["decoder_variables"].cpu().numpy()
    enc_var = list(best_tft.encoder_variables)
    dec_var = list(best_tft.decoder_variables)

    enc_df = pd.DataFrame({"Variable": enc_var, "Encoder": enc_w})
    dec_df = pd.DataFrame({"Variable": dec_var, "decoder": dec_w})
    importance_df = (
        pd.merge(enc_df, dec_df, on="Variable", how="outer")
          .fillna(0.0)
    )
    # Combined = mean of normalized encoder + decoder weights (each sums to 1)
    if importance_df["Encoder"].sum() > 0:
        importance_df["Encoder_n"] = importance_df["Encoder"] / importance_df["Encoder"].sum()
    else:
        importance_df["Encoder_n"] = 0.0
    if importance_df["decoder"].sum() > 0:
        importance_df["decoder_n"] = importance_df["decoder"] / importance_df["decoder"].sum()
    else:
        importance_df["decoder_n"] = 0.0
    importance_df["Importance"] = (importance_df["Encoder_n"] + importance_df["decoder_n"]) / 2
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print("✓ Feature importance extracted (encoder + decoder VSN).")

except Exception as e:
    print(f"  Importance extraction failed ({e}); writing uniform fallback.")
    all_vars = [TARGET] + FEATURES
    importance_df = pd.DataFrame({
        "Variable":   all_vars,
        "Encoder":    [0.0] * len(all_vars),
        "decoder":    [0.0] * len(all_vars),
        "Encoder_n":  [1.0 / len(all_vars)] * len(all_vars),
        "decoder_n":  [1.0 / len(all_vars)] * len(all_vars),
        "Importance": [1.0 / len(all_vars)] * len(all_vars),
    })

# ── Helper: build rows from a given sort order ──────────────────────────────
def _imp_rows(df_sorted):
    return [{
        "Variable": row["Variable"],
        "Encoder":  fmt4(row['Encoder_n']),
        "decoder":  fmt4(row['decoder_n']),
        "Combined": fmt4(row['Importance']),
    } for _, row in df_sorted.iterrows()]

NOTE_BASE = (
    r"\textit{Note}: Variable importance derived from the TFT Variable Selection "
    r"Networks. \textit{Encoder} weights reflect the network's allocation of "
    r"attention to lagged inputs (encoder context). \textit{decoder} weights reflect "
    r"the allocation to contemporaneous known covariates at the forecast horizon. "
    r"Each is normalized to sum to one across its respective input set. "
    r"\textit{Combined} = arithmetic mean of normalized encoder and decoder weights. "
)

# ── Table 1: sorted by Combined (descending) ────────────────────────────────
df_by_combined = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
imp_tex_combined = to_apa_latex_table(
    pd.DataFrame(_imp_rows(df_by_combined)),
    note=NOTE_BASE + r"Sorted descending by \textit{Combined} importance."
)
with open(f"{OUTPUT_PATH}/tft_importance_combined.tex", "w") as f:
    f.write(imp_tex_combined)
print("✓ Importance table (sorted by Combined) saved.")

# ── Table 2: sorted by Encoder (descending) ─────────────────────────────────
df_by_encoder = importance_df.sort_values("Encoder_n", ascending=False).reset_index(drop=True)
imp_tex_encoder = to_apa_latex_table(
    pd.DataFrame(_imp_rows(df_by_encoder)),
    note=NOTE_BASE + r"Sorted descending by \textit{Encoder} weight."
)
with open(f"{OUTPUT_PATH}/tft_importance_encoder.tex", "w") as f:
    f.write(imp_tex_encoder)
print("✓ Importance table (sorted by Encoder) saved.")

# ── Table 3: sorted by decoder (descending) ─────────────────────────────────
df_by_decoder = importance_df.sort_values("decoder_n", ascending=False).reset_index(drop=True)
imp_tex_decoder = to_apa_latex_table(
    pd.DataFrame(_imp_rows(df_by_decoder)),
    note=NOTE_BASE + r"Sorted descending by \textit{decoder} weight."
)
with open(f"{OUTPUT_PATH}/tft_importance_decoder.tex", "w") as f:
    f.write(imp_tex_decoder)
print("✓ Importance table (sorted by decoder) saved.")

# ─────────────────────────────────────────────
# PLOT: ACTUAL vs PREDICTED (test set)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates_al, y_test_al,  color="black",   linewidth=0.8, label="Actual")
ax.plot(dates_al, y_pred_test, color="#B22222", linewidth=0.8, linestyle="--", label="Predicted")
ax.set_ylabel("Log Return", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/tft_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# PLOT: COMBINED FEATURE IMPORTANCE (top 20)
# ─────────────────────────────────────────────
top_n   = 20
plot_df = importance_df.head(top_n).sort_values("Importance")
fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
ax.barh(plot_df["Variable"], plot_df["Importance"], color="#2C3E50", edgecolor="none")
ax.set_xlabel("Variable Selection Weight (Combined Encoder + decoder)", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/tft_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Feature importance plot saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"N Test predicted   : {len(y_pred_test)}  (full test: {n_test_n})")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV RMSE (best)     : {CV_RMSE_BEST:.6f}")
print(f"\nTop 5 features by combined importance:")
for _, row in importance_df.head(5).iterrows():
    print(f"  {row['Variable']:<45} {row['Importance']:.4f}")