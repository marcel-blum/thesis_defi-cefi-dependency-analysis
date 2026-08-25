"""
TFT Dry-Run Check
─────────────────
Verifies BEFORE the full CV/fit pipeline:
  (1) Whether `target_normalizer=None` is accepted by your installed
      pytorch-forecasting version (the #1 suspected failure point).
  (2) That the pytorch-forecasting 1.7.0 / lightning 2.6.5 / Python 3.13
      stack builds a TimeSeriesDataSet, fits a TFT, and runs predict()
      with the exact schema used in your three TFT scripts.

Runs in well under a minute (200 rows, 2 epochs, tiny model).
Nothing here is used for the thesis — it is purely a smoke test.
"""

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — mirrors your scripts (kept tiny for speed)
# ─────────────────────────────────────────────
INPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
TARGET     = "Global_TVL_USD_log_return"

N_ROWS     = 200    # small slice — just enough for a few rolling windows
ENC_LEN    = 5      # same as BEST_ENC default in your scripts
PRED_LEN   = 1
HID        = 8      # tiny model: we only care that it builds & runs
HEADS      = 1
DROP       = 0.1
MAX_EPOCHS = 2

pl.seed_everything(42, workers=True)

# ─────────────────────────────────────────────
# DATA (first N_ROWS only)
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
FEATURES = [c for c in df.columns if c not in ["date", TARGET]]

df_model = df[["date", TARGET] + FEATURES].dropna().reset_index(drop=True).iloc[:N_ROWS].copy()
df_model["time_idx"] = np.arange(len(df_model))
df_model["group"]    = "TVL"

print(f"Python torch : {torch.__version__}")
print(f"Rows used    : {len(df_model)}  |  Features: {len(FEATURES)}")


# ─────────────────────────────────────────────
# DATASET BUILDER — identical schema to your TFT scripts
# normalizer is the only thing we vary, to test point (1).
# ─────────────────────────────────────────────
def build_dataset(normalizer):
    return TimeSeriesDataSet(
        df_model,
        time_idx                   = "time_idx",
        target                     = TARGET,
        group_ids                  = ["group"],
        max_encoder_length         = ENC_LEN,
        max_prediction_length      = PRED_LEN,
        time_varying_unknown_reals = [TARGET],
        time_varying_known_reals   = FEATURES,
        target_normalizer          = normalizer,
    )


# ─────────────────────────────────────────────
# TEST (1): does target_normalizer=None build?
# Falls back to TorchNormalizer(method="identity") if it raises.
# ─────────────────────────────────────────────
chosen_label = None
training_ds  = None

try:
    training_ds  = build_dataset(None)
    chosen_label = "target_normalizer=None  (accepted ✓)"
    print("\n[1] target_normalizer=None  → ACCEPTED")
except Exception as e:
    print(f"\n[1] target_normalizer=None  → REJECTED ({type(e).__name__}: {e})")
    print("    retrying with TorchNormalizer(method='identity') ...")
    from pytorch_forecasting.data import TorchNormalizer
    training_ds  = build_dataset(TorchNormalizer(method="identity"))
    chosen_label = "TorchNormalizer(method='identity')  (use THIS in your scripts)"
    print("    → ACCEPTED")

# ─────────────────────────────────────────────
# BUILD LOADERS + TINY TFT
# ─────────────────────────────────────────────
train_loader = training_ds.to_dataloader(train=True,  batch_size=32, num_workers=0, shuffle=False)
val_loader   = training_ds.to_dataloader(train=False, batch_size=32, num_workers=0, shuffle=False)

tft = TemporalFusionTransformer.from_dataset(
    training_ds,
    learning_rate          = 1e-3,
    hidden_size            = HID,
    attention_head_size    = HEADS,
    dropout                = DROP,
    hidden_continuous_size = max(4, HID // 2),
    loss                   = MAE(),
    log_interval           = -1,
    reduce_on_plateau_patience = 3,
)

trainer = pl.Trainer(
    max_epochs           = MAX_EPOCHS,
    gradient_clip_val    = 0.1,
    enable_progress_bar  = False,
    enable_model_summary = False,
    logger               = False,
    callbacks            = [EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    accelerator          = "auto",
)

print("\n[2] Fitting tiny TFT (2 epochs) ...")
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("    → fit() completed")

# ─────────────────────────────────────────────
# TEST (3): predict path used in your fit/importance scripts
# ─────────────────────────────────────────────
print("\n[3] Testing predict(mode='prediction', return_index=True) ...")
pred_obj = tft.predict(val_loader, mode="prediction", return_index=True)
out = pred_obj.output if hasattr(pred_obj, "output") else pred_obj[0]
idx = pred_obj.index  if hasattr(pred_obj, "index")  else pred_obj[1]
out = out.cpu().numpy().flatten() if hasattr(out, "cpu") else np.asarray(out).flatten()
print(f"    → predictions: {out.shape[0]}  |  index rows: {len(idx)}")

# ─────────────────────────────────────────────
# TEST (4): raw output + interpret_output (importance path)
# Guarded exactly like your importance script.
# ─────────────────────────────────────────────
print("\n[4] Testing interpret_output (VSN importance path) ...")
try:
    raw = tft.predict(val_loader, mode="raw", return_x=True)
    raw_out = raw.output if hasattr(raw, "output") else raw[0]
    interp  = tft.interpret_output(raw_out, reduction="sum")
    enc = interp["encoder_variables"].cpu().numpy()
    dec = interp["decoder_variables"].cpu().numpy()
    print(f"    → encoder weights: {enc.shape}  |  decoder weights: {dec.shape}")
    print(f"    → encoder_variables: {len(list(tft.encoder_variables))} names")
    print(f"    → decoder_variables: {len(list(tft.decoder_variables))} names")
except Exception as e:
    print(f"    → interpret_output FAILED ({type(e).__name__}: {e})")
    print("      (your importance script has a try/except fallback for this,")
    print("       so the full run won't crash — but importances would be uniform.)")

# ─────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("DRY-RUN PASSED — your stack builds, fits, and predicts.")
print(f"NORMALIZER TO USE: {chosen_label}")
print("─" * 50)