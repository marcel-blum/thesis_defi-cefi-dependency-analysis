import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from joblib import Parallel, delayed

# --- 1. SETUP ---
input_path = "/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv"
output_dir = "/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/6_Deep_Learning_Model/Categorical_Dataset/Long_Short_Term_Memory_LSTM"
os.makedirs(output_dir, exist_ok=True)

# --- 2. DATA PREPARATION ---
df = pd.read_csv(input_path)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

feature_names = df.drop(columns=['date', 'Global_TVL_USD_log_return']).columns.tolist()
y = df['Global_TVL_USD_log_return'].values
X_raw = df.drop(columns=['date', 'Global_TVL_USD_log_return']).values

# Split first to avoid Data Leakage
split_idx = int(0.8 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

# Scaling based ONLY on Training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

def create_sequences(data, target, seq_length=7):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i: i + seq_length])
        y_seq.append(target[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

L = 7
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, L)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, L)

# --- 3. WORKER FUNCTION ---
def evaluate_config(config):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.regularizers import l2

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = Sequential()
    for i in range(config['num_layers']):
        is_last_lstm = (i == config['num_layers'] - 1)
        units = config['units'][i] if isinstance(config['units'], tuple) else config['units']
        model.add(LSTM(units, input_shape=(L, X_raw.shape[1]) if i == 0 else None,
                        return_sequences=not is_last_lstm, kernel_regularizer=l2(config['l2_lambda'])))
        model.add(Dropout(config['dropout']))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'],
                                                restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                        validation_split=0.15, callbacks=[callback], verbose=0)

    y_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_probs > config['threshold']).astype("int32")
    acc = accuracy_score(y_test, y_pred)

    return {
        'config': config, 'accuracy': acc, 'history': history.history,
        'y_pred': y_pred, 'y_probs': y_probs, 'model_weights': model.get_weights()
    }

# --- 4. TUNING ---
grid_1 = {
    'units': [(16, 8), (8, 4), (4, 4)],
    'dropout': [0.6, 0.7, 0.8],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64],
    'threshold': [0.5], 'num_layers': [2], 'l2_lambda': [0.05], 'patience': [5], 'epochs': [100]
}

def run_search(grid, n):
    keys, values = zip(*grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    subset = random.sample(combos, min(len(combos), n))
    return Parallel(n_jobs=-2)(delayed(evaluate_config)(c) for c in subset)

print("Starting Search Stage 1...")
res1 = run_search(grid_1, 15)
best_1 = max(res1, key=lambda x: x['accuracy'])['config']

grid_2 = {
    'units': [best_1['units']], 'dropout': [best_1['dropout']],
    'learning_rate': [best_1['learning_rate']], 'batch_size': [best_1['batch_size']],
    'threshold': [0.4, 0.5, 0.6], 'num_layers': [1, 2], 'l2_lambda': [0.1, 0.05, 0.01],
    'patience': [5], 'epochs': [100, 150]
}

print("Starting Search Stage 2...")
res2 = run_search(grid_2, 15)
best_res = max(res2, key=lambda x: x['accuracy'])
conf = best_res['config']

# --- 5. IMPORTANCE ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

champion_model = Sequential()
for i in range(conf['num_layers']):
    is_last_lstm = (i == conf['num_layers'] - 1)
    units = conf['units'][i] if isinstance(conf['units'], tuple) else conf['units']
    champion_model.add(LSTM(units, input_shape=(L, X_raw.shape[1]) if i == 0 else None,
                            return_sequences=not is_last_lstm, kernel_regularizer=l2(conf['l2_lambda'])))
    champion_model.add(Dropout(conf['dropout']))
champion_model.add(Dense(1, activation='sigmoid'))
champion_model.set_weights(best_res['model_weights'])

def get_permutation_importance(model, X_val, y_val, feature_names, threshold):
    baseline_acc = accuracy_score(y_val, (model.predict(X_val, verbose=0) > threshold).astype(int))
    importances = {}
    for i, name in enumerate(feature_names):
        X_permuted = X_val.copy()
        flat_col = X_permuted[:, :, i].flatten()
        np.random.shuffle(flat_col)
        X_permuted[:, :, i] = flat_col.reshape(X_val.shape[0], L)
        perm_acc = accuracy_score(y_val, (model.predict(X_permuted, verbose=0) > threshold).astype(int))
        importances[name] = baseline_acc - perm_acc
    return pd.Series(importances).sort_values(ascending=False)

importance_series = get_permutation_importance(champion_model, X_test, y_test, feature_names, conf['threshold'])
importance_series.to_csv(os.path.join(output_dir, 'feature_importance.csv'))

# --- 6. METRICS & PLOTTING ---
test_acc = best_res['accuracy']
test_error = 1 - test_acc  # CALCULATION OF CLASSIFICATION ERROR
train_acc = best_res['history']['accuracy'][-1]
val_acc = max(best_res['history']['val_accuracy'])
fpr, tpr, _ = roc_curve(y_test, best_res['y_probs'])
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, best_res['y_pred'])

plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
plt.plot(best_res['history']['accuracy'], label=f'Train (Final: {train_acc:.2%})')
plt.plot(best_res['history']['val_accuracy'], label=f'Val (Best: {val_acc:.2%})')
plt.title("I. Training History")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("II. ROC Curve")
plt.legend()

plt.subplot(2, 2, 3)
importance_series.head(15).plot(kind='barh', color='teal')
plt.title("III. Permutation Importance (Top 15)")

plt.subplot(2, 2, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Stress (1)'],
            yticklabels=['Normal (0)', 'Stress (1)'])
# UPDATED TITLE: Includes Accuracy and Classification Error
plt.title(f"IV. Confusion Matrix\nAccuracy: {test_acc:.2%} | Error: {test_error:.2%}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'LSTM_Final_Performance_Dashboard.png'))

# --- NEW: SAVE METRICS TO CSV ---
metrics_data = {
    'Metric': ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Classification Error', 'ROC AUC'],
    'Value': [train_acc, val_acc, test_acc, test_error, roc_auc]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(os.path.join(output_dir, 'model_performance_metrics.csv'), index=False)

# Save Confusion Matrix to CSV
cm_df = pd.DataFrame(cm,
                     index=['Actual Normal (0)', 'Actual Stress (1)'],
                     columns=['Predicted Normal (0)', 'Predicted Stress (1)'])
cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))

# Classification Report
report = classification_report(y_test, best_res['y_pred'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, 'classification_report_full.csv'))

print("\n--- Model Performance Summary ---")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Classification Error: {test_error:.4f}")
print(f"ROC AUC:             {roc_auc:.4f}")
print(f"Best Config: {conf}")