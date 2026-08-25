import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, roc_curve, accuracy_score, roc_auc_score)

# --- 1. SETUP PATHS ---
in_path = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv'
out_dir = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/5_Support_Vector_Machines/Categorical_Dataset/SVM_Radial_Basis_Function_RBF_Kernel'

os.makedirs(out_dir, exist_ok=True)

# --- 2. DATA PREPARATION ---
print("Loading data...")
df = pd.read_csv(in_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

X = df.drop(['date', 'Global_TVL_USD_log_return'], axis=1)
y = df['Global_TVL_USD_log_return']

# 80/20 Chronological Split
n = len(df)
split_idx = int(n * 0.80)
X_train_cv, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_cv, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scaling (Crucial for SVM RBF Kernel)
scaler = StandardScaler()
X_train_cv_s = scaler.fit_transform(X_train_cv)
X_test_s = scaler.transform(X_test)

# --- 3. SLIDING WINDOW CROSS-VALIDATION ---
tscv = TimeSeriesSplit(n_splits=5, max_train_size=365, gap=7)

param_dist = {
    'C': np.logspace(-3, 2, 20),
    'gamma': list(np.logspace(-4, 0, 15)) + ['scale', 'auto'],
    'kernel': ['rbf'],
    'class_weight': ['balanced', None]
}

random_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)

print("Starting Randomized Search with Sliding Window CV...")
random_search.fit(X_train_cv_s, y_train_cv)
best_svm = random_search.best_estimator_

# --- 4. METRICS COMPUTATION ---
y_train_pred = best_svm.predict(X_train_cv_s)
y_test_pred = best_svm.predict(X_test_s)
y_test_prob = best_svm.predict_proba(X_test_s)[:, 1]

train_acc = accuracy_score(y_train_cv, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_error = 1 - test_acc
test_roc_auc = roc_auc_score(y_test, y_test_prob)
cm_test = confusion_matrix(y_test, y_test_pred)

# --- 5. VISUALIZATION (Updated for Framing and No Color Legend) ---
def apply_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.grid(False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# A. Confusion Matrix (cbar=False removes the color legend)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax1,
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title(f'Test Confusion Matrix\nAccuracy: {test_acc:.4f} | Error: {test_error:.4f}', fontweight='bold', pad=10)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
apply_frame(ax1)

# B. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC AUC = {test_roc_auc:.4f}')
ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
ax2.set_title('Test Set ROC Curve', fontweight='bold', pad=10)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc="lower right", frameon=True, edgecolor='black')
apply_frame(ax2)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'SVM_Final_Performance.png'), dpi=300)

# --- 6. EXPORT RESULTS ---
# Results Summary
results_summary = pd.DataFrame({
    'Metric': ['Train Accuracy', 'Mean CV Accuracy', 'Test Accuracy', 'Test Classification Error', 'Test ROC AUC', 'Best C', 'Best Gamma'],
    'Value': [train_acc, random_search.best_score_, test_acc, test_error, test_roc_auc, random_search.best_params_['C'], random_search.best_params_['gamma']]
})
results_summary.to_csv(os.path.join(out_dir, 'SVM_SlidingCV_Metrics.csv'), index=False)

print(f"\nAnalysis complete. Files saved to: {out_dir}")