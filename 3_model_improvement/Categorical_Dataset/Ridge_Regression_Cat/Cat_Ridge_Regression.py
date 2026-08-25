import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc)

# --- 1. SET PATHS ---
base_path = '/Users/marcel/PycharmProjects/Uni_Pavia'
input_file = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv')
output_dir = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Ridge_Regression_Cat')

os.makedirs(output_dir, exist_ok=True)

# --- 2. LOAD & PREPARE DATA ---
df = pd.read_csv(input_file)
y_col = 'Global_TVL_USD_log_return'
X = df.drop(columns=['date', y_col])
y = df[y_col]

# Standardize features for Ridge (essential for fair penalty application)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-Series Split (80/20 split)
split_idx = int(len(df) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 3. TUNE RIDGE USING CV (L2 Penalty) ---
tscv = TimeSeriesSplit(n_splits=5)

# l1_ratios=[0.0] specifies Pure Ridge (L2)
ridge_cv = LogisticRegressionCV(
    l1_ratios=[0.0],
    Cs=50,
    cv=tscv,
    scoring='accuracy',
    class_weight='balanced',
    solver='saga',
    max_iter=20000,
    random_state=42,
    use_legacy_attributes=False
)
ridge_cv.fit(X_train, y_train)

# Extract best tuning parameters
best_c = float(np.atleast_1d(ridge_cv.C_)[0])
best_lambda = 1 / best_c

# --- 4. EVALUATE ON HOLD-OUT TEST SET ---
y_pred = ridge_cv.predict(X_test)
y_probs = ridge_cv.predict_proba(X_test)[:, 1]
# Decision function provides the Logit Scores (standardized beta * X)
logit_scores = ridge_cv.decision_function(X_test)

acc = accuracy_score(y_test, y_pred)
class_error = 1 - acc
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
total = len(y_test)

# --- 5. EXPORT RESULTS ---
perf_metrics = {
    'Metric': [
        'Best C (Inverse Lambda)', 'Best Lambda', 'AUC Score', 'Accuracy', 'Classification Error',
        'True Negatives (TN)', 'False Positives (FP)',
        'False Negatives (FN)', 'True Positives (TP)',
        'TN %', 'FP %', 'FN %', 'TP %'
    ],
    'Value': [
        best_c, best_lambda, roc_auc, acc, class_error,
        tn, fp, fn, tp,
        (tn / total) * 100, (fp / total) * 100,
        (fn / total) * 100, (tp / total) * 100
    ]
}
pd.DataFrame(perf_metrics).to_csv(os.path.join(output_dir, 'ridge_tuned_metrics.csv'), index=False)

coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Standardized_Coefficient': ridge_cv.coef_[0]
}).sort_values(by='Standardized_Coefficient', ascending=False)
coef_df.to_csv(os.path.join(output_dir, 'ridge_tuned_coefficients.csv'), index=False)

# --- 6. VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

# A. Confusion Matrix with Accuracy and Error
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1, annot_kws={"size": 14})
ax1.set_title(f'Confusion Matrix: Tuned Ridge\nAccuracy: {acc:.2%} | Error: {class_error:.2%}', fontsize=16, fontweight='bold')
ax1.set_xlabel('Predicted Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_ylabel('True Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_xticklabels(['Stable (0)', 'Stress (1)'])
ax1.set_yticklabels(['Stable (0)', 'Stress (1)'])

# B. ROC Curve
ax2.plot(fpr, tpr, color='rebeccapurple', lw=3, label=f'Ridge ROC curve (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve: Predictive Power', fontsize=16, fontweight='bold')
ax2.legend(loc="lower right", fontsize=12)

# C. RIDGE LOGISTIC SIGMOID CURVE
# Sort scores to ensure a smooth line
sort_idx = np.argsort(logit_scores)
logit_sorted = logit_scores[sort_idx]
probs_sorted = y_probs[sort_idx]

# Plot actual data points
ax3.scatter(logit_scores, y_test, alpha=0.3, color='rebeccapurple', label='Actual Test Points', edgecolors='k')
# Plot the probability curve
ax3.plot(logit_sorted, probs_sorted, color='red', lw=3, label='Ridge Probability Curve')
# Add decision boundary
ax3.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Decision Boundary (0.5)')

ax3.set_title('Ridge Probability Curve & Test Data', fontsize=16, fontweight='bold')
ax3.set_xlabel('Logit Score (Standardized Beta * X)', fontsize=12)
ax3.set_ylabel('Predicted Probability of Stress (1)', fontsize=12)
ax3.legend(loc="best", fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save final image
plot_filename = os.path.join(output_dir, 'ridge_tuned_diagnostics.png')
plt.savefig(plot_filename, dpi=300)

print(f"Analysis Complete. Ridge results saved to: {output_dir}")
print(f"Best Lambda: {best_lambda:.4f}")
print(f"Test Accuracy: {acc:.4f} | Classification Error: {class_error:.4f}")