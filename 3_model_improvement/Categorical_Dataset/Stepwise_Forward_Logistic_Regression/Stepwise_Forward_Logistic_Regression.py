import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, auc)

# --- 1. SET PATHS ---
base_path = '/Users/marcel/PycharmProjects/Uni_Pavia'
input_file = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv')
output_dir = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Stepwise_Forward_Logistic_Regression')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. LOAD DATA ---
df = pd.read_csv(input_file)
y = df['Global_TVL_USD_log_return']
X = df.drop(columns=['date', 'Global_TVL_USD_log_return'])

# Drop constant variables
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 4. FORWARD STEPWISE SELECTION FUNCTION ---
def forward_stepwise_selection(X, y, threshold_in=0.05):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit(disp=0)
            new_pval[new_col] = model.pvalues[new_col]

        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print(f'Added: {best_feature:30} (p-value: {best_pval:.6f})')
        if not changed:
            break
    return included


print("Starting Stepwise Forward Selection...")
selected_features = forward_stepwise_selection(X_train, y_train)

# --- 5. FINAL MODEL FITTING ---
X_train_selected = sm.add_constant(X_train[selected_features])
X_test_selected = sm.add_constant(X_test[selected_features])

final_logit = sm.Logit(y_train, X_train_selected).fit()

# Predictions & Probabilities
y_probs = final_logit.predict(X_test_selected)
y_pred = (y_probs > 0.5).astype(int)

# --- 5.1 CALCULATE LOGIT SCORES FOR CURVE ---
# Linear Predictor: XB
logit_scores = np.dot(X_test_selected, final_logit.params)

# --- 6. CALCULATE METRICS ---
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
total = len(y_test)

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
acc = accuracy_score(y_test, y_pred)
class_error = 1 - acc

# --- 7. EXPORT DATA ---
perf_metrics = {
    'Metric': ['AUC Score', 'Accuracy', 'Classification Error', 'TN', 'FP', 'FN', 'TP'],
    'Value': [roc_auc, acc, class_error, tn, fp, fn, tp]
}
pd.DataFrame(perf_metrics).to_csv(os.path.join(output_dir, 'forward_performance_metrics.csv'), index=False)
final_logit.summary2().tables[1].to_csv(os.path.join(output_dir, 'forward_coefficients.csv'))

# --- 8. VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

# A. Confusion Matrix with Classification Error
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax1, annot_kws={"size": 14})
ax1.set_title(f'Confusion Matrix: Stepwise Forward\nAccuracy: {acc:.2%} | Error: {class_error:.2%}', fontsize=15, fontweight='bold')
ax1.set_xlabel('Predicted Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_ylabel('True Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_xticklabels(['Gain (0)', 'Loss (1)'])
ax1.set_yticklabels(['Gain (0)', 'Loss (1)'])

# B. ROC Curve
ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'Forward ROC (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_title('ROC Curve: Predictive Power', fontsize=15, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.legend(loc="lower right", fontsize=11)

# C. STEPWISE LOGISTIC REGRESSION CURVE
sort_idx = np.argsort(logit_scores)
logit_sorted = logit_scores[sort_idx]
probs_sorted = y_probs.values[sort_idx]

# Plot actual binary outcomes from test set
ax3.scatter(logit_scores, y_test, alpha=0.3, color='darkorange', label='Actual Test Points', edgecolors='k')
# Plot the probability curve (Sigmoid)
ax3.plot(logit_sorted, probs_sorted, color='red', lw=3, label='Forward Sigmoid Curve')
ax3.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Decision Boundary (0.5)')

ax3.set_title('Stepwise Probability Curve & Data', fontsize=15, fontweight='bold')
ax3.set_xlabel('Logit Score (Selected Feature Combination)', fontsize=12)
ax3.set_ylabel('Predicted Probability of Stress (1)', fontsize=12)
ax3.legend(loc="best", fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save final image
plot_filename = os.path.join(output_dir, 'forward_performance_metrics.png')
plt.savefig(plot_filename, dpi=300)

print(f"\nStepwise Analysis Complete. Features Selected: {len(selected_features)}")
print(f"Final Test Accuracy: {acc:.4f} | Classification Error: {class_error:.4f}")