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
output_dir = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Stepwise_Backward_Logistic_Regression')

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


# --- 4. BACKWARD ELIMINATION FUNCTION ---
def backward_elimination(X, y, threshold_out=0.05):
    """
    Backward elimination based on p-values using sm.Logit.
    Starts with all variables and removes the one with max p-value > threshold_out.
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            print(f'Removed: {worst_feature:35} (p-value: {worst_pval:.6f})')

        if not changed:
            break
    return included


print("Starting Backward Elimination...")
selected_features = backward_elimination(X_train, y_train)

# --- 5. FINAL MODEL FITTING ---
X_train_final = sm.add_constant(X_train[selected_features])
X_test_final = sm.add_constant(X_test[selected_features])

final_logit = sm.Logit(y_train, X_train_final).fit()

# Predictions & Probabilities
y_probs = final_logit.predict(X_test_final)
y_pred = (y_probs > 0.5).astype(int)

# --- 5.1 CALCULATE LOGIT SCORES FOR CURVE ---
# The logit score (linear predictor) is the log-odds: XB
# We calculate this manually from the final model coefficients
logit_scores = np.dot(X_test_final, final_logit.params)

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
pd.DataFrame(perf_metrics).to_csv(os.path.join(output_dir, 'backward_performance_metrics.csv'), index=False)
final_logit.summary2().tables[1].to_csv(os.path.join(output_dir, 'backward_coefficients.csv'))

# --- 8. VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

# A. Confusion Matrix with Classification Error
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1, annot_kws={"size": 14})
ax1.set_title(f'Confusion Matrix: Backward Logistic\nAccuracy: {acc:.2%} | Error: {class_error:.2%}', fontsize=15, fontweight='bold')
ax1.set_xlabel('Predicted Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_ylabel('True Label (0=Gain, 1=Loss)', fontsize=12)
ax1.set_xticklabels(['Gain (0)', 'Loss (1)'])
ax1.set_yticklabels(['Gain (0)', 'Loss (1)'])

# B. ROC Curve
ax2.plot(fpr, tpr, color='rebeccapurple', lw=3, label=f'Backward ROC (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_title('ROC Curve: Predictive Power', fontsize=15, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.legend(loc="lower right", fontsize=11)

# C. BACKWARD LOGISTIC REGRESSION CURVE
sort_idx = np.argsort(logit_scores)
logit_sorted = logit_scores[sort_idx]
probs_sorted = y_probs.values[sort_idx] # y_probs is a pandas Series

# Plot actual test points
ax3.scatter(logit_scores, y_test, alpha=0.3, color='rebeccapurple', label='Actual Test Points', edgecolors='k')
# Plot the probability curve based on selected features
ax3.plot(logit_sorted, probs_sorted, color='red', lw=3, label='Stepwise Sigmoid Curve')
ax3.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Decision Boundary')

ax3.set_title('Stepwise Probability Curve & Data', fontsize=15, fontweight='bold')
ax3.set_xlabel('Logit Score (Linear Combination of Selected Features)', fontsize=12)
ax3.set_ylabel('Predicted Probability of Stress (1)', fontsize=12)
ax3.legend(loc="best", fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'backward_performance_metrics.png'), dpi=300)

print(f"\nBackward Analysis Complete. Selected {len(selected_features)} features.")
print(f"Final Test Accuracy: {acc:.4f} | Classification Error: {class_error:.4f}")