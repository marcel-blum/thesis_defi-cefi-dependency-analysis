import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

# ==========================================
# 0. Global Style Settings (Harvard Style)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# ==========================================
# 1. Define Paths
# ==========================================
base_path = "/Users/marcel/PycharmProjects/Uni_Pavia"
input_path = os.path.join(base_path, "Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv")
output_dir = os.path.join(base_path, "Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/4_Tree_Models/Categorical_Dataset/Random_Forest")
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. Load and Prepare Dataset
# ==========================================
df = pd.read_csv(input_path)
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col).reset_index(drop=True)
target_col = 'Global_TVL_USD_log_return'

# ==========================================
# 3. Preprocessing & Split (80/20 Chronological)
# ==========================================
y = df[target_col]
X = df.drop(columns=[date_col, target_col]).select_dtypes(include=[np.number])

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_dates = df.iloc[split_idx:][date_col]

# ==========================================
# 4. Training (Optimized Random_Forest_Regression)
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestClassifier(max_features='sqrt', random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [1000, 1200],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [5, 10]
}

random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist,
    n_iter=10, cv=tscv, scoring='accuracy', random_state=42, verbose=1
)

print("Training Optimized Random Forest...")
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Global Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
roc_auc = roc_auc_score(y_test, y_prob)

# Gini Importance Extraction
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Gini_Importance': best_rf.feature_importances_
}).sort_values(by='Gini_Importance', ascending=False)

# Export Full Gini Coefficients
importance_df.to_csv(os.path.join(output_dir, "gini_coefficients_full.csv"), index=False)

# ==========================================
# 5. CONSOLIDATED VISUALIZATION (Framed)
# ==========================================
def apply_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax.grid(False)

print("Generating Framed Figure 10 Dashboard...")
fig = plt.figure(figsize=(24, 8))
gs = GridSpec(1, 3, width_ratios=[0.8, 0.8, 1.2])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# --- Panel A: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
            annot_kws={"size": 16, "weight": "bold"})
ax1.set_title(f'A: Confusion Matrix\nAccuracy: {accuracy:.2%} | Error: {error_rate:.2%}', fontweight='bold', pad=15)
ax1.set_xticklabels(['Gain (0)', 'Stress (1)'])
ax1.set_yticklabels(['Gain (0)', 'Stress (1)'], rotation=0)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
apply_frame(ax1)

# --- Panel B: ROC Curve (Updated to 4 Decimals) ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.4f}') # 4 decimal precision
ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
ax2.set_title('B: ROC Curve', fontweight='bold', pad=15)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc="lower right", frameon=True, edgecolor='black', fancybox=False, framealpha=1)
apply_frame(ax2)

# --- Panel C: Top 15 Feature Importance ---
sns.barplot(x='Gini_Importance', y='Feature', data=importance_df.head(15),
            ax=ax3, palette='viridis', hue='Feature', legend=False)
ax3.set_title('C: Top 15 Predictors (Gini Importance)', fontweight='bold', pad=15)
ax3.set_xlabel('Mean Decrease Gini Score')
ax3.set_ylabel('')
apply_frame(ax3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rf_performance_dashboard_framed.png"), dpi=300)

# ==========================================
# 6. Timeline Plot (Framed)
# ==========================================
plt.figure(figsize=(14, 6))
plt.scatter(test_dates, y_test, color='black', s=15, label='Actual Stress', alpha=0.4)
plt.plot(test_dates, y_prob, color='blue', lw=1, label='Predicted Probability')
plt.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5)
plt.title('Out-of-Sample Probability Timeline', fontweight='bold', pad=15)
plt.ylabel('P(DeFi Stress)')
plt.legend(frameon=True, edgecolor='black', fancybox=False)
apply_frame(plt.gca())

plt.savefig(os.path.join(output_dir, "rf_timeline_fit.png"), dpi=300, bbox_inches='tight')

# ==========================================
# 7. Final Export of Metrics
# ==========================================
metrics_vertical = [
    ('Accuracy', accuracy),
    ('Classification_Error', error_rate),
    ('ROC_AUC_Score', roc_auc),
    ('Best_n_estimators', random_search.best_params_['n_estimators']),
    ('Best_min_leaf', random_search.best_params_['min_samples_leaf'])
]
pd.DataFrame(metrics_vertical, columns=['Metric', 'Value']).to_csv(
    os.path.join(output_dir, "performance_metrics.csv"), index=False)

print(f"\nAnalysis Complete. Files saved to: {output_dir}")