import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

# --- 1. SET PATHS ---
input_data = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv'
degree_file = '/Users/marcel/PycharmProjects/Uni_Pavia//Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/3_Non_Linear_Models/Categorical_Dataset/Polynomial_Selection/polynomial_degree_selection_summary.csv'
consensus_file = '/Users/marcel/PycharmProjects/Uni_Pavia//Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/DeFi_Consensus_Variables_4_Tiers.csv'
output_dir = '/Users/marcel/PycharmProjects/Uni_Pavia//Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/3_Non_Linear_Models/Categorical_Dataset/Polynomial_Regression'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. LOAD DATA ---
df = pd.read_csv(input_data)
y_col = 'Global_TVL_USD_log_return'
y = df[y_col]
X_raw = df.drop(columns=['date', y_col], errors='ignore')

poly_selection = pd.read_csv(degree_file)
consensus_df = pd.read_csv(consensus_file)

# Maps for metadata
tier_map = dict(zip(consensus_df['Variable'], consensus_df['Tier']))
degree_map = dict(zip(poly_selection['Variable'], poly_selection['Perfect_Degree']))

# --- 3. CONSTRUCT THE POLYNOMIAL FEATURE MATRIX ---
X_poly = pd.DataFrame()
for index, row in poly_selection.iterrows():
    var = row['Variable']
    degree = int(row['Perfect_Degree'])
    if var in X_raw.columns:
        X_poly[var] = X_raw[var]
        if degree >= 2: X_poly[f'{var}_pow2'] = X_raw[var] ** 2
        if degree >= 3: X_poly[f'{var}_pow3'] = X_raw[var] ** 3

# --- 4. STANDARDIZATION & SPLIT ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_poly), columns=X_poly.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. FIT THE LOGISTIC MODEL ---
X_train_const = sm.add_constant(X_train)
poly_model = sm.Logit(y_train, X_train_const).fit()

# --- 6. PREDICTION & METRICS ---
X_test_const = sm.add_constant(X_test)
y_probs = poly_model.predict(X_test_const)
y_pred = (y_probs > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
class_error = 1 - acc
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# --- 7. EXPORT DATA ---
poly_model.summary2().tables[1].to_csv(os.path.join(output_dir, 'final_poly_coefficients.csv'))

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Classification Error', 'AUC Score', 'True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
    'Value': [acc, class_error, roc_auc, tn, fp, fn, tp]
})
metrics_df.to_csv(os.path.join(output_dir, 'final_poly_performance_metrics.csv'), index=False)

cm_df = pd.DataFrame(cm, index=['True_Gain', 'True_Stress'], columns=['Pred_Gain', 'Pred_Stress'])
cm_df.to_csv(os.path.join(output_dir, 'final_poly_confusion_matrix.csv'))

# --- 8. VISUALIZATION 1: THE 3x3 RISK MAP ---

def plot_marginal_curve(target_var, ax, tier_label, degree):
    x_range = np.linspace(X_scaled[target_var].min(), X_scaled[target_var].max(), 100)
    dummy = pd.DataFrame(np.zeros((100, X_scaled.shape[1])), columns=X_scaled.columns)
    dummy[target_var] = x_range
    if f'{target_var}_pow2' in X_scaled.columns: dummy[f'{target_var}_pow2'] = x_range ** 2
    if f'{target_var}_pow3' in X_scaled.columns: dummy[f'{target_var}_pow3'] = x_range ** 3

    probs = poly_model.predict(sm.add_constant(dummy, has_constant='add'))
    ax.plot(x_range, probs, color='crimson', lw=2.5)
    sns.stripplot(x=X_test[target_var], y=y_test, hue=y_test, ax=ax, orient='h',
                  alpha=0.3, palette='viridis', size=3, jitter=0.08, legend=False)

    if '4-Vote' in str(tier_label):
        ax.set_facecolor('#eef7fa')
        tier_tag = "TIER 4 (CORE)"
    else:
        ax.set_facecolor('#ffffff')
        tier_tag = "TIER 3 (SATELLITE)"

    deg_names = {1: "Linear", 2: "Quadratic", 3: "Cubic"}
    ax.set_title(f'{tier_tag} | {deg_names.get(degree, "Linear")}\n{target_var}', fontsize=9, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='black', linestyle=':', alpha=0.3)
    ax.set_ylabel('Prob(Stress)')
    ax.set_xlabel('Std. Value')

target_list = list(poly_selection['Variable'].unique())
fig_grid, axes = plt.subplots(3, 3, figsize=(22, 18))
axes = axes.flatten()

for i, var in enumerate(target_list):
    if i < len(axes):
        plot_marginal_curve(var, axes[i], tier_map.get(var, "Unknown"), degree_map.get(var, 1))

plt.suptitle(f'Ensemble Polynomial Risk Map: Multi-Dimensional Effects\nClassification Error: {class_error:.2%}',
             fontsize=24, fontweight='bold', y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig(os.path.join(output_dir, '9_polynomial_effects.png'), dpi=300)

# --- 9. VISUALIZATION 2: STANDALONE CONFUSION MATRIX ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
            xticklabels=['Gain (0)', 'Stress (1)'],
            yticklabels=['Gain (0)', 'Stress (1)'])

plt.title(f'Final Model Performance\nAccuracy: {acc:.2%} | Error Rate: {class_error:.2%}',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)

print(f"Analysis complete. 3x3 Grid and Confusion Matrix saved in: {output_dir}")