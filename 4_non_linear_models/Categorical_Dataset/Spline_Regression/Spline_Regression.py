import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from patsy import dmatrix, build_design_matrices
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & DATA LOADING ---
base_path = '/Users/marcel/PycharmProjects/Uni_Pavia'
input_data = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv')
consensus_file = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/DeFi_Consensus_Variables_4_Tiers.csv')
poly_sel_file = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/3_Non_Linear_Models/Categorical_Dataset/Polynomial_Selection/polynomial_degree_selection_summary.csv')
output_dir = os.path.join(base_path, 'Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/3_Non_Linear_Models/Categorical_Dataset/Spline_Regression')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load datasets
df = pd.read_csv(input_data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

consensus_df = pd.read_csv(consensus_file)
poly_sel_df = pd.read_csv(poly_sel_file)

# Create Mappings
tier_map = dict(zip(consensus_df['Variable'], consensus_df['Tier']))
degree_map = dict(zip(poly_sel_df['Variable'], poly_sel_df['Perfect_Degree']))
target_vars = consensus_df[consensus_df['Vote_Count'] >= 3]['Variable'].tolist()

# --- 2. TEMPORAL SPLIT (80/20) ---
split_idx = int(len(df) * 0.8)
df_dev = df.iloc[:split_idx].reset_index(drop=True)
df_holdout = df.iloc[split_idx:].reset_index(drop=True)
y_col = 'Global_TVL_USD_log_return'

# --- 3. DYNAMIC FORMULA CONSTRUCTION ---
formula_parts = []
for var in target_vars:
    deg = degree_map.get(var, 1)
    if deg > 1:
        # cr() creates Natural Cubic Splines; df=3 is the standard baseline
        formula_parts.append(f"cr(Q('{var}'), df=3)")
    else:
        formula_parts.append(f"Q('{var}')")

formula = " + ".join(formula_parts)

# --- 4. MODEL EVALUATION ---
X_train_spline = dmatrix(formula, df_dev, return_type='dataframe')
d_info = X_train_spline.design_info
X_holdout_spline = pd.DataFrame(build_design_matrices([d_info], df_holdout)[0], columns=d_info.column_names)

scaler = StandardScaler()
cols = [c for c in X_train_spline.columns if c != 'Intercept']
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_spline[cols]), columns=cols)
X_holdout_scaled = pd.DataFrame(scaler.transform(X_holdout_spline[cols]), columns=cols)

X_train_final = sm.add_constant(X_train_scaled)
X_holdout_final = sm.add_constant(X_holdout_scaled, has_constant='add')

# Jitter for numerical stability
X_train_final += np.random.normal(0, 1e-9, size=X_train_final.shape)

# Fitting the Spline Logistic Regression
model = sm.Logit(df_dev[y_col], X_train_final).fit(method='bfgs', maxiter=500, disp=False)
holdout_probs = model.predict(X_holdout_final)
holdout_pred = (holdout_probs > 0.5).astype(int)

# --- 5. VISUALIZATION SUITE ---
def plot_results():
    # Metrics calculation
    acc = accuracy_score(df_holdout[y_col], holdout_pred)
    err = 1 - acc
    cm = confusion_matrix(df_holdout[y_col], holdout_pred)
    fpr, tpr, _ = roc_curve(df_holdout[y_col], holdout_probs)
    roc_auc = auc(fpr, tpr)

    # FIGURE 8: Performance Dashboard (CM + ROC AUC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Confusion Matrix (Color legend removed)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax1,
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=['Gain (0)', 'Stress (1)'],
                yticklabels=['Gain (0)', 'Stress (1)'],
                cbar=False)
    ax1.set_title(f'A: Confusion Matrix\nAccuracy: {acc:.2%} | Error: {err:.2%}',
                  fontweight='bold', fontsize=13)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.grid(False)

    # ROC Curve (Updated label to 4 decimals: .4f)
    ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (area = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (1 - Specificity)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.set_title('B: ROC Curve (Holdout Set)', fontweight='bold', fontsize=13)
    ax2.legend(loc="lower right")
    ax2.grid(False)

    plt.suptitle('Figure 8: Performance Dashboard - Spline Logistic Regression',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spline_performance_dashboard.png'), dpi=300)

    # FIGURE 9: 3x3 Marginal Effects Grid
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    for i, var in enumerate(target_vars[:9]):
        x_range = np.linspace(df_dev[var].min(), df_dev[var].max(), 100)
        dummy_df = pd.DataFrame({v: [df_dev[v].mean()] * 100 for v in target_vars})
        dummy_df[var] = x_range

        dummy_spline = pd.DataFrame(build_design_matrices([d_info], dummy_df)[0], columns=d_info.column_names)
        dummy_final = sm.add_constant(
            pd.DataFrame(scaler.transform(dummy_spline.drop(columns='Intercept')), columns=cols), has_constant='add')

        marginal_probs = model.predict(dummy_final)

        axes[i].plot(x_range, marginal_probs, color='darkblue', lw=2.5)
        axes[i].axhline(0.5, color='red', linestyle=':', alpha=0.5)

        tier = tier_map.get(var, "Satellite")
        poly_deg = degree_map.get(var, 1)
        axes[i].set_title(f"{var}\n{tier} | Polynomial Degree: {poly_deg}", fontsize=12, fontweight='bold')

        bg_color = '#f0f8ff' if poly_deg > 1 else '#ffffff'
        axes[i].set_facecolor(bg_color)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_ylabel("P(DeFi Stress)")
        axes[i].grid(False)

    plt.suptitle('Figure 9: Marginal Effects using Pre-Selected Polynomial Degrees',
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '9_variable_effects_grid.png'), dpi=300)

# Run visualization
plot_results()

print("Success: Spline Analysis Complete. Files saved to output directory.")