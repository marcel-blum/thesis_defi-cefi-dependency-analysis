import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# --- 1. SET PATHS ---
input_file = '/Master_Thesis/Statistics_for_Finance/Exam/Data/Final_Dataset_2020_2025_categorical_DV.csv'
consensus_file = '/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/DeFi_Consensus_Variables_4_Tiers.csv'
output_dir = '/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/3_Non_Linear_Models/Categorical_Dataset/Polynomial_Selection'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. LOAD DATA ---
df = pd.read_csv(input_file)
y = df['Global_TVL_USD_log_return']
X = df.drop(columns=['date', 'Global_TVL_USD_log_return'])

# Load Consensus and identify target variables (Tiers 3 and 4)
consensus_df = pd.read_csv(consensus_file)
target_vars = consensus_df[consensus_df['Vote_Count'] >= 3]['Variable'].tolist()

print(f"Targeting {len(target_vars)} variables for polynomial testing (Tiers 3 & 4).")

# --- 3. DECISION LOGIC LOOP ---
poly_results = []

for var in target_vars:
    # Baseline: use the 9 consensus variables as linear controls
    X_m1 = sm.add_constant(X[target_vars])
    m1 = sm.Logit(y, X_m1).fit(disp=0)
    aic1 = m1.aic

    # Quadratic Model (Degree 2)
    X_m2 = X[target_vars].copy()
    X_m2[f'{var}_2'] = X_m2[var] ** 2
    X_m2 = sm.add_constant(X_m2)
    m2 = sm.Logit(y, X_m2).fit(disp=0)
    aic2 = m2.aic
    p2 = m2.pvalues[f'{var}_2']

    # Cubic Model (Degree 3)
    X_m3 = X_m2.copy()
    X_m3[f'{var}_3'] = X[var] ** 3
    X_m3 = sm.add_constant(X_m3)
    m3 = sm.Logit(y, X_m3).fit(disp=0)
    aic3 = m3.aic
    p3 = m3.pvalues[f'{var}_3']

    # DECISION RULE:
    # 1. Delta AIC > 2 (Significant improvement in model fit)
    # 2. P-value of the higher term < 0.05 (Statistical significance)
    perfect_degree = 1
    if (aic2 < aic1 - 2) and (p2 < 0.05):
        perfect_degree = 2
        if (aic3 < aic2 - 2) and (p3 < 0.05):
            perfect_degree = 3

    poly_results.append({
        'Variable': var,
        'Perfect_Degree': perfect_degree,
        'Linear_AIC': aic1,
        'Quad_AIC': aic2,
        'Cubic_AIC': aic3,
        'P_Val_X2': p2,
        'P_Val_X3': p3
    })

# --- 4. EXPORT RESULTS ---
poly_selection_df = pd.DataFrame(poly_results)
output_path = os.path.join(output_dir, 'polynomial_degree_selection_summary.csv')
poly_selection_df.to_csv(output_path, index=False)

print(f"\nPolynomial Selection Complete. Results saved to: {output_path}")
print(poly_selection_df[['Variable', 'Perfect_Degree']])