import pandas as pd
import os

# --- 1. SET FILE PATHS ---
path_lasso = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Lasso_Regression_Cat/lasso_coefficients.csv'
path_ridge = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Ridge_Regression_Cat/ridge_tuned_coefficients.csv'
path_backward = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Stepwise_Backward_Logistic_Regression/backward_coefficients.csv'
path_forward = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset/Stepwise_Forward_Logistic_Regression/forward_coefficients.csv'

output_dir = '/Users/marcel/PycharmProjects/Uni_Pavia/Master_Thesis/Statistics_for_Finance/Exam/Data_Analysis/Models/2_Model_Improvement/Categorical_Dataset'
output_file = os.path.join(output_dir, 'DeFi_Consensus_Variables_4_Tiers.csv')

# --- 2. LOAD DATA & EXTRACT SELECTED VARIABLES ---

# Ridge: Baseline (all 29 variables are technically here)
df_ridge = pd.read_csv(path_ridge)
ridge_vars = df_ridge['Variable'].tolist()

# Lasso: Kept if coefficient != 0
df_lasso = pd.read_csv(path_lasso)
lasso_vars = df_lasso[df_lasso['Standardized_Coefficient'] != 0]['Variable'].tolist()

# Backward: Kept if present in file (assuming first column contains names)
df_back = pd.read_csv(path_backward)
back_vars = df_back.iloc[:, 0].tolist()

# Forward: Kept if present in file
df_forw = pd.read_csv(path_forward)
forw_vars = df_forw.iloc[:, 0].tolist()

# --- 3. BUILD VOTE MATRIX ---
consensus_list = []

for var in ridge_vars:
    # Check votes
    v_ridge = 1  # Ridge is baseline
    v_lasso = 1 if var in lasso_vars else 0
    v_back = 1 if var in back_vars else 0
    v_forw = 1 if var in forw_vars else 0

    total_votes = v_ridge + v_lasso + v_back + v_forw

    # Identify which models voted
    voters = []
    if v_ridge: voters.append('Ridge')
    if v_lasso: voters.append('Lasso')
    if v_back:  voters.append('Backward')
    if v_forw:  voters.append('Forward')

    consensus_list.append({
        'Variable': var,
        'Vote_Count': total_votes,
        'Models_Voting': ", ".join(voters),
        'Tier': f'{total_votes}-Vote Variable'
    })

# Convert to DataFrame and sort
consensus_df = pd.DataFrame(consensus_list).sort_values(by='Vote_Count', ascending=False)

# --- 4. EXPORT ---
consensus_df.to_csv(output_file, index=False)

print(f"Consensus file saved to: {output_file}")
print("\n--- Summary of Tiers ---")
print(consensus_df['Tier'].value_counts().sort_index(ascending=False))