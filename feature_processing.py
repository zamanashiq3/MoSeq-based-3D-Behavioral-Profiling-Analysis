import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ------------------------------------------------------------------------------

# Example file paths (adjust to match your actual files)
vf_file         = "vf-data.csv"  # Must contain column 'Label' (0=Control,1=DN) and 'VF_after'
velocity_file   = "Velocity-Moseq-data.csv"
length_file     = "length-Moseq-data.csv"
pose_file       = "MoSeq_pose.csv"
position_file   = "psition-Moseq.csv"

# Read data
vf_df          = pd.read_csv(vf_file)
velocity_df    = pd.read_csv(velocity_file).dropna(axis=1, how='all')
length_df      = pd.read_csv(length_file).dropna(axis=1, how='all')
pose_df        = pd.read_csv(pose_file).dropna(axis=1, how='all')
position_df    = pd.read_csv(position_file).dropna(axis=1, how='all')

# Merge MoSeq data on 'Label' (not 'Labels')
merged_df = velocity_df.merge(length_df,   on='Label', suffixes=('_velocity','_length'))
merged_df = merged_df.merge(pose_df,       on='Label', suffixes=('', '_pose'))
merged_df = merged_df.merge(position_df,   on='Label', suffixes=('', '_position'))

# Merge von Frey data
merged_df = merged_df.merge(vf_df, on='Label', how='left')

# Create a simplified group column
merged_df['group'] = merged_df['Label'].map({0:'Control', 1:'DN'})

# ------------------------------------------------------------------------------
# 2. NORMALIZE (STANDARD SCALE) THE FEATURES
# ------------------------------------------------------------------------------

# Identify columns that likely hold numeric features for MoSeq
feature_cols = [c for c in merged_df.columns if (
    ('MoSeq_bin_' in c) or
    ('position_bin_' in c) or
    ('length_bin'   in c) or
    ('velocity_bin' in c)
)]

# We'll also consider whether to scale the 'VF_after' metric.
# If you want to include VF in PCA, add 'VF_after' to this list.

scaler = StandardScaler()

# Drop rows with missing data in these columns
merged_for_scale = merged_df.dropna(subset=feature_cols)

# Standard scaling
scaled_features = scaler.fit_transform(merged_for_scale[feature_cols])

# Create a scaled DataFrame with the same index
scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=merged_for_scale.index)

# Put scaled columns back into a copy of merged_df
norm_merged_df = merged_for_scale.copy()
for col in feature_cols:
    norm_merged_df[col] = scaled_df[col]

# ------------------------------------------------------------------------------
# 3. PERFORM PCA & PLOT
# ------------------------------------------------------------------------------

# Perform PCA on normalized features
pca = PCA(n_components=2)
pca_result = pca.fit_transform(norm_merged_df[feature_cols])

# Create a PCA result DataFrame
pca_df = pd.DataFrame(
    data = pca_result,
    columns = ['PC1', 'PC2'],
    index = norm_merged_df.index
)

# Add group info
pca_df['group'] = norm_merged_df['group']

# Plot settings for a more "Nature-like" style
sns.set_theme(style="white", context="talk")

# Make a scatter plot of the first two PCs
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='group',
    palette=["#4c72b0", "#c44e52"],  # Provide a color palette if desired
    edgecolor="white",
    s=100
)
plt.title("PCA of MoSeq Features")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title="Group", loc="best").remove()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. CORRELATION WITH VON FREY
# ------------------------------------------------------------------------------

vf_col = 'VF_after'  # rename as needed
corr_results = {}

# Drop rows with missing VF for correlation analysis
corr_data = norm_merged_df.dropna(subset=[vf_col])

for col in feature_cols:
    if corr_data[col].dtype in [np.float64, np.int64]:
        valid_data = corr_data.dropna(subset=[col, vf_col])
        if len(valid_data) > 2:
            r, p = stats.pearsonr(valid_data[col], valid_data[vf_col])
            corr_results[col] = (r, p)

corr_df = pd.DataFrame.from_dict(corr_results, orient='index', columns=['Correlation','p-value'])

# ------------------------------------------------------------------------------
# 5. GROUP-BASED COMPARISON (Control vs. DN)
# ------------------------------------------------------------------------------

ttest_results = {}

for col in feature_cols:
    if norm_merged_df[col].dtype in [np.float64, np.int64]:
        ctrl_vals = norm_merged_df.loc[norm_merged_df['group']=='Control', col].dropna()
        dn_vals   = norm_merged_df.loc[norm_merged_df['group']=='DN', col].dropna()
        if (len(ctrl_vals) > 1) and (len(dn_vals) > 1):
            t_val, p_val = stats.ttest_ind(ctrl_vals, dn_vals)
            ttest_results[col] = (t_val, p_val)

# Put T-test results in a DataFrame
ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index', columns=['T-statistic','p-value'])

# ------------------------------------------------------------------------------
# 7. PRINT RESULTS
# ------------------------------------------------------------------------------

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nPearson Correlation with VF (Top 10 by p-value):")
print(corr_df.sort_values(by='p-value').head(10))

print("\nT-test (Control vs. DN) (Top 10 by p-value):")
print(ttest_df.sort_values(by='p-value').head(10))

print("\nDone.")


###############################################################################
# FIGURE 1: PCA SCATTER PLOT
###############################################################################
# We'll reuse pca_df created from your final code. It contains columns ['PC1','PC2','group'].

sns.set_theme(style="white", context="talk")  # Nature-like theme

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='group',
    palette=["#4c72b0", "#c44e52"],
    edgecolor='white', 
    s=100
)
plt.title("Figure 1: PCA of MoSeq Features")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title="Group", loc="best").remove()  # remove the legend box if desired
plt.tight_layout()
plt.show()

###############################################################################
# FIGURE 2: FEATURE-FEATURE CORRELATION HEATMAP
###############################################################################
# Correlation among normalized MoSeq features

corr_matrix = norm_merged_df[feature_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    cmap='RdBu_r',   # Divergent colormap
    center=0,
    square=True,
    linewidths=.5
)
plt.title("Figure 2: Correlation Matrix of Normalized MoSeq Features")
plt.tight_layout()
plt.show()

###############################################################################
# FIGURE 3: VOLCANO PLOT (GROUP COMPARISON)
###############################################################################
# We'll use ttest_df from your code for p-values. 
# We'll compute effect size = (DN mean - Control mean).

volcano_data = []
for col, (t_val, p_val) in ttest_df.iterrows():
    mean_control = norm_merged_df.loc[norm_merged_df['group']=='Control', col].mean()
    mean_dn      = norm_merged_df.loc[norm_merged_df['group']=='DN', col].mean()
    effect_size  = mean_dn - mean_control
    
    # Very small p-values can cause -log10(p) overflow; cap them
    capped_p = max(p_val, 1e-300)
    
    volcano_data.append({
        'Feature': col,
        'EffectSize': effect_size,
        'negLogP': -np.log10(capped_p)
    })

volcano_df = pd.DataFrame(volcano_data)

# Plot
plt.figure(figsize=(7,6))
sns.scatterplot(
    data=volcano_df,
    x='EffectSize', y='negLogP',
    color='gray', alpha=0.7
)
# Optionally highlight “significant” features (e.g., p<0.05 => negLogP>~1.3)
sig_hits = volcano_df[volcano_df['negLogP'] > 1.3]
sns.scatterplot(
    data=sig_hits,
    x='EffectSize', y='negLogP',
    color='#c44e52'
)

# Draw a reference line at p=0.05
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', lw=1)

plt.title("Figure 3: Volcano Plot (DN vs. Control)")
plt.xlabel("Mean (DN) - Mean (Control)")
plt.ylabel("-log10(p-value)")
plt.tight_layout()
plt.show()






























