# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import KMeans

# Adjust font sizes for plots
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # Default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # Axes title size
plt.rc('axes', labelsize=MEDIUM_SIZE)    # X and Y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)   # Tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)   # Tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # Legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # Figure title
plt.rcParams["font.family"] = "arial"

# Load CSV files
behavior_map = pd.read_csv('behavior_map_dn.csv')  # Contains descriptions and syllable IDs
CONTROL = pd.read_csv('Control_syllable_counts.csv')  # CTRL counts
DN = pd.read_csv('Diabetic_Neuropathy_syllable_counts.csv')  # DN counts

# Ensure column renaming consistency
CONTROL = CONTROL.rename(columns={'counts': 'Counts_CONTROL'})
DN = DN.rename(columns={'counts': 'Counts_DN'})

# Check that renaming is successful
print(CONTROL.columns)
print(DN.columns)

# Merge datasets on '# syllable id'
merged_data = pd.merge(CONTROL, DN, on='# syllable id')
final_data = pd.merge(behavior_map, merged_data, on='# syllable id')

# Standardize counts columns
counts_data = final_data[['Counts_CONTROL', 'Counts_DN']]
scaler = StandardScaler()
counts_data_scaled = scaler.fit_transform(counts_data)
final_data[['Counts_CONTROL_scaled', 'Counts_DN_scaled']] = counts_data_scaled

# Set index to behavior description for visualization
final_data.set_index('observation', inplace=True)

# Sort the data based on Counts_CONTROL_scaled (or Counts_DN_scaled) for easier interpretation
final_data_sorted = final_data.sort_values(by='Counts_CONTROL_scaled', ascending=False)


# Create a hierarchical heatmap
sns.clustermap(final_data[['Counts_CONTROL_scaled', 'Counts_DN_scaled']],
               standard_scale=1,cmap="PuBu",
               method='ward', metric='euclidean', figsize=(15,12), annot=True)
plt.show()

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(random_state=42)
umap_embedding = reducer.fit_transform(counts_data_scaled)

# Add UMAP coordinates to dataframe
final_data['UMAP1'] = umap_embedding[:, 0]
final_data['UMAP2'] = umap_embedding[:, 1]

# Ensure observation (index) is used as a string
final_data['observation_str'] = final_data.index.astype(str)

# Map unique string observations to numeric values for color mapping
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
final_data['observation_numeric'] = label_encoder.fit_transform(final_data['observation_str'])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
final_data['Cluster'] = kmeans.fit_predict(umap_embedding)

# Plot UMAP embedding with colorbar for string observations
plt.figure(figsize=(15,12))
scatter = plt.scatter(
    x=final_data['UMAP1'],
    y=final_data['UMAP2'],
    c=final_data['observation_numeric'],  # Use numeric mapping for color
    cmap='tab20',  # Choose colormap
    s=300,  # Size of points
    edgecolor='black',  # Border around points
    linewidth=0.5
)

# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(np.arange(len(label_encoder.classes_)))  # Set colorbar ticks to match observations
cbar.set_ticklabels(label_encoder.classes_)  # Replace ticks with observation strings
cbar.set_label('Observation')  # Label for colorbar

# Plot settings
plt.title('UMAP Clustering with Observation Color Mapping')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.show()

# Plot 2D scatter with UMAP embeddings, colored by cluster
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    x=final_data['UMAP1'],
    y=final_data['UMAP2'],
    c=final_data['Cluster'],  # Color by cluster
    cmap='tab20',  # Use a distinct colormap for clusters
    s=300,  # Size of point
    linewidth=0.5
)

# Annotate each point with its observation name
for i, row in final_data.iterrows():
    plt.text(
        x=row['UMAP1'],
        y=row['UMAP2'],
        s=row.name,  # Use the observation name
        fontsize=18,
        ha='center',
        va='center',
        color='black'
    )

# Set plot labels and title
plt.tight_layout()
plt.show()


# Calculate log2 fold change and classify regulation
final_data['log2_fold_change_DN'] = np.log2(final_data['Counts_DN_scaled'] / final_data['Counts_CONTROL_scaled'])
log2_threshold = 1
final_data['Regulation_DN'] = np.where(final_data['log2_fold_change_DN'] > log2_threshold, 'Upregulated',
                                       np.where(final_data['log2_fold_change_DN'] < -log2_threshold, 'Downregulated', 'Unchanged'))

# Simulate a meaningful -log10(p-value) for volcano plot
final_data['-log10(p-value)'] = np.abs(final_data['log2_fold_change_DN'])  # Placeholder for actual p-values

# Create volcano plot
plt.figure(figsize=(11, 6))
sns.scatterplot(x='log2_fold_change_DN', y='-log10(p-value)', hue='Regulation_DN', data=final_data,
                palette={'Upregulated': 'red', 'Downregulated': 'blue', 'Unchanged': 'gray'}, s=200, alpha=0.8)
plt.axvline(log2_threshold, linestyle='--', color='black', label=f'log2 fold change = {log2_threshold}')
plt.axvline(-log2_threshold, linestyle='--', color='black')
plt.title('DN vs Control Volcano Plot')
plt.xlabel('Log2 Fold Change')
plt.ylabel('|log2 Fold Change|')
plt.legend().remove()
plt.show()

# Create volcano plot
plt.figure(figsize=(11, 6))
sns.scatterplot(x='log2_fold_change_DN', y='-log10(p-value)', hue='Regulation_DN', data=final_data,
                palette={'Upregulated': 'red', 'Downregulated': 'blue', 'Unchanged': 'gray'}, s=200, alpha=0.8)
plt.axvline(log2_threshold, linestyle='--', color='black', label=f'log2 fold change = {log2_threshold}')
plt.axvline(-log2_threshold, linestyle='--', color='black')

# Annotate upregulated and downregulated points with observation text
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
for i, row in final_data.iterrows():
    if row['Regulation_DN'] in ['Upregulated', 'Downregulated']:
        x_pos = min(max(row['log2_fold_change_DN'], x_min), x_max)
        y_pos = min(max(row['-log10(p-value)'], y_min), y_max)
        plt.text(x=x_pos, y=y_pos, s=row.name, fontsize=14, color='black', ha='center', va='bottom')

# Finalize plot
plt.title('DN vs Control Volcano Plot')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-log10(p-value)')
plt.legend().remove()
plt.tight_layout()
plt.show()

# Sort the data based on the 'Counts_CONTROL_scaled' values
final_data_sorted = final_data.sort_values(by='Counts_CONTROL_scaled', ascending=False)

# Reset index so that 'observation' becomes a column
final_data_sorted = final_data_sorted.reset_index()

# Create a figure
plt.figure(figsize=(15, 12))

# Set bar width for side-by-side bars
bar_width = 0.35  # Width of each bar
x_positions = np.arange(len(final_data_sorted))  # Base positions for the bars

# Plot the bars for Control and Diabetic Neuropathy (DN)
ax = sns.barplot(x=x_positions + bar_width/2, y='Counts_CONTROL_scaled', data=final_data_sorted, color='blue', label='Control')
sns.barplot(x=x_positions + bar_width/2, y='Counts_DN_scaled', data=final_data_sorted, color='red', label='Diabetic Neuropathy (DN)', ax=ax)

# Rotate the x-axis labels to prevent overlap
plt.xticks(x_positions, final_data_sorted['observation'], rotation=90)
plt.yticks(rotation=90)

# Adjust layout and add a legend
plt.legend(title="Condition").remove()
plt.tight_layout()

# Show the plot
plt.show()


