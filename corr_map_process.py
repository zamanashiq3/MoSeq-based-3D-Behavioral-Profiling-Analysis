# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:01:55 2025

@author: 3D BIT
"""

import pandas as pd

# Load the provided Excel file
file_path = "new_data_corr.xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
xls.sheet_names

# Load the first sheet to inspect the data structure
df = pd.read_excel(xls, sheet_name="Sheet1")

# Display the first few rows of the dataset
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Separate data for Control (label=0) and Diabetic Neuropathy (label=1)
df_control = df[df['label'] == 0].drop(columns=['label'])
df_dn = df[df['label'] == 1].drop(columns=['label'])


# Compute Pearson correlation matrices
corr_control = df_control.corr()
corr_dn = df_dn.corr()
corr_all = df.drop(columns=['label']).corr()

# Compute the difference in correlation between DN and Control
corr_diff = corr_control - corr_dn
#corr_diff = corr_diff.abs()

# Function to plot correlation heatmaps without axes
def plot_correlation_map(corr_matrix, title):
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, cmap="RdBu_r", annot=False, linewidths=0.5, cbar=True)
    plt.title(title)
    #plt.xticks([])  # Remove x-axis labels
    #plt.yticks([])  # Remove y-axis labels
    plt.show()

# Plot separate heatmaps
plot_correlation_map(corr_control, "Control Group Correlation Map")
plot_correlation_map(corr_dn, "Diabetic Neuropathy Group Correlation Map")
plot_correlation_map(corr_all, "Pairwise Pearson Correlation (All Normalized)")
plot_correlation_map(corr_diff, "Difference in Correlation (DN - Control)")

# Select only the specified scalar features
selected_features = ['height', 'length', 'position', 'speed']





