# **MoSeq-based 3D Behavioral Profiling Analysis**
## **Supporting codebase and data for the publication titiled "MoSeq-based 3D Behavioral Profiling Uncovers Neuropathic Behavior Changes in Diabetic Neuropathy Mice Models "**
## `Ashiquzzaman et al. (2025)` [under review]
## **Overview**
This repository contains two main Python scripts used in the paper:

1. **`feature_processing.py`**  
   - Preprocesses MoSeq feature data, normalizes values, and performs PCA to explore differences in behavior profiles.  
   - Computes statistical comparisons and correlation with von Frey test results.  
   - Produces three **Nature-style figures**:  
     - **PCA scatter plot** to visualize group differences.  
     - **Correlation heatmap** to reveal feature-feature relationships.  
     - **Volcano plot** to highlight significantly altered behavioral features in **Diabetic Neuropathy (DN) vs. Control.**  

2. **`behavior_regulation.py`**  
   - Analyzes **MoSeq behavioral syllables**, focusing on frequency distributions and regulation changes.  
   - Performs **hierarchical clustering, UMAP dimensionality reduction, and KMeans clustering** to identify behavior patterns.  
   - Generates **volcano plots, heatmaps, and bar charts** to visualize behavioral shifts.  

This **novel MoSeq analysis framework** can be used to explore high-dimensional behavioral data in rodent models of **neurological disorders.**

---

## **1️⃣ feature_processing.py: PCA and Feature Analysis**
### **Functionality**
- Loads and **merges** MoSeq behavior features (`velocity`, `length`, `pose`, `position`) with von Frey test results.  
- **Normalizes** all features using **StandardScaler** to ensure comparability.  
- Performs **Principal Component Analysis (PCA)** to visualize group-level differences.  
- Computes **feature correlations** with von Frey test values.  
- Conducts **t-tests** to identify significant behavioral feature differences between **Diabetic Neuropathy (DN) and Control** groups.  
- Generates three key **figures** for scientific publication.

---

### **Key Steps**
#### **1. Data Loading & Merging**
- Reads **MoSeq feature files**:  
  - `Velocity-Moseq-data.csv`
  - `length-Moseq-data.csv`
  - `MoSeq_pose.csv`
  - `psition-Moseq.csv`
- Reads **von Frey data** (`vf-data.csv`) and merges with MoSeq features.  
- Creates a **"group" column** to distinguish **Control (0) vs. DN (1)**.

#### **2. Feature Normalization**
- Identifies all MoSeq behavioral features.  
- Uses **`StandardScaler()`** to normalize values across all features.  

#### **3. PCA Analysis & Visualization**
- **Performs PCA** to reduce feature space to **two principal components**.
- **Scatter plot**: Displays PCA results, with groups **colored for separation.**  

#### **4. Feature Correlations with von Frey**
- Computes **Pearson correlations** between MoSeq features and von Frey results.
- Generates a **correlation heatmap** of all features.

#### **5. Group-wise Comparisons (DN vs. Control)**
- Runs **t-tests** comparing **DN vs. Control** for each MoSeq feature.
- Generates a **volcano plot**:
  - X-axis: **Effect size (DN - Control)**
  - Y-axis: **Significance (-log10(p-value))**
  - **Highlights significantly altered behaviors.**

---

### **Generated Figures**
| **Figure** | **Description** |
|------------|----------------|
| **Figure 1** | **PCA scatter plot** showing behavioral separation between Control and DN. |
| **Figure 2** | **Correlation heatmap** of MoSeq features to visualize inter-feature relationships. |
| **Figure 3** | **Volcano plot** of significantly altered behavioral features in DN. |

---

## **2️⃣ behavior_regulation.py: Behavioral Clustering & Regulation Analysis**
### **Functionality**
- Processes **MoSeq syllable usage patterns**.
- **Clusters** behaviors via **hierarchical clustering, UMAP, and KMeans.**  
- **Compares** Control vs. DN **behavioral syllable distributions**.  
- **Identifies** significantly **upregulated/downregulated** behaviors.  
- **Generates multiple scientific figures**:  
  - **Heatmaps**
  - **UMAP embeddings**
  - **Volcano plots**
  - **Bar charts**

---

## **📂 File Structure**
```
/MoSeq_Behavior_Analysis
│── feature_processing.py       # PCA, t-tests, correlation analysis
│── behavior_regulation.py      # Clustering, UMAP, behavior regulation analysis
│── data/
│   ├── vf-data.csv             # Von Frey results
│   ├── Velocity-Moseq-data.csv # MoSeq feature data
│   ├── length-Moseq-data.csv   # MoSeq feature data
│   ├── MoSeq_pose.csv          # MoSeq feature data
│   ├── psition-Moseq.csv       # MoSeq feature data
│   ├── behavior_map_dn.csv     # MoSeq syllable mapping
│   ├── Control_syllable_counts.csv  # MoSeq syllable usage (Control)
│   ├── Diabetic_Neuropathy_syllable_counts.csv  # MoSeq syllable usage (DN)
│── figures/
│   ├── PCA_plot.png
│   ├── Heatmap.png
│   ├── Volcano_Plot.png
│   ├── UMAP_Plot.png
│   ├── Behavior_Frequencies.png
│── README.md                   # This document
```

---

## **🔧 Dependencies**
- Python 3.x
- NumPy, Pandas
- Matplotlib, Seaborn
- Scipy, Scikit-learn
- UMAP, KMeans (for behavior clustering)

To install all dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn umap-learn
```

---

## **📜 Citation**
If using this analysis, please cite:
> Ashiquzzaman et al., *MoSeq-based 3D Behavioral Profiling Uncovers Neuropathic Behavior Changes in Diabetic Neuropathy Mice Models*, under review in **Scientific Reports**.

---

This README provides **reviewers** with **clear documentation** on how your **MoSeq-based behavioral analysis** is implemented, making it easier to evaluate and reproduce.

Let me know if you need further refinements! 🚀

