#ALL TOPOLOGIES BY ARCHITECTURES with reductions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# Load the dataset and paths

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
base_save_folder = "/Volumes/dax-hd/project-data/images/pca_topology_2"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"

df = pd.read_csv(path)
original_columns = set(df.columns)

with open(cath_dict_path, 'r') as file:
    cath_dict = json.load(file)
if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)
# ---------------------------------------------------------------------------------------------------

selected_architectures = ["alpha_beta_complex (3,90)"]

destress_columns = [
    "hydrophobic_fitness",
    "isoelectric_point",
    "charge",
    "mass",
    "num_residues",
    "packing_density",
    "budeff_total",
    "budeff_steric",
    "budeff_desolvation",
    "budeff_charge",
    "evoef2_total",
    "evoef2_ref_total",
    "evoef2_intraR_total",
    "evoef2_interS_total",
    "evoef2_interD_total",
    "dfire2_total",
    "rosetta_total",
    "rosetta_fa_atr",
    "rosetta_fa_rep",
    "rosetta_fa_intra_rep",
    "rosetta_fa_elec",
    "rosetta_fa_sol",
    "rosetta_lk_ball_wtd",
    "rosetta_fa_intra_sol_xover4",
    "rosetta_hbond_lr_bb",
    "rosetta_hbond_sr_bb",
    "rosetta_hbond_bb_sc",
    "rosetta_hbond_sc",
    "rosetta_dslf_fa13",
    "rosetta_rama_prepro",
    "rosetta_p_aa_pp",
    "rosetta_fa_dun",
    "rosetta_omega",
    "rosetta_pro_close",
    "rosetta_yhh_planarity",
    "aggrescan3d_total_value",
    "aggrescan3d_avg_value",
    "aggrescan3d_min_value",
    "aggrescan3d_max_value"
]

# ---------------------------------------------------------------------------------------------------

def add_topology_description(df, cath_dict):
    def get_topology_description(row):
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        top_num = str(row['Topology number'])
        try:
            description = cath_dict[class_num][arch_num][top_num]['description']
            return description
        except KeyError:
            return "Unknown"
    
    df['topology_description'] = df.apply(get_topology_description, axis=1)
    return df

df = add_topology_description(df, cath_dict)

# ---------------------------------------------------------------------------------------------------

def remove_highly_correlated_features(df, tolerance, columns):
    if columns is None:
        columns = df.columns

    valid_columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    df_selected = df[valid_columns].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_selected)
    df_scaled = pd.DataFrame(scaled_features, columns=valid_columns)

    corr_matrix = df_scaled.corr(method='spearman').abs()
    dropped_features = []

    while True:
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > tolerance)]
        
        if not to_drop:
            break
        
        feature_to_remove = to_drop[0]
        df_selected.drop(columns=feature_to_remove, inplace=True)
        df_scaled.drop(columns=feature_to_remove, inplace=True)
        dropped_features.append(feature_to_remove)
        corr_matrix = df_scaled.corr(method='spearman').abs()

    return df.drop(columns=dropped_features), dropped_features

# ---------------------------------------------------------------------------------------------------
threshold = 0.2
missing_percentage = df.isnull().sum() / len(df)
columns_to_drop = missing_percentage[missing_percentage > threshold].index
df = df.drop(columns=columns_to_drop, axis=1)
df = df.dropna()

cleaned_columns = set(df.columns)
dropped_columns = list(original_columns - cleaned_columns)
print("Dropped columns (Missing Values):", dropped_columns)

# ---------------------------------------------------------------------------------------------------
# Normalise Data

normalise_columns = [
    "num_residues", "hydrophobic_fitness", "budeff_total", "budeff_steric", "budeff_desolvation", "budeff_charge",
    "evoef2_total", "evoef2_ref_total", "evoef2_intraR_total", "evoef2_interS_total", "evoef2_interD_total",
    "dfire2_total", "rosetta_total", "rosetta_fa_atr", "rosetta_fa_rep", "rosetta_fa_intra_rep", "rosetta_fa_elec",
    "rosetta_fa_sol", "rosetta_lk_ball_wtd", "rosetta_fa_intra_sol_xover4", "rosetta_hbond_lr_bb",
    "rosetta_hbond_sr_bb", "rosetta_hbond_bb_sc", "rosetta_hbond_sc", "rosetta_dslf_fa13", "rosetta_rama_prepro",
    "rosetta_p_aa_pp", "rosetta_fa_dun", "rosetta_omega", "rosetta_pro_close", "rosetta_yhh_planarity"
]

if 'num_residues' in df.columns:
    for field in normalise_columns:
        if field in df.columns:
            df[field] = df[field] / df['num_residues']

# ---------------------------------------------------------------------------------------------------
# Drop mass and residue number, removing highly correlated features, and scaling
            
df = df[df['architecture_name'].isin(selected_architectures)]

df = df.drop(['mass', 'num_residues'], axis=1)

df, dropped_features = remove_highly_correlated_features(df, tolerance=0.6, columns=destress_columns)

corr_columns = set(df.columns)
dropped_columns_corr = list(cleaned_columns - corr_columns)
print("Dropped columns (Correlation):", dropped_columns_corr)

nunique = df.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

nuq_columns = set(df.columns)
dropped_columns_nuq = list(corr_columns - nuq_columns)
print("Dropped columns (Little/no Variance):", dropped_columns_nuq)

pca_columns = [col for col in destress_columns if col in df.columns]
df_pca_ready = df[pca_columns].dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca_ready)

# ---------------------------------------------------------------------------------------------------
# Plotting

for architecture_name in selected_architectures:
    if architecture_name not in selected_architectures:
        continue

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    explained_variance = pca.explained_variance_ratio_ * 100
    
    save_folder = os.path.join(base_save_folder, architecture_name.replace('/', '_'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ---------------------------------------------------------------------------------------------------

    # PCA plotting
    pca_df['topology_description'] = df['topology_description'].values

    unique_topologies = pca_df['topology_description'].unique()
    topology_to_id = {topology: i % 3 for i, topology in enumerate(unique_topologies)}

    pca_df['marker_style'] = pca_df['topology_description'].map(topology_to_id).map({0: 'o', 1: '^', 2: 's'})
    plt.figure(figsize=(20, 12))
    ax = plt.subplot(111, aspect='equal')

    markers = ['o', '^', 's'] 
    palette = sns.color_palette('Spectral', n_colors=len(unique_topologies))

    pca_df.to_csv(os.path.join(base_save_folder, 'final_processed_data.csv'), index=False)

    sns.scatterplot(
        x='PC1', y='PC2',
        style='marker_style',
        hue='topology_description',
        data=pca_df,
        palette=palette, 
        markers=markers,
        s=100 
    )

    plt.title(f'PCA for {architecture_name} - PC1: {explained_variance[0]:.2f}%, PC2: {explained_variance[1]:.2f}% explained variance')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.figtext(0.5, 0.01, f"Features used: {', '.join(pca_columns)}", ha="center", fontsize=10)

    ax.legend_.remove()
    legend_items = []
    for i, (topology, marker) in enumerate(zip(unique_topologies, markers * (len(unique_topologies) // len(markers) + 1))):
        legend_items.append(mlines.Line2D([], [], color=palette[i % len(palette)], marker=marker, linestyle='None', markersize=10, label=topology))
    ax.legend(handles=legend_items, title='Topology Description', bbox_to_anchor=(1, 1), loc='upper left')

    plt.savefig(os.path.join(save_folder, f"{architecture_name.replace('/', '_')}-pca.png"), bbox_inches='tight')
    plt.close()

    # Plotting cumulative variance
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.tight_layout()
    cumulative_variance_plot_path = os.path.join(save_folder, 'cumulative_explained_variance.png')
    plt.savefig(cumulative_variance_plot_path)
    plt.close()

    # Component loading plot
    for i in range(pca.n_components_):
        plt.figure(figsize=(10, 6))
        component_loadings = pca.components_[i]
        indices = np.argsort(abs(component_loadings))[::-1]
        
        # Ensure feature names match the PCA input
        feature_names = np.array(pca_columns)[indices]
        
        plt.bar(range(len(component_loadings)), component_loadings[indices])
        plt.xticks(range(len(component_loadings)), feature_names, rotation=90)
        plt.title(f'PCA Component {i+1} Loadings')
        plt.ylabel('Loading Value')
        plt.tight_layout()
        feature_contribution_path = os.path.join(save_folder, f'pca_component_{i+1}_loadings.png')
        plt.savefig(feature_contribution_path)
        plt.close()

print("Variance explained by each component:")
print("Columns used:", pca_columns)
print("PCA explained variance:", pca.explained_variance_ratio_)

# ---------------------------------------------------------------------------------------------------