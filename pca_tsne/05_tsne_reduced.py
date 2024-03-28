import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from dpca import DensityPeakCluster
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# Load the dataset

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
base_save_folder = "/Volumes/dax-hd/project-data/images/tsne_topology_2/"
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
# Add the architecture name to df

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
# Removing correlating features

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
# Remove missing data

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

tsne_columns = [col for col in destress_columns if col in df.columns]
df_tsne_ready = df[tsne_columns].dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_tsne_ready)

df_scaled_array = scaler.fit_transform(df_tsne_ready)
df_scaled_2 = pd.DataFrame(df_scaled_array, columns=df_tsne_ready.columns)
df_scaled_2['topology_description'] = df['topology_description'].values[:len(df_scaled_2)]
df_scaled_2.to_csv(os.path.join(base_save_folder, 'final_processed_data.csv'), index=False)

print(f"Processed data saved")

# ---------------------------------------------------------------------------------------------------
# Plotting

for architecture_name in selected_architectures:
    save_folder = os.path.join(base_save_folder, architecture_name.replace('/', '_'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=800, n_iter=3000, random_state=42)
    tsne_results = tsne.fit_transform(df_scaled)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dimension 1', 'Dimension 2'])


    tsne_df['topology_description'] = df['topology_description'].values[:len(tsne_df)]
    print(f"Total number of datapoints for {architecture_name}: {len(tsne_df)}")

    # ---------------------------------------------------------------------------------------------------
    
    tsne_df['topology_description'] = df['topology_description'].values
    unique_topologies = tsne_df['topology_description'].unique()
    topology_to_id = {topology: i % 3 for i, topology in enumerate(unique_topologies)}

    tsne_df['marker_style'] = tsne_df['topology_description'].map(topology_to_id).map({0: 'o', 1: '^', 2: 's'})
    
    markers = ['o', '^', 's']
    palette = sns.color_palette('Spectral', n_colors=len(unique_topologies))
    
    plt.figure(figsize=(20, 12))
    ax = plt.subplot(111, aspect='equal')

    sns.scatterplot(
        x='Dimension 1', y='Dimension 2',
        hue='topology_description',
        style='marker_style',
        palette=palette,
        markers=markers,
        data=tsne_df, s=100,
        ax=ax
    )
    
    plt.title(f't-SNE for {architecture_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    ax.legend_.remove()
    legend_items = []
    for i, (topology, marker) in enumerate(zip(unique_topologies, markers * (len(unique_topologies) // len(markers) + 1))):
        legend_items.append(mlines.Line2D([], [], color=palette[i % len(palette)], marker=marker, linestyle='None', markersize=10, label=topology))
    ax.legend(handles=legend_items, title='Topology Description', bbox_to_anchor=(1, 1), loc='upper left')

    plt.figtext(0.5, 0.03, f"Perplexity: {tsne.perplexity}, Learning Rate: {tsne.learning_rate}, Iterations: {tsne.n_iter}", ha="center", fontsize=10)
    plt.figtext(0.5, 0.01, f"Features used: {tsne_columns}", ha="center", fontsize=10)
        
    plt.savefig(os.path.join(save_folder, f"{architecture_name}-tsne.png"), bbox_inches='tight')
    plt.close()

print("t-SNE analysis completed.")

# ---------------------------------------------------------------------------------------------------