# ALL DATA T-SNE ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import textwrap
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# Load the dataset

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"
base_save_folder = "/Volumes/dax-hd/project-data/images/figs/"
if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)

df = pd.read_csv(path)
original_columns = set(df.columns)

with open(cath_dict_path, 'r') as file:
    cath_dict = json.load(file)

# ---------------------------------------------------------------------------------------------------

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

normalise_columns = [
    "num_residues", "hydrophobic_fitness", "budeff_total", "budeff_steric", "budeff_desolvation", "budeff_charge",
    "evoef2_total", "evoef2_ref_total", "evoef2_intraR_total", "evoef2_interS_total", "evoef2_interD_total",
    "dfire2_total", "rosetta_total", "rosetta_fa_atr", "rosetta_fa_rep", "rosetta_fa_intra_rep", "rosetta_fa_elec",
    "rosetta_fa_sol", "rosetta_lk_ball_wtd", "rosetta_fa_intra_sol_xover4", "rosetta_hbond_lr_bb",
    "rosetta_hbond_sr_bb", "rosetta_hbond_bb_sc", "rosetta_hbond_sc", "rosetta_dslf_fa13", "rosetta_rama_prepro",
    "rosetta_p_aa_pp", "rosetta_fa_dun", "rosetta_omega", "rosetta_pro_close", "rosetta_yhh_planarity"
]

# ---------------------------------------------------------------------------------------------------

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# ---------------------------------------------------------------------------------------------------
# Add the architecture name to df

def add_cath_names(df, cath_dict):
    def get_descriptions(row):
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        top_num = str(row['Topology number'])
        super_num = str(row['Homologous superfamily number'])

        class_desc = cath_dict.get(class_num, {}).get('description', "Unknown")
        
        arch_desc = "Unknown"
        if arch_num in cath_dict.get(class_num, {}):
            arch_desc = cath_dict[class_num][arch_num].get('description', "Unknown")
        
        top_desc = "Unknown"
        if top_num in cath_dict.get(class_num, {}).get(arch_num, {}):
            top_desc = cath_dict[class_num][arch_num][top_num].get('description', "Unknown")
        
        super_desc = "Unknown"
        if super_num in cath_dict.get(class_num, {}).get(arch_num, {}).get(top_num, {}):
            super_desc = cath_dict[class_num][arch_num][top_num][super_num].get('description', "Unknown")

        return pd.Series([class_desc, arch_desc, top_desc, super_desc])

    descriptions = df.apply(get_descriptions, axis=1, result_type='expand')
    df[['class_description', 'arch_description', 'top_description', 'super_description']] = descriptions

    return df
df = add_cath_names(df, cath_dict)

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
# Tagging Class Archetype structures

def tag_class_archetypes(df, cath_dict):
    df['is_class_archetype'] = False
    for index, row in df.iterrows():
        class_num = str(row['Class number'])
        try:
            protein_id = cath_dict[class_num]['protein_id']
            if protein_id[:4] in row['design_name']:
                df.at[index, 'is_class_archetype'] = True
        except KeyError:
            continue
    return df
df = tag_class_archetypes(df, cath_dict)

# ---------------------------------------------------------------------------------------------------
# Tagging Arch Archetype structures

def tag_arch_archetypes(df, cath_dict):
    df['is_arch_archetype'] = False
    for index, row in df.iterrows():
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        try:
            protein_id = cath_dict[class_num][arch_num]['protein_id']
            if protein_id[:4] in row['design_name']:
                df.at[index, 'is_arch_archetype'] = True
        except KeyError:
            continue
    return df
df = tag_arch_archetypes(df, cath_dict)
# ---------------------------------------------------------------------------------------------------
# Tagging Top Archetypal structures

def tag_top_archetypes(df, cath_dict):
    df['is_top_archetype'] = False
    for index, row in df.iterrows():
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        top_num = str(row['Topology number'])
        try:
            protein_id = cath_dict[class_num][arch_num][top_num]['protein_id']
            if protein_id[:4] in row['design_name']:
                df.at[index, 'is_top_archetype'] = True
        except KeyError:
            continue
    return df
df = tag_top_archetypes(df, cath_dict)

# ---------------------------------------------------------------------------------------------------
# Remove unsubstantial columns and normalise data

def process_data(df):
    threshold = 0.2
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    df = df.drop(columns=columns_to_drop, axis=1)

    for feature in normalise_columns:
        if feature in df.columns:
            df[feature] = df[feature] / df['num_residues']
    return df

# ---------------------------------------------------------------------------------------------------
# ARCHITECTURE

n_components = 2 
perplexity = 20
learning_rate = 800
n_iter = 250
random_state = 42
min_datapoints_threshold = 20

path_to_log_file = os.path.join(base_save_folder, "arch_log.txt")
original_stdout = sys.stdout
sys.stdout = Logger(path_to_log_file)

for arch_name in df['arch_description'].unique():
    df_arch = df[df['arch_description'] == arch_name].copy()
    sanitized_arch_name = arch_name.replace('/', '_').replace(' ', '_')
    arch_name = sanitized_arch_name
    
    if len(df_arch) <= perplexity:
        print(f"\tNot enough data for t-SNE on {sanitized_arch_name}. Skipping...")
        continue
    
    if 1 < len(df_arch) <= 2 * perplexity:
        adjusted_perplexity = len(df_arch) / 2
        print(f"\tAdjusting perplexity for {sanitized_arch_name} to {adjusted_perplexity} due to low datapoint count.")
        perplexity = adjusted_perplexity

    print("--------------------------------\n")
    print(f"T-SNE results for {arch_name}:\n")
    print(f"\tInitial number of structures: {len(df_arch)}\n")
    
    # Drop mass and residue number, removing highly correlated features, and scaling
    df_processed = process_data(df_arch)         
    df_processed = df_processed.drop(['mass', 'num_residues'], axis=1)
    cleaned_columns = set(df_processed.columns)
    dropped_columns = list(original_columns - cleaned_columns)
    print("\tDropped features:\n\t\t- missing Values:", ", " .join(dropped_columns), "\n")
    
    df_processed, dropped_features = remove_highly_correlated_features(df_processed, tolerance=0.6, columns=destress_columns)

    corr_columns = set(df_processed.columns)
    dropped_columns_corr = list(cleaned_columns - corr_columns)
    print("\t\t- correlation:", ", " .join(dropped_columns_corr), "\n")

    df_destress = df_processed[[col for col in destress_columns if col in df_processed.columns]]

    nunique = df_destress.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_destress = df_destress.drop(cols_to_drop, axis=1)

    nuq_columns = set(df_destress.columns)
    dropped_columns_nuq = list((corr_columns) - (nuq_columns))
    print("\t\t- little/no variance:", ", " .join(dropped_columns_nuq), "\n")

    df_destress = df_destress.dropna()
    df_tsne_ready = df_destress
    print(f"\tUsed features:", ", ".join(df_destress.columns.tolist()), "\n")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_tsne_ready)

    # ---------------------------------------------------------------------------------------------------
    # Plotting

    # t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(df_scaled)

    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])

    tsne_df['arch_description'] = df['arch_description'].values[:len(tsne_df)]
    print(f"\tFinal number of structures: {len(tsne_df)}", "\n")

    if 'top_description' in df_processed.columns:
        tsne_df['top_description'] = df_processed['top_description'].values[:len(tsne_df)]
    else:
        print("\t!!! Warning !!!: 'top_description' column not found. Skipping topology description.")
    
    if 'is_arch_archetype' in df.columns:
        tsne_df['is_top_archetype'] = df['is_top_archetype'].values[:len(tsne_df)]
    else:
        print("\t!!! Warning !!!: 'is_top_archetype' column not found. Skipping topology identification.")
    
    print(f"\tTopologies used:", ", ".join(df_processed['top_description'].unique()), "\n")
        
    # Plotting
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, aspect='equal')
    
    # Plot non-archetypal structures
    sns.scatterplot(
        data=tsne_df[~tsne_df['is_top_archetype']],
        x='Dimension 1', y='Dimension 2',
        hue='top_description',
        palette="Spectral",
        marker='o',
        s=50,
        alpha=0.7,
        ax=ax,
        legend=False
    )

    # Archetypal structures
    sns.scatterplot(
        data=tsne_df[tsne_df['is_top_archetype']],
        x='Dimension 1', y='Dimension 2',
        hue='top_description',
        palette="Spectral",        
        marker='o',
        s=50,
        edgecolor='black',
        linewidth=1,
        ax=ax,
        legend=False
    )

    plt.title(f't-SNE for {arch_name} (Architecture) by Topology')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.1)

    top_counts = df_arch['top_description'].value_counts()
    print(f"\tTopology counts:")
    for top, count in top_counts.items():
        print(f"\t\t- {top}: {count}")

    print(f"\n\tPerplexity: {tsne.perplexity}, Learning Rate: {tsne.learning_rate}, Iterations: {tsne.n_iter}\n")

    arch_save_folder = os.path.join(base_save_folder, "by_arch", f"arch_{arch_name}")
    os.makedirs(arch_save_folder, exist_ok=True)
    plt.savefig(os.path.join(arch_save_folder, f"{arch_name.replace(' ', '_')}_tsne.png"), bbox_inches='tight')
    plt.close()

    print(f"\tAnalysis completed for {arch_name}.\n")
    print("--------------------------------\n")

print(f"\tAnalysis completed for Architectures.\n")
sys.stdout = original_stdout

# ---------------------------------------------------------------------------------------------------
# TOPOLOGY

# n_components = 2 
# perplexity = 100
# learning_rate = 800
# n_iter = 250
# random_state = 42
# min_datapoints_threshold = 1

# path_to_log_file = os.path.join(base_save_folder, "class_log.txt")
# original_stdout = sys.stdout
# sys.stdout = Logger(path_to_log_file)

# for class_name in df['class_description'].unique():
#     df_class = df[df['class_description'] == class_name].copy()
#     if len(df_class) < min_datapoints_threshold:
#         print(f"Skipping {class_name} due to insufficient datapoints ({len(df_class)}).")
#         continue
    
#     print("--------------------------------\n")
#     print(f"T-SNE results for {class_name}:\n")
#     print(f"\tInitial number of structures: {len(df_class)}\n")
    
#     # Drop mass and residue number, removing highly correlated features, and scaling
#     df_processed = process_data(df_class)         
#     df_processed = df_processed.drop(['mass', 'num_residues'], axis=1)
#     cleaned_columns = set(df_processed.columns)
#     dropped_columns = list(original_columns - cleaned_columns)
#     print("\tDropped features:\n\t\t- missing Values:", ", " .join(dropped_columns), "\n")
    
#     df_processed, dropped_features = remove_highly_correlated_features(df_processed, tolerance=0.6, columns=destress_columns)

#     corr_columns = set(df_processed.columns)
#     dropped_columns_corr = list(cleaned_columns - corr_columns)
#     print("\t\t- correlation:", ", " .join(dropped_columns_corr), "\n")

#     df_destress = df_processed[[col for col in destress_columns if col in df_processed.columns]]

#     nunique = df_destress.apply(pd.Series.nunique)
#     cols_to_drop = nunique[nunique == 1].index
#     df_destress = df_destress.drop(cols_to_drop, axis=1)

#     nuq_columns = set(df_destress.columns)
#     dropped_columns_nuq = list((corr_columns) - (nuq_columns))
#     print("\t\t- little/no variance:", ", " .join(dropped_columns_nuq), "\n")

#     df_destress = df_destress.dropna()
#     df_tsne_ready = df_destress
#     print(f"\tUsed features:", ", ".join(df_destress.columns.tolist()), "\n")
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df_tsne_ready)

#     # ---------------------------------------------------------------------------------------------------
#     # Plotting

#     save_folder = os.path.join(base_save_folder)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     # t-SNE
#     tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
#     tsne_results = tsne.fit_transform(df_scaled)

#     tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])

#     tsne_df['class_description'] = df['class_description'].values[:len(tsne_df)]
#     print(f"\tFinal number of structures: {len(tsne_df)}", "\n")

#     if 'arch_description' in df_processed.columns:
#         tsne_df['arch_description'] = df_processed['arch_description'].values[:len(tsne_df)]
#     else:
#         print("\t!!! Warning !!!: 'arch_description' column not found. Skipping architecture description.")
    
#     if 'is_arch_archetype' in df.columns:
#         tsne_df['is_arch_archetype'] = df['is_arch_archetype'].values[:len(tsne_df)]
#     else:
#         print("\t!!! Warning !!!: 'is_arch_archetype' column not found. Skipping archetype identification.")
    
#     print(f"\tArchitectures used:", ", ".join(df_processed['arch_description'].unique()), "\n")
        
#     # Plotting
#     plt.figure(figsize=(10, 10))
#     ax = plt.subplot(111, aspect='equal')
    
#     # Plot non-archetypal structures
#     sns.scatterplot(
#         data=tsne_df[~tsne_df['is_arch_archetype']],
#         x='Dimension 1', y='Dimension 2',
#         hue='arch_description',
#         palette="Spectral",
#         marker='o',
#         s=50,
#         alpha=0.7,
#         ax=ax
#     )

#     # Archetypal structures
#     sns.scatterplot(
#         data=tsne_df[tsne_df['is_arch_archetype']],
#         x='Dimension 1', y='Dimension 2',
#         hue='arch_description',
#         palette="Spectral",        
#         marker='o',
#         s=50,
#         edgecolor='black',
#         linewidth=1,
#         ax=ax,
#         legend=False
#     )

#     plt.title(f't-SNE for {class_name} (Class) by Architecture')
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     legend = plt.legend()
#     legend.get_frame().set_alpha(0.1)

#     arch_counts = df_class['arch_description'].value_counts()
#     print(f"\tArchitecture counts:")
#     for arch, count in arch_counts.items():
#         print(f"\t\t- {arch}: {count}")

#     print(f"\n\tPerplexity: {tsne.perplexity}, Learning Rate: {tsne.learning_rate}, Iterations: {tsne.n_iter}\n")
        
#     class_save_folder = os.path.join(base_save_folder, "by_class", f"class_{class_name.replace(' ', '_')}")
#     if not os.path.exists(class_save_folder):
#         os.makedirs(class_save_folder)
    
#     plt.savefig(os.path.join(class_save_folder, f"{class_name.replace(' ', '_')}_tsne.png"), bbox_inches='tight')
#     plt.close()

#     print(f"\tAnalysis completed for {class_name}.\n")
#     print("--------------------------------\n")

# sys.stdout = original_stdout

# print("All t-SNE complete")

# ---------------------------------------------------------------------------------------------------

    
