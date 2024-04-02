# ALL DATA UMAP ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
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
# CLASS by ARCHITECTURE

n_neighbors = 100
min_dist = 0.1
n_components = 2
metric = 'euclidean'

path_to_log_file = os.path.join(base_save_folder, "class_log_umap.txt")
original_stdout = sys.stdout
sys.stdout = Logger(path_to_log_file)

for class_name in df['class_description'].unique():
    df_class = df[df['class_description'] == class_name].copy()

    print("--------------------------------\n")
    print(f"UMAP results for {class_name}:\n")
    print(f"\tInitial number of structures: {len(df_class)}\n")
    
    # Drop mass and residue number, removing highly correlated features, and scaling
    df_processed = process_data(df_class)         
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
    df_umap_ready = df_destress
    print(f"\tUsed features:", ", ".join(df_destress.columns.tolist()), "\n")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_umap_ready)

    # ---------------------------------------------------------------------------------------------------
    # Plotting

    save_folder = os.path.join(base_save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric=metric,)
    umap_results = reducer.fit_transform(df_scaled)

    umap_df = pd.DataFrame(umap_results, columns=['UMAP 1', 'UMAP 2'])

    umap_df['class_description'] = df['class_description'].values[:len(umap_df)]
    print(f"\tFinal number of structures: {len(umap_df)}", "\n")

    if 'arch_description' in df_processed.columns:
        umap_df['arch_description'] = df_processed['arch_description'].values[:len(umap_df)]
    else:
        print("\t!!! Warning !!!: 'arch_description' column not found. Skipping architecture description.")
    
    if 'is_arch_archetype' in df.columns:
        umap_df['is_arch_archetype'] = df['is_arch_archetype'].values[:len(umap_df)]
    else:
        print("\t!!! Warning !!!: 'is_arch_archetype' column not found. Skipping archetype identification.")
    
    print(f"\tArchitectures used:", ", ".join(df_processed['arch_description'].unique()), "\n")
        
    # Plotting
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, aspect='equal')
    
    sns.scatterplot(
        data=umap_df[~umap_df['is_arch_archetype']],
        x='UMAP 1', y='UMAP 2',
        hue='arch_description',
        palette="Spectral",
        marker='o',
        s=50,
        alpha=0.7,
        ax=ax
    )

    # Archetypal structures
    sns.scatterplot(
        data=umap_df[umap_df['is_arch_archetype']],
        x='UMAP 1', y='UMAP 2',
        hue='arch_description',
        palette="Spectral",        
        marker='o',
        s=50,
        edgecolor='black',
        linewidth=1,
        ax=ax,
        legend=False
    )

    plt.title(f'UMAP for {class_name} (Class) by Architecture')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.1)

    arch_counts = df_class['arch_description'].value_counts()
    print(f"\tArchitecture counts:")
    for arch, count in arch_counts.items():
        print(f"\t\t- {arch}: {count}")

    print(f"\n\tNumber of Neighbours: {reducer.n_neighbors}, Minimum Distance: {reducer.min_dist}, Metric: {reducer.metric}\n")
        
    class_save_folder = os.path.join(base_save_folder, "by_class", f"class_{class_name.replace(' ', '_')}")
    if not os.path.exists(class_save_folder):
        os.makedirs(class_save_folder)
    
    plt.savefig(os.path.join(class_save_folder, f"{class_name.replace(' ', '_')}_umap.png"), bbox_inches='tight')
    plt.close()

    print(f"\tAnalysis completed for {class_name}.\n")
    print("--------------------------------\n")

print(f"\tAnalysis completed for Classes.\n")
sys.stdout = original_stdout

print("All UMAP complete")

# ---------------------------------------------------------------------------------------------------

    
