# PCA ANALYSIS by CLASS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

path_to_log_file = os.path.join(base_save_folder, "class_log_pca.txt")
original_stdout = sys.stdout
sys.stdout = Logger(path_to_log_file)

print("--------------------------------\n")

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
# Preprocess data

print(f"Initial number of structures: {len(df)}\n")

def process_data(df, normalise_columns, destress_columns, tolerance=0.6):
    valid_destress_columns = [col for col in destress_columns if col in df.columns]
    
    # Drop columns with a high percentage of missing values
    threshold = 0.2
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop_due_to_missing = missing_percentage[missing_percentage > threshold].index
    df = df.drop(columns=columns_to_drop_due_to_missing)
    
    # Normalize specified features
    if 'num_residues' in df.columns:
        for feature in [col for col in normalise_columns if col in df.columns]:
            df[feature] = df[feature] / df['num_residues']
    
    # Drop 'mass' and 'num_residues'
    df = df.drop(['mass', 'num_residues'], axis=1, errors='ignore')
    cleaned_columns = set(df.columns)
    dropped_columns = list(original_columns - cleaned_columns)
    print("\tDropped features:\n\n\t\t- missing Values:", ", " .join(dropped_columns), "\n")
        
    # Scale valid_destress_columns using StandardScaler
    if valid_destress_columns:
        existing_valid_destress_columns = [col for col in valid_destress_columns if col in df.columns]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[existing_valid_destress_columns])
        df[existing_valid_destress_columns] = df_scaled

    valid_destress_columns = [col for col in valid_destress_columns if col in df.columns]

    # Compute correlation matrix for valid_destress_columns
    if valid_destress_columns:
        corr_matrix = df[valid_destress_columns].corr(method='spearman').abs()
        dropped_features = []
        while True:
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > tolerance)]
            if not to_drop:
                break
            feature_to_remove = to_drop[0]
            df = df.drop(columns=[feature_to_remove], errors='ignore')
            valid_destress_columns.remove(feature_to_remove)
            dropped_features.append(feature_to_remove)
            if not valid_destress_columns: 
                break
            corr_matrix = df[valid_destress_columns].corr(method='spearman').abs()
        print("\tDropped features due to high correlation:", ", ".join(dropped_features), "\n")

    # Drop columns with little or no variance
    if valid_destress_columns:
        nunique = df[valid_destress_columns].apply(pd.Series.nunique)
        cols_to_drop_due_to_variance = nunique[nunique <= 1].index
        df = df.drop(cols_to_drop_due_to_variance, axis=1, errors='ignore')
        print("\tDropped features due to little/no variance:", ", ".join(cols_to_drop_due_to_variance), "\n")
    
    df = df.dropna()
    return df

df_processed = process_data(df, normalise_columns, destress_columns)

# ---------------------------------------------------------------------------------------------------
# PCA CLASS by ARCHITECTURE

for class_name in df['class_description'].unique():
    df_class_processed = df_processed[df_processed['class_description'] == class_name]

    print("--------------------------------\n")
    print(f"PCA results for {class_name}:\n")

    # ---------------------------------------------------------------------------------------------------
    # Plotting

    class_save_folder = os.path.join(base_save_folder, "by_class", f"class_{class_name.replace(' ', '_')}")
    if not os.path.exists(class_save_folder):
        os.makedirs(class_save_folder)

    df_class_numeric = df_class_processed.select_dtypes(include=[np.number])
    df_class_numeric = df_class_numeric.dropna()

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_class_numeric.drop(['class_description', 'arch_description', 'is_arch_archetype'], axis=1, errors='ignore'))
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df_class_numeric.index)
    
    pca_df['arch_description'] = df.loc[df_class_numeric.index, 'arch_description']
    pca_df['is_arch_archetype'] = df.loc[df_class_numeric.index, 'is_arch_archetype']

    print(f"\tArchitectures used:", ", ".join(pca_df['arch_description'].unique()), "\n")

    explained_variance = pca.explained_variance_ratio_ * 100

    # ---------------------------------------------------------------------------------------------------

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, aspect='equal')

    # Plot non-archetypal structures
    sns.scatterplot(
        x='PC1', y='PC2',
        data=pca_df[~pca_df['is_arch_archetype']],
        hue='arch_description',
        palette='Spectral',
        marker='o',
        s=50,
        alpha=0.7,
        ax=ax
    )
    # Archetypal structures
    sns.scatterplot(
        data=pca_df[pca_df['is_arch_archetype']],
        x='PC1', y='PC2',
        hue='arch_description',
        palette="Spectral",        
        marker='o',
        s=50,
        edgecolor='black',
        linewidth=1,
        ax=ax,
        legend=False
    )
    plt.title(f'PCA for {class_name} (Class) by Architecture')
    plt.xlabel(f'PC1: {explained_variance[0]:.2f}%')
    plt.ylabel(f' PC2: {explained_variance[1]:.2f}%')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.1)

    plt.savefig(os.path.join(class_save_folder, f"{class_name.replace(' ', '_')}_pca.png"), bbox_inches='tight')
    plt.close()

    arch_counts = df_class_processed['arch_description'].value_counts()
    print(f"\tArchitecture counts:")
    for arch, count in arch_counts.items():
        print(f"\t\t- {arch}: {count}")
    print("\n")    

    # # Component loading plot
    # pca_features = df_processed.columns

    # for i in range(pca.n_components_):
    #     plt.figure(figsize=(10, 6))
    #     component_loadings = pca.components_[i]
    #     sorted_loadings = component_loadings
    #     sorted_feature_names = pca_features

    #     plt.bar(x=range(len(sorted_feature_names)), height=sorted_loadings)
    #     plt.xticks(ticks=range(len(sorted_feature_names)), labels=sorted_feature_names, rotation=90)
    #     plt.title(f'PCA Component {i+1} Loadings for {class_name}')
    #     plt.xlabel('Features')
    #     plt.ylabel('Loading Value')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(class_save_folder, f'pca_component_{i+1}_loadings_{class_name}.png'))
    #     plt.close()

    print(f"\tClass number of structures: {len(pca_df)}", "\n")
    print("\tVariance explained by each component:", explained_variance, "\n")
    print(f"\tAnalysis completed for {class_name}.\n")
    print("--------------------------------\n")

print(f"\tAnalysis completed for Classes.\n")
sys.stdout = original_stdout

print("All PCA completed.")

# ---------------------------------------------------------------------------------------------------