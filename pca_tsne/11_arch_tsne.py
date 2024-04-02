#T-SNE by ARCHITECTURE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
import seaborn as sns
import textwrap
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# LOAD

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"
base_save_folder = "/Volumes/dax-hd/project-data/images/figs/"
if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)

# ---------------------------------------------------------------------------------------------------
# COLUMN RESTRICTIONS

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
# LOGGER

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

path_to_log_file = os.path.join(base_save_folder, "by_arch", "arch_log_tsne.txt")
original_stdout = sys.stdout
sys.stdout = Logger(path_to_log_file)

print("--------------------------------\n")
print("----- t-SNE by ARCHITECTURE ----\n")

# ---------------------------------------------------------------------------------------------------
# Add the architecture name to df

def add_cath_data(df, cath_dict):
    # Add the architecture name to df
    def get_descriptions(row):
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        top_num = str(row['Topology number'])
        super_num = str(row['Homologous superfamily number'])

        class_desc = cath_dict.get(class_num, {}).get('description', "Unknown")
        arch_desc = cath_dict.get(class_num, {}).get(arch_num, {}).get('description', "Unknown")
        top_desc = cath_dict.get(class_num, {}).get(arch_num, {}).get(top_num, {}).get('description', "Unknown")
        super_desc = cath_dict.get(class_num, {}).get(arch_num, {}).get(top_num, {}).get(super_num, {}).get('description', "Unknown")

        return pd.Series([class_desc, arch_desc, top_desc, super_desc])

    descriptions = df.apply(get_descriptions, axis=1, result_type='expand')
    df[['class_description', 'arch_description', 'top_description', 'super_description']] = descriptions
    
    # Initialize archetype tags
    df['is_class_archetype'] = False
    df['is_arch_archetype'] = False
    df['is_top_archetype'] = False

    # Tag archetype structures
    for index, row in df.iterrows():
        class_num, arch_num, top_num = str(row['Class number']), str(row['Architecture number']), str(row['Topology number'])
        
        # Tag class archetype
        class_archetype_protein_id = cath_dict.get(class_num, {}).get('protein_id', "")
        if class_archetype_protein_id and class_archetype_protein_id[:4] in row['design_name']:
            df.at[index, 'is_class_archetype'] = True
        
        # Tag architecture archetype
        arch_archetype_protein_id = cath_dict.get(class_num, {}).get(arch_num, {}).get('protein_id', "")
        if arch_archetype_protein_id and arch_archetype_protein_id[:4] in row['design_name']:
            df.at[index, 'is_arch_archetype'] = True
        
        # Tag topology archetype
        top_archetype_protein_id = cath_dict.get(class_num, {}).get(arch_num, {}).get(top_num, {}).get('protein_id', "")
        if top_archetype_protein_id and top_archetype_protein_id[:4] in row['design_name']:
            df.at[index, 'is_top_archetype'] = True

    return df

# ---------------------------------------------------------------------------------------------------
# PREPROCESSING

def process_data(df, normalise_columns, destress_columns, tolerance=0.39, variance_threshold = 0.1):
    non_destress_columns = df.drop(columns=destress_columns, errors='ignore')
    valid_destress_columns = [col for col in destress_columns if col in df.columns]
    
    # Drop columns with a high percentage of missing values
    threshold = 0.15
    missing_percentage = df[valid_destress_columns].isnull().sum() / len(df)
    columns_to_drop_due_to_missing = []

    print("\tDropping columns due to missing values > {:.0f}%:".format(threshold * 100))
    for col in valid_destress_columns:
        percentage = missing_percentage[col] * 100
        if percentage > threshold * 100:
            print(f"\t\t- {col}: {percentage:.2f}%")
            columns_to_drop_due_to_missing.append(col)

    df = df.drop(columns=columns_to_drop_due_to_missing)
    valid_destress_columns = [col for col in valid_destress_columns if col not in columns_to_drop_due_to_missing]


    if 'num_residues' in df.columns:
        for feature in [col for col in normalise_columns if col in valid_destress_columns]:
            df[feature] = df[feature] / df['num_residues']
    
    # Dropping 'mass' and 'num_residues' explicitly
    print("\n\tDropping columns explicitly: mass, num_residues\n")
    df = df.drop(['mass', 'num_residues'], axis=1, errors='ignore')
    valid_destress_columns = [col for col in valid_destress_columns if col not in ['mass', 'num_residues']]

    # Compute and plot the correlation matrix BEFORE removing highly correlated features
    corr_matrix_before = df[valid_destress_columns].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix_before, cmap='viridis', vmin=-1, vmax=1)
    plt.title("Correlation Matrix Before Removing Highly Correlated Features")
    plt.tight_layout()
    filename = "correlation_matrix_before.png"
    file_path = os.path.join(base_save_folder, filename)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Drop highly correlated features
    dropped_features = []
    while True:
        corr_matrix = df[valid_destress_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > tolerance)]
        if not to_drop:
            break
        feature_to_remove = to_drop[0]
        df = df.drop(columns=[feature_to_remove], errors='ignore')
        valid_destress_columns.remove(feature_to_remove)
        dropped_features.append(feature_to_remove)
    print(f"\tDropped features due to high correlation >{tolerance*100:.2f}%:\n\t\t- " + "\n\t\t- ".join(dropped_features) + "\n")

    # Compute and plot the correlation matrix AFTER removing highly correlated features
    corr_matrix_before = df[valid_destress_columns].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix_before, cmap='viridis', annot=True, fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": 10})
    plt.title("Correlation Matrix After Removing Highly Correlated Features")
    plt.tight_layout()
    filename = "correlation_matrix_after.png"
    file_path = os.path.join(base_save_folder, filename)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Drop columns with little or no variance
    variances = df[valid_destress_columns].var()
    if valid_destress_columns:
        selector = VarianceThreshold(threshold=variance_threshold)
        df_filtered = pd.DataFrame(selector.fit_transform(df[valid_destress_columns]),
                                columns=[valid_destress_columns[x] for x in selector.get_support(indices=True)],
                                index=df.index)
        df.update(df_filtered)
        valid_destress_columns = df_filtered.columns.tolist()
        cols_to_drop_due_to_variance = set(valid_destress_columns) ^ set(df_filtered.columns)
        print("\tDropped features due to little/no variance:", cols_to_drop_due_to_variance, "\n")

    dropped_features = ['rosetta_pro_close', 'aggrescan3d_min_value', 'aggrescan3d_max_value', 'rosetta_dslf_fa13', 'rosetta_yhh_planarity']
    df = df.drop(dropped_features, axis=1, errors='ignore')
    valid_destress_columns = [col for col in valid_destress_columns if col not in dropped_features]
    print(f"\tDropped features due to skew:", dropped_features, "\n")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[valid_destress_columns])
    df_scaled = pd.DataFrame(df_scaled, columns=valid_destress_columns, index=df.index)

    print(f"\tFeatures used: {list(df[valid_destress_columns].columns)}\n")

    df_processed = pd.concat([df_scaled, non_destress_columns], axis=1)
    df_processed = df_processed.dropna()

    return df_processed

# ---------------------------------------------------------------------------------------------------
# MANIPULATION

df = pd.read_csv(path)
original_columns = set(df.columns)

with open(cath_dict_path, 'r') as file:
    cath_dict = json.load(file)

print(f"Total number of structures: {len(df)}\n")

df_labelled = add_cath_data(df, cath_dict)
df_processed = process_data(df_labelled, normalise_columns, destress_columns)

print(f"Total number of processed structures: {len(df_processed)}\n")

# ---------------------------------------------------------------------------------------------------
# TSNE ARCHITECTURE by TOPOLOGY

n_components = 2 
perplexity = 20
learning_rate = 800
n_iter = 3000
random_state = 42

for arch_name in df_processed['arch_description'].unique():
    df_arch_processed = df_processed[df_processed['arch_description'] == arch_name]

    print("--------------------------------\n")
    print(f"{arch_name}:\n")
    if df_arch_processed.shape[0] <= perplexity:
        print(f"\tNot enough data for t-SNE on {arch_name}. Skipping...")
        continue
    print("--------------------------------\n")
    
    unsanitised_arch_name = arch_name
    arch_name = arch_name.replace(' ', '_').replace('/', '_')

    arch_save_folder = os.path.join(base_save_folder, "by_arch", f"arch_{arch_name.replace(' ', '_')}")
    if not os.path.exists(arch_save_folder):
        os.makedirs(arch_save_folder)

    numeric_destress_columns = [col for col in destress_columns if col in df_arch_processed.columns and np.issubdtype(df_arch_processed[col].dtype, np.number)]
    df_arch_numeric = df_arch_processed[numeric_destress_columns].dropna()

    # t-SNE
    if df_arch_numeric.shape[0] > 1 and df_arch_numeric.shape[1] > 1:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
        tsne_results = tsne.fit_transform(df_arch_numeric)
        tsne_df = pd.DataFrame(tsne_results, columns=['D1', 'D2'], index=df_arch_numeric.index)
    else:
        print(f"Skipping PCA for {unsanitised_arch_name}: Insufficient data. Shape: {df_arch_numeric.shape}")
        continue

    tsne_df['top_description'] = df_arch_processed.loc[df_arch_numeric.index, 'top_description']
    tsne_df['is_top_archetype'] = df_arch_processed.loc[df_arch_numeric.index, 'is_top_archetype']

    print(f"\tTopologies:", ", ".join(tsne_df['top_description'].unique()), "\n")

    # ---------------------------------------------------------------------------------------------------
        
    # Plotting
    plt.figure(figsize=(10, 10))
    
    # Plot non-archetypal structures
    sns.scatterplot(
        data=tsne_df[~tsne_df['is_top_archetype']],
        x='D1', y='D2',
        hue='top_description',
        palette="Spectral",
        marker='o',
        s=50,
        alpha=0.8,
    )

    # Archetypal structures
    sns.scatterplot(
        data=tsne_df[tsne_df['is_top_archetype']],
        x='D1', y='D2',
        hue='top_description',
        palette="Spectral",        
        marker='o',
        s=50,
        edgecolor='black',
        linewidth=2,
        legend=False
    )

    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize='small')
    legend.get_frame().set_alpha(0.1)
    plt.title(f't-SNE for {unsanitised_arch_name} (Architecture) by Topology')
    plt.xlabel('D1')
    plt.ylabel('D2')

    top_counts = df_arch_processed['top_description'].value_counts()
    print(f"\tTopology counts:")
    for top, count in top_counts.items():
        print(f"\t\t- {top}: {count}")

    print(f"\n\tPerplexity: {tsne.perplexity}, Learning Rate: {tsne.learning_rate}, Iterations: {tsne.n_iter}\n")
        
    arch_save_folder = os.path.join(base_save_folder, "by_arch", f"arch_{arch_name.replace(' ', '_')}")
    if not os.path.exists(arch_save_folder):
        os.makedirs(arch_save_folder)
    
    plt.savefig(os.path.join(arch_save_folder, f"{arch_name.replace(' ', '_')}_tsne.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\tArchitecture number of structures: {len(tsne_df)}", "\n")
    print(f"\tAnalysis completed for {unsanitised_arch_name}.\n")

print("--------------------------------\n")
print(f"\tAnalysis completed for Architectures.\n")
sys.stdout = original_stdout

print("All t-SNE complete")

# ---------------------------------------------------------------------------------------------------

    
