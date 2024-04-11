# SUPERFAMILY SPECIFIC PLOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import seaborn as sns
import textwrap
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# INPUT

cath_id = "1.10.510.10"

# ---------------------------------------------------------------------------------------------------
# LOAD

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"
uniprot_data_path = "/Volumes/dax-hd/project-data/search-files/uniprot-data.csv"
base_save_folder = "/Volumes/dax-hd/project-data/images/figs/"
log_directory = os.path.join(base_save_folder, "by_super")

if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

path_to_log_file = os.path.join(log_directory, "super_log_pca.txt")

with open(cath_dict_path, 'r') as file:
    cath_dict = json.load(file)

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

# ---------------------------------------------------------------------------------------------------
# CATH ID DF SEARCH

def filter_df_by_cath_id(df, cath_id):

    class_num, arch_num, topo_num, homsf_num = cath_id.split('.')
    
    class_num, arch_num, topo_num, homsf_num = int(class_num), int(arch_num), int(topo_num), int(homsf_num)
    
    df_filtered = df[(df['Class number'] == class_num) &
                     (df['Architecture number'] == arch_num) &
                     (df['Topology number'] == topo_num) &
                     (df['Homologous superfamily number'] == homsf_num)]
    
    return df_filtered

# ---------------------------------------------------------------------------------------------------
# ADD CATH NAMES

def add_cath_data(df, cath_dict):
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

def process_data(df, normalise_columns, destress_columns, tolerance=0.6, variance_threshold = 0.05):
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
    print("\n\tDropping columns explicitly: \n\t\t- mass\n\t\t- num_residues\n")
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
    original_columns_set = set(valid_destress_columns)
    selector = VarianceThreshold(threshold=variance_threshold)
    df_filtered = selector.fit_transform(df[valid_destress_columns])
    columns_mask = selector.get_support(indices=True)
    valid_destress_columns = [valid_destress_columns[i] for i in columns_mask]
    df = df[valid_destress_columns + list(non_destress_columns.columns)]
    dropped_columns_due_to_variance = original_columns_set - set(valid_destress_columns)
    if dropped_columns_due_to_variance:
        print(f"\tDropped features due to little/no variance: " + "\n\t\t- " + "\n\t\t- ".join(dropped_columns_due_to_variance) + "\n")
    else:
        print("\tNo features dropped due to little/no variance.\n")


    dropped_features = ['rosetta_pro_close', 'rosetta_dslf_fa13', 'rosetta_yhh_planarity', 'rosetta_omega', 'aggrescan3d_min_value', 'aggrescan3d_max_value']
    df = df.drop(dropped_features, axis=1, errors='ignore')
    valid_destress_columns = [col for col in valid_destress_columns if col not in dropped_features]
    print(f"\tDropped features due to skew:\n\t\t- " + "\n\t\t- ".join(dropped_features) + "\n")


    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[valid_destress_columns])
    df_scaled = pd.DataFrame(df_scaled, columns=valid_destress_columns, index=df.index)

    print(f"\tFeatures used:\n\t\t- " + "\n\t\t- ".join(df[valid_destress_columns].columns) + "\n")


    df_processed = pd.concat([df_scaled, non_destress_columns], axis=1)
    df_processed = df_processed.dropna()

    return df_processed

# ---------------------------------------------------------------------------------------------------
# PCA of SPECIFIC SUPERFAMILY

def add_protein_name(df, uniprot_data):
    df = pd.merge(df, uniprot_data[['PDB ID', 'Protein name']], left_on='design_name', right_on='PDB ID', how='left')
    df['Protein name'] = df['Protein name'].fillna('Unknown')
    df.drop(columns='PDB ID', inplace=True)
    return df

# ---------------------------------------------------------------------------------------------------
# PCA of SPECIFIC SUPERFAMILY

def plot_pca(df_processed, super_name):
    for super_name in df_processed['super_description'].unique():
        df_super_processed = df_processed[df_processed['super_description'] == super_name]

        print("--------------------------------\n")
        print(f"{super_name}:\n")
        print("--------------------------------\n")


        # Ensure the directory for saving plots exists
        super_save_folder = os.path.join(base_save_folder, "by_super", f"super_{super_name.replace(' ', '_')}")
        if not os.path.exists(super_save_folder):
            os.makedirs(super_save_folder)

        # Select only destress columns that are numeric
        numeric_destress_columns = [col for col in destress_columns if col in df_super_processed.columns and np.issubdtype(df_super_processed[col].dtype, np.number)]
        df_super_numeric = df_super_processed[numeric_destress_columns].dropna()

        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_super_numeric)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df_super_numeric.index)
        
        # Add superitecture descriptions for plotting
        pca_df['Protein name'] = df_super_processed.loc[df_super_numeric.index, 'Protein name']

        print(f"\tStructures:", ", ".join(pca_df['Protein name'].unique()), "\n")

        explained_variance = pca.explained_variance_ratio_ * 100

        # ---------------------------------------------------------------------------------------------------
        # figure
        num_unique_structures = len(pca_df['Protein name'].unique())
        color_indices = np.linspace(0, 1, num_unique_structures)
        spectral_colors = plt.cm.Spectral(color_indices)

        # Mapping each superitecture to a color
        colour_map = dict(zip(pca_df['Protein name'].unique(), spectral_colors))

        plt.figure(figsize=(10, 10))

        # Plot non-superetypal structures
        sns.scatterplot(
            x='PC1', y='PC2',
            data=pca_df,
            hue='Protein name',
            palette=colour_map,
            marker='o',
            s=50
        )
    
        legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize='small')
        legend.get_frame().set_alpha(0.1)
        plt.title(f'PCA for {super_name} (Superfamily)')
        plt.xlabel(f'PC1: {explained_variance[0]:.2f}%')
        plt.ylabel(f' PC2: {explained_variance[1]:.2f}%')

        plt.savefig(os.path.join(super_save_folder, f"{super_name.replace(' ', '_')}_pca.png"), dpi=300, bbox_inches='tight')
        plt.close()

        super_counts = df_super_processed['super_description'].value_counts()
        print(f"\tsuperitecture counts:\n")
        for super, count in super_counts.items():
            print(f"\t\t- {super}: {count}")
        print("\n")    

        # Component loading plot
        pca_feature_names = df_super_numeric.columns.tolist()

        # When plotting component loadings:
        for i in range(pca.n_components_):
            plt.figure(figsize=(10, 6))
            component_loadings = pca.components_[i]
            sorted_indices = np.argsort(np.abs(component_loadings))[::-1]
            sorted_loadings = component_loadings[sorted_indices]
            sorted_feature_names = np.array(pca_feature_names)[sorted_indices]

            plt.bar(x=range(len(sorted_feature_names)), height=sorted_loadings)
            plt.xticks(ticks=range(len(sorted_feature_names)), labels=sorted_feature_names, rotation=90)
            plt.title(f'PCA Component {i+1} Loadings for {super_name}')
            plt.xlabel('Features')
            plt.ylabel('Loading Value')
            plt.tight_layout()
            plt.savefig(os.path.join(super_save_folder, f'pc_{i+1}_loadings_{super_name}.png'))
            plt.close()

        print(f"\tSuperfamily number of structures: {len(pca_df)}", "\n")
        print("\tVariance explained by each component:", explained_variance, "\n")
        print(f"\tAnalysis completed for {super_name}.\n")

# ---------------------------------------------------------------------------------------------------
# MAIN

def main():

    print("--------------------------------\n")
    print("------ PCA for SUPERFAMILY -----\n")

    df = pd.read_csv(path)
    uniprot_data = pd.read_csv(uniprot_data_path)

    print(f"Total number of structures: {len(df)}\n")
    
    df_labelled = add_cath_data(df, cath_dict)
    df_processed = process_data(df_labelled, normalise_columns, destress_columns)
    df_filtered = filter_df_by_cath_id(df_processed, cath_id)
    df_final = add_protein_name(df_filtered, uniprot_data)
    structures_to_keep = [
    'Ephrin type-A receptor 2',
    'Tyrosine-protein kinase BTK',
    'Tyrosine-protein kinase JAK1',
    'Cyclin-dependent kinase 2',
    'Hepatocyte growth factor receptor',
    'Epidermal growth factor receptor',
    'Vascular endothelial growth factor receptor 2',
    'Tyrosine-protein kinase ABL1',
    'Mitogen-activated protein kinase 1',
    'Tyrosine-protein kinase JAK2',
    'Serine/threonine-protein kinase Chk1',
    'TGF-beta receptor type-1',
    'TGF-beta receptor type-2',
    'Serine/threonine-protein kinase ULK1',
    'Calcium/calmodulin-dependent protein kinase type II',
    'Dual specificity mitogen-activated protein kinase kinase 1',
    'Serine/threonine-protein kinase B-raf'
]
    df_final = df_final[df_final['Protein name'].isin(structures_to_keep)]

    super_name = df_filtered['super_description'].iloc[0]
    plot_pca(df_final, super_name)
    
    print(f"Total number of filtered structures: {len(df_filtered)}\n")
    print("--------------------------------\n")
    print("\tAnalysis completed for Homologous Superfamily.\n")
    
    print("All PCA completed.")

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()



