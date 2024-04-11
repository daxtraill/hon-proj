# SELECTIVE SUPERFAMILY SPECIFIC PLOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import seaborn as sns
import textwrap
import sys
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# INPUT

cath_id = "1.10.510.10,2.60.40.10,3.40.50.1820"

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
    'Serine/threonine-protein kinase B-raf',
    
    "Ephrin type-A receptor 2",
    "Tyrosine-protein kinase BTK",
    "Tyrosine-protein kinase JAK1",
    "Hepatocyte growth factor receptor",
    "Epidermal growth factor receptor",
    "Tyrosine-protein kinase ABL1",
    "Vascular endothelial growth factor receptor 2",
    "Tyrosine-protein kinase JAK2",
    "cAMP-dependent protein kinase catalytic subunit alpha",
    "Cyclin-dependent kinase 2",
    "RAC-alpha serine/threonine-protein kinase",
    "Serine/threonine-protein kinase MRCK beta",
    "LIM domain kinase 1",
    "Mitogen-activated protein kinase 1",
    "cAMP-dependent protein kinase inhibitor alpha",
    
    'Lipase B',
    'Triacylglycerol lipase',
    'Haloalkane dehalogenase',  # Environmental relevance but included due to its dehalogenase activity
    'Carboxylesterase 1',
    'Monoglyceride lipase',
    'Diacylglycerol acyltransferase/mycolyltransferase Ag85C',  # Relevant in lipid synthesis pathways
    'Feruloyl esterase A',
    'Lipase',
    'Esterase YbfF',
    'Acetyl esterase',
    'Pancreatic lipase-related protein 2',
    'Secreted mono- and diacylglycerol lipase LIP1',
    'triacylglycerol lipase',  # Despite being lowercase, included for functional relevance
    'Platelet-activating factor acetylhydrolase',
    'Thermostable monoacylglycerol lipase',
    'Prolyl endopeptidase',  # Included for its specificity towards proline bonds in peptides
    'Acetylxylan esterase 2',  # Relevant in breakdown of complex carbohydrates but included due to esterase activity
    'Cutinase 1',  # Relevant in degradation of cutin in plant cuticles but included for esterase activity
    'Secreted mono- and diacylglycerol lipase A',
    'Cutinase',
    'Cocaine esterase',  # Specific for cocaine but relevant due to esterase activity
    'L-serine/homoserine O-acetyltransferase',  # Relevant in amino acid metabolism
    'Protein phosphatase methylesterase 1',  # Involved in dephosphorylation but included for esterase activity
    'Acyl transferase',  # Relevant in lipid synthesis
    'Lysosomal protective protein',  # Although not directly a hydrolase, it's included due to its role in lysosomal function
    'Valacyclovir hydrolase',  # Specific but included due to hydrolase activity
    'Acyl-protein thioesterase 1',  # Involved in depalmitoylation, relevant for lipid modification
    'Palmitoyl-protein thioesterase ABHD10, mitochondrial',  # Similar to above
    'Serine carboxypeptidase 2',  # Included for peptidase activity
    'Palmitoyl-protein thioesterase 1',  # Involved in lysosomal degradation of lipid-modified proteins
    'Phospholipase A1-IIgamma',  # Relevant in phospholipid metabolism
    'Carboxypeptidase Y',  # Peptidase activity
    'Juvenile hormone epoxide hydrolase',  # Insect hormone metabolism but included for hydrolase activity
    'Carboxylesterase 2',  # Similar to carboxylesterase 1
    'Liver carboxylesterase 1',  # Similar to carboxylesterase 1
    'Bile salt-activated lipase',  # Relevant in digestion
    'Acetylcholinesterase'
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

def filter_df_by_cath_id(df, cath_ids):
    cath_id_list = cath_ids.split(',')
    filtered_dfs = []

    for cath_id in cath_id_list:
        class_num, arch_num, topo_num, homsf_num = cath_id.strip().split('.')
        filtered_df = df[(df['Class number'] == int(class_num)) &
                         (df['Architecture number'] == int(arch_num)) &
                         (df['Topology number'] == int(topo_num)) &
                         (df['Homologous superfamily number'] == int(homsf_num))]
        filtered_dfs.append(filtered_df)

    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    return combined_df

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
    # Adjust heatmap to only show the lower triangle
    sns.heatmap(corr_matrix_before, cmap='viridis', vmin=-1, vmax=1, annot=False, fmt=".2f", annot_kws={"size": 10})
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
    corr_matrix_after = df[valid_destress_columns].corr()

    plt.figure(figsize=(14, 12))
    # Adjust heatmap to only show the lower triangle
    sns.heatmap(corr_matrix_after, cmap='viridis', vmin=-1, vmax=1, annot=False, fmt=".2f", annot_kws={"size": 10})
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
    super_save_folder = os.path.join(base_save_folder, "combined_supers")
    if not os.path.exists(super_save_folder):
        os.makedirs(super_save_folder)

    # Initialize an empty DataFrame for PCA results
    pca_combined_df = pd.DataFrame()

    markers = ['o', 's', '^', 'D', 'p', '*', 'X']  # Markers for different superfamilies
    colours = ['#ed5054', '#3f93b4', '#e0ac24', '#fff785', '#dbcfc2']  # Colours for different superfamilies
    super_names = []  # To keep track of superfamilies

    for i, super_name in enumerate(df_processed['super_description'].unique()):
        df_super_processed = df_processed[df_processed['super_description'] == super_name]
        super_names.append(super_name)
        
        # Select only destress columns that are numeric
        numeric_destress_columns = [col for col in destress_columns if col in df_super_processed.columns and np.issubdtype(df_super_processed[col].dtype, np.number)]
        df_super_numeric = df_super_processed[numeric_destress_columns].dropna()

        # Add identifiers to the DataFrame
        df_super_numeric['Protein name'] = df_super_processed['Protein name']
        df_super_numeric['super_description'] = super_name
        df_super_numeric['Marker'] = markers[i % len(markers)]  # Assign a marker for the superfamily

        # Append the prepared DataFrame to the combined DataFrame
        pca_combined_df = pd.concat([pca_combined_df, df_super_numeric], ignore_index=True)

    # Perform PCA on the combined DataFrame
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(pca_combined_df[numeric_destress_columns])
    pca_combined_df[['PC1', 'PC2']] = pca_results
    
    explained_variance = pca.explained_variance_ratio_ * 100


    # Plotting
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='PC1', y='PC2',
        data=pca_combined_df,
        hue='super_description',
        style='super_description',
        palette=colours[:len(super_names)],  # Use the colours list
        markers={name: markers[i % len(markers)] for i, name in enumerate(super_names)},  # Map superfamily to marker
        s=75, alpha=0.7, edgecolor='black', linewidth=1
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='Protein name')
    plt.title(f'PCA for combined superfamilies: {", ".join(super_names)}')
    plt.xlabel(f'PC1: {explained_variance[0]:.2f}%')
    plt.ylabel(f' PC2: {explained_variance[1]:.2f}%')

    plt.savefig(os.path.join(super_save_folder, "combined_supers_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()   

    pca_feature_names = df_super_numeric.columns.tolist()

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
        plt.savefig(os.path.join(super_save_folder, f'pc_{i+1}_loadings.png'))
        plt.close()

        print(f"\tSuperfamily number of structures: {len(pca_combined_df)}", "\n")
        print("\tVariance explained by each component:", explained_variance, "\n")
        print(f"\tPCA completed.\n")

# ---------------------------------------------------------------------------------------------------
# t-SNE of SPECIFIC SUPERFAMILY

def plot_tsne(df_processed, super_name, perplexity, n_iter):
    super_save_folder = os.path.join(base_save_folder, "combined_supers")
    if not os.path.exists(super_save_folder):
        os.makedirs(super_save_folder)

    # Initialize an empty DataFrame for the t-SNE results
    tsne_combined_df = pd.DataFrame()

    markers = ['o', 's', '^', 'D', 'p', '*', 'X']  # Markers for different superfamilies
    colours = ['#ed5054', '#3f93b4', '#e0ac24', '#fff785', '#dbcfc2']  # Colours for different superfamilies
    super_names = []  # To keep track of superfamilies

    for i, super_name in enumerate(df_processed['super_description'].unique()):
        df_super_processed = df_processed[df_processed['super_description'] == super_name]
        super_names.append(super_name)
        
        # Select only destress columns that are numeric
        numeric_destress_columns = [col for col in destress_columns if col in df_super_processed.columns and np.issubdtype(df_super_processed[col].dtype, np.number)]
        df_super_numeric = df_super_processed[numeric_destress_columns].dropna()

        # Add identifiers to the DataFrame
        df_super_numeric['Protein name'] = df_super_processed['Protein name']
        df_super_numeric['super_description'] = super_name
        df_super_numeric['Marker'] = markers[i % len(markers)]  # Assign a marker for the superfamily

        # Append the prepared DataFrame to the combined DataFrame
        tsne_combined_df = pd.concat([tsne_combined_df, df_super_numeric], ignore_index=True)
    
    try:
        # Perform t-SNE on the combined DataFrame
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(tsne_combined_df[numeric_destress_columns])
        tsne_combined_df[['Dimension 1', 'Dimension 2']] = tsne_results
        fig = go.Figure()

        unique_super_descriptions = tsne_combined_df['super_description'].unique()
        color_map = {name: colours[i % len(colours)] for i, name in enumerate(unique_super_descriptions)}

        for name in unique_super_descriptions:
            df_subset = tsne_combined_df[tsne_combined_df['super_description'] == name]
            hover_text = df_subset['Protein name']
            fig.add_trace(go.Scatter(
                x=df_subset['Dimension 1'],
                y=df_subset['Dimension 2'],
                mode='markers',
                marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey'), color=color_map[name]),
                name=name,
                text=hover_text,
                hoverinfo='text'
            ))
        fig.update_layout(
            title=f't-SNE for combined superfamilies (Perplexity: {perplexity})',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            legend_title='Super Description',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01
            ))

        fig.write_html(os.path.join(super_save_folder, f"combined_supers_tsne_perplexity_{perplexity}.html"))

        print(f"Success plotting t-SNE for perplexity {perplexity}")

    except Exception as e:
        print(f"Failed to plot t-SNE for perplexity {perplexity} due to error: {e}")

# ---------------------------------------------------------------------------------------------------
# MAIN

def main():

    print("--------------------------------\n")
    print("--- ANALYSIS for SUPERFAMILY ---\n")

    df = pd.read_csv(path)
    uniprot_data = pd.read_csv(uniprot_data_path)

    print(f"Total number of structures: {len(df)}\n")
    
    df_labelled = add_cath_data(df, cath_dict)
    df_processed = process_data(df_labelled, normalise_columns, destress_columns)
    df_filtered = filter_df_by_cath_id(df_processed, cath_id)
    df_restricted = add_protein_name(df_filtered, uniprot_data)
    df_final = df_restricted.drop_duplicates()
    df_final = df_final[df_final['Protein name'].isin(structures_to_keep)]
    for super_name in df_final['super_description'].unique():
        df_filtered2 = df_final[df_final['super_description'] == super_name]
        unique_proteins = df_filtered2['Protein name'].unique()
        print(f"Structure names for {super_name}: {unique_proteins}")

    combined_csv_path = os.path.join(base_save_folder, "combined_supers", "combined_tsne_before_filtering.csv")
    df_final.to_csv(combined_csv_path, index=False)
    print(f"Combined DataFrame saved to CSV: {combined_csv_path}")

    super_name = df_filtered['super_description'].unique()
    plot_pca(df_final, super_name)
    for perplexity in range(5, 71, 5):
        plot_tsne(df_final, super_name, perplexity, n_iter=3000)

    print(f"Total number of filtered structures: {len(df_filtered)}\n")
    print("--------------------------------\n")
    print("\tAnalysis completed for Homologous Superfamily.\n")
    
    print("All completed.")

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()



