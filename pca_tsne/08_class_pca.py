import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# Load the dataset

path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"
base_save_folder = "/Volumes/dax-hd/project-data/images/figs/"

df = pd.read_csv(path)
original_columns = set(df.columns)

with open(cath_dict_path, 'r') as file:
    cath_dict = json.load(file)
if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)

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

# ---------------------------------------------------------------------------------------------------
# Add the architecture name to df

def add_class_name(df, cath_dict):
    def get_class_name(row):
        class_num = str(row['Class number'])
        try:
            description = cath_dict[class_num]['description']
            return description
        except KeyError:
            return "Unknown"
    
    df['class_description'] = df.apply(get_class_name, axis=1)
    return df

df = add_class_name(df, cath_dict)

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

def filter_for_archetypes(df, cath_dict):
    archetype_ids = []
    for _, row in df.iterrows():
        class_num = str(row['Class number'])
        arch_num = str(row['Architecture number'])
        top_num = str(row['Topology number'])
        try:
            protein_id = cath_dict[class_num][arch_num][top_num]['protein_id']
            if protein_id[:4] in row['design_name']:
                archetype_ids.append(row['design_name'])
        except KeyError:
            continue
    return df[df['design_name'].isin(archetype_ids)]

df = filter_for_archetypes(df, cath_dict)

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
            
df = df.drop([
    'mass', 'num_residues'
    ], axis=1)

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

df = df.drop([
    'evoef2_interD_total', 'aggrescan3d_min_value', 'aggrescan3d_max_value',
    'rosetta_fa_rep', 'rosetta_dslf_fa13', 'rosetta_omega', 'rosetta_yhh_planarity'
    ], axis=1)

pca_columns = [col for col in destress_columns if col in df.columns]
df_pca_ready = df[pca_columns].dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca_ready)

# ---------------------------------------------------------------------------------------------------
# Plotting

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

explained_variance = pca.explained_variance_ratio_ * 100

save_folder = os.path.join(base_save_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print(f"Total number of datapoints for class: {len(pca_df)}")

# ---------------------------------------------------------------------------------------------------

pca_df['class_description'] = df['class_description'].values
unique_class = pca_df['class_description'].unique()
class_to_id = {class_: i % 3 for i, class_ in enumerate(unique_class)}

palette = sns.color_palette(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, aspect='equal')

sns.scatterplot(
    x='PC1', y='PC2',
    hue='class_description',
    data=pca_df,
    palette=palette,
    s=100 
)

plt.title(f'PCA for Class')
plt.xlabel(f'PC1: {explained_variance[0]:.2f}%')
plt.ylabel(f' PC2: {explained_variance[1]:.2f}%')
plt.figtext(0.5, 0.01, f"Features used: {pca_columns}", ha="center", fontsize=10)
plt.savefig(os.path.join(save_folder, f"class-pca.png"), bbox_inches='tight')
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

print("PCA completed.")

# ---------------------------------------------------------------------------------------------------