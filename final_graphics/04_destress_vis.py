import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import json
import sys
import os

# ---------------------------------------------------------------------------------------------------
# LOAD DATA

feature_data_path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
base_save_folder = "/Volumes/dax-hd/project-data/final-figures/"

if not os.path.exists(base_save_folder):
    os.makedirs(base_save_folder)

save_folder = os.path.join(base_save_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

font = 'Andale Mono'

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
# DATA PROCESSING

def process_data(df, normalise_columns, destress_columns, tolerance=0.6, variance_threshold = 0.05):
    print("---------PREPROCESSING----------\n")
    non_destress_columns = df.drop(columns=destress_columns, errors='ignore')
    valid_destress_columns = [col for col in destress_columns if col in df.columns]

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
    
    print("\n\tDropping columns explicitly: \n\t\t- mass\n\t\t- num_residues\n")
    df = df.drop(['mass', 'num_residues'], axis=1, errors='ignore')
    valid_destress_columns = [col for col in valid_destress_columns if col not in ['mass', 'num_residues']]

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

    df_processed = df_scaled.dropna()
    print("---------PREPROCESSING----------\n")
    return df_processed

# ---------------------------------------------------------------------------------------------------
# MAIN

def main():
    df = pd.read_csv(feature_data_path)

    # Scale data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[destress_columns]), columns=destress_columns)

    # Remove outliers using z-score
    z_scores = np.abs(df_scaled)
    filtered_entries = (z_scores < 3).all(axis=1)
    df_filtered = df_scaled[filtered_entries]

    fig = go.Figure()

    # Use different colors for each violin
    colors = px.colors.qualitative.Plotly

    for index, feature in enumerate(destress_columns):
        color = colors[index % len(colors)]  # Cycle through colors if fewer than columns
        fig.add_trace(go.Violin(y=df_filtered[feature], name=feature, box_visible=False, line_color='black', fillcolor=color, opacity=0.6))

    # Update layout to fit all violins and make it scrollable if needed
    fig.update_layout(title="Violin Plots of Pre-processed Destress Features",
                      yaxis_zeroline=False,
                      violingap=0, violingroupgap=0, violinmode='overlay')

    # Save the figure
    save_path = os.path.join(base_save_folder, "all_destress_features_violin_plot.html")
    fig.write_html(save_path)
    print(f"All violin plots are generated and saved at {save_path}")

    df = pd.read_csv(feature_data_path)

    df_processed2 = process_data(df, normalise_columns, destress_columns)
    
    fig = go.Figure()
    columns = df_processed2.columns
    # Use different colors for each violin
    colors = px.colors.qualitative.Plotly

    for index, feature in enumerate(columns):
        color = colors[index % len(colors)]  # Cycle through colors if fewer than columns
        fig.add_trace(go.Violin(y=df_processed2[feature], name=feature, box_visible=False, line_color='black', fillcolor=color, opacity=0.6))

    # Update layout to fit all violins and make it scrollable if needed
    fig.update_layout(title="Violin Plots of Processed Destress Features",
                      yaxis_zeroline=False,
                      violingap=0, violingroupgap=0, violinmode='overlay')

    # Save the figure
    save_path = os.path.join(base_save_folder, "processed_destress_features_violin_plot.html")
    fig.write_html(save_path)
    print(f"Processed violin plots are generated and saved at {save_path}")

if __name__ == "__main__":
    main()