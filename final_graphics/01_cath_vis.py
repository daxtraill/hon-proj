import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import seaborn as sns
import os
import json

font = "Andale Mono"

# ---------------------------------------------------------------------------------------------------
# LOAD

data_path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"
cath_dict_path = "/Volumes/dax-hd/project-data/search-files/cath-archetype-dict.txt"
base_save_folder = "/Volumes/dax-hd/project-data/images/cath/"
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

def plot_architecture_counts(df_labelled):
    custom_palette = ['#ed5054', '#3f93b4', '#e0ac24', '#fff785', '#dbcfc2']

    
    class_order = ["Mainly Alpha", "Mainly Beta", "Alpha Beta", "Few Secondary Structures", "Special"]
    
    df_labelled['class_description'] = pd.Categorical(df_labelled['class_description'], categories=class_order, ordered=True)
    
    df_labelled['unique_arch'] = df_labelled['Class number'].astype(str) + '.' + \
                                 df_labelled['Architecture number'].astype(str) + ':   ' + \
                                 df_labelled['arch_description']
    
    arch_counts = df_labelled.groupby(['class_description', 'unique_arch'], observed=True).size().reset_index(name='counts')
    filtered_arch_counts = arch_counts[arch_counts['counts'] >= 40].copy()
    total_count = filtered_arch_counts['counts'].sum()
    filtered_arch_counts['percentage'] = (filtered_arch_counts['counts'] / total_count) * 100
    
    plt.figure(figsize=(8, 12), dpi=300)

    barplot = sns.barplot(data=filtered_arch_counts, y='unique_arch', x='percentage',
                          hue='class_description', dodge=False,
                          palette=custom_palette, edgecolor='none')

    new_labels = [label.get_text().split(':')[-1] for label in barplot.get_yticklabels()]
    barplot.set_yticklabels(new_labels, ha='left', va='center', weight='heavy', size='15', minor=False)
    barplot.tick_params(axis='y', length=0)
    
    for label in barplot.get_yticklabels():
        label_x_position = -0.038
        label_y_position = label.get_position()[1]
        label.set_position((label_x_position, label_y_position))
        
    plt.xlabel('% of structures', fontname=font)
    plt.ylabel(None)
    plt.title('Architecture Counts', fontname=font, size=16)

    for label in barplot.get_xticklabels() + barplot.get_yticklabels():
        label.set_fontname(font)

    legend = plt.legend(bbox_to_anchor=(0.45, 0.7), loc='center left', borderaxespad=0., frameon=False)
    plt.setp(legend.get_texts(), fontname=font, weight='heavy', size='15')
    
    barplot.spines['top'].set_visible(False)
    barplot.spines['right'].set_visible(False)
    barplot.spines['bottom'].set_visible(True)
    barplot.spines['left'].set_visible(True)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------

def plot_class_pie(df_labelled):
    custom_palette = ['#ed5054', '#3f93b4', '#e0ac24', '#fff785', '#dbcfc2']
    class_order = {1: "Mainly Alpha", 2: "Mainly Beta", 3: "Alpha Beta", 4: "Few Secondary Structures", 6: "Special"}

    # Counting occurrences of each class
    class_counts = df_labelled.groupby(['Class number']).size().reset_index(name='counts')
    class_counts['Class description'] = class_counts['Class number'].map(class_order)

    # Creating pie chart
    fig = go.Figure(data=[go.Pie(labels=class_counts['Class description'],
                                 values=class_counts['counts'],
                                 marker=dict(colors=custom_palette, 
                                              line=dict(color='black', width=1)),
                                 textinfo='percent+label',
                                 pull=[0.1] * len(class_counts))])  # Slightly pull slices out

    # Customizing layout
    fig.update_layout(title_text='Distribution of Class Numbers',
                      title_font=dict(size=20),  # Example to set the title font size
                      # Here you can set the global font used in the chart, if the font is available
                      font=dict(family='Andale Mono', size=18, color='black')) 

    fig.write_html("pichart.html")

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

def main():
    # Load data
    df = pd.read_csv(data_path)
    with open(cath_dict_path, 'r') as file:
        cath_dict = json.load(file)

    df_labelled = add_cath_data(df, cath_dict)
    df_processed = process_data(df_labelled, normalise_columns, destress_columns)
    # Plot architecture counts
    plot_architecture_counts(df_processed)
    # plot_class_pie(df_labelled)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------