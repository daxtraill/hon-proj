import pandas as pd
from io import StringIO
import csv

# Sample CSV data
input_csv_data = "/Volumes/dax-hd/project-data/brenda/cleaned_ph_optimum_brenda_results.csv"
output_csv_path = "/Volumes/dax-hd/project-data/brenda/grouped_ph_optimum_results.csv"


# Load data into a DataFrame
data = pd.read_csv((input_csv_data))

# Replace '-' with NaN to allow numerical operations and convert columns to numeric
data.replace('-', pd.NA, inplace=True)
data['pH Optimum Minimum'] = pd.to_numeric(data['pH Optimum Minimum'], errors='coerce')
data['pH Optimum Maximum'] = pd.to_numeric(data['pH Optimum Maximum'], errors='coerce')

# Fill missing Optimum Maximum values with the Minimum values for calculation
data['pH Optimum Maximum'].fillna(data['pH Optimum Minimum'], inplace=True)

# Calculate average, min, and max for each group
grouped = data.groupby(['EC ID', 'Enzyme Name']).agg(
    Average_Optimum_pH=('pH Optimum Minimum', 'mean'),
    Min_Optimum_pH=('pH Optimum Minimum', 'min'),
    Max_Optimum_pH=('pH Optimum Maximum', 'max')
).reset_index()

# Display the result
grouped.to_csv(output_csv_path, index=False)
print(f"Grouped data saved to {output_csv_path}")
