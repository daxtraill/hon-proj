cath_architype_dict = {
    1: {
        10: 'orthogonal_bundle', 20: 'up_down_bundle', 25: 'alpha_horseshoe', 40: 'alpha_solenoid', 50: 'alpha_alpha_barrel'
    },
    2: {
        10: 'ribbon', 20: 'single_sheet', 30: 'beta_roll', 40: 'beta_barrel', 50: 'clam', 60: 'sandwich', 70: 'distorted_sandwich',
        80: 'trefoil', 90: 'orthogonal_prism', 100: 'aligned_prism', 102: 'three_layer_sandwich', 105: 'three_propeller',
        110: 'four_propeller', 115: 'five_propeller', 120: 'six_propeller', 130: 'seven_propeller', 140: 'eight_propeller',
        150: 'two_solenoid', 160: 'three_solenoid', 170: 'beta_complex', 180: 'shell'
    },
    3: {
        10: 'alpha_beta_roll', 15: 'super_roll', 20: 'alpha_beta_barrel', 30: 'two_layer_sandwich', 40: 'three_layer_aba_sandwich',
        50: 'three_layer_bab_sandwich', 55: 'three_layer_bba_sandwich', 60: 'four_layer_sandwich', 65: 'alpha_beta_prism', 70: 'box',
        75: 'five_stranded_propeller', 80: 'alpha_beta_horseshoe', 90: 'alpha_beta_complex', 100: 'ribosomal_protein_l15_chain_k_domain_two'
    },
    4: {
        10: 'irregular'
    },
    6: {
        10: 'helix_non_globular', 20: 'other_non_globular'
    }
}

import json

def process_cath_file_to_json(input_file_path, output_json_path):
    # Initialize the root of our nested dictionary
    data = {}

    with open(input_file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Skip invalid lines

            cath_id = parts[0]
            protein_id = parts[1]
            description = " ".join(parts[2:])[1:]  # Remove leading ':' from description
            cath_id_parts = cath_id.split('.')

            # Drill down into data dictionary based on the parts of the CATH ID
            current_level = data
            for part in cath_id_parts:
                if part not in current_level:
                    current_level[part] = {}  # Initialize a new dictionary for this part if it doesn't exist
                current_level = current_level[part]

            # After drilling down, set the protein_id and description at the current level
            current_level["protein_id"] = protein_id
            current_level["description"] = description

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

input = '/Volumes/dax-hd/project-data/search-files/cath-names.txt'
output = '/Volumes/dax-hd/project-data/search-files/cath-architype-dict.txt'
process_cath_file_to_json(input, output)
