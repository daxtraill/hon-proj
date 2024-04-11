import requests
import logging
import json
import csv
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------------

cath_id_path = '/home/dax/project-data/search-files/cath-pdb-ids.txt'

logging.basicConfig(filename="uniprot_accessions.log", level=logging.INFO, format='%(message)s')

# ---------------------------------------------------------------------------------------------------

def scrape_uniprot_id(pdb_id):
    uniprot_sift_url = f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}'
    try:
        response = requests.get(uniprot_sift_url)
        if response.status_code == 200:
            data = response.json()
            uniprot_ids = []
            for pdb_info in data.values():
                for accession, details in pdb_info.get("UniProt", {}).items():
                    uniprot_ids.append(accession)
                    logging.info(f"Found UniProt Accession: {accession} for PDB ID {pdb_id} - {details['name']}")
            return uniprot_ids
        else:
            logging.info(f'Error fetching data from UniProt for PDB ID {pdb_id}. Status Code: {response.status_code}')
            return None
    except requests.RequestException as e:
        logging.info(f'Request failed: {e}')
        return None

# ---------------------------------------------------------------------------------------------------

def extract_comment_text(comments, comment_type):
    if comment_type == 'SUBCELLULAR_LOCATION':
        for comment in comments:
            if comment.get('type') == comment_type:
                locations = comment.get('locations', [])
                location_descriptions = [location.get('location', {}).get('value') for location in locations]
                return ', '.join(location_descriptions)
        return None
    else:
        for comment in comments:
            if comment.get('type') == comment_type:
                return ' '.join([text.get('value') for text in comment.get('text', [])])
        return None

# ---------------------------------------------------------------------------------------------------

def fetch_uniprot_data(uniprot_id, pdb_id):
    uniprot_api_url = f'https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}'
    headers = {'Accept': 'application/json'}
    try:
        response = requests.get(uniprot_api_url, headers=headers)
        if response.status_code == 200:
            uniprot_data = response.json()
            organism_names = [name.get('value') for name in uniprot_data.get('organism', {}).get('names', []) if name.get('type') in ['scientific', 'common']]
            
            selected_data = {
                'PDB ID': pdb_id, 
                'UniProt Accession': uniprot_data.get('accession'),
                'Protein name': uniprot_data.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                'Subcellular location': extract_comment_text(uniprot_data.get('comments', []), 'SUBCELLULAR_LOCATION'),
                'Organism name': ', '.join(organism_names),
                'Lineage': '; '.join(uniprot_data.get('organism', {}).get('lineage', [])),
                'Sequence length': uniprot_data.get('sequence', {}).get('length'),
                'Protein molecular weight': uniprot_data.get('sequence', {}).get('mass'),
            }
            return selected_data
        else:
            print(f'Error fetching data for UniProt ID {uniprot_id}. Status Code: {response.status_code}')
            return None
    except requests.RequestException as e:
        print(f'Request failed: {e}')
        return None

# ---------------------------------------------------------------------------------------------------

def save_to_csv(data, filename="uniprot_data.csv"):
    if data:
        keys = data[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

# ---------------------------------------------------------------------------------------------------

def main():
    pdb_ids = []
    all_uniprot_data = []

    with open(cath_id_path, 'r') as file:
        pdb_ids = [line.strip()[:4] for line in file.readlines()]

    pbar = tqdm(total=len(pdb_ids), desc="Overall Progress")

    for pdb_id in pdb_ids:
        uniprot_ids = scrape_uniprot_id(pdb_id)
        if uniprot_ids:
            for uniprot_id in uniprot_ids:
                uniprot_data = fetch_uniprot_data(uniprot_id, pdb_id)
                if uniprot_data:
                    all_uniprot_data.append(uniprot_data)
        else:
            logging.info(f"No UniProt Accessions found for PDB ID {pdb_id}.")
        pbar.update(1)

    pbar.close()
    save_to_csv(all_uniprot_data, "uniprot_data_combined.csv")
    print(f"Processing completed. Data for {len(all_uniprot_data)} entries saved to 'uniprot_data_combined.csv'.")

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
