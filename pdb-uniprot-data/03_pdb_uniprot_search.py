import requests
import json

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
                    print(f"Found UniProt Accession: {accession} for PDB ID {pdb_id} - {details['name']}")
            return uniprot_ids
        else:
            print(f'Error fetching data from UniProt for PDB ID {pdb_id}. Status Code: {response.status_code}')
            return None
    except requests.RequestException as e:
        print(f'Request failed: {e}')
        return None
    
# ---------------------------------------------------------------------------------------------------

def fetch_uniprot_data(uniprot_id):
    uniprot_api_url = f'https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}'
    headers = {'Accept': 'application/json'}
    try:
        response = requests.get(uniprot_api_url, headers=headers)
        if response.status_code == 200:
            uniprot_data = response.json()
            organism_names = [name.get('value') for name in uniprot_data.get('organism', 
        {}).get('names', []) if name.get('type') in ['scientific', 'common']]
        
            selected_data = {
                'UniProt Accession': uniprot_data.get('accession'),
                'Protein name': uniprot_data.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                'Subcellular location': extract_comment_text(uniprot_data.get('comments', []), 'SUBCELLULAR_LOCATION'),
                'Organism name': ', '.join(organism_names),
                'Function': extract_comment_text(uniprot_data.get('comments', []), 'FUNCTION'),
                'Lineage': uniprot_data.get('organism', {}).get('lineage'),
                'Sequence length': uniprot_data.get('sequence', {}).get('length'),
                'Protein molecular weight': uniprot_data.get('sequence', {}).get('mass'),
                'Amino acid sequence': uniprot_data.get('sequence', {}).get('sequence')
            }
            pretty_selected_data = json.dumps(selected_data, indent=4)
            return pretty_selected_data
        else:
            print(f'Error fetching data for UniProt ID {uniprot_id}. Status Code: {response.status_code}')
            return None
    except requests.RequestException as e:
        print(f'Request failed: {e}')
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

def main():
    while True:
        pdb_id = input("Enter PDB ID (or type 'exit' to quit): ")
        if pdb_id.lower() == 'exit':
            print("Exiting the program.")
            break
        uniprot_ids = scrape_uniprot_id(pdb_id)
        
        if uniprot_ids:
            for uniprot_id in uniprot_ids:
                uniprot_data = fetch_uniprot_data(uniprot_id)
                if uniprot_data:
                    print(f"UniProt Data for {uniprot_id}:")
                    print(uniprot_data)
        else:
            print(f"No UniProt Accessions found for PDB ID {pdb_id}.")
        input("Press Enter to search another...")

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()