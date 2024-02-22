# This code processes a FASTA file to extract unique Protein Data Bank (PDB) IDs,
# then downloads the corresponding PDB structure files concurrently to a specified directory,
# using Python's multiprocessing for efficiency.

import requests
import os
from os import path as p
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------------
# Import directories

def inputs():
    inDir = "/Volumes/dax-hd/project-data/sequence-data"
    outDir = "/Volumes/dax-hd/project-data/01_pdb_scraping"
    s35Fasta = p.join(inDir, "cath-domain-seqs-S35.fa")

    return inDir, outDir, s35Fasta

# ---------------------------------------------------------------------------------------------------
# Process FASTA files, extracting unique PDB IDs

def main():
    inDir, outDir, s35Fasta = inputs()
    os.makedirs(outDir, exist_ok=True)

    s35Df = fasta2df(s35Fasta)
    s35Df['PDB_ID'] = s35Df['ID'].apply(lambda x: x.split('|')[2].split('/')[0][:4])

    pdb_ids = s35Df["PDB_ID"].unique().tolist()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(scrape_pdb_structures, args=(pdb_id, outDir)) for pdb_id in pdb_ids]
        for r in results:
            r.get()

# ---------------------------------------------------------------------------------------------------
# Retrieve and download PDB files
        
def scrape_pdb_structures(pdb_id, out_dir):
    pdb_id = pdb_id[:4]
    pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response_pdb = requests.get(pdb_url)
    if response_pdb.status_code == 200:
        pdb_file_path = os.path.join(out_dir, f'{pdb_id}.pdb')
        with open(pdb_file_path, 'wb') as pdb_file:
            pdb_file.write(response_pdb.content)
        print(f'Downloaded PDB structure {pdb_id} to {pdb_file_path}')
    else:
        print(f'Error downloading PDB structure {pdb_id}. Status Code: {response_pdb.status_code}')

# ---------------------------------------------------------------------------------------------------
# Convert FASTA file into dataframe
                
def fasta2df(fastaFile):
    fastaList = []
    idLine = ""
    seqLine = ""
    firstLine = True
    dataDicts = []
    for line in open(fastaFile,"r"):
        line = line.strip()
        if line.startswith(">"):
            if not firstLine:
                dataDicts.append({"ID":idLine,"SEQ":seqLine})
            idLine = line[1:]
            seqLine = ""
            firstLine = False
        else:
            seqLine += line
    if not firstLine:
        dataDicts.append({"ID":idLine, "SEQ":seqLine})

    df = pd.DataFrame(dataDicts)
    return df

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()