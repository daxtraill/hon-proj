# This code processes PDB files to remove all non-essential atoms.

import os
from os import path as p
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------------
# Import directories

def inputs():
    # pdbDir ="/Volumes/dax-hd/project-data/01_pdb_scraping"
    # outDir = "/Volumes/dax-hd/project-data/02_pdb_cleanup"
    pdbDir ="/Users/daxtraill/Desktop/test-in"
    outDir = "/Users/datraill/Desktop/test-out"
    return pdbDir, outDir

# ---------------------------------------------------------------------------------------------------
# Clean PDB files

def clean_pdb(pdbFile, outDir, filename):
    excluded_residues = {"HOH"}

    outPdb = p.join(outDir, filename)
    with open(pdbFile, 'r') as inFile, open(outPdb, 'w') as outFile:
        for line in inFile:
            if line.startswith("ATOM"):
                outFile.write(line)
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name not in excluded_residues:
                    outFile.write(line)
            elif line.startswith(("TER", "END")):
                outFile.write(line)

# ---------------------------------------------------------------------------------------------------

def main():
    pdbDir, outDir = inputs()
    os.makedirs(outDir, exist_ok=True)

    pdb_files = [f for f in os.listdir(pdbDir) if f.endswith(".pdb")]
    for file in tqdm(pdb_files, desc="Processing PDB files"):
        pdbFile = p.join(pdbDir, file)
        clean_pdb(pdbFile, outDir, file)

# ---------------------------------------------------------------------------------------------------
        
if __name__ == "__main__":
    main()