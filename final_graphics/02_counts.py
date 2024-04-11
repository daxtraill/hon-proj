import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

s35_csv_path = "/Volumes/dax-hd/project-data/cath-classification-data/cath-domain-list-S35.txt"
s60_csv_path = "/Volumes/dax-hd/project-data/cath-classification-data/cath-domain-list-S60.txt"
s95_csv_path = "/Volumes/dax-hd/project-data/cath-classification-data/cath-domain-list-S95.txt"
s100_csv_path = "/Volumes/dax-hd/project-data/cath-classification-data/cath-domain-list-S100.txt"
total_csv_path = "/Volumes/dax-hd/project-data/cath-classification-data/cath-domain-list.txt"
merged_data_path = "/Volumes/dax-hd/project-data/search-files/merged-data.csv"

def count_csv(path):
    row_count = 0
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for _ in reader:
            row_count += 1
    print(f"Number of structures: {row_count}")
    return row_count

def main():
    csv_paths = [s35_csv_path, s60_csv_path, s95_csv_path, s100_csv_path, total_csv_path, merged_data_path]
    counts = [count_csv(path) for path in csv_paths]
    
    print(counts)
    
    # radii = np.sqrt(np.array(counts) / np.pi)

    # fig, ax = plt.subplots(dpi=300)
    # ax.set_aspect('equal')
    # ax.axis('off')

    # for radius in sorted(radii, reverse=True):
    #     circle = Circle((0, 0), radius, fill=False, edgecolor='black', linewidth=1.5)
    #     ax.add_artist(circle)

    # colors = ['#CCCCCC', '#999999', '#666666', '#333333', '#ed5054']
    
    # x_offset = -max(radii) / 2

    # for i, radius in enumerate(sorted(radii, reverse=True)):
    #     circle = Circle((0, 0), radius, fill=True, color=colors[i], edgecolor='black', linewidth=1.5)
    #     ax.add_artist(circle)
    # max_radius = max(radii)
    # ax.set_xlim(-max_radius, max_radius)
    # ax.set_ylim(-max_radius, max_radius)

    # plt.savefig('circles.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()

    