import pickle
import pandas as pd

def calculate_abundance(clusters_file):
    with open(clusters_file, 'rb') as f:
        clusters = pickle.load(f)
    
    abundance = {}
    for cluster_id, sequences in clusters.items():
        abundance[cluster_id] = len(sequences)
    
    df = pd.DataFrame(list(abundance.items()), columns=['Cluster', 'Abundance'])
    df.to_csv('results/abundance.csv', index=False)
    print("Abundance saved to results/abundance.csv")
    print(df)

if __name__ == "__main__":
    calculate_abundance('results/clusters.pkl')