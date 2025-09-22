import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_biodiversity(abundance_file):
    df = pd.read_csv(abundance_file)
    abundances = df['Abundance'].values
    total_reads = sum(abundances)
    
    # Species richness
    richness = len(abundances)
    
    # Shannon diversity
    proportions = abundances / total_reads
    shannon = entropy(proportions)
    
    # Simpson diversity
    simpson = 1 - sum(proportions ** 2)
    
    # Evenness
    evenness = shannon / np.log(richness) if richness > 1 else 0
    
    metrics = {
        'Richness': richness,
        'Shannon Diversity': shannon,
        'Simpson Diversity': simpson,
        'Evenness': evenness
    }
    
    with open('results/biodiversity.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print("Biodiversity metrics saved to results/biodiversity.txt")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    calculate_biodiversity('results/abundance.csv')