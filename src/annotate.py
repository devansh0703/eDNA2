import pickle
import pandas as pd

def gc_content(seq):
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) * 100 if seq else 0

def analyze_clusters(clusters_file):
    with open(clusters_file, 'rb') as f:
        clusters = pickle.load(f)
    
    annotations = []
    for cluster_id, sequences in clusters.items():
        if sequences:
            # Representative sequence (first one)
            rep_seq = sequences[0]
            
            # GC content
            gc_content_val = gc_content(rep_seq)
            
            # Sequence length
            seq_len = len(rep_seq)
            
            # Number of sequences
            count = len(sequences)
            
            annotations.append({
                'Cluster': cluster_id,
                'Representative_Sequence': rep_seq[:50] + '...' if len(rep_seq) > 50 else rep_seq,
                'GC_Content': gc_content_val,
                'Sequence_Length': seq_len,
                'Sequence_Count': count
            })
    
    df = pd.DataFrame(annotations)
    df.to_csv('results/cluster_annotations.csv', index=False)
    print("Cluster annotations saved to results/cluster_annotations.csv")
    print(df)

if __name__ == "__main__":
    analyze_clusters('results/clusters.pkl')