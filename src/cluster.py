import torch
from Bio import SeqIO
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from model import SequenceAutoencoder, one_hot_encode
import pickle

def load_model(model_path, latent_dim=64):
    model = SequenceAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def extract_embeddings(fastq_file, model):
    embeddings = []
    sequences = []
    for record in SeqIO.parse(fastq_file, "fastq"):
        seq = str(record.seq)
        encoded = one_hot_encode(seq).unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            emb = model.encode(encoded).squeeze(0).numpy()
        embeddings.append(emb)
        sequences.append(seq)
    return np.array(embeddings), sequences

def cluster_sequences(embeddings, n_clusters=5):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_scaled)
    return labels, scaler, kmeans

def save_clusters(labels, sequences, output_file):
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sequences[i])
    
    with open(output_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {output_file}")
    print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Number of noise points: {list(labels).count(-1)}")

if __name__ == "__main__":
    model = load_model('models/autoencoder.pth')
    embeddings, sequences = extract_embeddings('data/filtered.fastq', model)
    labels, scaler, kmeans = cluster_sequences(embeddings)
    save_clusters(labels, sequences, 'results/clusters.pkl')
    
    # Save scaler and kmeans for later
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)