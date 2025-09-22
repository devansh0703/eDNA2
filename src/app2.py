import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
from Bio import SeqIO
from model import SequenceAutoencoder, one_hot_encode

st.title("eDNA Analysis Results Viewer")

st.sidebar.header("Load Results")

if st.sidebar.button("Load Existing Results"):
    if not os.path.exists("results"):
        st.error("No results folder found. Run the pipeline first.")
    else:
        st.success("Loading results...")
        
        # Load model and embeddings for visualization
        if os.path.exists("models/autoencoder.pth"):
            model = SequenceAutoencoder()
            model.load_state_dict(torch.load('models/autoencoder.pth'))
            model.eval()
            
            # Get embeddings
            embeddings = []
            sequences = []
            for record in SeqIO.parse("data/filtered.fastq", "fastq"):
                seq = str(record.seq)
                encoded = one_hot_encode(seq).unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode(encoded).squeeze(0).numpy()
                embeddings.append(emb)
                sequences.append(seq)
            embeddings = np.array(embeddings)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Load clusters
            if os.path.exists("results/clusters.pkl"):
                with open("results/clusters.pkl", "rb") as f:
                    clusters = pickle.load(f)
                
                # Assign colors to clusters
                cluster_labels = []
                for i, seq in enumerate(sequences):
                    for cid, seqs in clusters.items():
                        if seq in seqs:
                            cluster_labels.append(cid)
                            break
                
                # Plot clusters
                st.header("Cluster Visualization")
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
                ax.set_title('Sequence Embeddings (PCA)')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                plt.colorbar(scatter, ax=ax, label='Cluster ID')
                st.pyplot(fig)
                
                # Cluster summary
                st.subheader("Cluster Summary")
                cluster_summary = []
                for cid, seqs in clusters.items():
                    cluster_summary.append({"Cluster": cid, "Sequences": len(seqs)})
                summary_df = pd.DataFrame(cluster_summary)
                st.dataframe(summary_df)
        
        # Abundance
        if os.path.exists("results/abundance.csv"):
            df = pd.read_csv("results/abundance.csv")
            st.header("Abundance Analysis")
            st.subheader("Abundance per Cluster")
            st.dataframe(df)
            
            # Plot
            fig, ax = plt.subplots()
            sns.barplot(x='Cluster', y='Abundance', data=df, ax=ax)
            ax.set_title('Abundance per Cluster')
            st.pyplot(fig)
        
        # Biodiversity
        if os.path.exists("results/biodiversity.txt"):
            st.header("Biodiversity Metrics")
            with open("results/biodiversity.txt", "r") as f:
                st.text(f.read())
        
        # Taxonomy (if exists)
        if os.path.exists("results/taxonomy.csv"):
            st.header("Taxonomy Assignment")
            tax_df = pd.read_csv("results/taxonomy.csv")
            st.dataframe(tax_df)

st.header("About")
st.write("This viewer displays results from the AI-driven eDNA analysis pipeline.")