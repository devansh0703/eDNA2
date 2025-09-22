import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
from Bio import SeqIO
from sklearn.decomposition import PCA
import pickle
import numpy as np
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import torch
from sklearn.decomposition import PCA
import pickle
import numpy as np
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

st.title("AI-Driven eDNA Analysis Pipeline")

st.sidebar.header("Data Input")
option = st.sidebar.radio("Choose input method:", ["Upload FASTQ", "Load Existing FASTA"])

input_file = None

if option == "Upload FASTQ":
    uploaded_file = st.sidebar.file_uploader("Upload FASTQ file", type=["fastq", "fq"])
    if uploaded_file is not None:
        with open("data/uploaded.fastq", "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_file = "data/uploaded.fastq"
        st.sidebar.success("File uploaded!")
else:
    fastq_files = [f for f in os.listdir("data") if f.endswith(('.fastq', '.fq'))]
    if fastq_files:
        selected_file = st.sidebar.selectbox("Select existing FASTQ file:", fastq_files)
        input_file = f"data/{selected_file}"
        st.sidebar.success(f"Selected: {selected_file}")
    else:
        st.sidebar.error("No FASTQ files found in data/ folder.")

if input_file and st.sidebar.button("Run Pipeline"):
    with st.spinner("Running pipeline... This may take a few minutes."):
        result = subprocess.run(["python", "src/pipeline.py", input_file], 
                              capture_output=True, text=True, cwd="/workspaces/codespaces-blank")
        
        if result.returncode == 0:
            st.success("Pipeline completed successfully!")
            
            # Load model and embeddings for visualization
            from model import SequenceAutoencoder, one_hot_encode
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
            
            # Taxonomy assignment (sample one sequence per cluster)
            st.header("Taxonomy Assignment")
            taxonomy_results = {}
            for cid, seqs in clusters.items():
                if seqs:
                    # Pick first sequence
                    rep_seq = seqs[0]
                    try:
                        # BLAST search
                        result_handle = NCBIWWW.qblast("blastn", "nt", rep_seq[:100])  # Short sequence for speed
                        blast_records = NCBIXML.parse(result_handle)
                        for blast_record in blast_records:
                            if blast_record.alignments:
                                top_hit = blast_record.alignments[0]
                                taxonomy_results[cid] = top_hit.hit_def.split()[0]  # Genus/species
                                break
                        else:
                            taxonomy_results[cid] = "Unknown"
                    except:
                        taxonomy_results[cid] = "BLAST failed"
            
            # Display taxonomy
            tax_df = pd.DataFrame(list(taxonomy_results.items()), columns=['Cluster', 'Taxonomy'])
            st.dataframe(tax_df)
            
            # Save taxonomy
            tax_df.to_csv("results/taxonomy.csv", index=False)
            
            # Abundance
            if os.path.exists("results/abundance.csv"):
                df = pd.read_csv("results/abundance.csv")
                st.subheader("Abundance per Cluster")
                st.dataframe(df)
                
                # Plot
                fig, ax = plt.subplots()
                sns.barplot(x='Cluster', y='Abundance', data=df, ax=ax)
                ax.set_title('Abundance per Cluster')
                st.pyplot(fig)
            
            # Biodiversity
            if os.path.exists("results/biodiversity.txt"):
                st.subheader("Biodiversity Metrics")
                with open("results/biodiversity.txt", "r") as f:
                    st.text(f.read())
            
        else:
            st.error("Pipeline failed!")
            st.text(result.stderr)

st.header("About")
st.write("""
This app runs an AI-driven pipeline for analyzing deep-sea eDNA data:
- Preprocesses sequencing reads
- Trains a deep learning model for sequence representation
- Clusters sequences into potential taxa
- Estimates abundance and biodiversity metrics
- Visualizes clusters and assigns taxonomy
""")