# AI-Driven eDNA Analysis Pipeline

This project implements an AI-driven pipeline for identifying eukaryotic taxa and assessing biodiversity from deep-sea environmental DNA (eDNA) datasets, minimizing reliance on reference databases.

## Overview

The pipeline uses deep learning (autoencoder) for sequence representation and unsupervised learning (KMeans clustering) to group sequences into potential taxa, followed by abundance estimation and biodiversity assessment.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install SRA toolkit for downloading data (optional, if using provided data):
   ```bash
   sudo apt update && sudo apt install -y sra-toolkit
   ```

## Usage

### Results Viewer
```bash
streamlit run src/app2.py
```
View existing analysis results without re-running the pipeline.

### Command Line
If you need to download your own data:
```bash
fastq-dump <SRA_ID> --maxSpotId 10000 -O data/
```

### Run Web UI
```bash
streamlit run src/app.py
```

This launches a web interface for uploading data and running the analysis interactively.

### Individual Steps
- Preprocess: `python src/preprocess.py data/SRR1105999.fastq data/filtered.fastq`
- Train Model: `python src/train_model.py`
- Cluster: `python src/cluster.py`
- Abundance: `python src/abundance.py`
- Biodiversity: `python src/biodiversity.py`
- Visualize: `python src/visualize.py`
- Annotate: `python src/annotate.py`

## Output

- `results/clusters.pkl`: Clustered sequences
- `results/abundance.csv`: Abundance per cluster
- `results/biodiversity.txt`: Biodiversity metrics
- `results/abundance_plot.png`: Abundance visualization
- `models/autoencoder.pth`: Trained model

## Dependencies

- Python 3.8+
- PyTorch
- BioPython
- scikit-learn
- pandas
- matplotlib
- seaborn
- cutadapt

## Notes

- Dataset is limited to 10,000 reads for codespaces compatibility
- Uses real eDNA data from NCBI SRA
- No simulated data or placeholders