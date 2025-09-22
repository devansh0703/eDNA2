#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and check for errors."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {command}")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)

def main(raw_fastq, output_dir='results'):
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    filtered_fastq = 'data/filtered.fastq'
    
    # Step 1: Preprocess
    print("Step 1: Preprocessing data...")
    run_command(f"python src/preprocess.py {raw_fastq} {filtered_fastq}")
    
    # Step 2: Train model
    print("Step 2: Training autoencoder...")
    run_command("python src/train_model.py")
    
    # Step 3: Cluster
    print("Step 3: Clustering sequences...")
    run_command("python src/cluster.py")
    
    # Step 4: Abundance
    print("Step 4: Estimating abundance...")
    run_command("python src/abundance.py")
    
    # Step 5: Biodiversity
    print("Step 5: Assessing biodiversity...")
    run_command("python src/biodiversity.py")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/pipeline.py <raw_fastq>")
        sys.exit(1)
    main(sys.argv[1])