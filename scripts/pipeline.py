#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd='/workspaces/codespaces-blank')
    if result.returncode != 0:
        print(f"Error in: {cmd}")
        sys.exit(1)

def main():
    # Preprocess
    run_command("python src/preprocess.py data/SRR1105999.fastq data/filtered.fastq")
    
    # Train model
    run_command("python src/train_model.py")
    
    # Cluster
    run_command("python src/cluster.py")
    
    # Abundance
    run_command("python src/abundance.py")
    
    # Biodiversity
    run_command("python src/biodiversity.py")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()