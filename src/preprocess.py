#!/usr/bin/env python3

import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os

def quality_filter(record, min_length=100, min_avg_quality=20):
    """
    Filter reads based on length and average quality.
    """
    if len(record.seq) < min_length:
        return False
    if record.letter_annotations.get('phred_quality'):
        avg_qual = sum(record.letter_annotations['phred_quality']) / len(record.letter_annotations['phred_quality'])
        if avg_qual < min_avg_quality:
            return False
    return True

def preprocess_fastq(input_file, output_file):
    """
    Preprocess FASTQ file: quality filter.
    """
    filtered_records = []
    for record in SeqIO.parse(input_file, "fastq"):
        if quality_filter(record):
            filtered_records.append(record)
    
    SeqIO.write(filtered_records, output_file, "fastq")
    print(f"Filtered {len(filtered_records)} reads out of {sum(1 for _ in SeqIO.parse(input_file, 'fastq'))}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input.fastq> <output.fastq>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)
    
    preprocess_fastq(input_file, output_file)