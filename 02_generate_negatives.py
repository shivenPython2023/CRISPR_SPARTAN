import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import csv
import os

# === Configuration ===
positives_file = 'training_data_K562_hg38.csv'
genome_file = 'hg38.fa'
output_file = 'final_training_dataset.csv'
MISMATCH_THRESHOLD = 6  # Standard for "Hard Negatives"

def hamming_distance(s1, s2):
    """Calculate mismatches between two strings."""
    if len(s1) != len(s2): return 999
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def generate_negatives():
    print("--- Starting Pro-Grade Negative Miner ---")
    
    # 1. Load Positives & Build a Lookup Set
    print(f"Loading positives from {positives_file}...")
    pos_df = pd.read_csv(positives_file)
    
    # Create a set of "known positives" for fast checking: "chrX:10000"
    # We use a string key "chr:start" to identify them
    positive_keys = set(
        (str(row['target_chr']) + ":" + str(row['target_start'])) 
        for _, row in pos_df.iterrows()
    )
    
    # Get the 19 unique guides we are studying
    unique_guides = pos_df['grna_target_sequence'].unique()
    print(f"Mining negatives for {len(unique_guides)} unique K562 guides.")

    # Prepare the output file with headers
    # We will append rows as we find them to save memory
    full_headers = list(pos_df.columns)
    if 'label' not in full_headers:
        full_headers.append('label')
    
    # Initialize output file with just the positives first
    # We give them Label = 1
    pos_df['label'] = 1
    pos_df.to_csv(output_file, index=False)
    
    print("Scanning full genome (this takes time)...")
    
    new_negatives_count = 0
    
    # 2. Iterate over the Genome (Chromosome by Chromosome)
    # This prevents loading 3GB into RAM all at once
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=full_headers)
        
        for record in SeqIO.parse(genome_file, "fasta"):
            chrom_name = record.id
            # Standardize 'chr1' vs '1' if needed. Assuming hg38 uses 'chr1'
            if "_" in chrom_name: continue # Skip weird contigs like chr1_KI270706v1
            
            print(f"Scanning {chrom_name}...")
            sequence = str(record.seq).upper()
            seq_len = len(sequence)
            
            # 3. Scanning Logic
            # Optimization: Instead of checking every base, we find the PAM "GG"
            # CRISPR needs 'NGG' at the end. 
            # So we look for 'GG' and grab the 21bp before it.
            
            # Find all occurrences of 'GG'
            start_search = 0
            while True:
                # Find next GG
                pam_index = sequence.find("GG", start_search)
                if pam_index == -1: break # No more GGs
                
                # Check if we have enough room upstream (21bp guide + PAM)
                # target_start is usually the start of the 23bp sequence
                target_start = pam_index - 21
                
                if target_start >= 0:
                    # Extract the potential target (23bp: 21bp guide + NG + G)
                    # Note: Usually sgRNA is 20bp + PAM. 
                    # Let's assume the CSV used 23bp context.
                    candidate_seq = sequence[target_start:pam_index+2]
                    
                    # Compare this candidate against ALL 19 guides
                    for guide in unique_guides:
                        # Guide is typically 23bp in your dataset (includes PAM)
                        # If guide in CSV is 23bp, compare directly
                        mismatches = hamming_distance(candidate_seq, guide)
                        
                        if mismatches <= MISMATCH_THRESHOLD:
                            # It's a match! Is it a known positive?
                            key = f"{chrom_name}:{target_start}"
                            
                            if key not in positive_keys:
                                # IT IS A NEGATIVE!
                                # Create the row
                                new_row = {
                                    'target_chr': chrom_name,
                                    'target_start': target_start,
                                    'target_end': target_start + 23,
                                    'target_sequence': candidate_seq,
                                    'grna_target_sequence': guide,
                                    'genome': 'hg38',
                                    'cell_line': 'K562', # It's in the K562 genome
                                    'cleavage_freq': 0, # Assumed 0
                                    'label': 0
                                }
                                # Fill other columns with NaN or 0
                                for col in full_headers:
                                    if col not in new_row:
                                        new_row[col] = 0
                                
                                writer.writerow(new_row)
                                new_negatives_count += 1
                
                # Move forward
                start_search = pam_index + 1
                
    print(f"--- Mining Complete ---")
    print(f"Found {new_negatives_count} new 'In-Silico' negatives.")
    print(f"Total dataset size: {len(pos_df) + new_negatives_count}")

if __name__ == "__main__":
    generate_negatives()