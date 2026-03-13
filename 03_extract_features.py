import pandas as pd
import numpy as np
import pyBigWig
import cooler
from Bio import SeqIO
import os
import time

# === Configuration ===
input_file = 'final_training_dataset.csv'
output_file = 'final_enriched_dataset.csv'

# Data Sources
genome_path = 'hg38.fa'
atac_path = 'K562_ATAC_Accessibility.bigWig'
hic_path = 'K562_HiC_3D_Contacts.mcool'

# Parameters
CONTEXT_WINDOW = 100  # +/- 100bp around target (Total 200bp context)
HIC_RESOLUTION = 10000 # 10kb resolution for 3D data

def extract_features_optimized():
    print("--- Starting Phase 2: Optimized Context Extraction ---")
    start_time = time.time()
    
    # 1. Load and Sort Data (Crucial for Speed)
    print(f"Loading and sorting {input_file}...")
    df = pd.read_csv(input_file)
    
    # Sorting ensures we read files sequentially, minimizing disk seeking
    # We map chromosomes to numbers to sort correctly (chr1, chr2... not chr1, chr10)
    df['sort_key'] = df['target_chr'].str.extract('(\d+)').astype(float)
    df = df.sort_values(by=['sort_key', 'target_start']).drop(columns=['sort_key'])
    
    print(f"Processing {len(df)} sorted sites...")

    # 2. Optimized Genome Loading (The RAM Saver)
    print("Indexing Genome (Zero RAM cost)...")
    # SeqIO.index creates a lookup table, it DOES NOT load the file into RAM
    genome_idx = SeqIO.index(genome_path, "fasta")

    # 3. Optimized Hi-C Loading 
    print(f"Pre-calculating 3D Density for the whole genome...")
    try:
        c = cooler.Cooler(f'{hic_path}::/resolutions/{HIC_RESOLUTION}')
        # We sum the 'count' of all interactions for every bin
        pixels = c.pixels()[:]
        # Sum contacts where the bin is on the left (bin1_id)
        cov1 = np.bincount(pixels['bin1_id'], weights=pixels['count'], minlength=c.bins().shape[0])
        # Sum contacts where the bin is on the right (bin2_id)
        cov2 = np.bincount(pixels['bin2_id'], weights=pixels['count'], minlength=c.bins().shape[0])
        
        # Total density = Left connections + Right connections
        hic_coverage_vector = cov1 + cov2
        
        print("Hi-C Vectorization complete. Lookups will now be instant.")
        
    except Exception as e:
        print(f"Warning: Hi-C optimization failed ({e}). 3D features will be 0.")
        hic_coverage_vector = None

    # 4. Open ATAC-seq
    print("Opening ATAC-seq track...")
    bw = pyBigWig.open(atac_path)

    # Storage lists
    new_contexts = []
    new_atac_signals = []
    new_hic_signals = []
    new_strands = []
    
    print("Starting Extraction Loop...")
    
    # 5. The Fast Loop
    for index, row in df.iterrows():
        chrom = row['target_chr']
        start = int(row['target_start'])
        end = int(row['target_end'])
        
        # --- A. Sequence Context (Indexed Fetch) ---
        center = (start + end) // 2
        ctx_start = max(0, center - CONTEXT_WINDOW)
        ctx_end = center + CONTEXT_WINDOW
        
        try:
            # Grab only the bytes we need from disk
            # SeqIO.index allows dict-like access but reads from disk
            raw_seq = genome_idx[chrom][ctx_start:ctx_end].seq
            new_contexts.append(str(raw_seq).upper())
            
            # Handle Strand
            current_strand = str(row['target_strand'])
            if current_strand == '0':
                new_strands.append('+')
            else:
                new_strands.append(current_strand)
                
        except KeyError:
            new_contexts.append("N" * (CONTEXT_WINDOW * 2))
            new_strands.append('+')

        # --- B. ATAC-seq Signal ---
        try:
            val = bw.stats(chrom, start, end, type="mean")[0]
            new_atac_signals.append(val if val is not None else 0.0)
        except:
            new_atac_signals.append(0.0)

        # --- C. Hi-C Signal (Instant Vector Lookup) ---
        if hic_coverage_vector is not None:
            try:
                # Find which bin this coordinate belongs to
                # Cooler has a fast helper for this
                bin_id = c.bins().fetch(f"{chrom}:{start}-{end}").index[0]
                
                # O(1) Lookup - Instant
                val = hic_coverage_vector[bin_id]
                new_hic_signals.append(val)
            except:
                new_hic_signals.append(0.0)
        else:
            new_hic_signals.append(0.0)

        # Progress every 10%
        if index % 4000 == 0 and index > 0:
            elapsed = time.time() - start_time
            print(f"Processed {index} rows in {elapsed:.1f}s ({(index/elapsed):.1f} rows/sec)")

    # 6. Save
    print("Updating DataFrame...")
    df['target_context'] = new_contexts
    df['target_strand'] = new_strands
    df['epigen_dnase'] = new_atac_signals
    df['energy_1'] = new_hic_signals
    
    # Clean up zeros
    for col in ['epigen_ctcf', 'epigen_rrbs', 'epigen_h3k4me3', 'epigen_drip']:
        df[col] = 0.0
        
    df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    print(f"--- SUCCESS ---")
    print(f"Finished in {total_time:.1f} seconds.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    extract_features_optimized()