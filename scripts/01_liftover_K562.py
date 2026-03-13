import pandas as pd
from pyliftover import LiftOver
import os

# === Configuration ===
input_file = 'CRISPR_Master_Dataset.csv'
chain_file = 'hg19ToHg38.over.chain.gz'
output_file = 'training_data_K562_hg38.csv'

def perform_liftover():
    # 1. Load Data
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 2. Filter for K562 (and ensure it has coordinates)
    # We select rows where cell_line is K562 AND it has a start coordinate
    k562_df = df[(df['cell_line'] == 'K562') & (df['target_start'].notna())].copy()
    print(f"Found {len(k562_df)} K562 sites to convert.")

    # 3. Initialize the Converter
    print("Loading chain file for coordinate conversion...")
    lo = LiftOver(chain_file)

    # 4. Define a helper function to convert one row
    def convert_coords(row):
        # LiftOver expects: (Chromosome, Position, Strand)
        # Note: Pandas imports as floats, so we convert to int
        chrom = row['target_chr']
        start = int(row['target_start'])
        
        # The conversion returns a list of matches (usually just one)
        new_coords = lo.convert_coordinate(chrom, start)
        
        if new_coords:
            # Return the new position (Target is usually a 23bp sequence)
            # We assume the length stays roughly the same
            new_start = new_coords[0][1]
            return new_start
        else:
            return None # Failed to map (deleted region)

    # 5. Apply the conversion
    print("Converting coordinates from hg19 to hg38...")
    k562_df['hg38_start'] = k562_df.apply(convert_coords, axis=1)

    # 6. Clean up
    # Remove rows that failed conversion (returned None)
    valid_df = k562_df.dropna(subset=['hg38_start']).copy()
    
    # Update the columns to be ready for the pipeline
    valid_df['target_start'] = valid_df['hg38_start'].astype(int)
    valid_df['target_end'] = valid_df['target_start'] + 23 # Approximate length of CRISPR target
    valid_df['genome'] = 'hg38' # Update the label
    
    # Drop the temporary column
    valid_df = valid_df.drop(columns=['hg38_start'])

    # 7. Save
    print(f"Liftover complete. Successfully converted {len(valid_df)} sites.")
    print(f"Dropped {len(k562_df) - len(valid_df)} sites that didn't map.")
    valid_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    if not os.path.exists(chain_file):
        print(f"Error: Missing chain file at {chain_file}")
        print("Please download 'hg19ToHg38.over.chain.gz' to the data folder.")
    else:
        perform_liftover()
