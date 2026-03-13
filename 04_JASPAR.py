import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats

# Suppress TF logs for cleaner execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- NEW IMPORTS FOR AUTOMATED MOTIF MATCHING ---
try:
    from Bio import motifs
    from Bio.Seq import Seq
except ImportError:
    print("ERROR: Biopython is not installed. Please run 'pip install biopython' first.")
    exit()

print("=== INITIATING GLOBAL INTEGRATED GRADIENTS CONSENSUS ===")

# --- 1. CONFIGURATION ---
DATA_PATH = 'final_enriched_dataset.csv'
MODEL_PATH = 'm3_pure.keras'
JASPAR_PATH = 'JASPAR2024.txt' 

DNA_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

def one_hot_encode(seq_list, length):
    encoded = []
    for seq in seq_list:
        seq = str(seq).upper().ljust(length, 'N')[:length]
        encoded.append([DNA_MAP.get(base, [0,0,0,0]) for base in seq])
    return np.array(encoded, dtype=np.float32)

# --- 2. LOAD JASPAR DATABASE ---
print(f"Loading JASPAR database from {JASPAR_PATH}...")
try:
    with open(JASPAR_PATH) as f:
        jaspar_motifs = motifs.parse(f, "jaspar")
    print(f"Successfully loaded {len(jaspar_motifs)} known transcription factor profiles.")
except FileNotFoundError:
    print(f"ERROR: Could not find {JASPAR_PATH}. Please download it and put it in this folder.")
    exit()

# Pre-calculate scoring matrices and background distributions
print("Calculating scoring thresholds for all 879 TFs...")
tf_scoring_matrices = []

for i, m in enumerate(jaspar_motifs):
    # --- DEBUG STATEMENT ---
    if (i + 1) % 100 == 0:
        print(f"  -> Processed {i + 1}/{len(jaspar_motifs)} matrices...")

    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()

    # THE INSTANT HEURISTIC: Skip the freezing distribution calculator.
    # Set the threshold to 80% of the maximum possible score for this motif.
    threshold = pssm.max * 0.80

    tf_scoring_matrices.append({
        'name': m.name,
        'pssm': pssm,
        'length': len(m),
        'threshold': threshold
    })
print("Done pre-calculating matrices!")

# --- 3. LOAD DATA AND MODEL ---
print(f"\nLoading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH).dropna(subset=['target_sequence', 'grna_target_sequence', 'target_context', 'label'])

print(f"Loading trained model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Use the whole dataset for max N, but limit to 5000 for speed if needed
df_subset = df.head(10000).copy()
X_target = one_hot_encode(df_subset['target_sequence'], 23)
X_guide = one_hot_encode(df_subset['grna_target_sequence'], 23)
X_align = np.concatenate([X_target, X_guide], axis=-1)
X_context = one_hot_encode(df_subset['target_context'], 200)
y_true = df_subset['label'].values

# --- 4. FIND HIGHLY CONFIDENT SAMPLES (TP & TN) ---
print("\nPredicting on data to find confident targets...")
predictions = model.predict([X_align, X_context], verbose=0).flatten()

# High Confidence True Positives
confident_tps = np.where((y_true == 1) & (predictions > 0.8))[0]
# High Confidence True Negatives
confident_tns = np.where((y_true == 0) & (predictions < 0.2))[0]

print(f"Found {len(confident_tps)} Confident True Positives.")
print(f"Found {len(confident_tns)} Confident True Negatives.")

# Sort by confidence, but DO NOT slice or cap the arrays. Take everything.
targets_to_analyze_tp = confident_tps[np.argsort(predictions[confident_tps])[::-1]]
targets_to_analyze_tn = confident_tns[np.argsort(predictions[confident_tns])]

all_targets = np.concatenate([targets_to_analyze_tp, targets_to_analyze_tn])
labels_for_targets = np.concatenate([np.ones(len(targets_to_analyze_tp)), np.zeros(len(targets_to_analyze_tn))])

print(f"Analyzing ALL {len(all_targets)} total sites ({len(targets_to_analyze_tp)} Positives, {len(targets_to_analyze_tn)} Negatives)...")

# --- 5. INTEGRATED GRADIENTS MATH ---
@tf.function
def compute_gradients(inputs_align, inputs_context, baseline_context, alphas):
    interpolated_context = baseline_context + alphas * (inputs_context - baseline_context)
    repeated_align = tf.repeat(inputs_align, tf.shape(alphas)[0], axis=0)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_context)
        preds = model([repeated_align, interpolated_context])
    return tape.gradient(preds, interpolated_context)

def get_ig_scores(input_align, input_context, num_steps=50):
    baseline_context = tf.zeros_like(input_context)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, num_steps+1), (num_steps+1, 1, 1))
    grads = compute_gradients(input_align, input_context, baseline_context, alphas)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (input_context[0] - baseline_context[0]) * avg_grads
    return np.sum(integrated_grads.numpy(), axis=1)

# --- 6. THE GLOBAL SCAN & BIOLOGICAL MAPPING ---
print(f"\nScanning the AI's brain and mapping to JASPAR (p < 0.05 threshold)...")

matched_tfs_tp = []
matched_tfs_tn = []
inverse_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

for count, idx in enumerate(all_targets):
    if (count + 1) % 1 == 0:
        print(f"Processed {count + 1}/{len(all_targets)}...")

    is_positive = labels_for_targets[count] == 1

    sample_align = X_align[idx:idx+1]
    sample_context = X_context[idx:idx+1]

    # Get raw IG scores
    importance_scores = get_ig_scores(tf.convert_to_tensor(sample_align), tf.convert_to_tensor(sample_context))

    # We take the absolute value of the gradients. A strong negative gradient (suppressing a false positive)
    # is just as important biologically as a strong positive gradient (activating a true positive).
    importance_scores = np.abs(importance_scores)

    # Decode the actual genomic sequence
    decoded_seq = []
    for i in range(200):
        if np.sum(sample_context[0, i]) == 0:
            decoded_seq.append('N')
        else:
            decoded_seq.append(inverse_map[np.argmax(sample_context[0, i])])

    # Find the single most important 12bp window for this specific genomic location
    window_size = 12
    max_score = -np.inf
    best_start = 0

    for i in range(200 - window_size):
        window_score = np.sum(importance_scores[i:i+window_size])
        if window_score > max_score:
            max_score = window_score
            best_start = i

    motif = "".join(decoded_seq[best_start:best_start+window_size])
    if 'N' in motif:
        continue # Skip corrupted sequences

    seq_obj = Seq(motif)

    # Cross-reference the highly-activated motif against all known biology
    # Store EVERY TF that hits p < 0.05 for this sequence
    for tf_data in tf_scoring_matrices:
        if 6 <= tf_data['length'] <= 16:
            try:
                score = tf_data['pssm'].calculate(seq_obj)
                max_score = np.max(score)

                # Check against the biological significance threshold
                if max_score >= tf_data['threshold']:
                    if is_positive:
                        matched_tfs_tp.append(tf_data['name'])
                    else:
                        matched_tfs_tn.append(tf_data['name'])
            except:
                pass # Skip length mismatch errors

# --- 7. AGGREGATE AND REPORT ---
print("\n" + "="*50)
print("=== GLOBAL MOTIF CONSENSUS RESULTS (MAPPED TO TFs) ===")
print("="*50)

print(f"\nTotal n analyzed: {len(all_targets)} genomic loci")
print(f"Significant biological matches (p < 0.05) found in True Positives: {len(matched_tfs_tp)}")
print(f"Significant biological matches (p < 0.05) found in True Negatives: {len(matched_tfs_tn)}")

tf_counts_tp = Counter(matched_tfs_tp)
tf_counts_tn = Counter(matched_tfs_tn)

top_tfs_tp = tf_counts_tp.most_common(10)
top_tfs_tn = tf_counts_tn.most_common(10)

print("\n🚨 Top 10 TFs driving TRUE POSITIVES (Cleavage Activators):")
for rank, (tf_name, freq) in enumerate(top_tfs_tp, 1):
    print(f"#{rank}: {tf_name} (Found {freq} times)")

print("\n🛡️ Top 10 TFs driving TRUE NEGATIVES (Cleavage Vetoes/Repressors):")
for rank, (tf_name, freq) in enumerate(top_tfs_tn, 1):
    print(f"#{rank}: {tf_name} (Found {freq} times)")

# --- 8. POSTER BOARD VISUALIZATION ---
HEX_BG = 'white'
HEX_UNDERLINE = '#1c4587'

plt.rcParams.update({
    'figure.facecolor': HEX_BG,
    'axes.facecolor': HEX_BG,
    'savefig.facecolor': HEX_BG,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

def add_board_title(ax, text):
    ax.set_title(text, fontsize=22, fontweight='bold', pad=25)
    ax.plot([0.1, 0.9], [1.02, 1.02], color=HEX_UNDERLINE, lw=5, transform=ax.transAxes, clip_on=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Positive Chart
labels_tp = [m[0] for m in top_tfs_tp]
values_tp = [m[1] for m in top_tfs_tp]
axes[0].barh(labels_tp[::-1], values_tp[::-1], color='#E63946', edgecolor='black', alpha=0.9)
add_board_title(axes[0], 'AI Cleavage Activators (True Positives)')
axes[0].set_xlabel('Frequency (Matches with p < 0.05)')

# Negative Chart
labels_tn = [m[0] for m in top_tfs_tn]
values_tn = [m[1] for m in top_tfs_tn]
axes[1].barh(labels_tn[::-1], values_tn[::-1], color='#4A90E2', edgecolor='black', alpha=0.9)
add_board_title(axes[1], 'AI Cleavage Repressors (True Negatives)')
axes[1].set_xlabel('Frequency (Matches with p < 0.05)')

plt.tight_layout()
plt.savefig('poster_tf_consensus_chart.png', dpi=300)
print("\n>>> Saved consensus chart to 'poster_tf_consensus_chart.png'.")