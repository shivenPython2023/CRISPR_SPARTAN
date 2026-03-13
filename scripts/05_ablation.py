import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import pandas as pd
import numpy as np
import tensorflow as tf
import random

# Global Seed for stability
np.random.seed(123)
tf.random.set_seed(123)
random.seed(123)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

# === Configuration ===
INPUT_FILE = 'sample_data/final_enriched_dataset.csv'
PLOT_SAVE_PATH = 'ablation_study_4way_comparison.png'
SPARTAN_PLOT_SAVE_PATH = 'spartan_specific_plots.png'
DNA_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

def one_hot_encode(seq_list, length):
    encoded = []
    for seq in seq_list:
        seq = str(seq).upper().ljust(length, 'N')[:length]
        encoded.append([DNA_MAP.get(base, [0,0,0,0]) for base in seq])
    return np.array(encoded, dtype=np.float32)

def reset_keras_state():
    tf.keras.backend.clear_session()

# =====================================================================
# MODEL 1: SEQUENCE ONLY
# =====================================================================
def build_model_1_seq_only():
    reg = l2(1e-4)
    align_input = Input(shape=(23, 8), name="Guide_Target_Alignment")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(align_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=reg)(x1)

    z = Dense(32, activation='relu', kernel_regularizer=reg)(x1)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name="Prediction")(z)

    model = Model(inputs=[align_input], outputs=output, name="Model_1_SeqOnly")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================================================================
# MODEL 2: SEQUENCE + PHYSICS
# =====================================================================
def build_model_2_seq_physics():
    reg = l2(1e-4)
    align_input = Input(shape=(23, 8), name="Guide_Target_Alignment")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(align_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=reg)(x1)

    phys_input = Input(shape=(2,), name="3D_and_ATAC")
    x3 = Dense(8, activation='relu')(phys_input)
    x3 = Dense(16, activation='sigmoid', bias_initializer='ones')(x3)

    gated_physics = Multiply()([x1, x3])
    merged = Concatenate()([x1, gated_physics])

    z = Dense(32, activation='relu', kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name="Prediction")(z)

    model = Model(inputs=[align_input, phys_input], outputs=output, name="Model_2_SeqPhysics")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================================================================
# MODEL 3: SEQUENCE + CONTEXT
# =====================================================================
def build_model_3_seq_context():
    reg = l2(1e-4)
    align_input = Input(shape=(23, 8), name="Guide_Target_Alignment")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(align_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=reg)(x1)

    context_input = Input(shape=(200, 4), name="DNA_Context")
    x2 = Conv1D(filters=32, kernel_size=8, activation='relu', kernel_regularizer=reg)(context_input)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(filters=16, kernel_size=4, activation='relu', kernel_regularizer=reg)(x2)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(16, activation='sigmoid', kernel_regularizer=reg, bias_initializer='ones')(x2)

    gated_context = Multiply()([x1, x2])
    merged = Concatenate()([x1, gated_context])

    z = Dense(32, activation='relu', kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name="Prediction")(z)

    model = Model(inputs=[align_input, context_input], outputs=output, name="Model_3_SeqContext")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================================================================
# MODEL 4: FULL SPARTAN
# =====================================================================
def build_model_4_full():
    reg = l2(1e-4)
    align_input = Input(shape=(23, 8), name="Guide_Target_Alignment")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(align_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=reg)(x1)

    context_input = Input(shape=(200, 4), name="DNA_Context")
    x2 = Conv1D(filters=32, kernel_size=8, activation='relu', kernel_regularizer=reg)(context_input)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(filters=16, kernel_size=4, activation='relu', kernel_regularizer=reg)(x2)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(16, activation='sigmoid', kernel_regularizer=reg, bias_initializer='ones')(x2)

    phys_input = Input(shape=(2,), name="3D_and_ATAC")
    x3 = Dense(8, activation='relu')(phys_input)
    x3 = Dense(16, activation='sigmoid', bias_initializer='ones')(x3)

    gated_context = Multiply()([x1, x2])
    gated_physics = Multiply()([x1, x3])
    merged = Concatenate()([x1, gated_context, gated_physics])

    z = Dense(32, activation='relu', kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name="Prediction")(z)

    model = Model(inputs=[align_input, context_input, phys_input], outputs=output, name="Model_4_Full")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=== 4-WAY ABLATION SHOWDOWN (WITH OPEN-GATE INIT) ===")

    df = pd.read_csv(INPUT_FILE).dropna(subset=['target_sequence', 'grna_target_sequence', 'target_context', 'label'])

    X_context = one_hot_encode(df['target_context'], 200)
    X_align = np.concatenate([one_hot_encode(df['target_sequence'], 23), one_hot_encode(df['grna_target_sequence'], 23)], axis=-1)
    X_phys = StandardScaler().fit_transform(df[['epigen_dnase', 'energy_1']].values)
    y = df['label'].values

    groups = df['grna_target_sequence'].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, y, groups=groups))

    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_idx, y[temp_idx], groups=temp_groups))
    val_idx, test_idx = temp_idx[val_rel_idx], temp_idx[test_rel_idx]

    neg_count = sum(y[train_idx] == 0)
    pos_count = sum(y[train_idx] == 1)
    class_weights = {0: 1.0, 1: min(15.0, (neg_count / pos_count) * 0.4)}

    def train_and_eval(model_func, train_data, val_data, test_data, save_name):
        reset_keras_state()
        model = model_func()

        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=4),
            ModelCheckpoint(save_name, monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        ]

        model.fit(train_data, y[train_idx], validation_data=(val_data, y[val_idx]),
                  epochs=45, batch_size=64, class_weight=class_weights, callbacks=callbacks, verbose=0)

        model.load_weights(save_name)
        y_probs = model.predict(test_data).flatten()

        roc_auc = roc_auc_score(y[test_idx], y_probs)
        pr_auc = average_precision_score(y[test_idx], y_probs)
        fpr, tpr, _ = roc_curve(y[test_idx], y_probs)
        precision, recall, _ = precision_recall_curve(y[test_idx], y_probs)

        return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall, 'y_probs': y_probs}

    print("Training models internally (Please wait)...")

    # Order maintained exactly as requested
    res_m1 = train_and_eval(build_model_1_seq_only,
                            [X_align[train_idx]],
                            [X_align[val_idx]], [X_align[test_idx]], "m1_pure.keras")

    res_m2 = train_and_eval(build_model_2_seq_physics,
                            [X_align[train_idx], X_phys[train_idx]],
                            [X_align[val_idx], X_phys[val_idx]], [X_align[test_idx], X_phys[test_idx]], "m2_pure.keras")

    res_m3 = train_and_eval(build_model_3_seq_context,
                            [X_align[train_idx], X_context[train_idx]],
                            [X_align[val_idx], X_context[val_idx]], [X_align[test_idx], X_context[test_idx]], "m3_pure.keras")

    res_m4 = train_and_eval(build_model_4_full,
                            [X_align[train_idx], X_context[train_idx], X_phys[train_idx]],
                            [X_align[val_idx], X_context[val_idx], X_phys[val_idx]], [X_align[test_idx], X_context[test_idx], X_phys[test_idx]], "m4_pure.keras")

    results_data = {
        'Model 1 (Sequence Only)': res_m1,
        'Model 2 (Seq + Physics)': res_m2,
        'Model 3 (Seq + Context)': res_m3,
        'Model 4 (Full Spartan)': res_m4
    }

    print("\n=== FINAL RESULTS ===")
    for name, data in results_data.items():
        print(f"{name} -> PR-AUC: {data['pr_auc']:.4f} | ROC-AUC: {data['roc_auc']:.4f}")

    print("\nGenerating Ablation Comparison Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'Model 1 (Sequence Only)': '#A0A0A0',
        'Model 2 (Seq + Physics)': '#F5A623',
        'Model 3 (Seq + Context)': '#4A90E2',
        'Model 4 (Full Spartan)': '#E63946'
    }
    line_styles = {
        'Model 1 (Sequence Only)': ':',
        'Model 2 (Seq + Physics)': '-.',
        'Model 3 (Seq + Context)': '--',
        'Model 4 (Full Spartan)': '-'
    }

    for name, data in results_data.items():
        axes[0].plot(data['fpr'], data['tpr'], color=colors[name], linestyle=line_styles[name], lw=2.5, label=f"{name} (AUC = {data['roc_auc']:.3f})")
        axes[1].plot(data['recall'], data['precision'], color=colors[name], linestyle=line_styles[name], lw=2.5, label=f"{name} (AUC = {data['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    axes[0].set_title('Ablation Study: ROC Curves', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Ablation Study: Precision-Recall Curves', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Saved visualization to: {PLOT_SAVE_PATH}")

    # === NEW: GENERATE SPECIFIC PLOTS FOR SPARTAN (MODEL 4) ===
    print("\nGenerating specific evaluations for Full Spartan model...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    m4_probs = res_m4['y_probs']
    m4_preds = (m4_probs >= 0.5).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y[test_idx], m4_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
                xticklabels=['Pred Safe', 'Pred Off-Target'],
                yticklabels=['Actual Safe', 'Actual Off-Target'])
    axes2[0].set_title('Spartan Confusion Matrix (Threshold = 0.5)', fontweight='bold')
    axes2[0].set_ylabel('True Biological Reality')
    axes2[0].set_xlabel('AI Prediction')

    # 2. ROC Curve
    axes2[1].plot(res_m4['fpr'], res_m4['tpr'], color='#E63946', lw=2, label=f"ROC curve (AUC = {res_m4['roc_auc']:.4f})")
    axes2[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes2[1].set_xlim([0.0, 1.0])
    axes2[1].set_ylim([0.0, 1.05])
    axes2[1].set_xlabel('False Positive Rate')
    axes2[1].set_ylabel('True Positive Rate')
    axes2[1].set_title('Spartan ROC Curve', fontweight='bold')
    axes2[1].legend(loc="lower right")
    axes2[1].grid(alpha=0.3)

    # 3. PR Curve
    axes2[2].plot(res_m4['recall'], res_m4['precision'], color='#0047AB', lw=2, label=f"PR curve (AUC = {res_m4['pr_auc']:.4f})")
    axes2[2].set_xlim([0.0, 1.0])
    axes2[2].set_ylim([0.0, 1.05])
    axes2[2].set_xlabel('Recall (Sensitivity)')
    axes2[2].set_ylabel('Precision (Positive Predictive Value)')
    axes2[2].set_title('Spartan Precision-Recall Curve', fontweight='bold')
    axes2[2].legend(loc="upper right")
    axes2[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SPARTAN_PLOT_SAVE_PATH, dpi=300)
    print(f"Saved Spartan visualization to: {SPARTAN_PLOT_SAVE_PATH}")

    # Checkpoint confirmation
    print("\n=== MODELS SAVED FOR DOWNLOAD ===")
    print(" - m1_pure.keras (Sequence Only)")
    print(" - m2_pure.keras (Sequence + Physics)")
    print(" - m3_pure.keras (Sequence + Context)")
    print(" - m4_pure.keras (Full Spartan Champion)")
    print("=================================================")

if __name__ == "__main__":
    main()
