# ============================================================
# IMPORTS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})
sns.set_palette('husl')

print(f'TensorFlow version: {tf.__version__}')
print(f'NumPy version:      {np.__version__}')
print(f'Pandas version:     {pd.__version__}')

# ============================================================
# CONSTANTS & HYPERPARAMETERS
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Sliding window
WINDOW_SIZE = 30

# Labeling thresholds
NORMAL_RUL_THRESHOLD  = 100   # RUL > 100  → NORMAL
ANOMALY_RUL_THRESHOLD = 30    # RUL < 30   → ANOMALOUS

# Directories
DATA_DIR      = './cmapss_data'
PROCESSED_DIR = './processed_data'
os.makedirs(DATA_DIR,      exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print('Configuration set.')
print(f'  Window Size:           {WINDOW_SIZE}')
print(f'  Normal RUL threshold:  > {NORMAL_RUL_THRESHOLD}')
print(f'  Anomaly RUL threshold: < {ANOMALY_RUL_THRESHOLD}')

# ============================================================
# COLUMN DEFINITIONS
# ============================================================

INDEX_COLS   = ['unit_id', 'cycle']
SETTING_COLS = [f'op_setting_{i}' for i in range(1, 4)]
SENSOR_COLS  = [f'sensor_{i}' for i in range(1, 22)]
ALL_COLS     = INDEX_COLS + SETTING_COLS + SENSOR_COLS

def load_cmapss_data(data_dir='.'):
    """Load C-MAPSS FD001 train, test, and RUL files."""
    train_df = pd.read_csv(
        os.path.join(data_dir, 'train_FD001.txt'),
        sep=r'\s+', header=None, names=ALL_COLS
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, 'test_FD001.txt'),
        sep=r'\s+', header=None, names=ALL_COLS
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, 'RUL_FD001.txt'),
        sep=r'\s+', header=None, names=['RUL']
    )
    return train_df, test_df, rul_df


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life (RUL) column to a training DataFrame."""
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df


# Load data
train_df, test_df, rul_df = load_cmapss_data(data_dir='.')

# Add RUL to training data
train_df = add_rul_column(train_df)

# Add RUL to test data (uses ground-truth RUL file)
test_max_cycles          = test_df.groupby('unit_id')['cycle'].max().reset_index()
test_max_cycles.columns  = ['unit_id', 'max_cycle']
rul_df['unit_id']        = test_max_cycles['unit_id']
rul_df['total_life']     = test_max_cycles['max_cycle'] + rul_df['RUL']
test_df                  = test_df.merge(rul_df[['unit_id', 'total_life']], on='unit_id', how='left')
test_df['RUL']           = test_df['total_life'] - test_df['cycle']
test_df.drop('total_life', axis=1, inplace=True)

print(f'Training data: {train_df.shape}')
print(f'Test data:     {test_df.shape}')
print(f'\nTraining engines: {train_df["unit_id"].nunique()}')
print(f'Test engines:     {test_df["unit_id"].nunique()}')
train_df.head()

# ============================================================
# SENSOR ANALYSIS — Identify Non-Informative Sensors
# ============================================================

print('Sensor variance analysis (training data):')
print('=' * 50)

sensor_variances = train_df[SENSOR_COLS].var()
print(sensor_variances.to_string())

VARIANCE_THRESHOLD = 0.01
non_informative    = sensor_variances[sensor_variances < VARIANCE_THRESHOLD].index.tolist()

print(f'\nNon-informative sensors (variance < {VARIANCE_THRESHOLD}): {non_informative}')

FEATURE_COLS = [col for col in SENSOR_COLS if col not in non_informative]
print(f'\nSelected features ({len(FEATURE_COLS)} sensors):')
print(f'  {FEATURE_COLS}')

# ============================================================
# VISUALIZE SENSOR DEGRADATION PATTERNS
# ============================================================

sample_unit = train_df['unit_id'].unique()[0]
sample_data = train_df[train_df['unit_id'] == sample_unit]

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
fig.suptitle(
    f'Sensor Readings for Engine Unit {sample_unit} (Run-to-Failure)',
    fontsize=16, fontweight='bold', y=1.01
)

for idx, sensor in enumerate(FEATURE_COLS[:16]):
    ax = axes[idx // 4, idx % 4]
    ax.plot(
        sample_data['cycle'], sample_data[sensor],
        linewidth=0.8, alpha=0.8,
        color=sns.color_palette('husl', 16)[idx]
    )
    ax.set_title(sensor, fontsize=10)
    ax.set_xlabel('Cycle')
    max_cycle = sample_data['cycle'].max()
    ax.axvspan(max_cycle - ANOMALY_RUL_THRESHOLD, max_cycle,
               alpha=0.15, color='red', label='Anomalous')

plt.tight_layout()
plt.show()
print('Red shaded region = last 30 cycles (anomalous/degraded zone).')

# ============================================================
# NORMALIZE FEATURES (StandardScaler)
# Fitted ONLY on training data — no leakage into test set.
# ============================================================

scaler = StandardScaler()

train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

print(f'Features normalized using StandardScaler.')
print(f'Scaler fitted on training data only: {len(FEATURE_COLS)} features.')
print(f'\nPost-normalization statistics (training):')
print(train_df[FEATURE_COLS].describe().loc[['mean', 'std']].round(3))

# ============================================================
# CREATE SLIDING WINDOWS
# ============================================================

def create_sequences(df, feature_cols, window_size, label_col='RUL'):
    """
    Create sliding window sequences from engine data.

    Returns:
        sequences  : (n_samples, window_size, n_features)
        labels     : (n_samples,)  0=normal, 1=anomalous, -1=transition
        rul_values : (n_samples,)  RUL at end of each window
    """
    sequences, labels, rul_values = [], [], []

    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id][feature_cols].values
        unit_rul  = df[df['unit_id'] == unit_id][label_col].values
        n_cycles  = len(unit_data)

        if n_cycles < window_size:
            continue

        for i in range(n_cycles - window_size + 1):
            window  = unit_data[i : i + window_size]
            end_rul = unit_rul[i + window_size - 1]

            sequences.append(window)
            rul_values.append(end_rul)

            if   end_rul > NORMAL_RUL_THRESHOLD:  labels.append(0)   # NORMAL
            elif end_rul < ANOMALY_RUL_THRESHOLD:  labels.append(1)   # ANOMALOUS
            else:                                   labels.append(-1)  # TRANSITION

    return np.array(sequences), np.array(labels), np.array(rul_values)


print('Creating training sequences...')
train_sequences, train_labels, train_rul = create_sequences(train_df, FEATURE_COLS, WINDOW_SIZE)

print('Creating test sequences...')
test_sequences, test_labels, test_rul = create_sequences(test_df, FEATURE_COLS, WINDOW_SIZE)

n_features = len(FEATURE_COLS)

print(f'\nTraining sequences shape: {train_sequences.shape}')
print(f'Test sequences shape:     {test_sequences.shape}')
print(f'\nSequence dimensions: (samples, timesteps={WINDOW_SIZE}, features={n_features})')

print(f'\nTraining label distribution:')
unique, counts = np.unique(train_labels, return_counts=True)
for u, c in zip(unique, counts):
    name = {0: 'NORMAL', 1: 'ANOMALOUS', -1: 'TRANSITION'}[u]
    print(f'  {name}: {c} ({100*c/len(train_labels):.1f}%)')

# ============================================================
# SPLIT DATA FOR TRAINING & EVALUATION (Paper-correct)
# ============================================================

normal_mask    = train_labels == 0
anomalous_mask = train_labels == 1

X_all_normal    = train_sequences[normal_mask]
X_all_anomalous = train_sequences[anomalous_mask]

# Shuffle
rng             = np.random.default_rng(SEED)
X_all_normal    = X_all_normal[rng.permutation(len(X_all_normal))]
X_all_anomalous = X_all_anomalous[rng.permutation(len(X_all_anomalous))]

# Normal splits: vN1 | vN2 | tN | sN
n_norm = len(X_all_normal)
n_vN1  = int(0.10 * n_norm)
n_vN2  = int(0.10 * n_norm)
n_tN   = int(0.15 * n_norm)

X_vN1 = X_all_normal[:n_vN1]
X_vN2 = X_all_normal[n_vN1 : n_vN1 + n_vN2]
X_tN  = X_all_normal[n_vN1 + n_vN2 : n_vN1 + n_vN2 + n_tN]
X_sN  = X_all_normal[n_vN1 + n_vN2 + n_tN:]

# Anomalous splits: vA | tA
n_anom = len(X_all_anomalous)
n_vA   = int(0.40 * n_anom)
X_vA   = X_all_anomalous[:n_vA]
X_tA   = X_all_anomalous[n_vA:]

# Derived sets used by model notebooks
X_train_final  = X_sN
X_train_target = X_train_final[:, ::-1, :]    # reversed targets for decoder
X_val_normal   = X_vN1
X_val_target   = X_val_normal[:, ::-1, :]
X_val_gauss    = X_vN2

X_val_thresh   = np.concatenate([X_vN2, X_vA], axis=0)
y_val_thresh   = np.concatenate([np.zeros(len(X_vN2)), np.ones(len(X_vA))], axis=0)

X_test         = np.concatenate([X_tN, X_tA], axis=0)
y_test         = np.concatenate([np.zeros(len(X_tN)), np.ones(len(X_tA))], axis=0)

print('Data split summary (paper-correct, all from train_df):')
print(f'  sN  — Training (normal only):   {X_train_final.shape}')
print(f'  vN1 — Early-stop validation:    {X_val_normal.shape}')
print(f'  vN2 — Gaussian fit validation:  {X_val_gauss.shape}')
print(f'  vA  — Anomalous threshold tune: {X_vA.shape}')
print(f'  tN  — Test normal:              {X_tN.shape}')
print(f'  tA  — Test anomalous:           {X_tA.shape}')

assert len(X_train_final) > 50, f'Too few training sequences: {len(X_train_final)}'
assert len(X_vA)          > 10, f'Too few anomalous validation sequences: {len(X_vA)}'
assert len(X_tA)          > 10, f'Too few anomalous test sequences: {len(X_tA)}'
print('\nSplit size checks passed.')

# ============================================================
# CREATE tf.data PIPELINES
# ============================================================

BATCH_SIZE = 64

def create_tf_dataset(X, y, batch_size, shuffle=True):
    """Create an efficient tf.data.Dataset pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000), seed=SEED)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = create_tf_dataset(X_train_final, X_train_target, BATCH_SIZE, shuffle=True)
val_dataset   = create_tf_dataset(X_val_normal,  X_val_target,   BATCH_SIZE, shuffle=False)

print('tf.data pipelines created.')
for batch_x, batch_y in train_dataset.take(1):
    print(f'  Training batch: input={batch_x.shape}, target={batch_y.shape}')

# ============================================================
# SAVE PREPROCESSED ARRAYS FOR DOWNSTREAM NOTEBOOKS
# ============================================================

np.save(os.path.join(PROCESSED_DIR, 'X_train_final.npy'),  X_train_final)
np.save(os.path.join(PROCESSED_DIR, 'X_train_target.npy'), X_train_target)
np.save(os.path.join(PROCESSED_DIR, 'X_val_normal.npy'),   X_val_normal)
np.save(os.path.join(PROCESSED_DIR, 'X_val_target.npy'),   X_val_target)
np.save(os.path.join(PROCESSED_DIR, 'X_val_gauss.npy'),    X_val_gauss)
np.save(os.path.join(PROCESSED_DIR, 'X_vA.npy'),           X_vA)
np.save(os.path.join(PROCESSED_DIR, 'X_vN2.npy'),          X_vN2)
np.save(os.path.join(PROCESSED_DIR, 'X_val_thresh.npy'),   X_val_thresh)
np.save(os.path.join(PROCESSED_DIR, 'y_val_thresh.npy'),   y_val_thresh)
np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'),         X_test)
np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'),         y_test)
np.save(os.path.join(PROCESSED_DIR, 'X_tN.npy'),           X_tN)
np.save(os.path.join(PROCESSED_DIR, 'X_tA.npy'),           X_tA)

# Save metadata
import json
metadata = {
    'WINDOW_SIZE':            WINDOW_SIZE,
    'n_features':             n_features,
    'FEATURE_COLS':           FEATURE_COLS,
    'NORMAL_RUL_THRESHOLD':   NORMAL_RUL_THRESHOLD,
    'ANOMALY_RUL_THRESHOLD':  ANOMALY_RUL_THRESHOLD,
    'SEED':                   SEED,
    'BATCH_SIZE':             BATCH_SIZE,
}
with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Preprocessed arrays saved to: {PROCESSED_DIR}/')
print('Files written:')
for fname in sorted(os.listdir(PROCESSED_DIR)):
    print(f'  {fname}')

