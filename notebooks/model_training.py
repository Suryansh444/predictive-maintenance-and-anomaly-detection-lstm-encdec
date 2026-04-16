# ============================================================
# IMPORTS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, auc
)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

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
print(f'TensorFlow: {tf.__version__}')

# ============================================================
# GPU CONFIGURATION
# ============================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU(s): {[g.name for g in gpus]}')
    except RuntimeError as e:
        print(f'GPU error: {e}')
else:
    print('No GPU detected — using CPU.')

# ============================================================
# LOAD METADATA & PREPROCESSED ARRAYS
# ============================================================

PROCESSED_DIR = './processed_data'
MODEL_DIR     = './saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(PROCESSED_DIR, 'metadata.json')) as f:
    meta = json.load(f)

WINDOW_SIZE  = meta['WINDOW_SIZE']
n_features   = meta['n_features']
FEATURE_COLS = meta['FEATURE_COLS']
SEED         = meta['SEED']
BATCH_SIZE   = meta['BATCH_SIZE']

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
LATENT_DIM      = 64
ATTENTION_UNITS = 32
LEARNING_RATE   = 5e-4
EPOCHS          = 80
PATIENCE        = 15
BETA            = 1.0

# Load splits
def load(name): return np.load(os.path.join(PROCESSED_DIR, name))

X_train_final  = load('X_train_final.npy')
X_train_target = load('X_train_target.npy')
X_val_normal   = load('X_val_normal.npy')
X_val_target   = load('X_val_target.npy')
X_val_gauss    = load('X_val_gauss.npy')
X_vN2          = load('X_vN2.npy')
X_vA           = load('X_vA.npy')
X_val_thresh   = load('X_val_thresh.npy')
y_val_thresh   = load('y_val_thresh.npy')
X_test         = load('X_test.npy')
y_test         = load('y_test.npy')
X_tN           = load('X_tN.npy')
X_tA           = load('X_tA.npy')

print(f'Training set (normal only): {X_train_final.shape}')
print(f'Validation set (vN1):       {X_val_normal.shape}')
print(f'Test set:                   {X_test.shape}')

# ============================================================
# RE-DEFINE MODEL COMPONENTS
# (copy from baseline_models.ipynb or import if refactored to .py)
# ============================================================

# --- Bahdanau Attention ---
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units, use_bias=False, name='attn_W1')
        self.W2 = Dense(units, use_bias=False, name='attn_W2')
        self.V  = Dense(1,     use_bias=False, name='attn_V')

    def call(self, encoder_outputs, decoder_outputs):
        enc_proj = tf.expand_dims(self.W1(encoder_outputs), axis=1)
        dec_proj = tf.expand_dims(self.W2(decoder_outputs), axis=2)
        score    = tf.squeeze(self.V(tf.nn.tanh(enc_proj + dec_proj)), axis=-1)
        weights  = tf.nn.softmax(score, axis=-1)
        context  = tf.matmul(weights, encoder_outputs)
        return context, weights

    def get_config(self):
        cfg = super().get_config(); cfg.update({'units': self.units}); return cfg


# --- Baseline Model ---
def build_baseline_encoder_decoder(timesteps, n_feat, latent_dim):
    enc_in  = Input(shape=(timesteps, n_feat), name='encoder_input')
    _, h, c = LSTM(latent_dim, return_sequences=False, return_state=True, name='encoder_lstm')(enc_in)
    dec_out = LSTM(latent_dim, return_sequences=True, name='decoder_lstm')(
                  RepeatVector(timesteps)(h), initial_state=[h, c])
    out     = TimeDistributed(Dense(n_feat))(dec_out)
    return Model(enc_in, out, name='EncDec_AD_Baseline')


# --- Attention Model ---
def build_attention_encoder_decoder(timesteps, n_feat, latent_dim, attn_units):
    enc_in         = Input(shape=(timesteps, n_feat), name='encoder_input')
    enc_out, h, c  = LSTM(latent_dim, return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2, name='encoder_lstm')(enc_in)
    enc_out        = Dropout(0.2)(enc_out)
    dec_out        = LSTM(latent_dim, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.2, name='decoder_lstm')(
                         RepeatVector(timesteps)(h), initial_state=[h, c])
    context, _     = BahdanauAttention(attn_units, name='bahdanau_attention')(enc_out, dec_out)
    context        = Dropout(0.3)(context)
    combined       = layers.Concatenate(axis=-1)([dec_out, context])
    out            = TimeDistributed(Dense(n_feat))(combined)
    return Model(enc_in, out, name='EncDec_AD_Attention')


# --- Anomaly Scoring Helpers ---
def compute_reconstruction_errors(model, X, batch_size=256):
    X_hat      = model.predict(X, batch_size=batch_size, verbose=0)[:, ::-1, :]
    errors_3d  = np.abs(X - X_hat)
    return errors_3d, errors_3d.reshape(-1, X.shape[2])

def fit_gaussian(error_flat):
    mu        = np.mean(error_flat, axis=0)
    sigma     = np.cov(error_flat, rowvar=False) + np.eye(error_flat.shape[1]) * 1e-5
    return mu, sigma, np.linalg.inv(sigma)

def compute_anomaly_scores(errors_3d, mu, sigma_inv):
    n, L, m = errors_3d.shape
    diff    = errors_3d.reshape(n * L, m) - mu
    return np.sum(diff @ sigma_inv * diff, axis=1).reshape(n, L).max(axis=1)

def optimize_threshold(scores, labels, beta=BETA, n_thresholds=500):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_tau, best_fb, best_results = thresholds[0], 0.0, {}
    precisions, recalls, fbetas = [], [], []
    for tau in thresholds:
        preds = (scores > tau).astype(int)
        tp = np.sum((preds==1)&(labels==1)); fp = np.sum((preds==1)&(labels==0))
        fn = np.sum((preds==0)&(labels==1))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fb = (1+beta**2)*p*r/(beta**2*p+r) if (p+r)>0 else 0.0
        precisions.append(p); recalls.append(r); fbetas.append(fb)
    precisions, recalls, fbetas = map(np.array, [precisions, recalls, fbetas])
    best_idx = np.argmax(fbetas)
    return thresholds[best_idx], fbetas[best_idx], {
        'thresholds': thresholds, 'precisions': precisions,
        'recalls': recalls, 'fbetas': fbetas, 'best_idx': best_idx
    }

print('All components loaded.')

# Build & compile models
baseline_model  = build_baseline_encoder_decoder(WINDOW_SIZE, n_features, LATENT_DIM)
attention_model = build_attention_encoder_decoder(WINDOW_SIZE, n_features, LATENT_DIM, ATTENTION_UNITS)

baseline_model.compile(optimizer=Adam(LEARNING_RATE),  loss='mse')
attention_model.compile(optimizer=Adam(LEARNING_RATE), loss='mse')

# tf.data pipelines
def make_dataset(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(min(len(X), 10000), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset = make_dataset(X_train_final, X_train_target, shuffle=True)
val_dataset   = make_dataset(X_val_normal,  X_val_target,   shuffle=False)

print('Models compiled. Datasets ready.')

# ============================================================
# TRAIN MODEL 1: BASELINE LSTM ENCODER-DECODER
# ============================================================

print('=' * 60)
print('TRAINING MODEL 1: Baseline LSTM Encoder-Decoder')
print('=' * 60)
print(f'Training samples (NORMAL only): {len(X_train_final)}')
print(f'Validation samples (NORMAL):    {len(X_val_normal)}')
print()

baseline_callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'baseline_best.keras'),
                    monitor='val_loss', save_best_only=True, verbose=0)
]

baseline_history = baseline_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=baseline_callbacks,
    verbose=1
)

print(f'\nBaseline training complete.')
print(f'Epochs trained:       {len(baseline_history.history["loss"])}')
print(f'Best validation loss: {min(baseline_history.history["val_loss"]):.6f}')

# ============================================================
# TRAIN MODEL 2: ATTENTION LSTM ENCODER-DECODER
# ============================================================

print('=' * 60)
print('TRAINING MODEL 2: Attention LSTM Encoder-Decoder')
print('=' * 60)
print(f'Training samples (NORMAL only): {len(X_train_final)}')
print()

attention_callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'attention_best.keras'),
                    monitor='val_loss', save_best_only=True, verbose=0)
]

attention_history = attention_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=attention_callbacks,
    verbose=1
)

print(f'\nAttention training complete.')
print(f'Epochs trained:       {len(attention_history.history["loss"])}')
print(f'Best validation loss: {min(attention_history.history["val_loss"]):.6f}')

# ============================================================
# PLOT TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, history, title, tc, vc in zip(
    axes,
    [baseline_history, attention_history],
    ['Model 1: Baseline LSTM EncDec-AD', 'Model 2: Attention LSTM EncDec-AD'],
    ['#2196F3', '#4CAF50'],
    ['#FF5722', '#E91E63']
):
    ax.plot(history.history['loss'],     label='Train Loss', linewidth=2, color=tc)
    ax.plot(history.history['val_loss'], label='Val Loss',   linewidth=2,
            linestyle='--', color=vc)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.set_yscale('log')

plt.suptitle('Training vs Validation Loss', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# COMPUTE RECONSTRUCTION ERRORS
# ============================================================

print('Computing errors — Baseline...')
baseline_err3d_vN2,  baseline_err_vN2  = compute_reconstruction_errors(baseline_model,  X_vN2)
baseline_err3d_vA,   _                 = compute_reconstruction_errors(baseline_model,  X_vA)
baseline_err3d_test, _                 = compute_reconstruction_errors(baseline_model,  X_test)

print('Computing errors — Attention...')
attention_err3d_vN2, attention_err_vN2 = compute_reconstruction_errors(attention_model, X_vN2)
attention_err3d_vA,  _                 = compute_reconstruction_errors(attention_model, X_vA)
attention_err3d_test,_                 = compute_reconstruction_errors(attention_model, X_test)

print('Done.')

# ============================================================
# FIT GAUSSIAN & COMPUTE ANOMALY SCORES
# ============================================================

# Baseline
baseline_mu,  baseline_sigma,  baseline_sigma_inv  = fit_gaussian(baseline_err_vN2)
baseline_scores_val  = np.concatenate([
    compute_anomaly_scores(baseline_err3d_vN2,  baseline_mu,  baseline_sigma_inv),
    compute_anomaly_scores(baseline_err3d_vA,   baseline_mu,  baseline_sigma_inv)
])
baseline_scores_test = compute_anomaly_scores(baseline_err3d_test, baseline_mu, baseline_sigma_inv)

# Attention
attention_mu, attention_sigma, attention_sigma_inv = fit_gaussian(attention_err_vN2)
attention_scores_val  = np.concatenate([
    compute_anomaly_scores(attention_err3d_vN2, attention_mu, attention_sigma_inv),
    compute_anomaly_scores(attention_err3d_vA,  attention_mu, attention_sigma_inv)
])
attention_scores_test = compute_anomaly_scores(attention_err3d_test, attention_mu, attention_sigma_inv)

n_vN2 = len(X_vN2)
print('Anomaly score statistics (Mahalanobis max-over-timesteps):')
print(f'  Baseline  — Val normal mean:    {baseline_scores_val[:n_vN2].mean():.2f}')
print(f'  Baseline  — Val anomalous mean: {baseline_scores_val[n_vN2:].mean():.2f}')
print(f'  Attention — Val normal mean:    {attention_scores_val[:n_vN2].mean():.2f}')
print(f'  Attention — Val anomalous mean: {attention_scores_val[n_vN2:].mean():.2f}')

# ============================================================
# THRESHOLD OPTIMISATION (on vN2 + vA)
# ============================================================

baseline_threshold,  baseline_best_fb,  baseline_thresh_results  = optimize_threshold(
    baseline_scores_val,  y_val_thresh, beta=BETA
)
attention_threshold, attention_best_fb, attention_thresh_results = optimize_threshold(
    attention_scores_val, y_val_thresh, beta=BETA
)

print(f'Baseline  — τ={baseline_threshold:.2f},  F{BETA}={baseline_best_fb:.4f}')
print(f'Attention — τ={attention_threshold:.2f}, F{BETA}={attention_best_fb:.4f}')

# ============================================================
# EVALUATE ON TEST SET (tN + tA)
# ============================================================

def evaluate_model(scores, labels, threshold, beta=BETA, model_name='Model'):
    preds = (scores > threshold).astype(int)
    tp = np.sum((preds==1)&(labels==1)); fp = np.sum((preds==1)&(labels==0))
    fn = np.sum((preds==0)&(labels==1)); tn = np.sum((preds==0)&(labels==0))
    p   = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r   = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fb  = (1+beta**2)*p*r/(beta**2*p+r) if (p+r)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    cm  = confusion_matrix(labels, preds)

    print(f'\n{"="*52}')
    print(f'{model_name} — Test Set Evaluation')
    print(f'{"="*52}')
    print(f'  Threshold τ  : {threshold:.4f}')
    print(f'  Precision    : {p:.4f}')
    print(f'  Recall (TPR) : {r:.4f}')
    print(f'  F-{beta}     : {fb:.4f}')
    print(f'  FPR          : {fpr:.4f}')
    print(f'  TPR/FPR      : {r/fpr if fpr>0 else float("inf"):.2f}')
    print(f'\n  Confusion Matrix:')
    print(f'                  Pred Normal  Pred Anomaly')
    print(f'  Actual Normal   {cm[0,0]:>10d}  {cm[0,1]:>12d}')
    print(f'  Actual Anomaly  {cm[1,0]:>10d}  {cm[1,1]:>12d}')

    return {'precision': p, 'recall': r, 'fbeta': fb,
            'fpr': fpr, 'confusion_matrix': cm, 'predictions': preds}


baseline_metrics  = evaluate_model(
    baseline_scores_test,  y_test, baseline_threshold,  BETA, 'Baseline LSTM EncDec-AD')
attention_metrics = evaluate_model(
    attention_scores_test, y_test, attention_threshold, BETA, 'Attention LSTM EncDec-AD')

# ============================================================
# VIZ 1: Anomaly Score Distribution (Mahalanobis)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, scores, threshold, title in zip(
    axes,
    [baseline_scores_val, attention_scores_val],
    [baseline_threshold, attention_threshold],
    ['Baseline EncDec-AD', 'Attention EncDec-AD']
):
    ax.hist(scores[y_val_thresh==0], bins=60, alpha=0.6, label='Normal',
            color='#4CAF50', density=True, edgecolor='white', linewidth=0.5)
    ax.hist(scores[y_val_thresh==1], bins=60, alpha=0.6, label='Anomalous',
            color='#E91E63', density=True, edgecolor='white', linewidth=0.5)
    ax.axvline(threshold, color='#FF9800', linestyle='--', linewidth=2.5,
               label=f'τ={threshold:.2f}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Anomaly Score (Mahalanobis Distance)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=10)

plt.suptitle('Anomaly Score (Mahalanobis Distance) with Gaussian Fit & Threshold',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# VIZ 2: Threshold vs F-β Curve
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, results, threshold, fb, title, color in zip(
    axes,
    [baseline_thresh_results, attention_thresh_results],
    [baseline_threshold, attention_threshold],
    [baseline_best_fb, attention_best_fb],
    ['Baseline EncDec-AD', 'Attention EncDec-AD'],
    ['#2196F3', '#4CAF50']
):
    ax.plot(results['thresholds'], results['fbetas'],     linewidth=2.5, color=color,    label=f'F-{BETA}')
    ax.plot(results['thresholds'], results['precisions'], linewidth=1.5, linestyle='--',
            color='#FF9800', alpha=0.8, label='Precision')
    ax.plot(results['thresholds'], results['recalls'],    linewidth=1.5, linestyle=':',
            color='#9C27B0', alpha=0.8, label='Recall')
    ax.axvline(threshold, color='red', linestyle='-.', linewidth=1.5, alpha=0.7,
               label=f'Best τ={threshold:.2f}')
    ax.scatter([threshold], [fb], color='red', s=100, zorder=5, edgecolors='black', linewidths=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Threshold (τ)'); ax.set_ylabel('Score')
    ax.legend(fontsize=10); ax.set_ylim(-0.05, 1.05)

plt.suptitle(f'Threshold vs F-{BETA} Score (Validation Set)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# VIZ 3: Confusion Matrices
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metrics, title, cmap in zip(
    axes,
    [baseline_metrics, attention_metrics],
    ['Baseline EncDec-AD', 'Attention EncDec-AD'],
    ['Blues', 'Greens']
):
    sns.heatmap(
        metrics['confusion_matrix'], annot=True, fmt='d', cmap=cmap, ax=ax,
        xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
        annot_kws={'size': 16, 'fontweight': 'bold'},
        linewidths=2, linecolor='white'
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)

plt.suptitle('Confusion Matrices (Test Set)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# VIZ 4: Precision-Recall Curve (Test Set)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

for scores, title, color, ls in zip(
    [baseline_scores_test, attention_scores_test],
    ['Baseline EncDec-AD', 'Attention EncDec-AD'],
    ['#2196F3', '#4CAF50'],
    ['-', '--']
):
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec_curve, prec_curve)
    ax.plot(rec_curve, prec_curve, linewidth=2.5, color=color, linestyle=ls,
            label=f'{title} (AUC={pr_auc:.4f})')

ax.set_xlabel('Recall', fontsize=13); ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve (Test Set)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
plt.tight_layout()
plt.show()

# ============================================================
# VIZ 5: Model Comparison Bar Chart
# ============================================================

metrics_names  = ['Precision', 'Recall', f'F-{BETA}']
baseline_vals  = [baseline_metrics['precision'],  baseline_metrics['recall'],  baseline_metrics['fbeta']]
attention_vals = [attention_metrics['precision'], attention_metrics['recall'], attention_metrics['fbeta']]

x, w = np.arange(len(metrics_names)), 0.32
fig, ax = plt.subplots(figsize=(10, 6))

b1 = ax.bar(x - w/2, baseline_vals,  w, label='Baseline EncDec-AD',
            color='#2196F3', edgecolor='white', linewidth=2, alpha=0.9)
b2 = ax.bar(x + w/2, attention_vals, w, label='Attention EncDec-AD',
            color='#4CAF50', edgecolor='white', linewidth=2, alpha=0.9)

for bar, col in [(b1, '#1565C0'), (b2, '#2E7D32')]:
    for b in bar:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=col)

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Comparison: Baseline vs Attention EncDec-AD (Test Set)',
             fontsize=15, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=12)
ax.legend(fontsize=12); ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# VIZ 6: Per-Engine Time-Series with Anomaly Scores
# ============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-load test_df (needed for per-engine plots)
INDEX_COLS   = ['unit_id', 'cycle']
SETTING_COLS = [f'op_setting_{i}' for i in range(1, 4)]
SENSOR_COLS  = [f'sensor_{i}' for i in range(1, 22)]
ALL_COLS     = INDEX_COLS + SETTING_COLS + SENSOR_COLS

train_df_raw = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=ALL_COLS)
test_df_raw  = pd.read_csv('test_FD001.txt',  sep=r'\s+', header=None, names=ALL_COLS)
rul_df_raw   = pd.read_csv('RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL'])

test_max = test_df_raw.groupby('unit_id')['cycle'].max().reset_index()
test_max.columns = ['unit_id', 'max_cycle']
rul_df_raw['unit_id']    = test_max['unit_id']
rul_df_raw['total_life'] = test_max['max_cycle'] + rul_df_raw['RUL']
test_df_raw = test_df_raw.merge(rul_df_raw[['unit_id','total_life']], on='unit_id', how='left')
test_df_raw['RUL'] = test_df_raw['total_life'] - test_df_raw['cycle']
test_df_raw.drop('total_life', axis=1, inplace=True)

scaler = StandardScaler()
scaler.fit(train_df_raw[FEATURE_COLS])
test_df_raw[FEATURE_COLS] = scaler.transform(test_df_raw[FEATURE_COLS])


def plot_engine_anomaly_scores(model, model_name, mu, sigma_inv, threshold,
                                df, feature_cols, window_size, unit_id, color):
    unit_data = df[df['unit_id'] == unit_id][feature_cols].values
    n_cycles  = len(unit_data)
    if n_cycles < window_size:
        print(f'Unit {unit_id}: fewer than {window_size} cycles, skipping.')
        return

    windows = np.array([unit_data[i:i+window_size] for i in range(n_cycles - window_size + 1)])
    errs3d, _ = compute_reconstruction_errors(model, windows)
    scores    = compute_anomaly_scores(errs3d, mu, sigma_inv)
    cycles    = np.arange(window_size, n_cycles + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[1, 1.2])
    fig.suptitle(f'{model_name} — Engine Unit {unit_id}', fontsize=14, fontweight='bold')

    for i, idx in enumerate([0, 2, 5, 8]):
        if idx < len(feature_cols):
            ax1.plot(range(1, n_cycles+1), unit_data[:, idx],
                     linewidth=0.8, alpha=0.7, label=feature_cols[idx])
    ax1.set_ylabel('Sensor Reading (normalized)')
    ax1.legend(fontsize=9, ncol=4, loc='upper left')
    ax1.set_title('Sensor Readings')

    ax2.plot(cycles, scores, linewidth=1.2, color=color, alpha=0.8, label='Anomaly Score')
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'τ={threshold:.2f}')
    ax2.fill_between(cycles, 0, scores, where=scores > threshold,
                     color='red', alpha=0.3, label='Detected Anomaly')
    ax2.set_xlabel('Cycle'); ax2.set_ylabel('Anomaly Score (Mahalanobis)')
    ax2.set_title('Anomaly Detection'); ax2.legend(fontsize=10)
    ax2.set_yscale('symlog', linthresh=10)
    plt.tight_layout()
    plt.show()


for unit_id in test_df_raw['unit_id'].unique()[:3]:
    print(f'\n--- Engine Unit {unit_id} ---')
    plot_engine_anomaly_scores(baseline_model,  'Baseline EncDec-AD',
        baseline_mu,  baseline_sigma_inv,  baseline_threshold,
        test_df_raw, FEATURE_COLS, WINDOW_SIZE, unit_id, '#2196F3')
    plot_engine_anomaly_scores(attention_model, 'Attention EncDec-AD',
        attention_mu, attention_sigma_inv, attention_threshold,
        test_df_raw, FEATURE_COLS, WINDOW_SIZE, unit_id, '#4CAF50')

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', f'F-{BETA} Score', 'FPR',
               'TPR/FPR', 'Threshold (τ)', 'Model Parameters', 'Training Epochs'],
    'Baseline EncDec-AD': [
        f"{baseline_metrics['precision']:.4f}",
        f"{baseline_metrics['recall']:.4f}",
        f"{baseline_metrics['fbeta']:.4f}",
        f"{baseline_metrics['fpr']:.4f}",
        f"{baseline_metrics['recall']/baseline_metrics['fpr']:.2f}" if baseline_metrics['fpr']>0 else 'inf',
        f"{baseline_threshold:.4f}",
        f"{baseline_model.count_params():,}",
        f"{len(baseline_history.history['loss'])}"
    ],
    'Attention EncDec-AD': [
        f"{attention_metrics['precision']:.4f}",
        f"{attention_metrics['recall']:.4f}",
        f"{attention_metrics['fbeta']:.4f}",
        f"{attention_metrics['fpr']:.4f}",
        f"{attention_metrics['recall']/attention_metrics['fpr']:.2f}" if attention_metrics['fpr']>0 else 'inf',
        f"{attention_threshold:.4f}",
        f"{attention_model.count_params():,}",
        f"{len(attention_history.history['loss'])}"
    ]
})

print('\n' + '='*70)
print('           ANOMALY DETECTION — FINAL RESULTS SUMMARY')
print('='*70)
print(f'Dataset:     NASA C-MAPSS FD001')
print(f'Window Size: {WINDOW_SIZE} | Latent Dim: {LATENT_DIM} | Beta: {BETA}')
print()
print(comparison_df.to_string(index=False))
print('\n' + '='*70)
print('Notebook execution complete.')
print('='*70)
