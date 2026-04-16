# ============================================================
# IMPORTS
# ============================================================

import os
import json
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    LSTM, Dense, RepeatVector, TimeDistributed,
    Concatenate, Layer, Dropout
)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

print(f'TensorFlow version: {tf.__version__}')

# ============================================================
# GPU CONFIGURATION
# ============================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU(s) available: {[gpu.name for gpu in gpus]}')
    except RuntimeError as e:
        print(f'GPU config error: {e}')
else:
    print('No GPU detected. Using CPU.')

# ============================================================
# LOAD PREPROCESSED DATA & METADATA
# ============================================================

PROCESSED_DIR = './processed_data'
MODEL_DIR     = './saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(PROCESSED_DIR, 'metadata.json')) as f:
    meta = json.load(f)

WINDOW_SIZE   = meta['WINDOW_SIZE']
n_features    = meta['n_features']
FEATURE_COLS  = meta['FEATURE_COLS']
SEED          = meta['SEED']
BATCH_SIZE    = meta['BATCH_SIZE']

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
LATENT_DIM      = 64
ATTENTION_UNITS = 32
LEARNING_RATE   = 5e-4
BETA            = 1.0   # balanced F1

print(f'Loaded metadata: window={WINDOW_SIZE}, features={n_features}')
print(f'Feature columns: {FEATURE_COLS}')

# ============================================================
# MODEL 1: BASELINE LSTM ENCODER-DECODER
# ============================================================

def build_baseline_encoder_decoder(timesteps, n_features, latent_dim):
    """
    Baseline LSTM Encoder-Decoder (EncDec-AD).

    Encoder: LSTM → final hidden state (h_T, c_T)
    Decoder: RepeatVector → LSTM (initialised with encoder state) → Dense
    Target : reversed input sequence
    """
    # --- Encoder ---
    encoder_inputs = Input(shape=(timesteps, n_features), name='encoder_input')

    encoder_lstm = LSTM(
        latent_dim, return_sequences=False, return_state=True,
        name='encoder_lstm'
    )
    encoder_output, state_h, state_c = encoder_lstm(encoder_inputs)

    # --- Decoder ---
    decoder_input = RepeatVector(timesteps, name='repeat_vector')(state_h)

    decoder_lstm = LSTM(
        latent_dim, return_sequences=True,
        name='decoder_lstm'
    )
    decoder_output = decoder_lstm(decoder_input, initial_state=[state_h, state_c])

    outputs = TimeDistributed(
        Dense(n_features, name='dense_output'),
        name='time_distributed_output'
    )(decoder_output)

    model = Model(encoder_inputs, outputs, name='EncDec_AD_Baseline')
    return model


baseline_model = build_baseline_encoder_decoder(WINDOW_SIZE, n_features, LATENT_DIM)
baseline_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
baseline_model.summary()

# ============================================================
# CUSTOM BAHDANAU ATTENTION LAYER
# ============================================================

class BahdanauAttention(Layer):
    """
    Bahdanau (Additive) Attention Layer.

    score(sᵢ, hⱼ) = Vᵀ tanh(W₁ hⱼ + W₂ sᵢ)
    αᵢⱼ = softmax(score) over j
    cᵢ  = Σⱼ αᵢⱼ hⱼ
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units, use_bias=False, name='attn_W1')  # encoder projection
        self.W2 = Dense(units, use_bias=False, name='attn_W2')  # decoder projection
        self.V  = Dense(1,     use_bias=False, name='attn_V')   # scalar score

    def call(self, encoder_outputs, decoder_outputs):
        """
        Args:
            encoder_outputs : (batch, T_enc, enc_dim)
            decoder_outputs : (batch, T_dec, dec_dim)
        Returns:
            context          : (batch, T_dec, enc_dim)
            attention_weights: (batch, T_dec, T_enc)
        """
        # Project to attention space
        enc_proj = self.W1(encoder_outputs)          # (batch, T_enc, units)
        dec_proj = self.W2(decoder_outputs)           # (batch, T_dec, units)

        # Broadcast: enc (batch,1,T_enc,units), dec (batch,T_dec,1,units)
        enc_proj = tf.expand_dims(enc_proj, axis=1)
        dec_proj = tf.expand_dims(dec_proj, axis=2)

        # Additive score → (batch, T_dec, T_enc)
        score = tf.squeeze(self.V(tf.nn.tanh(enc_proj + dec_proj)), axis=-1)

        # Softmax over encoder timesteps
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch, T_dec, T_enc)

        # Weighted sum of encoder outputs
        context = tf.matmul(attention_weights, encoder_outputs)  # (batch, T_dec, enc_dim)

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


print('BahdanauAttention layer defined.')

# ============================================================
# MODEL 2: ATTENTION-BASED LSTM ENCODER-DECODER
# ============================================================

def build_attention_encoder_decoder(timesteps, n_features, latent_dim, attention_units):
    """
    Attention-augmented LSTM Encoder-Decoder.

    Encoder: LSTM(return_sequences=True) — all hidden states available
    Decoder: LSTM → Bahdanau attention → Concatenate(decoder_out, context) → Dense
    """
    # --- Encoder ---
    encoder_inputs = Input(shape=(timesteps, n_features), name='encoder_input')

    encoder_lstm = LSTM(
        latent_dim,
        return_sequences=True, return_state=True,
        dropout=0.2, recurrent_dropout=0.2,
        name='encoder_lstm'
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_outputs = Dropout(0.2)(encoder_outputs)

    # --- Decoder ---
    decoder_input = RepeatVector(timesteps, name='repeat_vector')(state_h)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        dropout=0.2, recurrent_dropout=0.2,
        name='decoder_lstm'
    )
    decoder_outputs = decoder_lstm(decoder_input, initial_state=[state_h, state_c])

    # --- Attention ---
    attention_layer = BahdanauAttention(attention_units, name='bahdanau_attention')
    context, _      = attention_layer(encoder_outputs, decoder_outputs)
    context         = Dropout(0.3)(context)

    # Concatenate decoder output with context
    decoder_combined = layers.Concatenate(axis=-1, name='concat_context')(
        [decoder_outputs, context]
    )

    outputs = TimeDistributed(
        Dense(n_features, name='dense_output'),
        name='time_distributed_output'
    )(decoder_combined)

    model = Model(encoder_inputs, outputs, name='EncDec_AD_Attention')
    return model


attention_model = build_attention_encoder_decoder(
    WINDOW_SIZE, n_features, LATENT_DIM, ATTENTION_UNITS
)
attention_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
attention_model.summary()

# ============================================================
# RECONSTRUCTION ERROR COMPUTATION
# Per Malhotra et al. §2.2: e^(i) = |x^(i) − x'^(i)|
# ============================================================

def compute_reconstruction_errors(model, X, batch_size=256):
    """
    Compute per-timestep reconstruction errors.

    Returns:
        errors_3d  : (n_samples, timesteps, n_features) — full error tensor
        errors_flat: (n_samples * timesteps, n_features) — used for Gaussian fitting
    """
    X_hat_reversed = model.predict(X, batch_size=batch_size, verbose=0)
    X_hat          = X_hat_reversed[:, ::-1, :]     # un-reverse decoder output
    errors_3d      = np.abs(X - X_hat)              # e^(i) = |x^(i) − x'^(i)|
    errors_flat    = errors_3d.reshape(-1, X.shape[2])
    return errors_3d, errors_flat


print('compute_reconstruction_errors defined.')

# ============================================================
# MULTIVARIATE GAUSSIAN FITTING
# ============================================================

def fit_gaussian(error_flat):
    """
    Fit a multivariate Gaussian N(μ, Σ) to per-timestep reconstruction errors.

    Args:
        error_flat : (N*L, m) — per-timestep absolute errors from vN2
    Returns:
        mu       : (m,)
        sigma    : (m, m)
        sigma_inv: (m, m)
    """
    mu        = np.mean(error_flat, axis=0)
    sigma     = np.cov(error_flat, rowvar=False) + np.eye(error_flat.shape[1]) * 1e-5
    sigma_inv = np.linalg.inv(sigma)
    return mu, sigma, sigma_inv


print('fit_gaussian defined.')

# ============================================================
# MAHALANOBIS ANOMALY SCORING
# a^(i) = (e^(i) − μ)ᵀ Σ⁻¹ (e^(i) − μ)
# Sequence score = max over timesteps
# ============================================================

def compute_anomaly_scores(errors_3d, mu, sigma_inv):
    """
    Compute sequence-level Mahalanobis anomaly scores.

    Args:
        errors_3d : (n_samples, L, m)
    Returns:
        scores    : (n_samples,) — max Mahalanobis distance over L timesteps
    """
    n, L, m = errors_3d.shape
    diff    = errors_3d.reshape(n * L, m) - mu      # (N*L, m)
    maha    = np.sum(diff @ sigma_inv * diff, axis=1).reshape(n, L)  # (N, L)
    return maha.max(axis=1)                          # (N,)


print('compute_anomaly_scores defined.')

# ============================================================
# THRESHOLD OPTIMISATION
# τ chosen to maximise F_β on validation set (vN2 + vA)
# ============================================================

def optimize_threshold(scores, labels, beta=BETA, n_thresholds=500):
    """
    Find threshold τ that maximises F_β on the combined validation set.

    Args:
        scores       : (n_samples,) — Mahalanobis anomaly scores
        labels       : (n_samples,) — 0=normal, 1=anomalous
        beta         : float        — β for F_β  (1.0 = balanced F1)
        n_thresholds : int          — candidate grid size

    Returns:
        best_threshold     : float
        best_fbeta         : float
        threshold_results  : dict
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    precisions, recalls, fbetas = [], [], []

    for tau in thresholds:
        preds = (scores > tau).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fb        = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(fb)

    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    fbetas     = np.array(fbetas)

    best_idx       = np.argmax(fbetas)
    best_threshold = thresholds[best_idx]
    best_fbeta     = fbetas[best_idx]

    return best_threshold, best_fbeta, {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls':    recalls,
        'fbetas':     fbetas,
        'best_idx':   best_idx,
    }


print('optimize_threshold defined.')
print('\nAll model components ready. Proceed to model_training.ipynb.')
