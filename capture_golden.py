"""
Phase 0 golden-value capture script.

Loads the pre-trained SDF ensemble from sample_checkpoints/, runs inference on
the test split, and saves the outputs to .claude/golden/ as .npy files.

These files become the numerical regression targets for the PyTorch port.

Usage:
    python capture_golden.py

Requires:
    - datasets/char/Char_test.npz
    - datasets/macro/macro_test.npz
    - datasets/char/Char_train.npz   (for macro normalization stats)
    - datasets/macro/macro_train.npz
    - sample_checkpoints/ populated
"""

import json
import os
import numpy as np

from src.data import data_layer
from src.model.model_GAN import FeedForwardModelWithNA_GAN_Ensembled
from src.tf_compat import tf

GOLDEN_DIR = ".claude/golden"
CONFIG_PATH = "config/config.json"
CHECKPOINT_BASE = "sample_checkpoints/sample_checkpoints"
NUM_TRIALS = 9

os.makedirs(GOLDEN_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    config = json.load(f)
if "macro_idx" not in config:
    config["macro_idx"] = None

# Build normalization stats from training split
dl_train = data_layer.DataInRamInputLayer(
    config["individual_feature_file"],
    pathMacroFeature=config["macro_feature_file"],
    macroIdx=config["macro_idx"],
)
mean_macro, std_macro = dl_train.getMacroFeatureMeanStd()

dl_test = data_layer.DataInRamInputLayer(
    config["individual_feature_file_test"],
    pathMacroFeature=config["macro_feature_file_test"],
    macroIdx=config["macro_idx"],
    meanMacroFeature=mean_macro,
    stdMacroFeature=std_macro,
)

logdirs = [
    os.path.join(CHECKPOINT_BASE, f"Task_1_Trial_{i}", "sharpe")
    for i in range(NUM_TRIALS)
]

model = FeedForwardModelWithNA_GAN_Ensembled(logdirs, config, "test", config["tSize_test"])

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=sess_config)
model._model.randomInitialization(sess)

print("Capturing SDF on test split...")
sdf = model.getSDF(sess, dl_test)
np.save(os.path.join(GOLDEN_DIR, "sdf_test.npy"), sdf)
print(f"  sdf_test.npy  shape={sdf.shape}  mean={sdf.mean():.6f}  std={sdf.std():.6f}")

print("Capturing SDF weights on test split...")
weights = model.getWeightWithData(sess, dl_test)
np.save(os.path.join(GOLDEN_DIR, "weights_test.npy"), weights)
print(f"  weights_test.npy  shape={weights.shape}  mean={weights.mean():.6f}")

print("Copying pre-computed normalized SDF ensemble...")
src = os.path.join(CHECKPOINT_BASE, "sdf_normalized_ensemble.npy")
sdf_norm = np.load(src)
np.save(os.path.join(GOLDEN_DIR, "sdf_normalized_ensemble.npy"), sdf_norm)
print(f"  sdf_normalized_ensemble.npy  shape={sdf_norm.shape}")

print(f"\nAll golden files saved to {GOLDEN_DIR}/")
sess.close()