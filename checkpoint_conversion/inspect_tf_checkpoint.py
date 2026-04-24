"""Inspect a TF1 name-based checkpoint and print variable names and shapes.

Usage:
    python checkpoint_conversion/inspect_tf_checkpoint.py \
        sample_checkpoints/sample_checkpoints_RF/RF_1_Trial_0/model-best
"""

from __future__ import annotations

import argparse

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint prefix, e.g. .../model-best")
    return parser.parse_args()


def main():
    args = parse_args()
    reader = tf.train.load_checkpoint(args.checkpoint)
    for name, shape in sorted(reader.get_variable_to_shape_map().items()):
        print(f"{name} {shape}")


if __name__ == "__main__":
    main()

