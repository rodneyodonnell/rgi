#! /usr/bin/env python3
import torch
import jax
import tensorflow as tf

print("\n\nChecking GPU availability...")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
assert torch.cuda.is_available()

print(f"tf.config.list_physical_devices('GPU'): {tf.config.list_physical_devices('GPU')}")
assert "GPU" in str(tf.config.list_physical_devices("GPU"))

print(f"jax.devices(): {jax.devices()}")
assert "Cuda" in str(jax.devices())

print("\nAll GPU checks passed!")
