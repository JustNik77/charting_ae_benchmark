import numpy as np
import os

def generate_quasiperiodic_series(seq_len=15000):
    dt = 2 * np.pi / 24 
    t = np.arange(seq_len) * dt
    freq1 = 1.0
    freq2 = np.sqrt(2)
    signal = np.sin(freq1 * t) + 0.5 * np.cos(freq2 * t)
    noise = np.random.normal(0, 0.05, seq_len)
    return signal + noise

def create_sliding_windows(series, window_size, step=1):
    shape = ((series.shape[0] - window_size) // step + 1, window_size)
    strides = (series.strides[0] * step, series.strides[0])
    windows = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
    return windows


WINDOW_SIZE = 24
SEQ_LENGTH = 15000

print("generating signal")
raw_signal = generate_quasiperiodic_series(SEQ_LENGTH)

print(f"create sliding windows {WINDOW_SIZE}...")
dataset = create_sliding_windows(raw_signal, window_size=WINDOW_SIZE, step=1)

print(f"dataset shape: {dataset.shape}")

os.makedirs('data', exist_ok=True)
save_path = "data/synthetic_torus_data.npy"
np.save(save_path, dataset)
print(f"saved: {save_path}")

# loaded_data = np.load("synthetic_torus_data.npy")
# tensor_data = torch.from_numpy(loaded_data).float()
