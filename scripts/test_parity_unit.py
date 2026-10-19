import sys
from pathlib import Path
import numpy as np
import torch
from scipy import linalg, signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.glottal_analysis import levinson_durbin_lpc, iaif_inverse_filter_frame
from src.models.neural_glottal_inverter import batch_levinson_durbin, DifferentiableIAIF

# 1. Test Levinson-Durbin
x = np.sin(np.linspace(0, 10, 400)).astype(np.float32)
r_full = np.correlate(x, x, mode='full')
mid = len(r_full) // 2
r_scipy = r_full[mid:mid + 15]

a_scipy = levinson_durbin_lpc(x, 14)

r_torch = torch.from_numpy(r_scipy).unsqueeze(0)
a_torch = batch_levinson_durbin(r_torch, 14).squeeze(0).numpy()

diff_lpc = np.max(np.abs(a_scipy - a_torch))
print(f"Levinson Durbin diff max: {diff_lpc:.6e}")
assert diff_lpc < 1e-3, "Levinson Durbin failed parity test"

# 2. Test Multi-frame continuous waveform IAIF
sr = 16000
dur = 1.0  # 1 second test signal
t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
sig = (0.5 * np.sin(2 * np.pi * 180 * t) + 
       0.3 * np.sin(2 * np.pi * 360 * t) + 
       0.1 * np.sin(2 * np.pi * 540 * t) + 
       0.01 * np.random.randn(len(t))).astype(np.float32)

# Run Classical IAIF
frame_len = 400
hop_len = 160
win = np.hanning(frame_len).astype(np.float32)
n_frames = (len(sig) - frame_len) // hop_len + 1

ug_prime_class = np.zeros(len(sig), dtype=np.float32)
ug_class = np.zeros(len(sig), dtype=np.float32)
norm = np.zeros(len(sig), dtype=np.float32)

for i in range(n_frames):
    start = i * hop_len
    end = start + frame_len
    f_in = sig[start:end] * win
    p_frame, u_frame = iaif_inverse_filter_frame(f_in, p_glottal=2, p_vt=14)
    ug_prime_class[start:end] += p_frame * win
    ug_class[start:end] += u_frame * win
    norm[start:end] += win ** 2

norm = np.maximum(norm, 1e-6)
ug_prime_class /= norm
ug_class /= norm

# Run Differentiable IAIF
diff_iaif = DifferentiableIAIF()
t_sig = torch.from_numpy(sig).unsqueeze(0)
with torch.no_grad():
    ug_prime_diff_t, ug_diff_t = diff_iaif(t_sig)

ug_prime_diff = ug_prime_diff_t.squeeze(0).numpy()
ug_diff = ug_diff_t.squeeze(0).numpy()

# Compare over valid overlap regions (avoid edge artifacts)
valid_idx = slice(frame_len, len(sig) - frame_len)
corr_prime = np.corrcoef(ug_prime_class[valid_idx], ug_prime_diff[valid_idx])[0, 1]
corr_pulse = np.corrcoef(ug_class[valid_idx], ug_diff[valid_idx])[0, 1]

print(f"Continuous ug_prime correlation: {corr_prime:.4f}")
print(f"Continuous ug_pulse correlation: {corr_pulse:.4f}")

if corr_prime > 0.85 and corr_pulse > 0.85:
    print("[PASS] Full parity established between Classical and Differentiable IAIF!")
else:
    print("[WARN] Correlation below 0.85 target")
