import os
import numpy as np
import matplotlib.pyplot as plt
from signals import MODS, CONST, generate, channel

os.makedirs('results', exist_ok=True)
rng = np.random.default_rng(1)

# 1) Constellations: ideal symbols plus noise at two SNRs
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for r, snr in enumerate([20, 5]):
    for c, mod in enumerate(['BPSK', 'QPSK', '8PSK', '16QAM']):
        s = rng.choice(CONST[mod], 2000)
        noise = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
        s = s + np.sqrt(10 ** (-snr / 10) / 2) * noise
        a = ax[r, c]
        a.scatter(s.real, s.imag, s=3)
        a.set_aspect('equal')
        a.set_title(mod + ' @ ' + str(snr) + ' dB')
        a.grid(True)
plt.tight_layout()
plt.savefig('results/constellations.png', dpi=150)

# 2) Spectrograms of all six types
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, mod in zip(ax.flat, MODS):
    x = channel(generate(mod, n=2048, rng=rng), 15, rng)
    a.specgram(x, NFFT=64, Fs=1, noverlap=48, cmap='viridis')
    a.set_title(mod)
plt.tight_layout()
plt.savefig('results/spectrograms.png', dpi=150)

# 3) Time-domain look at one frame per type
fig, ax = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
for a, mod in zip(ax, MODS):
    x = channel(generate(mod, rng=rng), 20, rng)
    a.plot(x.real, label='I')
    a.plot(x.imag, label='Q')
    a.set_ylabel(mod)
ax[0].legend(loc='upper right')
plt.tight_layout()
plt.savefig('results/time_domain.png', dpi=150)

print('saved 3 images in results/')