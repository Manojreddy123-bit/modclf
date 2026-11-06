import json
import numpy as np
from tqdm import tqdm
from signals import MODS, generate, channel, to_iq

PER = 1000                      # frames per modulation per SNR (use 300 for a quick test)
SNRS = list(range(-10, 21, 2))  # -10, -8, ..., 20 dB
rng = np.random.default_rng(42)

X, y, snr = [], [], []
for ci, mod in enumerate(MODS):
    for s in tqdm(SNRS, desc=mod):
        for _ in range(PER):
            X.append(to_iq(channel(generate(mod, rng=rng), s, rng)))
            y.append(ci)
            snr.append(s)

X, y, snr = np.stack(X), np.array(y), np.array(snr)
idx = rng.permutation(len(X))
X, y, snr = X[idx], y[idx], snr[idx]

n1, n2 = int(0.7 * len(X)), int(0.85 * len(X))   # 70 / 15 / 15 split
np.savez_compressed('dataset.npz',
                    Xtr=X[:n1], ytr=y[:n1],
                    Xva=X[n1:n2], yva=y[n1:n2],
                    Xte=X[n2:], yte=y[n2:], snr_te=snr[n2:])
json.dump(MODS, open('labels.json', 'w'))
print('X shape:', X.shape)
print('train/val/test:', n1, n2 - n1, len(X) - n2)