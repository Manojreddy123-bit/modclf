# signals.py - synthetic modulation generator (complex baseband)
import numpy as np
from scipy.signal import firwin, lfilter

MODS = ['AM', 'FM', 'BPSK', 'QPSK', '8PSK', '16QAM']
N = 128      # samples per frame
SPS = 8      # samples per symbol (digital)
SPAN = 8     # RRC filter span in symbols

CONST = {
    'BPSK':  np.array([-1, 1], dtype=complex),
    'QPSK':  np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))),
    '8PSK':  np.exp(1j * 2 * np.pi * np.arange(8) / 8),
    '16QAM': np.array([x + 1j * y for x in (-3, -1, 1, 3) for y in (-3, -1, 1, 3)]) / np.sqrt(10),
}

def rrc(beta=0.35, sps=SPS, span=SPAN):
    # root-raised-cosine pulse shaping filter
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(x) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))

RRC = rrc()

def _norm(x):
    return x / np.sqrt(np.mean(np.abs(x) ** 2))

def gen_digital(mod, n, rng):
    nsym = n // SPS + 2 * SPAN
    sym = rng.choice(CONST[mod], nsym)
    up = np.zeros(nsym * SPS, dtype=complex)
    up[::SPS] = sym                                  # one symbol every SPS samples
    x = np.convolve(up, RRC, mode='same')            # pulse shaping
    start = SPAN * SPS + rng.integers(0, SPS)        # random timing offset
    return _norm(x[start:start + n])

def _message(n, rng):
    # random low-pass 'voice-like' message signal
    m = lfilter(firwin(65, 0.05), 1, rng.standard_normal(n + 200))[200:]
    return m / np.max(np.abs(m))

def gen_am(n, rng):
    depth = rng.uniform(0.3, 0.9)
    return _norm((1 + depth * _message(n, rng)).astype(complex))

def gen_fm(n, rng):
    dev = rng.uniform(0.05, 0.15)                    # peak deviation, cycles/sample
    return np.exp(1j * 2 * np.pi * dev * np.cumsum(_message(n, rng)))

def generate(mod, n=N, rng=None):
    rng = rng or np.random.default_rng()
    if mod == 'AM': return gen_am(n, rng)
    if mod == 'FM': return gen_fm(n, rng)
    return gen_digital(mod, n, rng)

def channel(x, snr_db, rng, cfo_max=0.01, fading=True):
    n = len(x)
    # random carrier frequency offset and phase
    x = x * np.exp(1j * (2 * np.pi * rng.uniform(-cfo_max, cfo_max) * np.arange(n) + rng.uniform(0, 2 * np.pi)))
    if fading:
        # simple 3-tap multipath channel
        cn = lambda: (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        taps = np.array([1.0, 0.3 * cn(), 0.15 * cn()])
        x = np.convolve(x, taps)[:n]
    x = _norm(x)
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)        # noise std per I/Q component
    return x + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

def to_iq(x):
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)   # shape (n, 2)