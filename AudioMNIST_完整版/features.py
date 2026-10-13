"""Audio-only frontend. Classic signal transforms, not a newly invented method.

每条录音独立计算，不使用标签、不跨样本拟合。归一化在训练折内部另做。
The formulas match the preceding exploratory pipeline, but caches are rebuilt.
"""
import numpy as np
from scipy.signal import stft
from scipy.fft import dct


def mel_bank(sr=16000, nfft=512, bands=32):
    """HTK-style Mel scale; triangular filters normalised to unit row sum."""
    to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(to_mel(20), to_mel(sr/2), bands+2)/2595)-1)
    hz = np.linspace(0, sr/2, nfft//2+1)
    bank = np.zeros((bands, len(hz)))
    for i in range(bands):
        bank[i] = np.maximum(0, np.minimum((hz-edges[i])/(edges[i+1]-edges[i]),
                                          (edges[i+2]-hz)/(edges[i+2]-edges[i+1])))
        bank[i] /= max(bank[i].sum(), 1e-12)
    return bank.astype(np.float32)


BANK = mel_bank()


def audio_features(waveforms):
    """批量输入 shape=(n,16000)，返回 256 / 320 维两套独立输入矩阵。"""
    _, _, spectrum = stft(waveforms, fs=16000, window='hann', nperseg=400,
                           noverlap=240, nfft=512, boundary=None, padded=False, axis=-1)
    power = np.abs(spectrum)**2
    logmel = np.log(np.maximum(np.einsum('mf,nft->nmt', BANK, power), 1e-12))
    mfcc = dct(logmel, type=2, axis=1, norm='ortho')[:, :20, :]
    # Per-frame central differences in the interior; one-sided at endpoints.
    delta = np.gradient(mfcc, axis=2)
    def pool(x):
        # Ordering: segment1 means, segment1 stds, segment2 means, ...
        return np.concatenate([v for block in np.array_split(x, 4, axis=2)
                               for v in (block.mean(2), block.std(2))], axis=1).astype(np.float32)
    return {'logmel': pool(logmel), 'mfcc_delta': pool(np.concatenate([mfcc, delta], axis=1))}


def feature_names(representation):
    channels = ([f'mel_band_{i:02}' for i in range(1,33)] if representation == 'logmel'
                else [f'mfcc_{i:02}' for i in range(20)] + [f'delta_mfcc_{i:02}' for i in range(20)])
    return [f'segment_{s}_{stat}_{channel}' for s in range(1,5)
            for stat in ('mean','std') for channel in channels]


def noisy_audio(waveforms, snr, seed, pink=False):
    """合成噪声：SNR 按补零后的完整 1 秒计算，不是按有效语音段计算。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=waveforms.shape).astype(np.float32)
    if pink:
        fourier = np.fft.rfft(noise, axis=1)
        fourier /= np.sqrt(np.maximum(np.arange(fourier.shape[1]), 1))[None,:]
        noise = np.fft.irfft(fourier, n=noise.shape[1], axis=1).astype(np.float32)
    target = np.sqrt(np.mean(waveforms**2, 1)) / 10**(snr/20)
    noise *= (target/np.maximum(np.sqrt(np.mean(noise**2, 1)), 1e-12))[:,None]
    return waveforms + noise
