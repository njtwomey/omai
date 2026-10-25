"""Audio metrics: SNR, SDR, SI-SDR, SI-SNR, permutation-invariant training and the DNSMOS calibration from
torchmetrics; STOI and ESTOI from pystoi; the log-spectral distance on scipy.signal.stft; the mel-cepstral distortion
with dtw-python alignment; the segmental SNR from its definition (pysepm's SNRseg).
"""

from importlib.metadata import version
from typing import Any

import dtw
import numpy as np
import scipy
import scipy.signal as ss
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from pystoi import stoi
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)
from torchmetrics.functional.audio.dnsmos import _polyfit_val  # pyright: ignore[reportPrivateUsage]

REFERENCE: dict[str, str] = {
    "torchmetrics": version("torchmetrics"),
    "pystoi": version("pystoi"),
    "scipy": scipy.__version__,
    "dtw-python": version("dtw-python"),
}


def speech(fs: int, seconds: float, seed: int) -> FloatArray:
    """A speech-like test signal: syllable-rate modulated harmonics with pauses."""
    g = rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    voiced = sum((0.6**k) * np.sin(k * phase + g.uniform(0, 2 * np.pi)) for k in range(1, 12))
    envelope = np.clip(np.sin(2 * np.pi * 3.1 * t), 0, None) ** 2
    envelope[(t > 0.55) & (t < 0.8)] = 0
    return np.asarray(voiced * envelope + 0.01 * g.normal(size=t.size))


def segmental_snr(clean: FloatArray, processed: FloatArray, fs: int) -> float:
    eps = np.finfo(np.float64).eps
    m = round(0.03 * fs)
    skip = int(np.floor(0.25 * 0.03 * fs))
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (m + 1)))
    count = (len(clean) - (m - skip)) // skip
    cf = np.array([w * clean[i * skip : i * skip + m] for i in range(count)])
    pf = np.array([w * processed[i * skip : i * skip + m] for i in range(count)])
    energy = (cf**2).sum(-1)
    noise = ((cf - pf) ** 2).sum(-1)
    snr = np.clip(10 * np.log10(energy / (noise + eps) + eps), -10, 35)[:-1]
    return float(np.mean(snr))


def lsd(target: FloatArray, preds: FloatArray, nfft: int, hop: int, eps: float) -> float:
    def spec(x: FloatArray) -> FloatArray:
        _, _, z = ss.stft(x, window="hann", nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
        return np.abs(z) ** 2

    ratio = 10 * np.log10((spec(target) + eps) / (spec(preds) + eps))
    return float(np.mean(np.sqrt(np.mean(ratio**2, axis=0))))


def mcd(a: FloatArray, b: FloatArray, align: bool) -> float:
    k = 10 / np.log(10) * np.sqrt(2)

    def dist(x: FloatArray, y: FloatArray) -> float:
        return float(k * np.sqrt(((x[1:] - y[1:]) ** 2).sum()))

    if not align:
        return float(np.mean([dist(x, y) for x, y in zip(a, b, strict=True)]))
    # dtw-python's stubs type step_pattern as a string and omit the path attributes.
    options: dict[str, Any] = {"dist_method": "euclidean", "step_pattern": dtw.symmetric1}
    path: Any = dtw.dtw(a[:, 1:], b[:, 1:], **options)
    return float(np.mean([dist(a[i], b[j]) for i, j in zip(path.index1, path.index2, strict=True)]))


def p835(segment: FloatArray) -> tuple[float, float, float]:
    """A stand-in for the P.835 network: smooth functions of segment statistics."""
    rms = float(np.sqrt(np.mean(segment**2)))
    peak = float(np.max(np.abs(segment)))
    return 2 + rms, 3 - 0.5 * peak, 1.5 + rms * peak


def p808(segment: FloatArray) -> float:
    return 2.5 + float(np.mean(np.abs(segment))) + len(segment) / 1e6


def dnsmos(audio: FloatArray, personalized: bool) -> list[float]:
    fs = 16000
    length = int(9.01 * fs)
    while audio.shape[-1] < length:
        audio = np.concatenate([audio, audio])
    hops = int(np.floor(audio.shape[-1] / fs) - 9.01) + 1
    moss: list[FloatArray] = []
    for idx in range(hops):
        seg = audio[int(idx * fs) : int((idx + 9.01) * fs)]
        if seg.shape[-1] < length:
            continue
        raw = np.array([[p808(seg[:-160]), *p835(seg)]], dtype=float)
        moss.append(_polyfit_val(raw, personalized)[0])
    return [float(v) for v in np.mean(np.stack(moss), axis=0)]


def cases() -> list[Case]:
    g = rng(90)
    out: list[Case] = []
    t = g.normal(size=(3, 800))
    p = 0.8 * t + 0.3 * g.normal(size=(3, 800)) + 0.1
    p[:, 1:] += 0.4 * t[:, :-1]
    tt, pt = torch.tensor(t), torch.tensor(p)
    out.append(
        case("signalNoiseRatio", "signalNoiseRatio", [t, p], np.asarray(float(signal_noise_ratio(pt, tt).mean())))
    )
    out.append(
        case(
            "signalNoiseRatio/zeroMean",
            "signalNoiseRatio",
            [t, p],
            np.asarray(float(signal_noise_ratio(pt, tt, zero_mean=True).mean())),
            zeroMean=True,
        )
    )
    si = float(scale_invariant_signal_distortion_ratio(pt, tt).mean())
    out.append(case("scaleInvariantSdr", "scaleInvariantSdr", [t, p], np.asarray(si)))
    out.append(
        case(
            "scaleInvariantSnr",
            "scaleInvariantSnr",
            [t, p],
            np.asarray(float(scale_invariant_signal_noise_ratio(pt, tt).mean())),
        )
    )
    for length, zero in [(512, False), (32, True)]:
        sdr = float(signal_distortion_ratio(pt, tt, filter_length=length, zero_mean=zero).mean())
        out.append(
            case(
                f"signalDistortionRatio/{length}",
                "signalDistortionRatio",
                [t, p],
                np.asarray(sdr),
                rtol=1e-9,
                filterLength=length,
                zeroMean=zero,
            )
        )
    loaded = float(signal_distortion_ratio(pt, tt, filter_length=64, load_diag=1e-3).mean())
    out.append(
        case(
            "signalDistortionRatio/loadDiag",
            "signalDistortionRatio",
            [t, p],
            np.asarray(loaded),
            rtol=1e-9,
            filterLength=64,
            loadDiag=1e-3,
        )
    )

    # Permutation-invariant SI-SDR of three mixed-up sources.
    order = [2, 0, 1]
    est = p[order]
    best, perm = permutation_invariant_training(
        torch.tensor(est)[None], tt[None], scale_invariant_signal_distortion_ratio, mode="speaker-wise", eval_func="max"
    )
    out.append(case("permutationInvariant/value", "pitValue", [t, est], np.asarray(float(best[0]))))
    out.append(case("permutationInvariant/permutation", "pitPermutation", [t, est], perm[0].numpy().astype(float)))

    for fs in [10000, 16000, 8000]:
        clean = speech(fs, 1.3, 91)
        noisy = clean + 0.3 * rng(92).normal(size=clean.size) * np.std(clean)
        np.random.seed(0)  # ESTOI adds eps-scale noise from the global generator
        values = [float(stoi(clean, noisy, fs, extended=extended)) for extended in [False, True]]
        out.append(case(f"stoi/{fs}", "stoi", [clean, noisy], np.asarray(values), rtol=1e-9, atol=1e-10, fs=fs))

    clean = speech(8000, 0.5, 93)
    noisy = clean + 0.05 * rng(94).normal(size=clean.size)
    out.append(
        case("segmentalSnr", "segmentalSnr", [clean, noisy], np.asarray(segmental_snr(clean, noisy, 8000)), fs=8000)
    )
    out.append(
        case(
            "logSpectralDistance",
            "logSpectralDistance",
            [clean, noisy],
            np.asarray(lsd(clean, noisy, 512, 128, 1e-10)),
            rtol=1e-10,
        )
    )
    out.append(
        case(
            "logSpectralDistance/256",
            "logSpectralDistance",
            [clean, noisy],
            np.asarray(lsd(clean, noisy, 256, 64, 1e-6)),
            rtol=1e-10,
            nfft=256,
            hop=64,
            eps=1e-6,
        )
    )

    ca = g.normal(size=(20, 13))
    cb = ca + 0.2 * g.normal(size=(20, 13))
    cc = np.repeat(ca, 2, axis=0)[:33] + 0.1 * g.normal(size=(33, 13))
    out.append(case("melCepstralDistortion", "melCepstralDistortion", [ca, cb], np.asarray(mcd(ca, cb, False))))
    out.append(
        case("melCepstralDistortion/dtw", "melCepstralDistortion", [ca, cc], np.asarray(mcd(ca, cc, True)), align="dtw")
    )

    audio = speech(16000, 0.4, 95)  # repeated to 12.8 s: three windows
    for personalized in [False, True]:
        out.append(
            case(
                f"dnsmos/{'personalized' if personalized else 'default'}",
                "dnsmos",
                [audio],
                np.asarray(dnsmos(audio, personalized)),
                personalized=personalized,
            )
        )
    return out
