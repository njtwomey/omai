# Signal processing (part 2, signals, systems and information)

Signals filtered, transformed and analysed, with audio as its own section: sampling and aliasing, Fourier analysis, filter design and application, multirate processing, time-frequency and wavelet transforms, sparse representations, spectral estimation, adaptive filters and audio features, as sessions over `signal`.

## Capabilities

- Signals and systems: sampling and aliasing, convolution stepped, the z-transform; options: the signal (named, synthesised, a recorded clip), the sample rate.
- Fourier: the DFT built column by column, windowing and leakage, the FFT's butterflies.
- Filters: FIR by windowing and Parks-McClellan, IIR by Butterworth and friends; the response as the order changes; the filter applied sample by sample.
- Multirate: decimation, interpolation, polyphase and filter banks with perfect reconstruction.
- Time-frequency: the STFT with window and hop as knobs, constant-Q, wavelets and denoising by thresholding, empirical mode decomposition.
- Sparse: matching and orthogonal matching pursuit, basis pursuit, dictionary learning (K-SVD) stepped.
- Spectral estimation: periodogram, Welch, multitaper, Lomb-Scargle, AR models.
- Adaptive: LMS and RLS tracking a changing system, the error on the tape.
- Audio: mel and gammatone banks, MFCCs, chroma, pitch and beat tracking, source separation by NMF.

## Examples to make

- Aliasing: a sine sampled below Nyquist, the reconstructed signal against the original, the rate as the knob.
- A filter designed live: the order and cutoff as knobs, the response and the filtered signal.
- The STFT's trade-off: window length against time and frequency resolution on a chirp.
- LMS converging on an unknown filter, the weights against the truth.
- NMF separating two instruments, the components and their activations.

## Rendering

- Lines, spectrograms as heatmaps (`image`), pole-zero plots, filter responses; the player as an audio-aligned cursor.
- Missing: audio playback in sync with the cursor; a pole-zero view with the unit circle.

## Shared

- `shared/`: the signal slot (named signals and clips with their truths: frequencies, the clean signal), the spectrogram record.

## Problems

- `signal/` and `audio/` move here. Audio data on the tape is large: keep derived features, not samples, and fetch clips lazily (`datasets` remote assets).
