# Signal processing (part 2, signals, systems and information)

Signals sampled, filtered, transformed and analysed, with audio as its own section: sampling and quantisation, convolution and LTI systems in the z domain, Fourier analysis and windows, filter design and application, multirate processing and filter banks, time-frequency and wavelet transforms, sparse representations and dictionary learning, spectral estimation, adaptive filters and linear prediction, detection, and the audio chain (features, pitch, rhythm, enhancement, separation, perceptual coding), as sessions over `signal`, `systems`, `numerics/fourier`, `numerics/convolution`, `optim/sparse` and `metrics/audio`. The series methods (autocorrelation, decomposition, change points, matrix profiles) are the time-series chapter's and share this root.

## Structure

Everything here is an operation on a signal watched, so it lives under `process/signals/` (section 3 of `methods/README.md`), one folder per group, with `audio/` a section rather than a chapter and `series/` the folder the time-series chapter fills. Where a library routine runs whole (a filter designed, a spectrum estimated), the session's step is the knob swept (the order, the rate, the bits); where it iterates (an adaptive filter, a sparse solver, Parks-McClellan's exchanges) the step is the iteration.

```
process/signals/
  sampling/         sampling and aliasing, reconstruction, quantisation
  systems/          convolution, LTI systems in the z domain (poles and zeros, responses)
  fourier/          the DFT and FFT, windows and leakage, the analytic signal
  filters/          FIR and IIR design, filtering sample by sample, zero-phase and smoothing filters, Wiener filters
  multirate/        decimation and interpolation, polyphase, filter banks and wavelet packets
  time-frequency/   STFT, reassignment and synchrosqueezing, Wigner-Ville, constant-Q, wavelets and denoising
  sparse/           sparse recovery, compressed sensing, dictionary learning
  spectral/         non-parametric and parametric spectral estimation, uneven sampling, coherence
  adaptive/         LMS, NLMS and RLS; linear prediction
  detection/        detectors and their ROC, matched filters, peaks, the cepstrum
  audio/            features, pitch, rhythm, enhancement, separation, perceptual coding, augmentation, keyword spotting
  series/           the time-series chapter's: foundations, decomposition, change points, matrix profiles
  shared/           the signal slot, the spectrogram and response records, the sample-per-step base, audio playback (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Signals on the tape are kept in full where they are short (a few thousand samples); audio keeps its features and fetches its samples (Problems).

### sampling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sampling-and-aliasing` | A sine or a chirp sampled at a rate that is the knob: below Nyquist the alias at \|f − k·fs\|, the reconstruction by sinc interpolation against the original, the zero-order hold's droop; the error against the rate as a sweep. | `signal/samples` t!, `signal/reconstructed` t!, `signal/aliasFrequency` s, `signal/error` s, `spectrum/magnitude` p | `signal/sampling.ts`: `alias`, `sincInterpolate`, `zeroOrderHold`, `zeroOrderHoldResponse`; `signal/signal.ts`: `createSignal`, `sampleTimes` | |
| `quantisation` | Bits as the step (1 to 16): the staircase, the error signal, SQNR against the 6 dB-per-bit rule, dither turning distortion into noise, μ-law's companding on a quiet passage. | `quantiser/bits` s, `quantiser/sqnr` s, `quantiser/idealSqnr` s, `signal/quantised` t!, `signal/error` t!, `quantiser/histogram` p | `signal/sampling.ts`: `quantise`, `dither`, `sqnr`, `idealSqnr`, `muLawEncode`, `muLawDecode` | |

### systems/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `convolution` | One output sample per step: the flipped kernel sliding over the input, the products summed, the output growing; direct against FFT and overlap-add on the same pair with the cost counted; correlation as the unflipped twin with the lag of the best match. | `conv/n` s, `conv/output` t!, `conv/window` p (the overlap at the step), `conv/cost` s, `corr/lag` s | `numerics/convolution/one-d.ts`: `convolve`, `correlate`, `correlationLags`, `fftConvolve`, `oaConvolve` (whole; the sample loop is the method's) | |
| `lti-systems` | A transfer function in z (the pairs table as the slot): poles and zeros on the plane, the impulse and step responses one sample per step, the frequency response read off the pole-zero geometry, stability by the poles inside the unit circle; the difference equation and the inverse z-transform side by side. | `response/impulse` t!, `response/step` t!, `system/poles` t, `system/zeros` t, `system/stable` s, `system/frequencyResponse` p, `system/planeView` p | `systems/representations.ts`: `tf`, `zpk`, `toZpk`; `systems/stability.ts`: `poles`, `zeros`, `isStable`, `jury`; `systems/responses.ts`: `lsimAlgorithm`, `impulse`, `step`; `systems/filters.ts`: `freqz`; `systems/pairs.ts`: `zTransformPairs`; `systems/transforms.ts`: `inverseZ`, `tfToDifference` | |
| `see control-theory` | Feedback, margins, root loci and state space are the control chapter's; here a system is a filter. | | | |

### fourier/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dft` | The DFT matrix built one row per step (the twiddle factors as phasors), a signal's coefficient as its projection on that row; the FFT's butterflies one stage per step with the operation count against n²; resolution against length. | `dft/row` s, `dft/matrix` t!, `dft/coefficients` t!, `fft/stage` s, `fft/operations` s, `fft/butterflies` p | `numerics/fourier/helpers.ts`: `dftMatrix`, `dft`, `fftfreq`; `numerics/fourier/fft.ts`: `fft`, `rfft` (whole; TODO a radix-2 `fftAlgorithm` exposing the stages) | |
| `windows-and-leakage` | A sinusoid between bins: leakage through the rectangular window's sidelobes; the window as a mode (Hann, Hamming, Blackman, Kaiser with β as the knob, flat top, DPSS) with its main-lobe width against sidelobe level; zero padding interpolating the spectrum without resolving; two close tones resolved or not. | `window/shape` t!, `window/metrics` t (ENBW, coherent gain), `spectrum/magnitude` t!, `spectrum/leakage` s, `spectrum/resolved` s | `signal/windows/get-window.ts`: `getWindow`; `signal/windows/metrics.ts`: `windowMetrics`, `enbw`; `signal/windows/dpss.ts`: `dpss`; `signal/fourier.ts`: `dtft`, `zeroPad`, `frequencyResolution`, `leakageFraction`, `peakFrequency`, `parseval` | |
| `analytic-signal` | The Hilbert transform's quadrature, the envelope and instantaneous frequency of an AM or FM signal against the planted modulation; the failure on a broadband signal. | `signal/envelope` t!, `signal/instantaneousFrequency` t!, `signal/error` s, `signal/analytic` p | `signal/fourier.ts`: `hilbert`, `hilbertTransform`, `envelope`; `signal/time-frequency.ts`: `instantaneousPhase`, `instantaneousFrequency` | |
| `see mathematics` | Fourier series, the Gibbs overshoot and epicycles are `simulate/numerics/number-theory/` (series-and-fourier-series). | | | |

### filters/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `fir-design` | The method as a mode (the window method with the window as a knob, `firwin2`'s arbitrary response, Parks-McClellan's equiripple with its exchanges as steps, Kaiser's order formula): the response against the specification, the order as the knob, the ripple equioscillating; linear phase and its delay; minimum phase as a transform. | `design/order` s, `design/taps` t!, `design/response` t!, `design/ripple` s, `design/passed` s, `remez/exchange` e, `design/specification` p | `signal/filters/fir.ts`: `firwin`, `firwin2`, `remez`, `kaiserord`, `kaiserBeta`, `minimumPhase` (remez whole; TODO per-exchange records, as `mathematics` asks of `best-approximation`); `signal/filters/response.ts`: `frequencyResponse`, `groupDelay`, `checkSpecification` | |
| `iir-design` | The family as a mode (Butterworth, Chebyshev I and II, elliptic, Bessel; notch, peak and comb): the analogue prototype's poles, the frequency transformation and the bilinear warp as steps, the order from the specification; the responses compared at equal order (the elliptic's steepness against its ripple and phase), poles and zeros on the plane, the group delay. | `design/order` s, `design/poles` t, `design/zeros` t, `design/response` t!, `design/groupDelay` t!, `design/stage` m (prototype, transformed, discretised), `design/planeView` p | `signal/filters/iir.ts`: `iirfilter`, `butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirnotch`, `iirpeak`, `iircomb`; `signal/filters/order.ts`: `buttord`, `cheb1ord`, `cheb2ord`, `ellipord`, `iirdesign`; `signal/filters/prototypes.ts`: `buttap`, `cheb1ap`, `ellipap`, `lp2lpZpk`, `lp2bpZpk`, `bilinearZpk`; `signal/filters/response.ts`: `sosfreqz`, `freqzZpk`, `groupDelay` | |
| `filtering` | A designed filter applied one sample per step with its state carried (the transient, then the steady state), the form as a mode (direct, second-order sections for numerical safety at high order, zero-phase `filtfilt` with its padding), the delay visible against the clean signal; the median and Savitzky-Golay filters as the nonlinear and polynomial cousins on an impulsive signal. | `filter/output` t!, `filter/state` t, `filter/n` s, `filter/error` s, `filter/delay` s, `signal/picture` p | `signal/filters/apply.ts`: `lfilter`, `lfilterZi`, `lfiltic`, `sosfilt`, `sosfiltZi`, `filtfilt`, `sosfiltfilt`, `medfilt`, `savgolFilter`, `movingAverage`, `exponentialMovingAverage` (the sample loop runs `lfilter` with its carried state) | |
| `wiener` | The optimal linear filter for a signal in noise with known spectra: the non-causal frequency-domain gain, the causal FIR Wiener from the Wiener-Hopf equations (its length as the knob), the error against the clean signal; the adaptive filter converging to the same taps as the comparison. | `wiener/taps` t!, `wiener/gain` t!, `wiener/error` s, `signal/filtered` t!, `signal/picture` p | `signal/estimation.ts`: `nonCausalWiener`, `firWiener`, `wiener` | |

### multirate/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `resampling` | Decimation (the anti-aliasing filter, then every Mth sample) and interpolation (zeros, then the image-rejecting filter) one block per step with the spectra at each stage; polyphase doing the same at a fraction of the cost (operations counted); rational resampling with `upfirdn`. | `stage/spectrum` t!, `stage/name` m, `resample/output` t!, `resample/error` s, `resample/operations` s | `signal/multirate.ts`: `decimate`, `interpolate`, `resamplePoly`, `resample`, `upfirdn`, `downsample`, `upsample`, `polyphase`, `polyphaseRecombine`, `expandFilter` | |
| `filter-banks` | Analysis and synthesis banks: the DFT bank's channels, a two-channel QMF pair with perfect reconstruction checked (aliasing cancelled, the error against the input), the wavelet bank as a tree, wavelet packets choosing the best basis by entropy one split per step. | `bank/channels` t!, `bank/reconstruction` t!, `bank/error` s, `bank/aliasing` s, `packet/tree` p, `packet/entropy` s | `signal/filter-banks.ts`: `dftAnalysis`, `dftSynthesis`, `qmf`, `checkReconstruction`, `powerComplementarity`, `waveletBank`, `waveletPacket`, `packetPaths`, `packetReconstruct` | |

### time-frequency/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `stft` | The spectrogram of a chirp with the window length and hop as knobs: time against frequency resolution (the uncertainty product), the frame drawn on the signal at the viewed step, the inverse reconstructing the signal under the overlap-add constraint; a two-component signal resolved in one dimension or the other. | `frame/index` s, `frame/spectrum` t!, `stft/spectrogram` t!, `stft/timeBandwidth` s, `stft/reconstructionError` s, `frame/picture` p | `signal/time-frequency.ts`: `stft`, `istft`, `spectrogram`, `timeSpread`, `frequencySpread`, `timeBandwidthProduct` | |
| `reassignment` | The reassigned spectrogram and synchrosqueezing sharpening the ridges of the plain STFT (the three side by side), the inverse of synchrosqueezing recovering a mode; the Wigner-Ville distribution's cross terms and the smoothing that kills them (the pseudo and smoothed pseudo forms as modes). | `tf/plain` t!, `tf/reassigned` t!, `tf/synchrosqueezed` t!, `tf/wigner` t!, `tf/ridge` p, `tf/concentration` s | `signal/reassignment.ts`: `reassignedSpectrogram`, `synchrosqueeze`, `inverseSynchrosqueeze`; `signal/wigner.ts`: `wignerVille`, `pseudoWignerVille`, `smoothedPseudoWignerVille` | |
| `constant-q` | Log-spaced bins with constant Q: the kernels' lengths falling with frequency, a musical signal's harmonics equally spaced in the CQT where the STFT's are not; bins per octave as the knob. | `cqt/magnitude` t!, `cqt/frequencies` t, `cqt/kernels` p | `signal/constant-q.ts`: `cqtFrequencies`, `cqtFilters`, `cqt` | |
| `wavelets` | The discrete wavelet transform level by level (approximation and details), the family as a mode (Haar, Daubechies, Symlets, Coiflets, biorthogonal), reconstruction exact; the stationary transform's redundancy; the continuous transform's scalogram of a chirp against the STFT; denoising by thresholding the details (soft or hard, the universal and SURE thresholds as modes) with PSNR against the clean signal. | `dwt/level` s, `dwt/coefficients` t!, `dwt/reconstructionError` s, `cwt/scalogram` t!, `denoise/threshold` s, `denoise/psnr` s, `denoise/kept` s | `signal/wavelets/dwt.ts`: `dwt`, `idwt`, `wavedec`, `waverec`, `wavedec2`; `signal/wavelets/swt.ts`: `swt`; `signal/wavelets/cwt.ts`: `cwt`, `centralFrequency`; `signal/wavelets/families.ts`: `wavelet`, `waveletList`, `dwtMaxLevel`; `signal/wavelets/denoise.ts`: `waveletDenoise`, `threshold`, `estimateNoiseSigma` | |
| `see time-series` | EMD, VMD and SSA are the series chapter's decompositions (`process/signals/series/decomposition/`). | | | |

### sparse/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sparse-recovery` | A sparse signal in a random dictionary: the solver as a mode (matching pursuit, orthogonal matching pursuit, CoSaMP, iterative hard thresholding, basis pursuit by linear programming, the lasso by coordinate descent) one atom or iteration per step; the support recovered against the truth, the residual falling, the dictionary's coherence as what decides. | `solver/residual` s, `solver/support` t!, `solver/coefficients` t!, `solver/supportError` s, `solver/atom` m, `dictionary/coherence` s | `optim/sparse.ts`: `matchingPursuitAlgorithm`, `orthogonalMatchingPursuitAlgorithm`, `cosampAlgorithm`, `iterativeHardThresholdingAlgorithm`, `basisPursuit`, `basisPursuitDenoising`, `lassoCoordinateDescentAlgorithm`; `optim/dictionary.ts`: `mutualCoherence`; `datasets/matrices.ts`: `makeSparseCodedSignal` | |
| `compressed-sensing` | The phase transition: sparsity k against measurements m, the recovery probability over a grid (one cell per step, many trials), the empirical boundary against the coherence- and RIP-based bounds. | `grid/k` s, `grid/m` s, `grid/recovered` s, `study/phaseTransition` proj, `bound/coherence` s, `bound/rip` s | the solvers of `optim/sparse.ts`; `optim/dictionary.ts`: `coherenceSparsityBound`, `ripEstimate`, `babelFunction` | |
| `dictionary-learning` | K-SVD against the method of optimal directions and online dictionary learning as modes: the sparse coding step then the dictionary update, one alternation per step, the atoms drawn as a grid (image patches learn edges), the representation error falling; a noisy patch set reconstructed as denoising. | `solver/objective` s, `dictionary/atoms` t!, `dictionary/change` s, `codes/sparsity` s, `dictionary/picture` p | `optim/dictionary.ts`: `ksvdAlgorithm`, `modAlgorithm`, `onlineDictionaryLearningAlgorithm`, `sparseEncode` | |

### spectral/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `spectral-estimation` | The estimator as a mode (periodogram, Bartlett, Welch with its segment length and overlap as knobs, Blackman-Tukey, multitaper with its bandwidth) on a process with a known spectrum: variance against resolution, the confidence interval from the degrees of freedom, the periodogram's variance refusing to fall as n grows (the step adds data). | `estimate/spectrum` t!, `estimate/dof` s, `estimate/interval` t, `estimate/error` s, `truth/spectrum` p | `signal/spectral.ts`: `periodogram`, `bartlettSpectrum`, `welch`, `blackmanTukey`, `multitaper`, `welchDof`, `spectralConfidenceInterval`, `detrend`; `datasets/timeseries.ts`: `armaProcess` (its spectral density) | |
| `parametric-spectrum` | AR spectra by Yule-Walker and Burg with the order as the step (peaks sharpening, then spurious ones), MUSIC and ESPRIT resolving two close sinusoids where the periodogram cannot (the subspace dimension as the knob), Prony's damped exponentials. | `fit/order` s, `estimate/spectrum` t!, `estimate/frequencies` t, `estimate/error` s, `subspace/eigenvalues` t | `signal/parametric.ts`: `yuleWalker`, `burg`, `arSpectrum`, `music`, `esprit`, `prony`; `datasets/timeseries.ts`: `sinusoids` | |
| `lomb-scargle` | Unevenly sampled sinusoids (a gap as the knob): the Lomb-Scargle periodogram against the naive one on interpolated samples, the false-alarm probability of the peak, the amplitude recovered. | `estimate/periodogram` t!, `estimate/peak` s, `estimate/falseAlarm` s, `estimate/amplitude` s | `signal/lomb-scargle.ts`: `lombScargle`, `lombScargleAmplitude`, `falseAlarmProbability`, `falseAlarmLevel`; `datasets/timeseries.ts`: `subsampleSeries` | |
| `coherence` | Two signals sharing a band: the cross-spectral density and the coherence as the shared band's width or the noise is the knob; the phase of the cross spectrum as the delay. | `estimate/csd` t!, `estimate/coherence` t!, `estimate/phase` t!, `estimate/delay` s | `signal/spectral.ts`: `csd`, `coherence` | |

### adaptive/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `adaptive-filter` | LMS, NLMS and RLS as modes identifying an unknown filter that changes at a planted step: one sample per step, the taps converging to the truth, the error on the tape, the step size μ (or the forgetting factor λ) as the knob trading speed for misadjustment; adaptive noise cancellation with a reference as the second setup. | `filter/taps` t!, `filter/error` s, `filter/misadjustment` s, `filter/tapError` s, `truth/changed` e | `signal/adaptive.ts`: `adaptiveFilterAlgorithm` (lms, nlms, rls ready), `adaptiveNoiseCancel` | |
| `linear-prediction` | The order as the step: Levinson-Durbin's reflection coefficients, the prediction-error variance falling, the LPC spectrum enveloping the periodogram of a vowel, the residual whitening. | `fit/order` s, `fit/reflection` s, `fit/predictionError` s, `fit/coefficients` t!, `fit/spectrum` p, `signal/residual` t! | `signal/estimation.ts`: `lpc`, `linearPredict`, `predictionResidual`; `signal/parametric.ts`: `levinsonDurbin`, `arSpectrum` | |

### detection/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `detectors` | A pulse in noise: the energy detector against the matched filter (the SNR gain of knowing the pulse), CFAR's threshold following the noise floor; the ROC traced as the threshold sweeps with the operating point at the viewed step; the SNR as the knob. | `detector/statistic` t!, `detector/threshold` t!, `detector/alarms` e, `detector/detected` s, `study/roc` proj | `signal/detection.ts`: `energyDetector`, `matchedFilterDetector`, `cfar`, `rocKnownSignal`, `rocEnergyDetector`; `signal/estimation.ts`: `matchedFilter`, `matchedFilterSnr` | |
| `peaks` | Peaks found with height, prominence, distance and width as knobs on a noisy multi-peak signal: which peaks survive each criterion, the prominences' bases drawn; zero crossings and local extrema as the simpler tools. | `peaks/indices` t!, `peaks/prominences` t, `peaks/widths` t, `peaks/count` s, `peaks/picture` p | `signal/peaks.ts`: `findPeaks`, `peakProminences`, `peakWidths`, `argrelextrema`, `zeroCrossings` | |
| `cepstrum` | The real cepstrum of a voiced sound: the pitch peak at the quefrency of the period, the vocal-tract envelope recovered by liftering (the cutoff as the knob); an echo as a cepstral spike; the complex cepstrum and its inverse. | `cepstrum/values` t!, `cepstrum/pitch` s, `cepstrum/envelope` t!, `cepstrum/liftered` p | `signal/cepstrum.ts`: `realCepstrum`, `complexCepstrum`, `inverseComplexCepstrum`, `lifter`, `cepstralPitch` | |

### audio/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `features` | The chain on a clip: frames, the mel spectrogram (the bank drawn), decibels, MFCCs by the DCT with liftering, chroma, the spectral shape features (centroid, bandwidth, roll-off, flatness, flux), RMS and zero-crossing rate as curves aligned with the audio; the gammatone bank as the auditory alternative. | `features/melSpectrogram` t!, `features/mfcc` t!, `features/chroma` t!, `features/centroid` t!, `features/rms` t!, `bank/filters` p | `signal/audio/features.ts`: `melSpectrogram`, `mfcc`, `chromaStft`, `spectralCentroid`, `spectralBandwidth`, `spectralRolloff`, `spectralFlatness`, `spectralFlux`, `rms`, `zeroCrossingRate`, `powerToDb`; `signal/audio/filterbanks.ts`: `melFilterBank`, `chromaFilterBank`, `gammatoneBank`; `signal/audio/scales.ts`: `hzToMel`, `melFrequencies` | |
| `pitch` | The estimator as a mode (autocorrelation, YIN with its threshold, probabilistic YIN decoding voicing by Viterbi, harmonic product spectrum, cepstral) one frame per step on a synthesised tone with vibrato and octave traps: the track against the planted f0, octave errors as events, the frame's YIN difference function as a probe. | `pitch/f0` t!, `pitch/voiced` t!, `pitch/error` s, `pitch/octaveError` e, `frame/difference` p | `signal/audio/pitch.ts`: `yin`, `pyin`, `autocorrelationPitch`, `harmonicProductSpectrum`, `autocorrelate`; `signal/cepstrum.ts`: `cepstralPitch` | |
| `rhythm` | The onset strength envelope from spectral flux, peaks picked as onsets, the tempogram's autocorrelation and the tempo under the log-normal prior, Ellis's dynamic-programming beat tracker (its tightness as the knob) placing beats against a synthesised click track; the F-measure of onsets and beats. | `onsets/strength` t!, `onsets/detected` e, `tempo/tempogram` t!, `tempo/bpm` s, `beats/times` t!, `eval/onsetF1` s, `eval/beatF1` s | `signal/audio/rhythm.ts`: `onsetStrength`, `peakPick`, `onsetDetect`, `tempogram`, `tempo`, `beatTrack`, `beatTrackFromEnvelope` | |
| `enhancement` | Speech in noise: spectral subtraction (its over-subtraction factor as the knob, musical noise as the artefact), decision-directed Wiener denoising, energy voice-activity detection with hangover; SNR and segmental SNR against the clean clip, the spectrograms before and after. | `enhance/spectrogram` t!, `enhance/gain` t!, `enhance/snr` s, `enhance/segmentalSnr` s, `vad/active` t!, `vad/accuracy` s | `signal/audio/enhancement.ts`: `spectralSubtraction`, `wienerDenoise`, `voiceActivity`; `metrics/audio.ts`: `signalNoiseRatio`, `segmentalSnr`, `logSpectralDistance` | |
| `separation` | Two sources mixed: the oracle masks (binary, ratio, phase-sensitive as modes) bounding what masking can do (SDR, SI-SDR), then NMF on the magnitude spectrogram one multiplicative update per step, the components and activations assigned to sources, the permutation-invariant score against the oracle. | `nmf/objective` s, `nmf/components` t!, `nmf/activations` t!, `separation/sdr` s, `separation/siSdr` s, `mask/picture` p | `signal/audio/separation.ts`: `idealBinaryMask`, `idealRatioMask`, `phaseSensitiveMask`, `applyMask`, `oracleSeparation`; `metrics/audio.ts`: `signalDistortionRatio`, `scaleInvariantSdr`, `permutationInvariant`; TODO `optim/nmf.ts`: `nmfAlgorithm` (multiplicative updates under the Frobenius and KL objectives) | |
| `perceptual-coding` | The absolute threshold of hearing, the Bark-domain masking threshold of a frame (the spreading function drawn), MDCT coefficients quantised below the threshold (bits as the knob), the entropy estimate against the bits spent, artefacts where the threshold is crossed. | `frame/threshold` t!, `frame/spectrum` t!, `coder/bits` s, `coder/entropy` s, `coder/error` s | `signal/audio/coding.ts`: `absoluteThreshold`, `spreadingFunction`, `maskingThreshold`, `mdctCode`; `numerics/fourier/trig.ts`: `mdct`, `imdct` | |
| `augmentation` | SpecAugment's masks, the phase vocoder's time stretch and pitch shift (the factor as the knob, artefacts at the extremes), mixing at an SNR, each drawn as the spectrogram before and after with the feature chain's output changing. | `augment/spectrogram` t!, `augment/factor` s, `augment/snr` s, `features/mfcc` t! | `signal/audio/augmentation.ts`: `specAugment`, `phaseVocoder`, `timeStretch`, `pitchShift`, `mixAtSnr`, `addNoise` | |
| `keyword-spotting` | MFCC sequences compared by DTW: a spoken digit against templates one candidate per step, the warping path on the cost matrix, the Sakoe-Chiba band as the knob; the simplest recogniser, with its confusion table. | `query/candidate` s, `query/distance` s, `query/path` p, `eval/confusion` p | `signal/audio/features.ts`: `mfcc`; `signal/similarity.ts`: `dtw`, `lbKeogh`; TODO a spoken-digit set in `datasets/` (7.11: the Free Spoken Digit Dataset) | |

### series/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `see time-series` | Autocorrelation and stationarity, decomposition, change points and the matrix-profile family are `process/signals/series/`, the time-series chapter's. | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `signals.ts` | The signal slot: synthesised signals with their truths (sinusoids with known frequencies and amplitudes, chirps, pulses in noise, AM and FM with their modulation, a sparse signal in a random dictionary, a known-spectrum process, vowels from an LPC model, two sources for separation, a click track), uneven sampling and noise as modifiers; real clips fetched lazily once 7.11's ingestion lands | every group |
| `records.ts` | The spectrogram and scalogram record (time against frequency as a heatmap aligned with the signal, the frame marked), the frequency-response record (magnitude, phase and group delay against a specification), the pole-zero record, the ROC of a detector sweep | `fourier/`, `filters/`, `time-frequency/`, `detection/`, `audio/` |
| `stream.ts` | The one-sample-per-step base (a sample arriving, the state carried), shared with the online detectors of `series/` and with `shift-and-drift` and `online-learning` through `methods/shared/streams.ts` | `filters/filtering`, `adaptive/`, `detection/` |
| `audio.ts` | Playback: the clip and the step cursor aligned, the sample rate and the frame hop as one knob set; what the page shows (features on the tape) and what it fetches (the samples) | `audio/` |

## Examples to make

- Aliasing: a sine sampled below Nyquist, the reconstruction against the original, the rate as the knob.
- A filter designed live: the family, order and cutoff as knobs, the response against the specification, the signal filtered sample by sample.
- The STFT's trade-off on a chirp: window length against time and frequency resolution; the reassigned spectrogram beside it.
- LMS converging on an unknown filter that changes halfway, against RLS.
- OMP recovering a sparse signal atom by atom; the phase transition of compressed sensing filling in.
- Beat tracking on a synthesised click track with the tempo as the knob.

## Rendering

- Lines, spectrograms as heatmaps (`image`), filter responses, pole-zero plots; the player as the audio-aligned cursor.
- Missing: audio playback in sync with the cursor; a pole-zero view with the unit circle; a response-against-specification panel; a signal-with-spectrogram panel (the signal above, the heatmap beneath, the frame marked).

## Problems

- `signal/` and `audio/` of the old plan move here; audio stays a section of this chapter, and `series/` is the time-series chapter's under the same root.
- Audio data: no clip is vendored, so every audio row runs on synthesised sound until 7.11's ingestion lands (spoken digits, speech commands). Samples are large: the tape keeps features, the page fetches the clip.
- Whole routines whose steps a page wants: the FFT's stages, `remez`'s exchanges; NMF is missing (`optim/nmf.ts`). Everything else named here exists.
- Most pages are a one-shot computation with a knob (a filter designed, a spectrum estimated); they run as one-step sessions, and the knob sweeps (order, rate, bits) are the step where the method has no iteration.
