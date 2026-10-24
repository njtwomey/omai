import { goldenSuite, type Params } from '../../../testing/suite.ts'
import { stack, tensor } from '../../tensor/index.ts'
import { stft } from '../time-frequency.ts'
import * as A from './index.ts'

const n = (o: Params, k: string): number | undefined => (typeof o[k] === 'number' ? (o[k] as number) : undefined)
const defined = <T extends Record<string, unknown>>(o: T): T =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as T

const spec = (o: Params): A.SpectrogramOptions =>
  defined({
    nFft: n(o, 'nFft'),
    hopLength: n(o, 'hopLength'),
    winLength: n(o, 'winLength'),
    power: n(o, 'power'),
    center: o['center'] === false ? false : undefined,
    padMode: o['padMode'] as A.AudioPadMode | undefined,
  }) as A.SpectrogramOptions

goldenSuite('perceptual scales against librosa and their formulas', 'signal/audio/scales', {
  hzToMel: ([f], o) => A.hzToMel(f, o['htk'] === true),
  melToHz: ([m], o) => A.melToHz(m, o['htk'] === true),
  melFrequencies: (_x, o) => A.melFrequencies(n(o, 'n'), n(o, 'fmin'), n(o, 'fmax'), o['htk'] === true),
  fftFrequencies: (_x, o) => A.fftFrequencies(n(o, 'sr'), n(o, 'nfft')),
  hzToBark: ([f], o) => A.hzToBark(f, o['formula'] as A.BarkFormula),
  erbBandwidth: ([f]) => A.erbBandwidth(f),
  hzToErb: ([f]) => A.hzToErb(f),
})

goldenSuite('auditory filter banks against librosa and scipy', 'signal/audio/filterbanks', {
  melFilterBank: (_x, o) =>
    A.melFilterBank(o['sr'] as number, o['nfft'] as number, {
      nMels: o['n_mels'] as number,
      ...(o['fmin'] === undefined ? {} : { fmin: o['fmin'] as number }),
      ...(o['fmax'] === undefined ? {} : { fmax: o['fmax'] as number }),
      htk: o['htk'] === true,
      ...('norm' in o ? { norm: o['norm'] as A.FilterNorm } : {}),
    }),
  chromaFilterBank: (_x, o) =>
    A.chromaFilterBank(o['sr'] as number, o['nfft'] as number, {
      ...(o['n_chroma'] === undefined ? {} : { nChroma: o['n_chroma'] as number }),
      ...(o['tuning'] === undefined ? {} : { tuning: o['tuning'] as number }),
      ...('octwidth' in o ? { octwidth: o['octwidth'] as number | null } : {}),
      ...(o['norm'] === undefined ? {} : { norm: o['norm'] as number }),
      ...(o['base_c'] === undefined ? {} : { baseC: o['base_c'] as boolean }),
    }),
  gammatone: (_x, o) => {
    const g = A.gammatone(o['freq'] as number, o['ftype'] as 'fir' | 'iir', {
      fs: o['fs'] as number,
      ...(o['order'] === undefined ? {} : { order: o['order'] as number }),
      ...(o['numtaps'] === undefined ? {} : { numtaps: o['numtaps'] as number }),
    })
    return tensor([...g.b, ...g.a])
  },
  gammatoneBank: ([x], o) => A.gammatoneBank(x, o['centres'] as number[], o['fs'] as number),
})

goldenSuite('spectral audio features against librosa', 'signal/audio/features', {
  spectrogram: ([y], o) => A.spectrogram(y, spec(o)),
  melSpectrogram: ([y], o) => A.melSpectrogram(y, o['sr'] as number, { ...spec(o), nMels: o['nMels'] as number }),
  powerToDb: ([s], o) =>
    A.powerToDb(s, {
      ...(o['ref'] === 'max' ? { ref: 'max' as const } : {}),
      ...('topDb' in o ? { topDb: o['topDb'] as number | null } : {}),
    }),
  amplitudeToDb: ([s], o) => A.amplitudeToDb(s, o['ref'] === 'max' ? { ref: 'max' } : {}),
  mfcc: ([y], o) =>
    A.mfcc(y, o['sr'] as number, {
      ...spec(o),
      nMels: o['nMels'] as number,
      nMfcc: o['nMfcc'] as number,
      lifter: o['lifter'] as number,
    }),
  spectralCentroid: ([s], o) => A.spectralCentroid(s, o['sr'] as number),
  spectralBandwidth: ([s], o) => A.spectralBandwidth(s, o['sr'] as number, o['p'] as number),
  spectralRolloff: ([s], o) => A.spectralRolloff(s, o['sr'] as number, o['rollPercent'] as number),
  spectralFlatness: ([s]) => A.spectralFlatness(s),
  chromaStft: ([y], o) => A.chromaStft(y, o['sr'] as number, spec(o)),
  rms: ([y], o) =>
    A.rms(
      y,
      defined({
        frameLength: n(o, 'frameLength'),
        hopLength: n(o, 'hopLength'),
        center: o['center'] === false ? false : undefined,
      }) as A.FrameOptions,
    ),
  rmsFromSpectrogram: ([s], o) => A.rmsFromSpectrogram(s, o['frameLength'] as number),
  zeroCrossingRate: ([y], o) =>
    A.zeroCrossingRate(y, { frameLength: o['frameLength'] as number, hopLength: o['hopLength'] as number }),
  frame: ([y], o) => A.frame(y, o['frameLength'] as number, o['hop'] as number),
})

goldenSuite('augmentation against librosa', 'signal/audio/augmentation', {
  phaseVocoder: ([y], o) => {
    const nFft = o['nFft'] as number
    const hop = o['hopLength'] as number
    const z = stft(y, { nperseg: nFft, noverlap: nFft - hop, padded: false, scaling: 'none' }).values
    const r = A.phaseVocoder(z, o['rate'] as number)
    return stack([r.re, r.im])
  },
  timeStretch: ([y], o) =>
    tensor(
      Array.from(
        A.timeStretch(y, o['rate'] as number, { nFft: o['nFft'] as number, hopLength: o['hopLength'] as number }),
      ),
    ),
  pitchShift: ([y], o) =>
    tensor(
      Array.from(
        A.pitchShift(y, o['sr'] as number, o['nSteps'] as number, {
          nFft: o['nFft'] as number,
          hopLength: o['hopLength'] as number,
        }),
      ),
    ),
  mixAtSnr: ([y, n], o) => tensor(Array.from(A.mixAtSnr(y, n as never, o['snrDb'] as number))),
})

goldenSuite('rhythm against librosa', 'signal/audio/rhythm', {
  onsetStrength: ([y], o) =>
    tensor(
      Array.from(
        A.onsetStrength(y, {
          ...(o['aggregate'] === undefined ? {} : { aggregate: o['aggregate'] as 'mean' | 'median' }),
          ...(o['lag'] === undefined ? {} : { lag: o['lag'] as number }),
        }),
      ),
    ),
  peakPick: ([x], o) => tensor(A.peakPick(x?.data ?? [], o as unknown as A.PeakPickOptions)),
  onsetDetect: ([e], o) =>
    tensor(A.onsetDetectFromEnvelope(e?.data ?? [], o['sr'] as number, o['hopLength'] as number)),
  tempogram: ([e], o) => tensor(A.tempogram(e?.data ?? [], o['winLength'] as number).map((r) => Array.from(r))),
  tempo: ([e]) => tensor([A.tempo(e?.data ?? [])]),
  beatTrack: ([e], o) => {
    const r = A.beatTrackFromEnvelope(e?.data ?? [], {
      ...(o['tightness'] === undefined ? {} : { tightness: o['tightness'] as number }),
      ...(o['trim'] === undefined ? {} : { trim: o['trim'] as boolean }),
      ...(o['bpm'] === undefined ? {} : { bpm: o['bpm'] as number }),
    })
    return tensor([r.bpm, ...r.beats])
  },
})

goldenSuite('pitch estimation against librosa', 'signal/audio/pitch', {
  autocorrelate: ([x], o) => tensor(Array.from(A.autocorrelate(x, o['maxSize'] as number))),
  yin: ([y], o) => tensor(Array.from(A.yin(y, o as unknown as A.YinOptions & { troughThreshold: number }))),
  pyin: ([y], o) => {
    const r = A.pyin(y, o as unknown as A.PyinOptions)
    return tensor([...Array.from(r.f0), ...Array.from(r.voiced), ...Array.from(r.voicedProb)])
  },
})
