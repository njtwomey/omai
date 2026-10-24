import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as M from './index.ts'

const mean = (x: Float64Array): number => x.reduce((a, b) => a + b, 0) / x.length

/** The stand-in networks of the oracle (oracle/cases/metrics/audio.py). */
const model: M.DnsmosModel = {
  p835: (s) => {
    const rms = Math.sqrt(mean(s.map((v) => v * v)))
    const peak = s.reduce((a, v) => Math.max(a, Math.abs(v)), 0)
    return [2 + rms, 3 - 0.5 * peak, 1.5 + rms * peak]
  },
  p808: (s) => 2.5 + mean(s.map(Math.abs)) + s.length / 1e6,
}

goldenSuite('audio metrics against torchmetrics, pystoi, scipy and dtw-python', 'metrics/audio', {
  signalNoiseRatio: ([t, p], o) => M.signalNoiseRatio(t, p, o).estimate.value,
  scaleInvariantSdr: ([t, p], o) => M.scaleInvariantSdr(t, p, o).estimate.value,
  scaleInvariantSnr: ([t, p]) => M.scaleInvariantSnr(t, p).estimate.value,
  signalDistortionRatio: ([t, p], o) => M.signalDistortionRatio(t, p, o).estimate.value,
  pitValue: ([t, p]) => M.permutationInvariant(t, p, (a, b) => M.scaleInvariantSdr(a, b).estimate.value).value,
  pitPermutation: ([t, p]) =>
    tensor([...M.permutationInvariant(t, p, (a, b) => M.scaleInvariantSdr(a, b).estimate.value).permutation]),
  stoi: ([x, y], o) =>
    tensor(
      [false, true].map(
        (extended) => M.shortTimeObjectiveIntelligibility(x, y, { fs: o['fs'] as number, extended }).estimate.value,
      ),
    ),
  segmentalSnr: ([x, y], o) => M.segmentalSnr(x, y, { fs: o['fs'] as number }).estimate.value,
  logSpectralDistance: ([x, y], o) => M.logSpectralDistance(x, y, o).estimate.value,
  melCepstralDistortion: ([a, b], o) => M.melCepstralDistortion(a, b, o).estimate.value,
  dnsmos: ([a], o) => {
    const b = M.dnsmos(a, model, o).breakdown ?? {}
    return tensor([b['p808'] ?? 0, b['sig'] ?? 0, b['bak'] ?? 0, b['ovrl'] ?? 0])
  },
})
