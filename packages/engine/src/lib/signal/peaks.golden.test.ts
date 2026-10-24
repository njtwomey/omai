import { goldenSuite, type Params } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const camel = (k: string): string => k.replace(/_([a-z])/g, (_m, c: string) => c.toUpperCase())

const options = (o: Params): S.FindPeaksOptions => {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(o)) if (k !== 'prop') out[k] = v
  return out as S.FindPeaksOptions
}

goldenSuite('peaks, zero crossings and DTW against scipy, librosa and dtw-python', 'signal/peaks', {
  findPeaks: ([x], o) => tensor(S.findPeaks(x, options(o)).peaks),
  findPeaksProp: ([x], o) => tensor(S.findPeaks(x, options(o)).properties[camel(o['prop'] as string)] ?? []),
  peakProminences: ([x, p], o) => {
    const r = S.peakProminences(x, Array.from(p.data), o['wlen'] as number)
    return tensor([r.prominences, r.leftBases, r.rightBases])
  },
  peakWidths: ([x, p], o) => {
    const r = S.peakWidths(x, Array.from(p.data), { relHeight: o['relHeight'] as number })
    return tensor([r.widths, r.widthHeights, r.leftIps, r.rightIps])
  },
  argrelmax: ([x], o) => tensor(S.argrelextrema(x, 'max', o as never)),
  argrelmin: ([x], o) => tensor(S.argrelextrema(x, 'min', o as never)),
  zeroCrossings: ([y], o) => tensor(S.zeroCrossings(y, o as S.ZeroCrossingOptions).map((b) => (b ? 1 : 0))),
  dtw: ([a, b], o) => {
    const r = S.dtw(a, b, o as S.DtwOptions)
    return tensor([r.distance, r.normalizedDistance, ...r.index1, ...r.index2])
  },
})
