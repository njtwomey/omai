import { goldenSuite, type Params } from '../../testing/suite.ts'
import * as S from './index.ts'

const n = (o: Params, k: string): number => o[k] as number

goldenSuite('multirate processing against scipy.signal', 'signal/multirate', {
  upfirdn: ([h, x], o) =>
    S.upfirdn(h, x, {
      up: n(o, 'up'),
      down: n(o, 'down'),
      ...(o['mode'] === undefined ? {} : { mode: o['mode'] as S.UpfirdnMode, cval: n(o, 'cval') }),
      ...(o['axis'] === undefined ? {} : { axis: n(o, 'axis') }),
    }),
  resamplePoly: ([x], o) =>
    S.resamplePoly(x, n(o, 'up'), n(o, 'down'), {
      ...(o['padtype'] === undefined ? {} : { padtype: o['padtype'] as S.ResamplePadType }),
      ...(o['window'] === undefined ? {} : { window: o['window'] as [string, number] }),
      ...(o['cval'] === undefined ? {} : { cval: n(o, 'cval') }),
    }),
  resamplePolyTaps: ([x, taps], o) => S.resamplePoly(x, n(o, 'up'), n(o, 'down'), { window: taps }),
  resample: ([x], o) => S.resample(x, n(o, 'num'), o['window'] === undefined ? {} : { window: o['window'] as string }),
  decimate: ([x], o) =>
    S.decimate(x, n(o, 'q'), { ftype: o['ftype'] as 'iir' | 'fir', zeroPhase: o['zeroPhase'] as boolean }),
})
