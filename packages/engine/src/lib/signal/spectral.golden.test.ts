import { goldenSuite, type Params } from '../../testing/suite.ts'
import { type ComplexValue, stack, type Tensor, tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const pair = (z: ComplexValue): Value => stack([z.re, z.im])

const welchOptions = (o: Params): S.WelchOptions => o as S.WelchOptions

goldenSuite('spectral estimation against scipy.signal', 'signal/spectral', {
  detrend: ([x], o) => S.detrend(x, o as S.DetrendOptions),
  periodogram: ([x], o) => S.periodogram(x, o as S.PeriodogramOptions).values,
  periodogramFreqs: ([x], o) => S.periodogram(x, o as S.PeriodogramOptions).frequencies,
  welch: ([x], o) => S.welch(x, welchOptions(o)).values,
  csd: ([x, y], o) => pair(S.csd(x, y, welchOptions(o)).values),
  coherence: ([x, y], o) => S.coherence(x, y, welchOptions(o)).values,
  confidence: ([p], o) => {
    const c = S.spectralConfidenceInterval(p, o['dof'] as number, o['level'] as number)
    return stack([c.lower, c.upper])
  },
})

goldenSuite('Lomb-Scargle against scipy.signal and astropy', 'signal/lomb_scargle', {
  lombScargle: ([t, y, f], o) => S.lombScargle(t, y, f as Tensor, o as S.LombScargleOptions),
  lombScargleWeighted: ([t, y, f, w], o) =>
    S.lombScargle(t, y, f as Tensor, { ...(o as S.LombScargleOptions), weights: w as Tensor }),
  lombScargleAmplitude: ([t, y, f], o) => pair(S.lombScargleAmplitude(t, y, f as Tensor, o as S.LombScargleOptions)),
  falseAlarmProbability: ([t, z], o) =>
    tensor(
      S.falseAlarmProbability(Array.from(z.data), {
        t,
        fmax: o['fmax'] as number,
        method: o['method'] as S.FalseAlarmMethod,
      }),
    ),
  falseAlarmLevel: ([t, p], o) =>
    tensor(
      S.falseAlarmLevel(Array.from(p.data), {
        t,
        fmax: o['fmax'] as number,
        method: o['method'] as S.FalseAlarmMethod,
      }),
    ),
})
