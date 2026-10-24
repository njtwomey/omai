import { goldenSuite, type Params } from '../../testing/suite.ts'
import { concat, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const n = (o: Params, k: string): number => o[k] as number

goldenSuite('estimation, detection, adaptive filters and cepstra', 'signal/estimation', {
  wiener: ([x], o) => S.wiener(x, (o['size'] as number | number[] | undefined) ?? 3, o['noise'] as number | undefined),
  lpc: ([x], o) => S.lpc(x, n(o, 'order')),
  rocKnownSignal: ([p], o) => tensor(S.rocKnownSignal(Array.from(p.data), n(o, 'd2'))),
  rocEnergyDetector: ([p], o) => tensor(S.rocEnergyDetector(Array.from(p.data), n(o, 'n'), n(o, 'snr'))),
  energyThreshold: (_i, o) =>
    S.energyDetector(new Array(n(o, 'n')).fill(0), { noiseVariance: n(o, 'noiseVariance'), pfa: n(o, 'pfa') })
      .threshold,
  adaptive: ([x, d], o) => {
    const r = S.adaptiveFilter(o['method'] as S.AdaptiveMethod, x, d, { mu: n(o, 'mu'), eps: n(o, 'eps') })
    return concat([r.y, r.e, r.w])
  },
  complexCepstrum: ([x], o) => {
    const c = S.complexCepstrum(x, n(o, 'n'))
    return concat([c.cepstrum, tensor([c.delay])])
  },
  realCepstrum: ([x]) => S.realCepstrum(x),
})
