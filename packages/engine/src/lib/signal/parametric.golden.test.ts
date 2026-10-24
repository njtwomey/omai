import { goldenSuite } from '../../testing/suite.ts'
import { concat, reshape } from '../tensor/index.ts'
import * as S from './index.ts'

const flat = (m: S.ArModel) => concat([m.ar, reshape(m.sigma2, [1])])

goldenSuite('autoregressive estimation against statsmodels', 'signal/parametric', {
  yuleWalker: ([x], o) => flat(S.yuleWalker(x, o['order'] as number, { method: o['method'] as 'mle' })),
  burg: ([x], o) => flat(S.burg(x, o['order'] as number)),
  levinsonDurbin: ([r]) => {
    const m = S.levinsonDurbin(r)
    return concat([m.ar, reshape(m.sigma2, [1]), m.reflection])
  },
  arSpectrum: ([phi], o) =>
    S.arSpectrum(
      { ar: phi, sigma2: o['sigma2'] as number },
      { n: o['n'] as number, onesided: o['onesided'] as boolean },
    ).values,
})
