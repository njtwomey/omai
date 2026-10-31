import { goldenSuite } from '../../testing/suite.ts'
import { asTensor, type Tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('ANOVA against scipy and statsmodels', 'stats/design', {
  oneWay: (groups) => {
    const r = S.oneWayAnova(groups.filter((g): g is Tensor => g !== undefined))
    return asTensor([r.statistic, r.pValue, r.etaSquared])
  },
  twoWay: ([y, a, b], p) => {
    const r = S.twoWayAnova(y, a, b ?? a, { ssType: p['ssType'] as 'I' | 'II' | 'III' })
    const terms = [r.factorA, r.factorB, r.interaction]
    return asTensor([
      ...terms.map((t) => t?.ss ?? Number.NaN),
      ...terms.map((t) => t?.statistic ?? Number.NaN),
      ...terms.map((t) => t?.pValue ?? Number.NaN),
      r.residual.ss,
    ])
  },
})
