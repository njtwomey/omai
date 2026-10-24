import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const tri = (e: S.EffectSize): Tensor => tensor([e.value, e.interval?.[0] ?? Number.NaN, e.interval?.[1] ?? Number.NaN])

goldenSuite('effect sizes against pingouin, statsmodels and scipy', 'stats/effect_size', {
  cohen: ([x, y], p) => tri(S.cohensD(x, y, { paired: p['paired'] as boolean, level: 0.9 })),
  hedges: ([x, y], p) => tri(S.hedgesG(x, y, { paired: p['paired'] as boolean, level: 0.9 })),
  cohenOne: ([x], p) => tri(S.cohensD(x, Number(p['mu']), { level: 0.9 })),
  hedgesOne: ([x], p) => tri(S.hedgesG(x, Number(p['mu']), { level: 0.9 })),
  glassDelta: ([x, y]) => tri(S.glassDelta(x, y, { level: 0.9 })),
  cohensH: (_, p) =>
    tri(S.cohensH(Number(p['p1']), Number(p['p2']), { n1: Number(p['n1']), n2: Number(p['n2']), level: 0.9 })),
  cramersV: ([t]) => tri(S.cramersV(t, { level: 0.9 })),
  cramersVCorrected: ([t]) => S.cramersV(t, { correction: true }).value,
  etaSquared: ([a, b, c]) => tri(S.etaSquared([a, b, c as Tensor], { level: 0.9 })),
  omegaSquared: ([a, b, c]) => tri(S.omegaSquared([a, b, c as Tensor], { level: 0.9 })),
  cliffsDelta: ([x, y]) => tri(S.cliffsDelta(x, y, { level: 0.9 })),
})
