import { goldenSuite, type Params } from '../../testing/suite.ts'
import type { Alternative } from '../base/index.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const alt = (p: Params): Alternative => (p['alternative'] as Alternative | undefined) ?? 'two-sided'
const sp = (r: S.ResamplingTestResult): Tensor => tensor([r.statistic, r.pValue])

// The one-way F statistic, computed directly (the full ANOVA also checks assumptions and builds intervals).
const fStatistic = (...groups: Float64Array[]): number => {
  const all = groups.flatMap((g) => Array.from(g))
  const grand = S.mean(all)
  let ssb = 0
  let ssw = 0
  for (const g of groups) {
    const m = S.mean(g)
    ssb += g.length * (m - grand) ** 2
    for (const v of g) ssw += (v - m) ** 2
  }
  return ssb / (groups.length - 1) / (ssw / (all.length - groups.length))
}

goldenSuite('exact permutation tests against scipy.stats.permutation_test', 'stats/resampling', {
  independent: ([a, b], p) => sp(S.permutationTest([a, b], S.meanDifference, { alternative: alt(p) })),
  samplesOne: ([d], p) =>
    sp(S.permutationTest([d], (x) => S.mean(x), { permutationType: 'samples', alternative: alt(p) })),
  samplesTwo: ([x, y], p) =>
    sp(S.permutationTest([x, y], S.meanDifference, { permutationType: 'samples', alternative: alt(p) })),
  pairings: ([x, y], p) =>
    sp(
      S.permutationTest([x, y], (a, b) => S.pearson(a, b).statistic, {
        permutationType: 'pairings',
        alternative: alt(p),
      }),
    ),
  independentThree: ([a, b, c]) => sp(S.permutationTest([a, b, c as Tensor], fStatistic, { alternative: 'greater' })),
  jackknife: ([x]) => {
    const r = S.jackknife(x, (v) => S.variance(v, { ddof: 1 }))
    return tensor([r.estimate.value, r.estimate.standardError ?? Number.NaN, r.bias])
  },
})
