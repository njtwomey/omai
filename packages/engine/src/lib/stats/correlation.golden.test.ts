import { goldenSuite, type Params } from '../../testing/suite.ts'
import type { Alternative, TestResult } from '../base/index.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const alt = (p: Params): Alternative => (p['alternative'] as Alternative | undefined) ?? 'two-sided'
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('correlations against scipy.stats, numpy and pingouin', 'stats/correlation', {
  pearson: ([x, y], p) => {
    const r = S.pearson(x, y, { alternative: alt(p), level: 0.9 })
    return tensor([r.statistic, r.pValue, r.estimate?.interval?.[0] ?? 0, r.estimate?.interval?.[1] ?? 0])
  },
  spearman: ([x, y], p) => sp(S.spearman(x, y, { alternative: alt(p) })),
  kendallTau: ([x, y], p) => sp(S.kendallTau(x, y, { alternative: alt(p), variant: p['variant'] as 'b' })),
  partialCorrelation: ([x, y, z], p) =>
    sp(S.partialCorrelation(x, y, z as Tensor, { alternative: alt(p), method: p['method'] as 'pearson' })),
  partialCorrelationCi: ([x, y, z]) => tensor([...(S.partialCorrelation(x, y, z as Tensor).estimate?.interval ?? [])]),
  pointBiserial: ([b, x]) => sp(S.pointBiserial(b, x)),
  cov: ([x], p) => S.cov(x, undefined, { ddof: Number(p['ddof'] ?? 1) }),
  covColumns: ([x]) => S.cov(x, undefined, { rowvar: false }),
  covXY: ([x, y]) => S.cov(x, y),
  covWeights: ([x, fw, aw]) => S.cov(x, undefined, { fweights: fw, aweights: aw as Tensor }),
  covAweights: ([x, aw]) => S.cov(x, undefined, { aweights: aw }),
  corrcoef: ([x]) => S.corrcoef(x),
  distanceCorrelation: ([x, y]) => S.distanceCorrelation(x, y).value,
})
