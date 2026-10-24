import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative, TestResult } from '../../base/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => p['alternative'] as Alternative
const row = (r: TestResult): Tensor =>
  tensor([
    r.statistic,
    r.pValue,
    Number(r.df),
    r.estimate?.interval?.[0] ?? Number.NaN,
    r.estimate?.interval?.[1] ?? Number.NaN,
  ])
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('tests of means against scipy.stats, statsmodels and pingouin', 'stats/tests/means', {
  ttest1samp: ([x], p) => row(T.ttest1samp(x, Number(p['popmean']), { alternative: alt(p), level: 0.9 })),
  ttestInd: ([x, y], p) =>
    row(T.ttestInd(x, y, { equalVar: p['equalVar'] as boolean, alternative: alt(p), level: 0.9 })),
  ttestRel: ([x, y], p) => row(T.ttestRel(x, y, { alternative: alt(p), level: 0.9 })),
  ztest1: ([x], p) => sp(T.ztest(x, undefined, { value: Number(p['value']), alternative: alt(p) })),
  ztest2: ([x, y], p) =>
    sp(T.ztest(x, y, { value: Number(p['value']), alternative: alt(p), usevar: p['usevar'] as 'pooled' })),
  fOneway: ([a, b, c]) => sp(T.fOneway([a, b, c as Tensor])),
  welchAnova: ([a, b, c]) => {
    const r = T.welchAnova([a, b, c as Tensor])
    return tensor([r.statistic, r.pValue, (r.df as readonly number[])[1] ?? 0, r.effectSize?.value ?? 0])
  },
  tukeyHsd: ([a, b, c], p) => {
    const r = T.tukeyHsd([a, b, c as Tensor])
    const part = p['part'] as string
    return part === 'statistic' ? r.statistic : part === 'pvalue' ? r.pValue : part === 'low' ? r.low : r.high
  },
  rmAnova: ([wide], p) => {
    const r = T.rmAnova(wide, { effsize: p['effsize'] as 'ng2' })
    const a = r.assumptions ?? {}
    return tensor([
      r.statistic,
      r.pValue,
      r.effectSize?.value ?? 0,
      Number(a['epsilon']),
      Number(a['pGreenhouseGeisser']),
      Number(a['mauchlyW']),
      Number(a['mauchlyP']),
    ])
  },
})
