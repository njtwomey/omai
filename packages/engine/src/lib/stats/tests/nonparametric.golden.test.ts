import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative, TestResult } from '../../base/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => p['alternative'] as Alternative
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('rank and sign tests against scipy.stats and statsmodels', 'stats/tests/nonparametric', {
  mannWhitneyU: ([x, y], p) =>
    sp(T.mannWhitneyU(x, y, { alternative: alt(p), useContinuity: p['useContinuity'] !== false })),
  wilcoxon: ([d], p) =>
    sp(
      T.wilcoxon(d, undefined, {
        alternative: alt(p),
        ...(p['zeroMethod'] === undefined ? {} : { zeroMethod: p['zeroMethod'] as T.ZeroMethod }),
        ...(p['method'] === undefined ? {} : { method: p['method'] as T.RankTestMethod }),
        correction: p['correction'] === true,
      }),
    ),
  brunnerMunzel: ([x, y], p) =>
    sp(T.brunnerMunzel(x, y, { alternative: alt(p), distribution: (p['distribution'] as 'normal') ?? 't' })),
  kruskal: ([a, b, c]) => sp(T.kruskal([a, b, c as Tensor])),
  friedman: ([x]) => sp(T.friedman(x)),
  nemenyiFriedmanQ: ([x]) => T.nemenyiFriedman(x).statistic,
  nemenyiFriedmanP: ([x]) => T.nemenyiFriedman(x).pValue,
  nemenyiQ: ([a, b, c]) => T.nemenyi([a, b, c as Tensor]).statistic,
  nemenyiP: ([a, b, c]) => T.nemenyi([a, b, c as Tensor]).pValue,
  signTest: ([d], p) => sp(T.signTest(d, Number(p['mu0']))),
})
