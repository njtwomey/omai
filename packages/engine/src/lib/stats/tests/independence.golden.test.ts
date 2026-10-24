import { goldenSuite, type Params } from '../../../testing/suite.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const z = (p: Params): number[] => toFlat(p['z'] as Tensor)

goldenSuite('independence tests against pingouin, sklearn and causal-learn', 'stats/tests/independence', {
  distanceCovariance: ([x, y]) => T.distanceCovarianceTest(x, y, { nResamples: 9 }).statistic,
  hoeffding: ([x, y]) => T.hoeffdingD(x, y, { nResamples: 9 }).statistic,
  miDiscrete: ([x, y]) => T.mutualInformation(x, y, { discrete: true }),
  miBinned: ([x, y], p) => T.mutualInformation(x, y, { bins: Number(p['bins']) }),
  fisherz: ([d], p) => T.partialCorrelationTest(d, Number(p['x']), Number(p['y']), z(p)).pValue,
  gTestConditional: ([d], p) =>
    T.gTestConditional(d, Number(p['x']), Number(p['y']), z(p), { statistic: p['statistic'] as 'g' }).pValue,
  kci: ([d], p) => T.kernelCITest(d, Number(p['x']), Number(p['y']), z(p)).pValue,
})
