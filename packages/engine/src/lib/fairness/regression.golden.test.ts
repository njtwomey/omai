import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { boundedGroupLoss, distributionParity, equalOpportunityInDistribution } from './regression.ts'

const v = (t: Tensor | undefined): number[] => Array.from(toFlat(t as Tensor))

goldenSuite('regression fairness against scipy and fairlearn', 'fairness/regression', {
  ks: ([s, g]) => distributionParity(v(s), g as Tensor).value,
  wasserstein: ([s, g]) => distributionParity(v(s), g as Tensor, { distance: 'wasserstein' }).value,
  eoKs: ([s, y, g]) => equalOpportunityInDistribution(v(s), v(y), g as Tensor).value,
  groupLoss: ([y, p, g]) => tensor([...boundedGroupLoss(v(y), v(p), g as Tensor).losses]),
})
