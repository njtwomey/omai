import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { reweighingWeights } from './preprocessing.ts'

const ints = (t: Tensor | undefined): number[] => Array.from(toFlat(t as Tensor), (v) => Math.round(v))

goldenSuite('reweighing against AIF360', 'fairness/preprocessing', {
  reweighing: ([y, s]) => tensor(reweighingWeights(ints(y), ints(s)).weights),
})
