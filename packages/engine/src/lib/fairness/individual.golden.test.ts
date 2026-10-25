import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { consistency } from './individual.ts'

goldenSuite('consistency against AIF360', 'fairness/individual', {
  consistency: ([x, y], o) => {
    const t = x as Tensor
    const [n = 0, d = 0] = t.shape
    const flat = toFlat(t)
    const rows = Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))
    return consistency(rows, Array.from(toFlat(y as Tensor)), { k: o['k'] as number })
  },
})
