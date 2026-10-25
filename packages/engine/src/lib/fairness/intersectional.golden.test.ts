import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { entropyDecomposition, richSubgroupAudit } from './intersectional.ts'

const rows = (x: Tensor): number[][] => {
  const [n = 0, d = 0] = x.shape
  const v = toFlat(x)
  return Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => v[i * d + j] ?? 0))
}
const decompose = ([y, p, s]: readonly (Tensor | undefined)[], o: Readonly<Record<string, unknown>>) =>
  entropyDecomposition(y as Tensor, p as Tensor, s as Tensor, { alpha: o['alpha'] as number })
const audit = ([x, y, p]: readonly (Tensor | undefined)[], o: Readonly<Record<string, unknown>>) =>
  richSubgroupAudit(rows(x as Tensor), y as Tensor, toFlat(p as Tensor), {
    rate: o['rate'] as 'falsePositive' | 'falseNegative',
  })

goldenSuite('entropy indices and rich-subgroup audits against AIF360', 'fairness/intersectional', {
  entropyTotal: (inputs, o) => decompose(inputs, o).total,
  entropyBetween: (inputs, o) => decompose(inputs, o).between,
  auditMembers: (inputs, o) => tensor([...audit(inputs, o).members]),
  auditDisparity: (inputs, o) => audit(inputs, o).weightedDisparity,
})
