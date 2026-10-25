import { type CaseFn, goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { analyseThresholds } from './contingency.ts'

type Sweep = ReturnType<typeof analyseThresholds>
const sweep = ([y, s, w]: readonly (Tensor | undefined)[]): Sweep =>
  analyseThresholds(y as Tensor, s as Tensor, w === undefined ? {} : { sampleWeight: w })
const column =
  (pick: (r: Sweep) => readonly number[]): CaseFn =>
  (inputs) =>
    tensor([...pick(sweep(inputs))])

goldenSuite('threshold sweep against sklearn', 'metrics/contingency', {
  thresholds: column((r) => r.thresholds),
  tp: column((r) => r.tp),
  fp: column((r) => r.fp),
  fpr: column((r) => r.fpr),
  tpr: column((r) => r.tpr),
  rocAuc: (inputs) => sweep(inputs).rocAuc,
  averagePrecision: (inputs) => sweep(inputs).averagePrecision,
})
