import { learnGolden, num, str } from '../../testing/learn.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { type MetricInit, ncaValueAndGradient, neighbourhoodComponentsAnalysis } from './metric-learning.ts'
import { encodeLabels, matrixView } from './validation.ts'

learnGolden('neighbourhood components analysis against sklearn', 'learn/metric_learning', {
  objective: (inputs) => {
    const [x, y, a] = inputs as [Tensor, Tensor, Tensor]
    const { codes } = encodeLabels(y, 'golden')
    const r = ncaValueAndGradient(toFlat(a), a.shape[0] ?? 0, matrixView(x, 'golden'), codes)
    return {
      value: r.value,
      gradient: fromStorage(r.gradient, [...a.shape], 'float64'),
      probabilities: Array.from(r.probabilities),
    }
  },
  fit: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = neighbourhoodComponentsAnalysis({
      nComponents: num(p, 'components'),
      init: str(p, 'init') as MetricInit,
      maxIter: 500,
      gradTol: 1e-10,
    }).fit({ x, y })
    return { value: f.params.objective }
  },
})
