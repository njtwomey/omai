import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { truncatedSvd } from './truncated-svd.ts'

learnGolden('TruncatedSVD against scikit-learn, signs fixed', 'decomposition/truncated_svd', {
  svd: ([x, held], p) => {
    const f = truncatedSvd
      .with({ nComponents: num(p, 'nComponents'), solver: str(p, 'solver') as 'full' | 'randomized', nIter: 7 })
      .fit({ x: x as Tensor })
    const scores = f.transform(held as Tensor)
    return {
      components: f.params.components,
      singularValues: f.params.singularValues,
      explainedVariance: f.params.explainedVariance,
      explainedVarianceRatio: f.params.explainedVarianceRatio,
      transform: scores,
      inverse: f.inverseTransform(scores),
    }
  },
})
