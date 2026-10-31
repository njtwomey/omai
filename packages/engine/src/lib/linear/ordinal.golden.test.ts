import { learnGolden, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type OrdinalLink, ordinal } from './ordinal.ts'

learnGolden('cumulative-link ordinal regression against statsmodels OrderedModel', 'linear/ordinal', {
  cumulative: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = ordinal({ link: str(p, 'link') as OrdinalLink, tol: 1e-11, maxIter: 200 }).fit({ x, y })
    const d = f.params.coef.length
    const head = toFlat(x).slice(0, 10 * d)
    return {
      coef: f.params.coef,
      thresholds: f.params.thresholds,
      logLikelihood: f.params.logLikelihood,
      bse: f.params.coef.map((_, j) => Math.sqrt(f.params.covariance[j]?.[j] ?? Number.NaN)),
      proba: f.predictProba(Array.from({ length: 10 }, (_, i) => Array.from(head.slice(i * d, (i + 1) * d)))),
    }
  },
})
