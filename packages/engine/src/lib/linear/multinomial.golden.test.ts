import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { multinomial } from './multinomial.ts'

learnGolden('multinomial logit against statsmodels MNLogit and scikit-learn', 'linear/multinomial', {
  multinomial: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = multinomial({ alpha: num(p, 'alpha'), tol: 1e-12 }).fit({ x, y })
    const d = f.params.coef[0]?.length ?? 0
    const head = Array.from({ length: 8 }, (_, i) => Array.from(toFlat(x).slice(i * d, (i + 1) * d)))
    // Standard errors in the layout of the reference: class by class over the free classes, slopes then intercept.
    const m = f.params.covariance.length
    const shift = num(p, 'alpha') > 0 ? f.params.intercept.reduce((a, b) => a + b, 0) / f.params.intercept.length : 0
    return {
      coef: f.params.coef,
      intercept: f.params.intercept.map((v) => v - shift),
      logLikelihood: f.params.logLikelihood,
      bse: Array.from({ length: m }, (_, j) => Math.sqrt(f.params.covariance[j]?.[j] ?? Number.NaN)),
      proba: f.predictProba(head),
    }
  },
})
