import { learnGolden, num, rows, str } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { glsar, glsFit, heteroscedasticRegressor } from './gls.ts'

const bseOf = (cov: number[][]): number[] => cov.map((r, j) => Math.sqrt(r[j] ?? Number.NaN))

learnGolden('generalised least squares against statsmodels GLS, GLSAR and reference loops', 'linear/gls', {
  gls: (inputs) => {
    const [x, y, s] = inputs as [Tensor, Tensor, Tensor]
    const sigma = s.shape.length === 1 ? Array.from(toFlat(s)) : rows(s)
    const f = glsFit({ x, y }, { sigma })
    return {
      coef: f.coef,
      intercept: f.intercept,
      scale: f.scale,
      logLikelihood: f.logLikelihood,
      bse: bseOf(f.covariance),
    }
  },
  glsar: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = glsar({ order: num(p, 'order'), maxIter: num(p, 'maxIter'), rtol: 1e-4 }).fit({ x, y })
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      rho: f.params.rho,
      scale: f.params.scale,
      bse: bseOf(f.params.covariance),
    }
  },
  heteroscedastic: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const method = str(p, 'method') as 'fgls' | 'ml'
    // FGLS runs a fixed number of rounds (the iteration is not guaranteed to converge); ML runs to its tolerance.
    const maxIter = method === 'fgls' ? num(p, 'rounds') : 2000
    const f = heteroscedasticRegressor({ method, tol: method === 'fgls' ? 0 : 1e-12, maxIter }).fit({ x, y })
    const out: Record<string, number | number[]> = {
      coef: f.params.coef,
      intercept: f.params.intercept,
      gamma: f.params.gamma,
    }
    if (method === 'ml') out['logLikelihood'] = f.params.logLikelihood
    return out
  },
})
