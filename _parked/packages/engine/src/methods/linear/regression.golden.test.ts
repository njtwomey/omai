import { kFold } from '../../lib/learn/model-selection/index.ts'
import { type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { learnGolden, num } from '../../testing/learn.ts'
import { olsSummary } from './inference.ts'
import { cvPath, lassoPath } from './paths.ts'
import { elasticNet, orthogonalMatchingPursuit, ridge, ridgeCV } from './regression.ts'

type In = readonly [Tensor, Tensor, Tensor?]
const tight = { tol: 1e-12, maxIter: 100000 }

// statsmodels has the constant last here (it was appended to exog), as olsSummary does.
learnGolden('linear regression against scikit-learn and statsmodels', 'methods/linear/regression', {
  lassoPath: (inputs) => {
    const [x, y, alphas] = inputs as In
    return { coefs: lassoPath({ x, y }, { alphas: toFlat(alphas as Tensor), ...tight }).coefs }
  },
  ridge: (inputs, p) => {
    const [x, y, w] = inputs as In
    const f = ridge({ alpha: num(p, 'alpha') }).fit({ x, y, ...(w === undefined ? {} : { weights: w }) })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  elasticNet: (inputs, p) => {
    const [x, y] = inputs as In
    const f = elasticNet({ alpha: num(p, 'alpha'), l1Ratio: num(p, 'l1Ratio'), ...tight }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  lassoCV: (inputs) => {
    const [x, y, alphas] = inputs as In
    const cv = cvPath({ x, y }, { alphas: toFlat(alphas as Tensor), cv: kFold(5), ...tight })
    const f = elasticNet({ alpha: cv.bestAlpha, l1Ratio: 1, ...tight }).fit({ x, y })
    return { mse: cv.mse, alpha: cv.bestAlpha, coef: f.params.coef }
  },
  ridgeCV: (inputs) => {
    const [x, y] = inputs as In
    const f = ridgeCV({ alphas: [0.01, 0.1, 1, 10, 100] }).fit({ x, y })
    return { alpha: f.alpha, coef: f.params.coef }
  },
  omp: (inputs, p) => {
    const [x, y] = inputs as In
    const f = orthogonalMatchingPursuit({ nNonzeroCoefs: num(p, 'k') }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  olsSummary: (inputs) => {
    const [x, y] = inputs as In
    const s = olsSummary({ x, y })
    const hc3 = olsSummary({ x, y }, { covType: 'HC3' })
    return {
      params: s.table.estimates,
      bse: s.table.standardErrors ?? [],
      pvalues: s.table.pValues ?? [],
      hc3: hc3.table.standardErrors ?? [],
      summary: [s.rSquared, s.adjRSquared, s.fStatistic, s.fPValue, s.logLikelihood, s.aic, s.bic],
      cooks: s.cooksDistance,
      leverage: s.leverage,
      vif: s.vif,
    }
  },
})
