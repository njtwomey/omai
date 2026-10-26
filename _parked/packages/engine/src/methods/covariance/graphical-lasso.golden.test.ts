import { tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { graphicalLasso } from './graphical-lasso.ts'

const fit = (s: Parameters<typeof graphicalLasso>[0], alpha: unknown) =>
  graphicalLasso(s, alpha as number, { tol: 1e-10, enetTol: 1e-12, maxIter: 500 })

goldenSuite('the graphical lasso against scikit-learn', 'methods/covariance/graphical_lasso', {
  precision: ([s], o) => tensor(fit(s, o['alpha']).precision),
  covariance: ([s], o) => tensor(fit(s, o['alpha']).covariance),
})
