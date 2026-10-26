import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type CovarianceType, gaussianEmAlgorithm, mixturePosterior } from './gaussian.ts'

const fit = (inputs: readonly (Tensor | undefined)[], kind: CovarianceType) => {
  const [x, w, mu, cov] = inputs as Tensor[]
  const start = {
    weights: toArray(w as Tensor) as number[],
    means: toArray(mu as Tensor) as number[][],
    covariances: toArray(cov as Tensor) as number[][][],
  }
  const alg = gaussianEmAlgorithm(x as Tensor, start, { covarianceType: kind, tol: 1e-12 })
  return { x: x as Tensor, params: runAlgorithm(alg, { maxSteps: 15 }).state }
}

/** Covariances in sklearn's per-type shape. */
const shaped = (covs: number[][][], kind: CovarianceType): Tensor => {
  if (kind === 'full') return tensor(covs)
  if (kind === 'tied') return tensor(covs[0] as number[][])
  const diag = covs.map((c) => c.map((r, a) => r[a] ?? 0))
  return tensor(kind === 'diag' ? diag : diag.map((d) => d[0] ?? 0))
}

goldenSuite('Gaussian mixtures by EM against scikit-learn', 'methods/mixture/gaussian', {
  weights: (i, o) => tensor(fit(i, o['kind'] as CovarianceType).params.weights),
  means: (i, o) => tensor(fit(i, o['kind'] as CovarianceType).params.means),
  covariances: (i, o) => shaped(fit(i, o['kind'] as CovarianceType).params.covariances, o['kind'] as CovarianceType),
  score: (i, o) => {
    const { x, params } = fit(i, o['kind'] as CovarianceType)
    const { logNorm } = mixturePosterior(x, params)
    return logNorm.reduce((a, v) => a + v, 0) / logNorm.length
  },
})
