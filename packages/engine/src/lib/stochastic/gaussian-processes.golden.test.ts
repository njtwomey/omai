import { goldenSuite, type Params } from '../../testing/suite.ts'
import { exp, mul, square, sub, toFlat, transpose, type Value } from '../tensor/index.ts'
import {
  bridgeCovariance,
  brownianCovariance,
  type CovarianceFunction,
  conditionGaussianProcess,
  covarianceMatrix,
  karhunenLoeve,
  ouCovariance,
} from './gaussian-processes.ts'

/** sklearn's ConstantKernel(variance) * RBF(length) on one-dimensional inputs. */
const rbf =
  (variance: number, length: number): CovarianceFunction =>
  (x: Value, y: Value) =>
    mul(exp(mul(square(sub(x, transpose(y))), -0.5 / length ** 2)), variance)

const posterior = (inputs: readonly Value[], p: Params) =>
  conditionGaussianProcess(
    rbf(p['variance'] as number, p['length'] as number),
    inputs[0] as Value,
    inputs[1] as Value,
    inputs[2] as Value,
    {
      noise: p['noise'] as number,
    },
  )

goldenSuite('stochastic/gaussian-processes against scikit-learn and numpy', 'stochastic/gaussian-processes', {
  'gp-mean': (inputs, p) => posterior(inputs as readonly Value[], p).mean,
  'gp-cov': (inputs, p) => posterior(inputs as readonly Value[], p).covariance,
  'gp-std': (inputs, p) => posterior(inputs as readonly Value[], p).std,
  'gp-lml': (inputs, p) => posterior(inputs as readonly Value[], p).logMarginalLikelihood,
  'cov-brownian': ([t], p) => covarianceMatrix(brownianCovariance(p['sigma'] as number), t),
  'cov-bridge': ([t], p) => covarianceMatrix(bridgeCovariance(p['horizon'] as number, p['sigma'] as number), t),
  'cov-ou': ([t], p) => covarianceMatrix(ouCovariance({ theta: p['theta'] as number, sigma: p['sigma'] as number }), t),
  'kl-values': ([grid], p) => karhunenLoeve(brownianCovariance(), toFlat(grid), p['terms'] as number).eigenvalues,
  'kl-functions': ([grid], p) => karhunenLoeve(brownianCovariance(), toFlat(grid), p['terms'] as number).eigenfunctions,
})
