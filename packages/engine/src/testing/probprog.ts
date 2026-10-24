// The probabilistic programs the probprog golden tests share with the oracle (oracle/cases/probprog/_models.py
// defines the same programs under the same names, with the same data). Node only; not published.
import {
  bernoulli,
  beta,
  categorical,
  compose,
  dirichlet,
  gamma,
  halfCauchy,
  logNormal,
  multivariateNormal,
  normal,
  poisson,
  uniform,
} from '../lib/dists/index.ts'
import {
  deterministic,
  factor,
  type Model,
  markov,
  mask,
  observe,
  plate,
  sample,
  scale,
} from '../lib/probprog/index.ts'
import { add, matmul, mul, square, take, tensor, type Value, zeros } from '../lib/tensor/index.ts'

export const Y4: readonly number[] = [0.3, -1.2, 2.1, 0.8]
const SIGMA = [15, 10, 16, 11]
const SCHOOLS = [28, 8, -3, 7]
const X = [
  [0.5, -1],
  [1.2, 0.3],
  [-0.7, 0.8],
  [0.1, -0.4],
  [1.5, 1.1],
]
const YB = [1, 0, 0, 1, 1]
const COUNTS = [3, 5, 4, 6]
const CATS = [0, 2, 2, 1, 2]
const FLIPS = [1, 0, 1, 1]
const Y3 = [0.5, 1.5, -0.5]
const MASK3 = [true, false, true]
const Y10 = [0.2, -0.4, 1.1, 0.7, -1.3, 0.5, 2.0, -0.1, 0.9, 0.3]
const YN = [
  [0.1, -0.2],
  [0.4, 0.9],
  [-1.0, 0.3],
]
const SIMPLEX_DATA = [3, 0, 2, 3, 1, 3]
export const MIX_W: readonly number[] = [0.3, 0.7]
export const MIX_Y: readonly number[] = [-1.1, 0.4, 2.3]
export const HMM_T: readonly (readonly number[])[] = [
  [0.8, 0.2],
  [0.3, 0.7],
]
export const HMM_Y: readonly number[] = [0.2, 1.7, 2.2, -0.3]

export const MODELS: Readonly<Record<string, Model>> = {
  normalMean: () => {
    const mu = sample('mu', normal(0, 2))
    plate('data', 4, () => observe('y', normal(mu, 1.5), Y4))
  },
  schools: () => {
    const mu = sample('mu', normal(0, 5))
    const tau = sample('tau', halfCauchy(5))
    plate('J', 4, () => {
      const theta = sample('theta', normal(mu, tau))
      observe('y', normal(theta, SIGMA), SCHOOLS)
    })
  },
  logistic: () => {
    const alpha = sample('alpha', normal(0, 1))
    const b = plate('features', 2, () => sample('beta', normal(0, 1)))
    const logits = add(alpha, matmul(tensor(X), b))
    plate('data', 5, () => observe('y', bernoulli({ logits }), YB))
  },
  gammaPoisson: () => {
    const rate = sample('rate', gamma(2, 0.5))
    deterministic('double', mul(rate, 2))
    plate('data', 4, () => observe('k', poisson(rate), COUNTS))
  },
  dirichletCategorical: () => {
    const p = sample('p', dirichlet([1, 2, 3]))
    plate('data', 5, () => observe('z', categorical({ probs: p }), CATS))
  },
  betaFactor: () => {
    const p = sample('p', beta(2, 3))
    factor('penalty', mul(square(p), -1))
    plate('data', 4, () => observe('y', bernoulli({ probs: p }), FLIPS))
  },
  scaleMask: () => {
    const mu = sample('mu', normal(0, 1))
    scale(
      mask(() => plate('data', 3, () => observe('y', normal(mu, 1), Y3)), MASK3),
      2,
    )(undefined)
  },
  subsampled: () => {
    const mu = sample('mu', normal(0, 1))
    plate('data', 10, (idx) => observe('y', normal(mu, 1), take(tensor(Y10), idx)), { subsampleSize: 3 })
  },
  nested: () => {
    plate('a', 2, () =>
      plate('b', 3, () => {
        const x = sample('x', normal(0, 1))
        observe('y', normal(x, 0.5), YN)
      }),
    )
  },
  bounded: () => {
    const u = sample('u', uniform(-1, 4))
    const s = sample('s', logNormal(0, 1))
    observe('y', normal(u, s), 1.3)
  },
  simplex: () => {
    const w = sample('w', dirichlet([2, 1, 1.5, 3]))
    plate('data', 6, () => observe('z', categorical({ probs: w }), SIMPLEX_DATA))
  },
  mvn: () => {
    const x = sample(
      'x',
      multivariateNormal([0, 1], {
        covariance: [
          [1, 0.3],
          [0.3, 2],
        ],
      }),
    )
    observe('y', normal(add(take(x, 0), mul(take(x, 1), 2)), 1), 0.7)
  },
}

export const DISCRETE: Readonly<Record<string, Model>> = {
  mixture: () => {
    const locs = sample('locs', compose.independent(normal(zeros([2]), 3), 1))
    plate('data', 3, () => {
      const z = sample('z', categorical({ probs: MIX_W }))
      observe('y', normal(take(locs, z), 1), MIX_Y)
    })
  },
  hmm: () => {
    markov(
      'time',
      HMM_Y.length,
      (t, prev: Value | undefined) => {
        const probs = prev === undefined ? tensor([0.5, 0.5]) : take(tensor(HMM_T), prev)
        const z = sample(`z${t}`, categorical({ probs }))
        observe(`y${t}`, normal(take(tensor([0, 2]), z), 1), HMM_Y[t] ?? 0)
        return z
      },
      undefined as Value | undefined,
    )
  },
}
