/**
 * A linear-Gaussian state-space model written as a spec with a `scan` node: the stationary AR(1) state with Gaussian
 * observations. The scan compiles to one `markov` loop, so the engines that follow a chain run it (the Kalman smoother of
 * `gaussian`, the particle filter, expectation propagation on the unrolled factor graph, and every sampler and
 * approximation on the compiled density), and the spec-level analyses see a Gaussian chain. The truth is dense Gaussian
 * conditioning of the joint of the states and the observations (independent of any filter in the library): the exact
 * posterior of every state and the evidence. `ar1MarkovProgram` is the same model written by hand as a `markov` program, the
 * form the scan replaced, kept as the reference the equivalence tests compare the compiled scan with.
 */

import { normal } from '../../dists/index.ts'
import { dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { normal as drawNormal, foldIn, type Key } from '../../random/index.ts'
import { mul, toFlat, type Value } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { type Model, markov, observe, sample } from '../primitives.ts'
import type { MarginalOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { exactPlan, laplacePlan, nutsPlan, particleFilterPlan, sviPlan } from './plans.ts'
import { mixtureOracle } from './truth.ts'

const AR1 = { phi: 0.8, sigmaX: 0.5, sigmaY: 0.6 }
/** The sd of the stationary distribution of the state, which the first state is drawn from. */
const S0 = AR1.sigmaX / Math.sqrt(1 - AR1.phi ** 2)

function generateSeries(key: Key, size = 12): BenchmarkData {
  const e = toFlat(drawNormal(foldIn(key, 0), [size]))
  const o = toFlat(drawNormal(foldIn(key, 1), [size]))
  const x: number[] = []
  for (let t = 0; t < size; t++)
    x.push(t === 0 ? S0 * (e[0] ?? 0) : AR1.phi * (x[t - 1] ?? 0) + AR1.sigmaX * (e[t] ?? 0))
  return { data: { y: x.map((v, t) => v + AR1.sigmaY * (o[t] ?? 0)) }, generating: { ...AR1, x } }
}

/**
 * The exact posterior of the AR(1) states by dense conditioning: the states are jointly normal with covariance
 * `S_ij = s0² phi^|i-j|`, the observations add `sigma_y²` on the diagonal, so the posterior mean is `S (S + R)^-1 y`, the
 * covariance `S - S (S + R)^-1 S` and the evidence `N(y; 0, S + R)`.
 *
 * @param y - The observations.
 * @returns The truth: a normal marginal for each state `x[0]`, `x[1]`, ... and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { ar1Truth } from 'omai-engine/probprog/models'
 * Object.keys(ar1Truth([0.1, 0.4, -0.2]).marginals) // => ['x[0]', 'x[1]', 'x[2]']
 * ```
 */
export function ar1Truth(y: readonly number[]): BenchmarkTruth {
  const T = y.length
  const S = new Float64Array(T * T)
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) S[i * T + j] = S0 * S0 * AR1.phi ** Math.abs(i - j)
  const C = Float64Array.from(S)
  for (let i = 0; i < T; i++) C[i * T + i] = (C[i * T + i] ?? 0) + AR1.sigmaY ** 2
  const { factor } = dpotrf(C, T)
  const alpha = dpotrs(factor, y, T)
  const mean = new Float64Array(T)
  for (let i = 0; i < T; i++)
    for (let j = 0; j < T; j++) mean[i] = (mean[i] ?? 0) + (S[i * T + j] ?? 0) * (alpha[j] ?? 0)
  let logDet = 0
  for (let i = 0; i < T; i++) logDet += 2 * Math.log(factor[i * T + i] ?? 1)
  const quad = y.reduce((a, v, i) => a + v * (alpha[i] ?? 0), 0)
  const marginals: Record<string, MarginalOracle> = {}
  for (let i = 0; i < T; i++) {
    // The posterior variance of x_i: S_ii - S_i (S + R)^-1 S_i'.
    const row = Float64Array.from({ length: T }, (_, j) => S[i * T + j] ?? 0)
    const solved = dpotrs(factor, row, T)
    const reduction = row.reduce((a, v, j) => a + v * (solved[j] ?? 0), 0)
    marginals[`x[${i}]`] = mixtureOracle(`x[${i}]`, [
      { weight: 1, mean: mean[i] ?? 0, sd: Math.sqrt((S[i * T + i] ?? 0) - reduction) },
    ])
  }
  return {
    kind: 'exact',
    source:
      'dense Gaussian conditioning of the joint of the states and the observations: posterior S (S + R)^-1 y, evidence N(y; 0, S + R)',
    logEvidence: -0.5 * (T * Math.log(2 * Math.PI) + logDet + quad),
    marginals,
  }
}

/**
 * The AR(1) state space written by hand as a `markov` program (sites `x0`, `y0`, ...): the form the spec's scan compiles
 * to, with the sites named without the brackets. It has the log joint of `ar1StateSpace` at the same values.
 *
 * @param y - The observations.
 * @returns The program.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { ar1MarkovProgram } from 'omai-engine/probprog/models'
 * import { logJoint } from 'omai-engine/probprog'
 * Number.isFinite(logJoint(ar1MarkovProgram([0.1, 0.4]), undefined, { x0: 0, x1: 0.2 }).logJoint.data[0]) // => true
 * ```
 */
export function ar1MarkovProgram(y: readonly number[]): Model {
  return () => {
    markov(
      'time',
      y.length,
      (t, prev: Value | undefined) => {
        const x = sample(`x${t}`, normal(prev === undefined ? 0 : mul(prev, AR1.phi), t === 0 ? S0 : AR1.sigmaX))
        observe(`y${t}`, normal(x, AR1.sigmaY), y[t] ?? 0)
        return x
      },
      undefined as Value | undefined,
    )
  }
}

/**
 * A stationary AR(1) state observed with noise: `x_0 ~ N(0, s0²)`, `x_t = 0.8 x_{t-1} + N(0, 0.5²)`, `y_t = x_t + N(0, 0.6²)`, twelve
 * steps, written as a `scan` over `time`. The posterior of the states is exactly Gaussian: the Kalman smoother, the Laplace
 * fit and the mean-field guide's family are all right, and the particle filter and NUTS approximate it; the smoothing
 * marginals come from dense conditioning. The sites are `x[t]` and `y[t]`.
 *
 * @concept posterior-validation
 * @index dataset Stationary AR(1) state with noisy observations as a scan; exact Gaussian smoothing truth.
 * @example
 * ```ts
 * import { ar1StateSpace } from 'omai-engine/probprog/models'
 * [ar1StateSpace.definition?.spec.nodes.map((n) => n.kind), ar1StateSpace.truth().kind] // => [['data', 'scan'], 'exact']
 * ```
 */
export const ar1StateSpace: BenchmarkModel = defineBenchmark({
  name: 'ar1-state-space',
  title: 'AR(1) state space',
  description:
    'A stationary AR(1) latent state observed with Gaussian noise, as a scan: the linear-Gaussian chain whose exact smoothing posterior is the truth.',
  tags: ['state-space', 'scan', 'markov', 'linear-gaussian', 'time-series', 'exact'],
  references: [
    'Kalman (1960), A new approach to linear filtering and prediction problems',
    'Durbin and Koopman (2012), Time Series Analysis by State Space Methods, chapter 4',
    'Doucet and Johansen (2011), A tutorial on particle filtering and smoothing',
  ],
  definition: defineModel('ar1-state-space', (b) => {
    const y = b.data('y', { dims: ['time'], role: 'response' })
    b.scan('x', {
      over: 'time',
      init: d.normal(0, S0),
      step: ({ prev }) => d.normal(prev.mul(AR1.phi), AR1.sigmaX),
      emit: ({ state, observe: obs }) => obs('y', d.normal(state, AR1.sigmaY), y),
    })
  }),
  generate: generateSeries,
  truth: (data) => ar1Truth(data.data['y'] as number[]),
  plans: (scale) => [
    exactPlan('gaussian', { config: { method: 'kalman' }, note: 'the Kalman smoother: exact' }),
    nutsPlan(scale),
    laplacePlan(scale, { expect: 'agree', note: 'the posterior is Gaussian, so a normal at the mode is exact' }),
    particleFilterPlan(scale),
    sviPlan(scale, { note: 'a mean-field normal ignores the correlation of neighbouring states: too narrow' }),
  ],
})
