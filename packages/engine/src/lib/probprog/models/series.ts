/**
 * Two time-series benchmarks written as `scan` nodes. The stochastic volatility model (a log-variance that follows an AR(1)
 * and scales Gaussian returns) is non-linear and non-Gaussian in the state, so no filter is exact; with the parameters known
 * its smoothing posterior is a one-dimensional state per step, whose truth is a point-mass filter and smoother on a fine
 * grid (a deterministic quadrature, accurate to the grid). The Gaussian hidden Markov model has three states, known
 * transitions and Gaussian emissions with overlapping means, whose truth is forward-backward written out independently of
 * every engine, with the exact distribution of the number of visits to a state, which depends on the joint posterior of the
 * steps and so tests more than the marginals.
 */

import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type MarginalOracle, weightedOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { hmmSmoothing, pmfOracle } from './discrete-truth.ts'
import { exactPlan, gibbsPlan, laplacePlan, nutsPlan, particleFilterPlan, sviPlan, temperedPlan } from './plans.ts'

const SV = { mu: -1, phi: 0.95, sigma: 0.25 }
/** The stationary standard deviation of the log-variance, from which the first state is drawn. */
const SV_S0 = SV.sigma / Math.sqrt(1 - SV.phi ** 2)

function generateReturns(key: Key, size = 30): BenchmarkData {
  const e = toFlat(drawNormal(foldIn(key, 0), [size]))
  const r = toFlat(drawNormal(foldIn(key, 1), [size]))
  const h: number[] = []
  for (let t = 0; t < size; t++)
    h.push(t === 0 ? SV.mu + SV_S0 * (e[0] ?? 0) : SV.mu + SV.phi * ((h[t - 1] ?? 0) - SV.mu) + SV.sigma * (e[t] ?? 0))
  return { data: { y: h.map((v, t) => Math.exp(v / 2) * (r[t] ?? 0)) }, generating: { ...SV, h } }
}

/**
 * The smoothing posterior of the stochastic volatility states by a point-mass filter and smoother: the log-variance is put on a
 * grid of `points` values over six stationary standard deviations either side of `mu`, the transition is the normal
 * density between grid points and the likelihood the normal density of the return, and forward-backward on the grid gives
 * every state's posterior (as weighted grid points) and the evidence. The error is that of a trapezoid rule on a normal
 * density of standard deviation `sigma` sampled at a spacing of `12 s0 / points`, far below Monte Carlo error at the default
 * 800 points.
 *
 * @param y - The returns.
 * @param points - The grid size (default 800).
 * @param parameters - The known `mu`, `phi` and `sigma` (default the benchmark's).
 * @returns The truth: an oracle for each state `h[t]` and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { stochasticVolatilityTruth } from 'omai-engine/probprog/models'
 * Object.keys(stochasticVolatilityTruth([0.3, -0.1, 0.5], 100).marginals) // => ['h[0]', 'h[1]', 'h[2]']
 * ```
 */
export function stochasticVolatilityTruth(
  y: readonly number[],
  points = 800,
  parameters: { readonly mu: number; readonly phi: number; readonly sigma: number } = SV,
): BenchmarkTruth {
  const SV = parameters
  const SV_S0 = SV.sigma / Math.sqrt(1 - SV.phi ** 2)
  const G = points
  const lo = SV.mu - 6 * SV_S0
  const step = (12 * SV_S0) / (G - 1)
  const h = Float64Array.from({ length: G }, (_, g) => lo + g * step)
  // Transition: T[a][b] = N(h_b; mu + phi (h_a - mu), sigma) * step.
  const T = new Float64Array(G * G)
  const c = step / (SV.sigma * Math.sqrt(2 * Math.PI))
  for (let a = 0; a < G; a++) {
    const m = SV.mu + SV.phi * ((h[a] as number) - SV.mu)
    for (let b = 0; b < G; b++) T[a * G + b] = c * Math.exp(-0.5 * (((h[b] as number) - m) / SV.sigma) ** 2)
  }
  const lik = (t: number, g: number): number => {
    const sd = Math.exp((h[g] as number) / 2)
    return Math.exp(-0.5 * ((y[t] as number) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
  }
  const alpha: Float64Array[] = []
  const scale: number[] = []
  for (let t = 0; t < y.length; t++) {
    const a = new Float64Array(G)
    if (t === 0)
      for (let g = 0; g < G; g++)
        a[g] =
          ((Math.exp(-0.5 * (((h[g] as number) - SV.mu) / SV_S0) ** 2) * step) / (SV_S0 * Math.sqrt(2 * Math.PI))) *
          lik(0, g)
    else {
      const prev = alpha[t - 1] as Float64Array
      for (let b = 0; b < G; b++) {
        let s = 0
        for (let g = 0; g < G; g++) s += (prev[g] as number) * (T[g * G + b] as number)
        a[b] = s * lik(t, b)
      }
    }
    const z = a.reduce((s, v) => s + v, 0)
    scale[t] = z
    alpha[t] = a.map((v) => v / z)
  }
  const beta: Float64Array[] = Array.from({ length: y.length }, () => new Float64Array(G).fill(1))
  for (let t = y.length - 2; t >= 0; t--) {
    const next = beta[t + 1] as Float64Array
    const w = Float64Array.from({ length: G }, (_, g) => (next[g] as number) * lik(t + 1, g))
    const b = beta[t] as Float64Array
    for (let a = 0; a < G; a++) {
      let s = 0
      for (let g = 0; g < G; g++) s += (T[a * G + g] as number) * (w[g] as number)
      b[a] = s / (scale[t + 1] as number)
    }
  }
  const marginals: Record<string, MarginalOracle> = {}
  for (let t = 0; t < y.length; t++) {
    const weights = Float64Array.from(
      { length: G },
      (_, g) => ((alpha[t] as Float64Array)[g] as number) * ((beta[t] as Float64Array)[g] as number),
    )
    marginals[`h[${t}]`] = weightedOracle(`h[${t}]`, h, weights)
  }
  return {
    kind: 'quadrature',
    source: `point-mass filter and smoother on a grid of ${G} log-variance values over mu +- 6 stationary sd (spacing ${step.toFixed(4)}, transition sd ${SV.sigma}), the parameters known; the evidence is the product of the filter's normalising constants`,
    logEvidence: scale.reduce((s, v) => s + Math.log(v), 0),
    marginals,
  }
}

/**
 * Stochastic volatility with known parameters: the log-variance `h_t = mu + phi (h_{t-1} - mu) + sigma e_t` (`mu = -1`, `phi =
 * 0.95`, `sigma = 0.25`, stationary start) scales the returns `y_t ~ N(0, exp(h_t / 2))`, thirty steps, written as a `scan`.
 * The posterior of the states is not Gaussian (the likelihood is not), so the Kalman smoother does not apply and the
 * filter-based methods are approximations: the truth is a point-mass smoother. NUTS differentiates through the unrolled
 * recurrence; the bootstrap filter's smoothing marginals thin out toward the start of the series. With the parameters unknown
 * this is the standard particle-MCMC target; here they are fixed so that the states have an independent truth.
 *
 * @concept posterior-validation
 * @index dataset Stochastic volatility as a scan, parameters known; the truth is a point-mass smoother on a grid.
 * @example
 * ```ts
 * import { stochasticVolatility } from 'omai-engine/probprog/models'
 * [stochasticVolatility.bound().sizes, stochasticVolatility.truth(undefined, 'fast').kind] // => [{ time: 30 }, 'quadrature']
 * ```
 */
export const stochasticVolatility: BenchmarkModel = defineBenchmark({
  name: 'stochastic-volatility',
  title: 'Stochastic volatility',
  description:
    'A log-variance following an AR(1) scales Gaussian returns, the parameters known: a non-linear, non-Gaussian state-space model written as a scan, with a point-mass smoother as the truth.',
  tags: ['state-space', 'scan', 'markov', 'time-series', 'non-gaussian', 'volatility', 'quadrature'],
  references: [
    'Kim, Shephard and Chib (1998), Stochastic volatility: likelihood inference and comparison with ARCH models',
    'Taylor (1982), Financial returns modelled by the product of two stochastic processes',
    'Andrieu, Doucet and Holenstein (2010), Particle Markov chain Monte Carlo methods (the stochastic volatility example)',
  ],
  definition: defineModel('stochastic-volatility', (b) => {
    const y = b.data('y', { dims: ['time'], role: 'response' })
    b.scan('h', {
      over: 'time',
      init: d.normal(SV.mu, SV_S0),
      step: ({ prev }) => d.normal(prev.sub(SV.mu).mul(SV.phi).add(SV.mu), SV.sigma),
      emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(state.mul(0.5))), y),
      doc: 'the log-variance of the returns',
    })
  }),
  generate: generateReturns,
  truth: (data, quality) => stochasticVolatilityTruth(data.data['y'] as number[], quality === 'fast' ? 200 : 800),
  plans: (scale) => [
    nutsPlan(scale, {
      warmup: 300,
      draws: 300,
      note: 'a consistent sampler through the unrolled recurrence: agrees within Monte Carlo error',
      tolerances: { meanError: 0.2, meanZ: 6, sdRatio: [0.85, 1.15], ks: 0.15 },
    }),
    temperedPlan(scale, {
      particles: 500,
      expect: 'report',
      note: 'tempered particles over thirty correlated states: consistent, with a noisy tail',
    }),
    particleFilterPlan(scale, {
      note: 'a bootstrap filter: unbiased evidence and good filtering marginals; the smoothing marginals are the genealogies of the final particles, thin toward the start',
    }),
    laplacePlan(scale, {
      note: 'a Gaussian at the mode of the joint density: the log-variance posterior is close to Gaussian, the tails of the return likelihood are not',
    }),
    sviPlan(scale, {
      steps: 800,
      note: 'a mean-field normal ignores the correlation between neighbouring states: spreads too small',
    }),
  ],
})

const HMM = {
  initial: [0.5, 0.3, 0.2],
  transition: [
    [0.8, 0.15, 0.05],
    [0.1, 0.75, 0.15],
    [0.05, 0.15, 0.8],
  ],
  means: [-1.5, 0, 1.5],
  sd: 1,
}

function generateHmm(key: Key, size = 16): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  const pick = (probs: readonly number[], v: number): number => {
    let acc = 0
    for (let k = 0; k < probs.length; k++) {
      acc += probs[k] as number
      if (v < acc) return k
    }
    return probs.length - 1
  }
  const z: number[] = []
  for (let t = 0; t < size; t++)
    z.push(pick(t === 0 ? HMM.initial : (HMM.transition[z[t - 1] as number] as number[]), u[t] ?? 0.5))
  return {
    data: { y: z.map((k, t) => (HMM.means[k] as number) + HMM.sd * (e[t] ?? 0)) },
    generating: { ...HMM, z },
  }
}

/** A Kolmogorov distance between draws on a few atoms and a step CDF is not small for a correct sampler: compare by the mean and the spread. */
const DISCRETE = { ks: 1 } as const

/** The number of steps spent in state 2 in each draw, from the draws of the states `z[t]`. */
function visitsOfLastState(draws: Readonly<Record<string, ArrayLike<number>>>): Record<string, ArrayLike<number>> {
  const sites = Object.keys(draws).filter((n) => /^z\[\d+\]$/.test(n))
  const first = draws[sites[0] ?? '']
  if (first === undefined) return {}
  const out = new Float64Array(first.length)
  for (const s of sites) {
    const v = draws[s] as ArrayLike<number>
    for (let i = 0; i < out.length; i++) if (v[i] === 2) out[i] = (out[i] as number) + 1
  }
  return { visits_2: out }
}

/**
 * The exact smoothing of the Gaussian HMM benchmark: the marginal of each state `z[t]` and the number of steps in state 2,
 * by forward-backward written out here (`hmmSmoothing`), and the log evidence.
 *
 * @param y - The observations.
 * @returns The truth.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { gaussianHmmTruth } from 'omai-engine/probprog/models'
 * Object.keys(gaussianHmmTruth([0.1, 1.4, 1.2]).marginals) // => ['z[0]', 'z[1]', 'z[2]', 'visits_2']
 * ```
 */
export function gaussianHmmTruth(y: readonly number[]): BenchmarkTruth {
  const s = hmmSmoothing(y, HMM.initial, HMM.transition, HMM.means, HMM.sd)
  const marginals: Record<string, MarginalOracle> = {}
  s.marginals.forEach((p, t) => {
    marginals[`z[${t}]`] = pmfOracle(`z[${t}]`, [0, 1, 2], p)
  })
  marginals['visits_2'] = pmfOracle(
    'visits_2',
    (s.visits[2] as number[]).map((_, c) => c),
    s.visits[2] as number[],
  )
  return {
    kind: 'exact',
    source:
      'forward-backward for the marginals and the evidence, and a forward pass over (state, number of visits) for the exact distribution of the visits to state 2, written out in discrete-truth.ts (independent of the chain engine)',
    logEvidence: s.logEvidence,
    marginals,
  }
}

/**
 * A three-state Gaussian hidden Markov model with known parameters, sixteen steps, written as a `scan` over a categorical
 * state: the chain engine (forward-backward) and the junction tree are exact, Gibbs and the particle filter approximate.
 * The emissions overlap (means -1.5, 0, 1.5, sd 1), so the posterior of each state is spread, and the number of visits to
 * a state is a quantity of the joint posterior with an exact distribution.
 *
 * @concept hidden-markov-models
 * @index dataset Three-state Gaussian HMM as a scan, parameters known; exact forward-backward truth with visit counts.
 * @example
 * ```ts
 * import { gaussianHmm } from 'omai-engine/probprog/models'
 * [gaussianHmm.bound().sizes, gaussianHmm.truth().kind] // => [{ time: 16, from: 3, state: 3 }, 'exact']
 * ```
 */
export const gaussianHmm: BenchmarkModel = defineBenchmark({
  name: 'gaussian-hmm',
  title: 'Gaussian hidden Markov model',
  description:
    'Three discrete states with known transitions and overlapping Gaussian emissions, written as a scan: forward-backward is the exact truth, and the number of visits to a state tests the joint posterior.',
  tags: ['state-space', 'scan', 'markov', 'hmm', 'discrete-latent', 'time-series', 'exact', 'enumeration'],
  references: [
    'Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition',
    'Murphy (2012), Machine Learning: a probabilistic perspective, chapter 17',
  ],
  definition: defineModel('gaussian-hmm', (b) => {
    const y = b.data('y', { dims: ['time'], role: 'response' })
    const A = b.const(HMM.transition, ['from'])
    const means = b.const(HMM.means, ['state'])
    b.scan('z', {
      over: 'time',
      init: d.categorical({ probs: HMM.initial }),
      step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
      emit: ({ state, observe }) => observe('y', d.normal(ops.at(means, state), HMM.sd), y),
      doc: 'the hidden state',
    })
  }),
  generate: generateHmm,
  summarise: visitsOfLastState,
  truth: (data) => gaussianHmmTruth(data.data['y'] as number[]),
  plans: (scale) => {
    const marginalsOnly = Array.from({ length: 16 }, (_, t) => `z[${t}]`)
    return [
      exactPlan('chain', {
        tolerances: DISCRETE,
        note: 'forward-backward and forward filtering backward sampling: exact marginals, and joint draws that keep the visit counts',
      }),
      exactPlan('enumeration', {
        quantities: marginalsOnly,
        tolerances: DISCRETE,
        note: 'a junction tree over the unrolled chain: exact marginals (its draws are independent per state, so the visit count is not compared)',
      }),
      gibbsPlan(scale, {
        draws: 500,
        expect: 'report',
        note: 'single-site Gibbs on a sticky chain: consistent, slow to move between runs of states',
      }),
      particleFilterPlan(scale, {
        note: 'a bootstrap filter over the categorical states: unbiased evidence, filtering marginals right, early smoothing marginals thinned by the genealogy',
      }),
    ]
  },
})
