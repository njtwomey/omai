/**
 * State-space models: hidden Markov models with categorical or Gaussian emissions (hmmlearn style), with their chain
 * potentials, sampling and conversion to a dynamic Bayes network, and linear dynamical systems (linear-Gaussian
 * state-space models), with unrolling to a linear-Gaussian network, sampling and the exact smoothing posterior.
 */

import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import { checkSquare, choleskyLower, dotVec, logdetPositive, type Matrix, matVec, solveVector, sub } from '../_dense.ts'
import { gaussianFromMoments, gaussianLogValue } from '../gaussian-factor.ts'
import type { ChainModel } from '../inference/chain.ts'
import { type DynamicBayesNet, dynamicBayesNet } from './dynamic.ts'
import {
  type LinearGaussianBayesNet,
  type LinearGaussianNode,
  linearGaussianBayesNet,
  linearGaussianPosterior,
} from './linear-gaussian.ts'

/** The emission model of a hidden Markov model: a table over symbols, or a Gaussian per state. */
export type HmmEmission =
  | { readonly kind: 'categorical'; readonly symbols: readonly string[]; readonly probabilities: Matrix }
  | { readonly kind: 'gaussian'; readonly means: Matrix; readonly covariances: readonly Matrix[] }

/**
 * A hidden Markov model: hidden states with an initial distribution and transition matrix, and categorical or Gaussian
 * emissions (hmmlearn's CategoricalHMM and GaussianHMM). As a DBN it has two variables per slice.
 *
 * @concept hidden-markov-models
 */
export interface HiddenMarkovModel {
  readonly kind: 'hmm'
  readonly states: readonly string[]
  readonly initial: readonly number[]
  /** transition[i][j] = P(z_{t+1} = j | z_t = i). */
  readonly transition: Matrix
  readonly emission: HmmEmission
}

const isDistribution = (row: readonly number[]): boolean =>
  row.every((v) => v >= 0) && Math.abs(row.reduce((a, b) => a + b, 0) - 1) <= 1e-9

/**
 * Build and validate a hidden Markov model.
 *
 * @param spec - States, initial distribution, transition matrix and emission model.
 * @returns The model.
 * @throws WhiteboxError with code `shape` for mismatched sizes, `value` for rows that are not distributions or
 * covariances that are not positive definite.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { hiddenMarkovModel } from 'whitebox-engine/pgm'
 * const hmm = hiddenMarkovModel({
 *   states: ['fair', 'loaded'],
 *   initial: [0.5, 0.5],
 *   transition: [[0.95, 0.05], [0.1, 0.9]],
 *   emission: { kind: 'categorical', symbols: ['low', 'high'], probabilities: [[0.5, 0.5], [0.2, 0.8]] },
 * })
 * hmm.states.length // => 2
 * ```
 */
export function hiddenMarkovModel(spec: Omit<HiddenMarkovModel, 'kind'>): HiddenMarkovModel {
  const K = spec.states.length
  if (spec.initial.length !== K || !isDistribution(spec.initial))
    throw new WhiteboxError('value', 'hiddenMarkovModel: the initial distribution needs one probability per state')
  checkSquare(spec.transition, K, 'hiddenMarkovModel')
  if (!spec.transition.every(isDistribution))
    throw new WhiteboxError('value', 'hiddenMarkovModel: transition rows must be distributions')
  const e = spec.emission
  if (e.kind === 'categorical') {
    if (
      e.probabilities.length !== K ||
      e.probabilities.some((r) => r.length !== e.symbols.length || !isDistribution(r))
    )
      throw new WhiteboxError('value', 'hiddenMarkovModel: one emission distribution over the symbols per state')
  } else {
    const D = e.means[0]?.length ?? 0
    if (e.means.length !== K || e.covariances.length !== K || e.means.some((m) => m.length !== D))
      throw new WhiteboxError('shape', 'hiddenMarkovModel: one mean and one covariance per state')
    for (const c of e.covariances) {
      checkSquare(c, D, 'hiddenMarkovModel')
      logdetPositive(c, 'hiddenMarkovModel')
    }
  }
  return { kind: 'hmm', ...spec }
}

/**
 * The chain potentials of a hidden Markov model for a sequence of observations: log initial and transition
 * probabilities and the log emission probability (or density) of each observation under each state, ready for
 * forward-backward, Viterbi and sampling.
 *
 * @param hmm - The model.
 * @param observations - Symbols (names or indices) for categorical emissions, vectors for Gaussian ones.
 * @returns The chain.
 * @throws WhiteboxError with code `value` for unknown symbols or vectors of the wrong length.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { hiddenMarkovModel, hmmChain, forwardBackward } from 'whitebox-engine/pgm'
 * const hmm = hiddenMarkovModel({
 *   states: ['a', 'b'], initial: [0.6, 0.4], transition: [[0.7, 0.3], [0.4, 0.6]],
 *   emission: { kind: 'categorical', symbols: ['x', 'y', 'z'], probabilities: [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]] },
 * })
 * Math.round(forwardBackward(hmmChain(hmm, ['x', 'y', 'z', 'x'])).logLikelihood * 1e6) / 1e6 // => -4.700712
 * ```
 */
export function hmmChain(
  hmm: HiddenMarkovModel,
  observations: readonly (string | number | readonly number[])[],
): ChainModel {
  const logTransition = hmm.transition.map((r) => r.map(Math.log))
  const logInitial = hmm.initial.map(Math.log)
  const e = hmm.emission
  let logEmission: number[][]
  if (e.kind === 'categorical') {
    logEmission = observations.map((o) => {
      const s = typeof o === 'number' ? o : typeof o === 'string' ? e.symbols.indexOf(o) : -1
      if (!(s >= 0 && s < e.symbols.length)) throw new WhiteboxError('value', `hmmChain: unknown symbol ${String(o)}`)
      return e.probabilities.map((r) => Math.log(r[s] ?? 0))
    })
  } else {
    const names = (e.means[0] ?? []).map((_, d) => `d${d}`)
    const densities = e.means.map((m, k) => gaussianFromMoments(names, m, e.covariances[k] ?? []))
    logEmission = observations.map((o) => {
      if (typeof o === 'string' || typeof o === 'number' || o.length !== names.length)
        throw new WhiteboxError('value', `hmmChain: Gaussian emissions need vectors of length ${names.length}`)
      const x = Object.fromEntries(names.map((n, d) => [n, o[d] ?? 0]))
      return densities.map((f) => gaussianLogValue(f, x))
    })
  }
  return { logInitial, logTransition, logEmission }
}

/** Draw an index from a distribution with a uniform number. */
function drawIndex(p: readonly number[], u: number): number {
  let acc = 0
  for (const [k, v] of p.entries()) {
    acc += v
    if (u < acc) return k
  }
  return p.length - 1
}

/**
 * Sample a state path and observations from a hidden Markov model.
 *
 * @param hmm - The model.
 * @param key - The random key.
 * @param T - Sequence length.
 * @returns The hidden states (indices) and the observations (symbol indices or vectors).
 * @concept sampling
 * @example
 * ```ts
 * import { hiddenMarkovModel, hmmSample } from 'whitebox-engine/pgm'
 * import { key } from 'whitebox-engine/random'
 * const hmm = hiddenMarkovModel({
 *   states: ['a', 'b'], initial: [1, 0], transition: [[0, 1], [1, 0]],
 *   emission: { kind: 'categorical', symbols: ['x', 'y'], probabilities: [[1, 0], [0, 1]] },
 * })
 * hmmSample(hmm, key(0), 4) // => { states: [0, 1, 0, 1], observations: [0, 1, 0, 1] }
 * ```
 */
export function hmmSample(
  hmm: HiddenMarkovModel,
  key: Key,
  T: number,
): { readonly states: number[]; readonly observations: (number | number[])[] } {
  const u = uniform(foldIn(key, 0), [2 * T]).data
  const states: number[] = []
  const observations: (number | number[])[] = []
  const e = hmm.emission
  const factors = e.kind === 'gaussian' ? e.covariances.map((c) => choleskyLower(c)) : []
  const z = e.kind === 'gaussian' ? normal(foldIn(key, 1), [T, e.means[0]?.length ?? 0]).data : new Float64Array(0)
  for (let t = 0; t < T; t++) {
    const p = t === 0 ? hmm.initial : (hmm.transition[states[t - 1] ?? 0] ?? [])
    const s = drawIndex(p, u[2 * t] ?? 0)
    states.push(s)
    if (e.kind === 'categorical') observations.push(drawIndex(e.probabilities[s] ?? [], u[2 * t + 1] ?? 0))
    else {
      const D = e.means[0]?.length ?? 0
      const L = factors[s] ?? []
      const eps = Array.from({ length: D }, (_, d) => z[t * D + d] ?? 0)
      const shift = matVec(L, eps)
      observations.push((e.means[s] ?? []).map((m, d) => m + (shift[d] ?? 0)))
    }
  }
  return { states, observations }
}

/**
 * A categorical hidden Markov model as a dynamic Bayes network with a hidden variable `z` and an observed variable `x`
 * per slice.
 *
 * @param hmm - The model (categorical emissions).
 * @returns The two-slice network.
 * @throws WhiteboxError with code `value` for Gaussian emissions.
 * @concept dynamic-bayes-networks
 * @example
 * ```ts
 * import { hiddenMarkovModel, hmmToDbn, unroll } from 'whitebox-engine/pgm'
 * const hmm = hiddenMarkovModel({
 *   states: ['a', 'b'], initial: [0.5, 0.5], transition: [[0.9, 0.1], [0.1, 0.9]],
 *   emission: { kind: 'categorical', symbols: ['x', 'y'], probabilities: [[0.8, 0.2], [0.3, 0.7]] },
 * })
 * unroll(hmmToDbn(hmm), 2).nodes.map((n) => n.name) // => ['z_0', 'x_0', 'z_1', 'x_1']
 * ```
 */
export function hmmToDbn(hmm: HiddenMarkovModel): DynamicBayesNet {
  const e = hmm.emission
  if (e.kind !== 'categorical') throw new WhiteboxError('value', 'hmmToDbn: only categorical emissions are discrete')
  const emission = { name: 'x', states: e.symbols, parents: ['z'], cpt: e.probabilities.flat() }
  return dynamicBayesNet(
    [{ name: 'z', states: hmm.states, parents: [], cpt: [...hmm.initial] }, emission],
    [
      { name: 'z', states: hmm.states, parents: [], previous: ['z'], cpt: hmm.transition.flat() },
      { ...emission, previous: [] },
    ],
  )
}

/**
 * A linear dynamical system (linear-Gaussian state-space model): x_0 ~ N(m0, P0), x_t = A x_{t-1} + N(0, Q),
 * y_t = C x_t + N(0, R).
 *
 * @concept linear-dynamical-systems
 */
export interface LinearDynamicalSystem {
  readonly kind: 'lds'
  readonly transition: Matrix
  readonly observation: Matrix
  readonly transitionCovariance: Matrix
  readonly observationCovariance: Matrix
  readonly initialMean: readonly number[]
  readonly initialCovariance: Matrix
}

/**
 * Build and validate a linear dynamical system.
 *
 * @param spec - A, C, Q, R, the initial mean and covariance.
 * @returns The system.
 * @throws WhiteboxError with code `shape` for mismatched sizes, `value` for covariances that are not positive definite.
 * @concept linear-dynamical-systems
 * @example
 * ```ts
 * import { linearDynamicalSystem } from 'whitebox-engine/pgm'
 * const lds = linearDynamicalSystem({ transition: [[1]], observation: [[1]], transitionCovariance: [[0.1]], observationCovariance: [[1]], initialMean: [0], initialCovariance: [[1]] })
 * lds.kind // => 'lds'
 * ```
 */
export function linearDynamicalSystem(spec: Omit<LinearDynamicalSystem, 'kind'>): LinearDynamicalSystem {
  const n = spec.initialMean.length
  const m = spec.observation.length
  checkSquare(spec.transition, n, 'linearDynamicalSystem')
  checkSquare(spec.transitionCovariance, n, 'linearDynamicalSystem')
  checkSquare(spec.initialCovariance, n, 'linearDynamicalSystem')
  checkSquare(spec.observationCovariance, m, 'linearDynamicalSystem')
  if (spec.observation.some((r) => r.length !== n))
    throw new WhiteboxError('shape', `linearDynamicalSystem: C must be ${m} x ${n}`)
  for (const c of [spec.transitionCovariance, spec.observationCovariance, spec.initialCovariance])
    logdetPositive(c, 'linearDynamicalSystem')
  return { kind: 'lds', ...spec }
}

/**
 * Scalar linear-Gaussian nodes for a vector v ~ N(B u + c, S) given parent values u: component i depends on u and on
 * v_0..v_{i-1}, with the weights and variance of the Gaussian conditional (so correlated noise is represented exactly).
 */
function vectorNodes(
  names: readonly string[],
  parents: readonly string[],
  B: Matrix,
  c: readonly number[],
  S: Matrix,
): LinearGaussianNode[] {
  return names.map((name, i) => {
    const before = Array.from({ length: i }, (_, k) => k)
    const beta =
      i === 0
        ? []
        : solveVector(
            sub(S, before, before),
            before.map((k) => S[i]?.[k] ?? 0),
          )
    const onParents = parents.map((_, j) => (B[i]?.[j] ?? 0) - beta.reduce((s, b, k) => s + b * (B[k]?.[j] ?? 0), 0))
    const intercept = (c[i] ?? 0) - beta.reduce((s, b, k) => s + b * (c[k] ?? 0), 0)
    const variance =
      (S[i]?.[i] ?? 0) -
      dotVec(
        beta,
        before.map((k) => S[k]?.[i] ?? 0),
      )
    return {
      name,
      parents: [...parents, ...before.map((k) => names[k] ?? '')],
      weights: [...onParents, ...beta],
      intercept,
      variance,
    }
  })
}

/**
 * A linear dynamical system unrolled over T steps as a linear-Gaussian Bayes network with scalar nodes `x<i>_<t>` and
 * `y<j>_<t>` (correlated noise becomes within-slice arcs).
 *
 * @param lds - The system.
 * @param T - Number of steps.
 * @returns The network.
 * @concept linear-dynamical-systems
 * @example
 * ```ts
 * import { linearDynamicalSystem, ldsToLinearGaussian } from 'whitebox-engine/pgm'
 * const lds = linearDynamicalSystem({ transition: [[1]], observation: [[1]], transitionCovariance: [[0.1]], observationCovariance: [[1]], initialMean: [0], initialCovariance: [[1]] })
 * ldsToLinearGaussian(lds, 2).nodes.map((n) => n.name) // => ['x0_0', 'y0_0', 'x0_1', 'y0_1']
 * ```
 */
export function ldsToLinearGaussian(lds: LinearDynamicalSystem, T: number): LinearGaussianBayesNet {
  const n = lds.initialMean.length
  const m = lds.observation.length
  const xs = (t: number): string[] => Array.from({ length: n }, (_, i) => `x${i}_${t}`)
  const ys = (t: number): string[] => Array.from({ length: m }, (_, j) => `y${j}_${t}`)
  const nodes: LinearGaussianNode[] = []
  for (let t = 0; t < T; t++) {
    if (t === 0) nodes.push(...vectorNodes(xs(0), [], [], lds.initialMean, lds.initialCovariance))
    else
      nodes.push(
        ...vectorNodes(xs(t), xs(t - 1), lds.transition, new Array<number>(n).fill(0), lds.transitionCovariance),
      )
    nodes.push(...vectorNodes(ys(t), xs(t), lds.observation, new Array<number>(m).fill(0), lds.observationCovariance))
  }
  return linearGaussianBayesNet(nodes)
}

/**
 * Sample a state trajectory and observations from a linear dynamical system.
 *
 * @param lds - The system.
 * @param key - The random key.
 * @param T - Number of steps.
 * @returns States `[T][n]` and observations `[T][m]`.
 * @concept sampling
 * @example
 * ```ts
 * import { linearDynamicalSystem, ldsSample } from 'whitebox-engine/pgm'
 * import { key } from 'whitebox-engine/random'
 * const lds = linearDynamicalSystem({ transition: [[1]], observation: [[1]], transitionCovariance: [[0.1]], observationCovariance: [[1]], initialMean: [0], initialCovariance: [[1]] })
 * ldsSample(lds, key(0), 5).observations.length // => 5
 * ```
 */
export function ldsSample(
  lds: LinearDynamicalSystem,
  key: Key,
  T: number,
): { readonly states: number[][]; readonly observations: number[][] } {
  const n = lds.initialMean.length
  const m = lds.observation.length
  const L0 = choleskyLower(lds.initialCovariance)
  const LQ = choleskyLower(lds.transitionCovariance)
  const LR = choleskyLower(lds.observationCovariance)
  const zx = normal(foldIn(key, 0), [T, n]).data
  const zy = normal(foldIn(key, 1), [T, m]).data
  const states: number[][] = []
  const observations: number[][] = []
  for (let t = 0; t < T; t++) {
    const ex = Array.from({ length: n }, (_, i) => zx[t * n + i] ?? 0)
    const base = t === 0 ? [...lds.initialMean] : matVec(lds.transition, states[t - 1] ?? [])
    const noise = matVec(t === 0 ? L0 : LQ, ex)
    const x = base.map((v, i) => v + (noise[i] ?? 0))
    states.push(x)
    const ey = Array.from({ length: m }, (_, j) => zy[t * m + j] ?? 0)
    const ny = matVec(LR, ey)
    observations.push(matVec(lds.observation, x).map((v, j) => v + (ny[j] ?? 0)))
  }
  return { states, observations }
}

/**
 * The smoothing posterior of a linear dynamical system, p(x_t | y_1..y_T) for every t, and the log-likelihood of the
 * observations, by exact Gaussian conditioning of the unrolled linear-Gaussian network (what the Kalman smoother
 * computes recursively).
 *
 * @param lds - The system.
 * @param observations - Observations `[T][m]`.
 * @returns Smoothed means `[T][n]`, covariances `[T][n][n]` and log p(y).
 * @concept linear-dynamical-systems
 * @example
 * ```ts
 * import { linearDynamicalSystem, ldsPosterior } from 'whitebox-engine/pgm'
 * const lds = linearDynamicalSystem({ transition: [[1]], observation: [[1]], transitionCovariance: [[1e-12]], observationCovariance: [[1]], initialMean: [0], initialCovariance: [[1]] })
 * ldsPosterior(lds, [[2]]).means.map((m) => Math.round((m[0] ?? 0) * 1e9) / 1e9) // => [1]
 * ```
 */
export function ldsPosterior(
  lds: LinearDynamicalSystem,
  observations: readonly (readonly number[])[],
): { readonly means: number[][]; readonly covariances: number[][][]; readonly logLikelihood: number } {
  const T = observations.length
  const n = lds.initialMean.length
  const g = ldsToLinearGaussian(lds, T)
  const evidence: Record<string, number> = {}
  for (const [t, y] of observations.entries()) for (const [j, v] of y.entries()) evidence[`y${j}_${t}`] = v
  const query = Array.from({ length: T }, (_, t) => Array.from({ length: n }, (_, i) => `x${i}_${t}`)).flat()
  const post = linearGaussianPosterior(g, query, evidence)
  const means = Array.from({ length: T }, (_, t) => post.mean.slice(t * n, (t + 1) * n))
  const covariances = Array.from({ length: T }, (_, t) =>
    Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, k) => post.covariance[t * n + i]?.[t * n + k] ?? 0),
    ),
  )
  return { means, covariances, logLikelihood: post.logMass }
}
