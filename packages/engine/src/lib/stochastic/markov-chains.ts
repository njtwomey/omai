/**
 * Discrete-time Markov chains given by a transition matrix: classification of states into communicating classes with
 * periods, stationary distributions, n-step distributions, absorption probabilities and times, hitting probabilities
 * and times, mean return times, mixing times and the spectral gap, reversibility and time reversal, and simulation of
 * paths.
 */

import { WhiteboxError } from '../base/index.ts'
import { totalVariation } from '../information/index.ts'
import { eigh, eigvals, matrixPower, solve } from '../numerics/linalg/index.ts'
import { categorical, foldIn, type Key } from '../random/index.ts'
import {
  asFloat,
  asTensor,
  div,
  expandDims,
  eye,
  fromStorage,
  matmul,
  max,
  mul,
  ones,
  reshape,
  sliceAxis,
  sqrt,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import {
  assemble,
  block,
  classify,
  maybeStochasticRows,
  reach,
  stationarySolve,
  stochasticRows,
  targetMask,
  tensorOf,
} from './_chains.ts'
import { positiveInteger, primalNumbers, traceOf, vector } from './_shared.ts'
import { type Paths, simulatePaths, type TraceOptions } from './paths.ts'

/** How the states of a chain fall into communicating classes. */
export interface ChainClassification {
  /** The communicating classes (strongly connected components of the transition graph), ordered by smallest state. */
  readonly classes: readonly (readonly number[])[]
  /** The class index of every state. */
  readonly classOf: Int32Array
  /** Whether each class is closed (no probability leaves it): its states are recurrent, the others transient. */
  readonly closed: readonly boolean[]
  /** The period of each class (1 for aperiodic classes). */
  readonly periods: readonly number[]
  /** Whether each state is recurrent. */
  readonly recurrent: readonly boolean[]
  /** The absorbing states (P_ii = 1). */
  readonly absorbing: readonly number[]
  /** One communicating class. */
  readonly irreducible: boolean
  /** Irreducible with period 1. */
  readonly aperiodic: boolean
}

/**
 * Classify the states of a discrete-time Markov chain: communicating classes, which are closed (recurrent) and which
 * transient, their periods, the absorbing states, irreducibility and aperiodicity.
 *
 * @param p - The transition matrix `[n, n]` (rows sum to one).
 * @returns The classification.
 * @throws WhiteboxError with code `value` or `shape` for a matrix that is not stochastic.
 * @concept markov-chains
 * @example
 * ```ts
 * import { classifyStates } from 'whitebox-engine/stochastic'
 * const c = classifyStates([[0, 1, 0], [1, 0, 0], [0.5, 0, 0.5]])
 * c.classes // => [[0, 1], [2]]
 * c.periods // => [2, 1]
 * c.recurrent // => [true, true, false]
 * ```
 */
export function classifyStates(p: TensorLike): ChainClassification {
  const { n, rows } = stochasticRows(p, 'classifyStates')
  const c = classify(n, rows)
  const order = c.classes.map((_, k) => k).sort((a, b) => (c.classes[a]?.[0] ?? 0) - (c.classes[b]?.[0] ?? 0))
  const remap = new Int32Array(order.length)
  for (const [i, k] of order.entries()) remap[k] = i
  const periods = order.map((k) => c.periods[k] ?? 1)
  return {
    classes: order.map((k) => c.classes[k] ?? []),
    classOf: Int32Array.from(c.classOf, (k) => remap[k] ?? 0),
    closed: order.map((k) => c.closed[k] ?? false),
    periods,
    recurrent: c.recurrent,
    absorbing: c.absorbing,
    irreducible: c.classes.length === 1,
    aperiodic: c.classes.length === 1 && periods[0] === 1,
  }
}

/**
 * The stationary distribution pi = pi P of a chain with a single closed class (irreducible, possibly with transient
 * states), by solving the balance equations with sum(pi) = 1. Differentiable in P.
 *
 * @param p - The transition matrix.
 * @returns pi `[n]` (zero on transient states).
 * @throws WhiteboxError with code `value` when there are several closed classes (no unique stationary law; see
 *   {@link stationaryDistributions}).
 * @concept stationary-distribution
 * @example
 * ```ts
 * import { stationaryDistribution } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(stationaryDistribution([[0.9, 0.1], [0.5, 0.5]])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.833333333333, 0.166666666667]
 * ```
 */
export function stationaryDistribution(p: TensorLike): Value {
  const { n, rows } = maybeStochasticRows(p, 'stationaryDistribution')
  if (rows !== undefined && classify(n, rows).closed.filter((x) => x).length !== 1) {
    throw new WhiteboxError('value', 'stationaryDistribution: several closed classes; use stationaryDistributions')
  }
  return stationarySolve(sub(asFloat(asTensor(p)), eye(n)), n)
}

/**
 * One stationary distribution per closed class (the extreme points of the set of stationary laws), each supported on
 * its class.
 *
 * @param p - The transition matrix.
 * @returns `[classes, n]`, one row per closed class in order of its smallest state.
 * @concept stationary-distribution
 * @example
 * ```ts
 * import { stationaryDistributions } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(stationaryDistributions([[1, 0, 0], [0.2, 0.6, 0.2], [0, 0, 1]])) // => [[1, 0, 0], [0, 0, 1]]
 * ```
 */
export function stationaryDistributions(p: TensorLike): Tensor<'float64'> {
  const { n, rows } = stochasticRows(p, 'stationaryDistributions')
  const c = classifyStates(p)
  const out: number[] = []
  for (const [k, cls] of c.classes.entries()) {
    if (c.closed[k] !== true) continue
    const local = cls.map((i) => cls.map((j) => rows[i]?.[j] ?? 0))
    const pi = toFlat(stationarySolve(sub(tensorOf(local), eye(cls.length)), cls.length))
    const row = new Array<number>(n).fill(0)
    for (const [i, s] of cls.entries()) row[s] = pi[i] ?? 0
    out.push(...row)
  }
  return fromStorage(Float64Array.from(out), [out.length / n, n], 'float64')
}

/**
 * The distribution after n steps, p0 P^n (Chapman-Kolmogorov). Differentiable in p0 and P.
 *
 * @param p - The transition matrix.
 * @param p0 - The initial distribution `[n]` (or several `[k, n]`).
 * @param steps - The number of steps (a non-negative integer).
 * @returns The distribution(s) after `steps` steps.
 * @concept markov-chains
 * @example
 * ```ts
 * import { distributionAfter } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(distributionAfter([[0, 1], [1, 0]], [1, 0], 3)) // => [0, 1]
 * ```
 */
export function distributionAfter(p: TensorLike, p0: TensorLike, steps: number): Value {
  if (!(Number.isInteger(steps) && steps >= 0)) throw new WhiteboxError('value', 'distributionAfter: steps >= 0')
  maybeStochasticRows(p, 'distributionAfter')
  return matmul(asFloat(asTensor(p0)), matrixPower(asFloat(asTensor(p)), steps))
}

/** The result of {@link absorption}. */
export interface Absorption {
  /** The transient states, in order. */
  readonly transient: readonly number[]
  /** The absorbing states, in order. */
  readonly absorbing: readonly number[]
  /** The fundamental matrix N = (I - Q)^{-1}: expected visits to each transient state. */
  readonly fundamental: Value
  /** B = N R: the probability of ending in each absorbing state, from each transient state. */
  readonly probabilities: Value
  /** t = N 1: the expected number of steps before absorption. */
  readonly expectedSteps: Value
  /** The variance of the number of steps, (2 N - I) t - t^2. */
  readonly varianceSteps: Value
}

/**
 * Absorption analysis of an absorbing chain: the fundamental matrix, absorption probabilities and the mean and variance
 * of the time to absorption (Kemeny and Snell). Differentiable in P.
 *
 * @param p - The transition matrix, with an absorbing state reachable from every other state.
 * @returns The analysis, indexed by the transient and absorbing states.
 * @throws WhiteboxError with code `value` when the chain has no absorbing state or some state cannot reach one.
 * @concept absorbing-markov-chains
 * @example
 * ```ts
 * import { absorption } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * const a = absorption([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]])
 * toArray(a.probabilities) // => [[0.5, 0.5]]
 * toArray(a.expectedSteps) // => [1]
 * ```
 */
export function absorption(p: TensorLike): Absorption {
  const { n, rows } = stochasticRows(p, 'absorption')
  const c = classify(n, rows)
  const absorbing = [...c.absorbing]
  if (absorbing.length === 0) throw new WhiteboxError('value', 'absorption: the chain has no absorbing state')
  const transient = Array.from({ length: n }, (_, i) => i).filter((i) => !absorbing.includes(i))
  if (transient.some((i) => c.recurrent[i] === true)) {
    throw new WhiteboxError('value', 'absorption: some non-absorbing states never reach an absorbing state')
  }
  const k = transient.length
  if (k === 0) {
    const none = fromStorage(new Float64Array(0), [0], 'float64')
    return {
      transient,
      absorbing,
      fundamental: fromStorage(new Float64Array(0), [0, 0], 'float64'),
      probabilities: fromStorage(new Float64Array(0), [0, absorbing.length], 'float64'),
      expectedSteps: none,
      varianceSteps: none,
    }
  }
  const pt = asFloat(asTensor(p))
  const iq = sub(eye(k), block(pt, transient, transient))
  const fundamental = solve(iq, eye(k))
  const probabilities = solve(iq, block(pt, transient, absorbing))
  const expectedSteps = solve(iq, ones([k]))
  const nt = reshape(matmul(fundamental, expandDims(expectedSteps, -1)), [k])
  const varianceSteps = sub(sub(mul(nt, 2), expectedSteps), mul(expectedSteps, expectedSteps))
  return { transient, absorbing, fundamental, probabilities, expectedSteps, varianceSteps }
}

/**
 * The probability of ever reaching a set of states from each state: the minimal non-negative solution of h = 1 on the
 * target and h = P h elsewhere (zero where the target is unreachable). Differentiable in P.
 *
 * @param p - The transition matrix.
 * @param target - The target states.
 * @returns h `[n]`.
 * @concept hitting-times
 * @example
 * ```ts
 * import { hittingProbabilities } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * // a fair gambler's ruin on {0, 1, 2, 3}: reach 3 before 0
 * toArray(hittingProbabilities([[1, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 1]], [3])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0.333333333333, 0.666666666667, 1]
 * ```
 */
export function hittingProbabilities(p: TensorLike, target: readonly number[]): Value {
  const { n, rows } = stochasticRows(p, 'hittingProbabilities')
  const mask = targetMask(n, target, 'hittingProbabilities')
  const can = reach(n, rows, mask)
  const free = Array.from({ length: n }, (_, i) => i).filter((i) => !mask[i] && can[i] === true)
  const parts = new Array<Value | number>(n).fill(0)
  for (const s of target) parts[s] = 1
  if (free.length > 0) {
    const pt = asFloat(asTensor(p))
    const h = solve(sub(eye(free.length), block(pt, free, free)), sum(block(pt, free, [...target]), 1))
    for (const [i, s] of free.entries()) parts[s] = sliceAxis(h, 0, i, i + 1)
  }
  return assemble(parts)
}

/**
 * The expected number of steps to reach a set of states from each state: k = 0 on the target and k = 1 + P k
 * elsewhere, Infinity from states that miss the target with positive probability. Differentiable in P.
 *
 * @param p - The transition matrix.
 * @param target - The target states.
 * @returns k `[n]`.
 * @concept hitting-times
 * @example
 * ```ts
 * import { expectedHittingTimes } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(expectedHittingTimes([[0.5, 0.5], [0, 1]], [1])) // => [2, 0]
 * ```
 */
export function expectedHittingTimes(p: TensorLike, target: readonly number[]): Value {
  const { n } = stochasticRows(p, 'expectedHittingTimes')
  const mask = targetMask(n, target, 'expectedHittingTimes')
  const h = primalNumbers(hittingProbabilities(p, target))
  const sure = Array.from({ length: n }, (_, i) => i).filter((i) => !mask[i] && Math.abs((h[i] ?? 0) - 1) < 1e-9)
  const parts = new Array<Value | number>(n).fill(Number.POSITIVE_INFINITY)
  for (const s of target) parts[s] = 0
  if (sure.length > 0) {
    const pt = asFloat(asTensor(p))
    const k = solve(sub(eye(sure.length), block(pt, sure, sure)), ones([sure.length]))
    for (const [i, s] of sure.entries()) parts[s] = sliceAxis(k, 0, i, i + 1)
  }
  return assemble(parts)
}

/**
 * The mean return times E_i[T_i^+] = 1 / pi_i of an irreducible chain (Kac's formula). Differentiable in P.
 *
 * @param p - The transition matrix (irreducible).
 * @returns `[n]`.
 * @concept hitting-times
 * @example
 * ```ts
 * import { meanReturnTimes } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(meanReturnTimes([[0, 1], [1, 0]])) // => [2, 2]
 * ```
 */
export function meanReturnTimes(p: TensorLike): Value {
  return div(1, stationaryDistribution(p))
}

/** The result of {@link mixingTime}. */
export interface MixingTime {
  /** The first t with max_x TV(P^t(x, .), pi) <= epsilon (Infinity when the cap is reached first). */
  readonly time: number
  /** The worst-case distance d(t) for t = 0, 1, ..., time (or up to the cap). */
  readonly distances: readonly number[]
}

/**
 * The mixing time t_mix(epsilon) = min{t : max_x TV(P^t(x, .), pi) <= epsilon} by powering the matrix, with the
 * worst-case distance curve d(t). Periodic chains never mix; the search stops at `maxSteps`.
 *
 * @param p - The transition matrix (with a unique stationary law).
 * @param options - `epsilon` (default 1/4) and `maxSteps` (default 10000).
 * @returns The mixing time and the distance curve.
 * @concept mixing-time
 * @example
 * ```ts
 * import { mixingTime } from 'whitebox-engine/stochastic'
 * mixingTime([[0.5, 0.5], [0.5, 0.5]]).time // => 1
 * mixingTime([[0, 1], [1, 0]], { maxSteps: 50 }).time // => Infinity
 * ```
 */
export function mixingTime(
  p: TensorLike,
  options: { readonly epsilon?: number; readonly maxSteps?: number } = {},
): MixingTime {
  const eps = options.epsilon ?? 0.25
  const cap = options.maxSteps ?? 10000
  if (!(eps > 0 && eps < 1)) throw new WhiteboxError('value', 'mixingTime: epsilon must be in (0, 1)')
  const { n, rows } = stochasticRows(p, 'mixingTime')
  const pi = asTensor(primalNumbers(stationaryDistribution(p)))
  const m = tensorOf(rows)
  let power: Value = eye(n)
  const distances: number[] = []
  for (let t = 0; t <= cap; t++) {
    const d = toFlat(max(totalVariation(power, pi, { normalise: false })))[0] ?? 1
    distances.push(d)
    if (d <= eps) return { time: t, distances }
    power = matmul(power, m)
  }
  return { time: Number.POSITIVE_INFINITY, distances }
}

/** The spectrum summary of {@link spectralGap}. */
export interface SpectralGap {
  /** The eigenvalue moduli in decreasing order (the first is 1). */
  readonly moduli: readonly number[]
  /** 1 - lambda_2 for a reversible chain (lambda_2 the second largest eigenvalue); 1 - |lambda_2| otherwise. */
  readonly gap: number
  /** 1 - max_{i >= 2} |lambda_i|: governs the geometric rate of convergence. */
  readonly absoluteGap: number
  /** 1 / absoluteGap. */
  readonly relaxationTime: number
}

/**
 * The spectral gap and relaxation time of a chain. A reversible chain is symmetrised
 * ({@link symmetrisedTransition}) and diagonalised by `eigh`; otherwise the general eigenvalues are used.
 *
 * @param p - The transition matrix (with a unique stationary law).
 * @returns Moduli, gap, absolute gap and relaxation time.
 * @concept spectral-gap
 * @example
 * ```ts
 * import { spectralGap } from 'whitebox-engine/stochastic'
 * spectralGap([[0.75, 0.25], [0.25, 0.75]]).gap // => 0.5
 * ```
 */
export function spectralGap(p: TensorLike): SpectralGap {
  const { n, rows } = stochasticRows(p, 'spectralGap')
  if (n === 1) return { moduli: [1], gap: 1, absoluteGap: 1, relaxationTime: 1 }
  if (isReversible(p)) {
    const s = toFlat(symmetrisedTransition(p))
    const sym = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => ((s[i * n + j] ?? 0) + (s[j * n + i] ?? 0)) / 2),
    )
    const values = toFlat(eigh(tensorOf(sym)).eigenvalues).sort((a, b) => b - a)
    const absolute = 1 - Math.max(Math.abs(values[1] ?? 0), Math.abs(values[n - 1] ?? 0))
    return {
      moduli: values.map(Math.abs).sort((a, b) => b - a),
      gap: 1 - (values[1] ?? 0),
      absoluteGap: absolute,
      relaxationTime: 1 / absolute,
    }
  }
  const ev = eigvals(tensorOf(rows))
  const re = toFlat(ev.re)
  const im = toFlat(ev.im)
  const moduli = re.map((r, i) => Math.hypot(r, im[i] ?? 0)).sort((a, b) => b - a)
  const absolute = 1 - (moduli[1] ?? 0)
  return { moduli, gap: absolute, absoluteGap: absolute, relaxationTime: 1 / absolute }
}

/**
 * The symmetrised transition matrix D^{1/2} P D^{-1/2} (D = diag pi): symmetric exactly when the chain is reversible,
 * with the same eigenvalues as P. Differentiable in P.
 *
 * @param p - The transition matrix (with a unique stationary law, positive on every state).
 * @returns `[n, n]`.
 * @concept reversibility
 * @example
 * ```ts
 * import { symmetrisedTransition } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(symmetrisedTransition([[0.5, 0.5], [0.25, 0.75]])).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[0.5, 0.353553390593], [0.353553390593, 0.75]]
 * ```
 */
export function symmetrisedTransition(p: TensorLike): Value {
  const root = sqrt(stationaryDistribution(p))
  return div(mul(asFloat(asTensor(p)), expandDims(root, 1)), expandDims(root, 0))
}

/**
 * Whether a chain is reversible: detailed balance pi_i P_ij = pi_j P_ji with its stationary law.
 *
 * @param p - The transition matrix (with a unique stationary law).
 * @param tol - The tolerance (default 1e-10).
 * @returns True for a reversible chain.
 * @concept reversibility
 * @example
 * ```ts
 * import { isReversible } from 'whitebox-engine/stochastic'
 * isReversible([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]]) // => true
 * isReversible([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) // => false
 * ```
 */
export function isReversible(p: TensorLike, tol = 1e-10): boolean {
  const { n, rows } = stochasticRows(p, 'isReversible')
  const pi = primalNumbers(stationaryDistribution(p))
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs((pi[i] ?? 0) * (rows[i]?.[j] ?? 0) - (pi[j] ?? 0) * (rows[j]?.[i] ?? 0)) > tol) return false
    }
  }
  return true
}

/**
 * The time reversal P^_ij = pi_j P_ji / pi_i of a chain: equal to P exactly when the chain is reversible.
 * Differentiable in P.
 *
 * @param p - The transition matrix (irreducible).
 * @returns The reversed chain's transition matrix.
 * @concept reversibility
 * @example
 * ```ts
 * import { timeReversal } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(timeReversal([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
 * ```
 */
export function timeReversal(p: TensorLike): Value {
  const pi = stationaryDistribution(p)
  return div(mul(transpose(asFloat(asTensor(p))), expandDims(pi, 0)), expandDims(pi, 1))
}

/**
 * Simulate a discrete-time Markov chain as a traced path simulation: each step draws the next state of every path from
 * its row of P.
 *
 * @param key - The key.
 * @param p - The transition matrix.
 * @param x0 - The initial state of every path, or one per path.
 * @param steps - The number of steps.
 * @param options - The number of paths (default 1, ignored when x0 lists one state per path) and tracing.
 * @returns Paths `[paths, steps + 1]` of state indices (as floats) on the times 0, 1, ..., steps.
 * @concept markov-chains
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { simulateChain } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(simulateChain(key(0), [[0, 1], [1, 0]], 0, 4).values) // => [[0, 1, 0, 1, 0]]
 * ```
 */
export function simulateChain(
  key: Key,
  p: TensorLike,
  x0: number | readonly number[],
  steps: number,
  options: { readonly paths?: number } & TraceOptions = {},
): Paths {
  const { rows } = stochasticRows(p, 'simulateChain')
  const paths = typeof x0 === 'number' ? positiveInteger(options.paths ?? 1, 'paths', 'simulateChain') : x0.length
  const start = typeof x0 === 'number' ? new Array<number>(paths).fill(x0) : [...x0]
  for (const s of start) {
    if (!(Number.isInteger(s) && s >= 0 && s < rows.length)) throw new WhiteboxError('value', 'simulateChain: bad x0')
  }
  if (!(Number.isInteger(steps) && steps >= 0)) throw new WhiteboxError('value', 'simulateChain: steps >= 0')
  return simulatePaths(
    key,
    {
      name: 'markov-chain',
      config: { states: rows.length, paths, steps },
      times: Array.from({ length: steps + 1 }, (_, i) => i),
      x0: vector(start),
      step: (x, c) => nextStates(x, rows, c.key),
    },
    traceOf(options),
  )
}

/**
 * Draw the next state of every path from the rows of a transition matrix (one categorical draw per path).
 *
 * @param x - The current states `[paths]` (state indices as floats).
 * @param rows - The transition probabilities, one row per state.
 * @param key - The key.
 * @returns The next states `[paths]`.
 * @concept markov-chains
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { nextStates } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(nextStates([0, 1], [[0, 1], [1, 0]], key(0))) // => [1, 0]
 * ```
 */
export function nextStates(x: TensorLike, rows: readonly (readonly number[])[], key: Key): Tensor<'float64'> {
  return vector(primalNumbers(x).map((s, j) => categorical(foldIn(key, j), [...(rows[s] ?? [])], [1]).data[0] ?? 0))
}

/**
 * The empirical occupation frequencies of states along simulated paths (a check of ergodic averages).
 *
 * @param paths - Paths of states (from {@link simulateChain} or `ctmcPaths`).
 * @param n - The number of states.
 * @returns The fraction of grid points spent in each state.
 * @concept markov-chains
 * @example
 * ```ts
 * import { occupation } from 'whitebox-engine/stochastic'
 * Array.from(occupation({ times: Float64Array.from([0, 1, 2, 3]), values: [[0, 1, 1, 1]] }, 2)) // => [0.25, 0.75]
 * ```
 */
export function occupation(paths: Paths, n: number): Float64Array {
  const xs = primalNumbers(paths.values)
  const out = new Float64Array(n)
  for (const s of xs) out[s] = (out[s] ?? 0) + 1
  for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) / xs.length
  return out
}
