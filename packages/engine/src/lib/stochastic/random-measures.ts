/**
 * Random measures and the combinatorial processes of Bayesian nonparametrics: stick-breaking weights and truncated
 * Dirichlet and Pitman-Yor processes, the Chinese restaurant process with its expected table count and partition
 * probabilities, the Polya urn, the Indian buffet process for latent feature allocations, and beta and gamma processes.
 */

import { OmaiError } from '../base/index.ts'
import { beta as betaLaw, type Distribution, gamma as gammaLaw } from '../dists/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { categorical, foldIn, type Key, poisson, uniform } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  cumprod,
  div,
  exp,
  fromStorage,
  log,
  mul,
  promoteFloat,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { positiveInteger, primalNumbers, traceOf } from './_shared.ts'
import { type History, historyArray, type Paths, simulatePaths, type TraceOptions } from './paths.ts'

/** A discrete random measure: atoms with weights (summing to at most one for a truncated probability measure). */
export interface DiscreteMeasure {
  readonly atoms: Value
  readonly weights: Value
}

function checkConcentration(alpha: number, discount: number, where: string): void {
  if (!(discount >= 0 && discount < 1)) throw new OmaiError('value', `${where}: the discount must be in [0, 1)`)
  if (!(alpha > -discount)) throw new OmaiError('value', `${where}: need alpha > -discount`)
}

/**
 * Stick-breaking weights (Sethuraman): V_k ~ Beta(1 - d, alpha + k d), w_k = V_k prod_{j < k} (1 - V_j), the GEM law of
 * the Dirichlet process (d = 0) or the Pitman-Yor process (discount d), truncated to K sticks. Reparameterised (implicit
 * gradients of the beta draws), so the weights differentiate in alpha.
 *
 * @param key - The key.
 * @param alpha - The concentration.
 * @param truncation - The number of sticks K.
 * @param options - The discount d (default 0).
 * @returns The K weights; 1 - sum is the mass left beyond the truncation.
 * @concept dirichlet-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { stickBreaking } from 'omai-engine/stochastic'
 * import { sum, item } from 'omai-engine/tensor'
 * item(sum(stickBreaking(key(0), 1, 200))) > 0.999 // => true
 * ```
 */
export function stickBreaking(
  key: Key,
  alpha: TensorLike,
  truncation: number,
  options: { readonly discount?: number } = {},
): Value {
  positiveInteger(truncation, 'truncation', 'stickBreaking')
  const [a] = promoteFloat(alpha) as [Value]
  const d = options.discount ?? 0
  checkConcentration(primalNumbers(a)[0] ?? Number.NaN, d, 'stickBreaking')
  const ks = fromStorage(
    Float64Array.from({ length: truncation }, (_, k) => k + 1),
    [truncation],
    'float64',
  )
  const v = betaLaw(add(mul(ks, 0), 1 - d), add(a, mul(ks, d))).rsample(key, [])
  const left = cumprod(sub(1, v), 0)
  const before = concat([fromStorage(Float64Array.of(1), [1], 'float64'), sliceAxis(left, 0, 0, truncation - 1)], 0)
  return mul(v, before)
}

/**
 * A draw from a Dirichlet process DP(alpha, H) (or a Pitman-Yor process with a discount), truncated: stick-breaking
 * weights on atoms drawn independently from the base measure H.
 *
 * @param key - The key.
 * @param alpha - The concentration.
 * @param base - The base measure H.
 * @param truncation - The number of atoms.
 * @param options - The discount (default 0).
 * @returns The atoms and weights.
 * @concept dirichlet-process
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { dirichletProcess } from 'omai-engine/stochastic'
 * dirichletProcess(key(0), 2, normal(0, 1), 50).atoms.shape // => [50]
 * ```
 */
export function dirichletProcess(
  key: Key,
  alpha: TensorLike,
  base: Distribution,
  truncation: number,
  options: { readonly discount?: number } = {},
): DiscreteMeasure {
  return {
    atoms: base.sample(foldIn(key, 1), [truncation]),
    weights: stickBreaking(foldIn(key, 0), alpha, truncation, options),
  }
}

/**
 * Draw values from a discrete random measure (normalising its weights).
 *
 * @param key - The key.
 * @param measure - Atoms and weights.
 * @param n - The number of draws.
 * @returns The drawn atoms `[n, ...]`.
 * @concept dirichlet-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { sampleFromMeasure } from 'omai-engine/stochastic'
 * import { toArray, tensor } from 'omai-engine/tensor'
 * toArray(sampleFromMeasure(key(0), { atoms: tensor([5, 7]), weights: tensor([1, 0]) }, 3)) // => [5, 5, 5]
 * ```
 */
export function sampleFromMeasure(key: Key, measure: DiscreteMeasure, n: number): Tensor<'float64'> {
  positiveInteger(n, 'n', 'sampleFromMeasure')
  const w = primalNumbers(measure.weights)
  const atoms = primalNumbers(measure.atoms)
  const per = atoms.length / Math.max(1, w.length)
  const idx = categorical(key, w, [n]).data
  const out = new Float64Array(n * per)
  for (let i = 0; i < n; i++) for (let j = 0; j < per; j++) out[i * per + j] = atoms[(idx[i] ?? 0) * per + j] ?? 0
  const shape = asTensor(measure.atoms).shape
  return fromStorage(out, [n, ...shape.slice(1)], 'float64')
}

/** The state of {@link chineseRestaurantAlgorithm}: table sizes and every customer's table. */
export interface RestaurantState {
  readonly tables: readonly number[]
  readonly seating: History<number> | undefined
  readonly customers: number
}

/**
 * The Chinese restaurant process as a traced {@link Algorithm} (one customer per step): customer n + 1 joins table k
 * with probability (n_k - d) / (n + alpha) and opens a new table with probability (alpha + K d) / (n + alpha) (the
 * Pitman-Yor two-parameter process; d = 0 is the Dirichlet process's Ewens partition).
 *
 * @param alpha - The concentration.
 * @param options - The discount (default 0).
 * @returns The algorithm.
 * @concept chinese-restaurant-process
 * @example
 * ```ts
 * import { chineseRestaurantAlgorithm } from 'omai-engine/stochastic'
 * chineseRestaurantAlgorithm(1).name // => 'chinese-restaurant'
 * ```
 */
export function chineseRestaurantAlgorithm(
  alpha: number,
  options: { readonly discount?: number } = {},
): Algorithm<RestaurantState> {
  const d = options.discount ?? 0
  checkConcentration(alpha, d, 'chineseRestaurant')
  return {
    name: 'chinese-restaurant',
    config: { alpha, discount: d },
    schema: channels({ metrics: { tables: { description: 'occupied tables' } } }),
    init: () => ({ tables: [], seating: undefined, customers: 0 }),
    step: (s, ctx) => {
      const k = s.tables.length
      const weights = [...s.tables.map((c) => c - d), alpha + k * d]
      const table = s.customers === 0 ? 0 : (categorical(ctx.key, weights, [1]).data[0] ?? k)
      const tables = table === k ? [...s.tables, 1] : s.tables.map((c, i) => (i === table ? c + 1 : c))
      return { tables, seating: { item: table, previous: s.seating }, customers: s.customers + 1 }
    },
    observe: (s) => ({ metrics: { tables: s.tables.length } }),
  }
}

/**
 * Seat n customers by the Chinese restaurant process ({@link chineseRestaurantAlgorithm}).
 *
 * @param key - The key.
 * @param alpha - The concentration.
 * @param n - The number of customers.
 * @param options - The discount and tracing.
 * @returns Each customer's table (in order of opening) and the table sizes.
 * @concept chinese-restaurant-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { chineseRestaurant } from 'omai-engine/stochastic'
 * const r = chineseRestaurant(key(0), 1, 20)
 * r.sizes.reduce((a, b) => a + b, 0) // => 20
 * ```
 */
export function chineseRestaurant(
  key: Key,
  alpha: number,
  n: number,
  options: { readonly discount?: number } & TraceOptions = {},
): { readonly tables: Int32Array; readonly sizes: Int32Array } {
  positiveInteger(n, 'n', 'chineseRestaurant')
  const alg = chineseRestaurantAlgorithm(alpha, options.discount === undefined ? {} : { discount: options.discount })
  const r = runAlgorithm(alg, { ...traceOf(options), key, maxSteps: n })
  return { tables: Int32Array.from(historyArray(r.state.seating)), sizes: Int32Array.from(r.state.tables) }
}

/**
 * The expected number of tables after n customers: sum_{i<n} alpha / (alpha + i) for the Dirichlet process (about
 * alpha log n) and (alpha / d)(Gamma(alpha + d + n) Gamma(alpha) / (Gamma(alpha + d) Gamma(alpha + n)) - 1) for the
 * Pitman-Yor process (growing like n^d). Differentiable in alpha.
 *
 * @param alpha - The concentration.
 * @param n - The number of customers.
 * @param options - The discount (default 0).
 * @returns E K_n.
 * @concept chinese-restaurant-process
 * @example
 * ```ts
 * import { expectedTables } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(expectedTables(1, 3)) // => 1.8333333333333333
 * ```
 */
export function expectedTables(alpha: TensorLike, n: number, options: { readonly discount?: number } = {}): Value {
  positiveInteger(n, 'n', 'expectedTables')
  const [a] = promoteFloat(alpha) as [Value]
  const d = options.discount ?? 0
  checkConcentration(primalNumbers(a)[0] ?? Number.NaN, d, 'expectedTables')
  if (d === 0) {
    const i = fromStorage(
      Float64Array.from({ length: n }, (_, k) => k),
      [n],
      'float64',
    )
    return sum(div(a, add(a, i)))
  }
  const ratio = exp(sub(add(lgamma(add(add(a, d), n)), lgamma(a)), add(lgamma(add(a, d)), lgamma(add(a, n)))))
  return mul(div(a, d), sub(ratio, 1))
}

/**
 * The exchangeable partition probability function of the Chinese restaurant (Pitman-Yor) process: the probability of
 * a particular seating with table sizes n_1, ..., n_K, prod_{i<K} (alpha + i d) / (alpha + 1)_{n-1} prod_k (1 - d)_{n_k
 * - 1} (Ewens's formula alpha^K Gamma(alpha) / Gamma(alpha + n) prod_k Gamma(n_k) when d = 0). Differentiable in alpha.
 *
 * @param sizes - The table sizes.
 * @param alpha - The concentration.
 * @param options - The discount (default 0).
 * @returns The log probability.
 * @concept chinese-restaurant-process
 * @example
 * ```ts
 * import { partitionLogProbability } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * // two customers at one table: 1 / (1 + alpha)
 * Math.round(Math.exp(item(partitionLogProbability([2], 1))) * 1e12) / 1e12 // => 0.5
 * ```
 */
export function partitionLogProbability(
  sizes: readonly number[],
  alpha: TensorLike,
  options: { readonly discount?: number } = {},
): Value {
  const [a] = promoteFloat(alpha) as [Value]
  const d = options.discount ?? 0
  checkConcentration(primalNumbers(a)[0] ?? Number.NaN, d, 'partitionLogProbability')
  for (const s of sizes) if (!(Number.isInteger(s) && s >= 1)) throw new OmaiError('value', 'partition: bad size')
  const n = sizes.reduce((x, y) => x + y, 0)
  const k = sizes.length
  // log prod_{i=1}^{K-1} (alpha + i d)
  let top: Value = mul(a, 0)
  for (let i = 1; i < k; i++) top = add(top, log(add(a, i * d)))
  // log rising factorial (alpha + 1)_{n-1} = lgamma(alpha + n) - lgamma(alpha + 1)
  const bottom = sub(lgamma(add(a, n)), lgamma(add(a, 1)))
  let tables = 0
  for (const s of sizes) tables += (primalNumbers(lgamma(s - d))[0] ?? 0) - (primalNumbers(lgamma(1 - d))[0] ?? 0)
  return add(sub(top, bottom), tables)
}

/** The state of {@link polyaUrnAlgorithm}. */
export interface UrnState {
  readonly draws: History<number> | undefined
  readonly count: number
}

/**
 * The Blackwell-MacQueen Polya urn as a traced {@link Algorithm}: draw n + 1 is a fresh draw from the base law with
 * probability alpha / (alpha + n) and otherwise repeats one of the previous draws chosen uniformly; the draws are
 * exchangeable with the law of i.i.d. samples from a DP(alpha, H) draw.
 *
 * @param alpha - The concentration.
 * @param base - The base law (scalar).
 * @returns The algorithm.
 * @concept polya-urn
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { polyaUrnAlgorithm } from 'omai-engine/stochastic'
 * polyaUrnAlgorithm(1, normal(0, 1)).name // => 'polya-urn'
 * ```
 */
export function polyaUrnAlgorithm(alpha: number, base: Distribution): Algorithm<UrnState> {
  checkConcentration(alpha, 0, 'polyaUrn')
  return {
    name: 'polya-urn',
    config: { alpha },
    init: () => ({ draws: undefined, count: 0 }),
    step: (s, ctx) => {
      const u = (uniform(foldIn(ctx.key, 0), [1]).data[0] ?? 0) * (alpha + s.count)
      let value: number
      if (u < alpha || s.draws === undefined) value = primalNumbers(base.sample(foldIn(ctx.key, 1), [1]))[0] ?? 0
      else {
        // repeat draw number floor((u - alpha)) counted from the oldest: walk back from the newest
        const back = s.count - 1 - Math.min(s.count - 1, Math.floor(u - alpha))
        let h: History<number> | undefined = s.draws
        for (let i = 0; i < back && h !== undefined; i++) h = h.previous
        value = h?.item ?? 0
      }
      return { draws: { item: value, previous: s.draws }, count: s.count + 1 }
    },
  }
}

/**
 * Draw n values by the Polya urn ({@link polyaUrnAlgorithm}).
 *
 * @param key - The key.
 * @param alpha - The concentration.
 * @param base - The base law.
 * @param n - The number of draws.
 * @param options - Tracing.
 * @returns The draws (ties mark shared atoms).
 * @concept polya-urn
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { polyaUrn } from 'omai-engine/stochastic'
 * new Set(polyaUrn(key(0), 1e-9, normal(0, 1), 10)).size // => 1
 * ```
 */
export function polyaUrn(
  key: Key,
  alpha: number,
  base: Distribution,
  n: number,
  options: TraceOptions = {},
): Float64Array {
  positiveInteger(n, 'n', 'polyaUrn')
  const r = runAlgorithm(polyaUrnAlgorithm(alpha, base), { ...traceOf(options), key, maxSteps: n })
  return Float64Array.from(historyArray(r.state.draws))
}

/** The state of {@link indianBuffetAlgorithm}: the dish counts and every customer's dishes. */
export interface BuffetState {
  readonly counts: readonly number[]
  readonly rows: History<readonly number[]> | undefined
  readonly customers: number
}

/**
 * The Indian buffet process (Griffiths and Ghahramani) as a traced {@link Algorithm}: customer i takes each dish
 * already tried by m_k customers with probability m_k / i and then Poisson(alpha / i) new dishes; the binary matrix is
 * the latent feature allocation of the beta process.
 *
 * @param alpha - The mass parameter.
 * @returns The algorithm.
 * @concept indian-buffet-process
 * @example
 * ```ts
 * import { indianBuffetAlgorithm } from 'omai-engine/stochastic'
 * indianBuffetAlgorithm(2).name // => 'indian-buffet'
 * ```
 */
export function indianBuffetAlgorithm(alpha: number): Algorithm<BuffetState> {
  if (!(alpha > 0)) throw new OmaiError('value', 'indianBuffet: alpha must be positive')
  return {
    name: 'indian-buffet',
    config: { alpha },
    schema: channels({ metrics: { dishes: { description: 'dishes tried so far' } } }),
    init: () => ({ counts: [], rows: undefined, customers: 0 }),
    step: (s, ctx) => {
      const i = s.customers + 1
      const u = uniform(foldIn(ctx.key, 0), [Math.max(1, s.counts.length)]).data
      const taken = s.counts.map((m, k) => ((u[k] ?? 1) < m / i ? 1 : 0))
      const fresh = poisson(foldIn(ctx.key, 1), alpha / i, [1]).data[0] ?? 0
      const row = [...taken, ...new Array<number>(fresh).fill(1)]
      const counts = [...s.counts.map((m, k) => m + (taken[k] ?? 0)), ...new Array<number>(fresh).fill(1)]
      return { counts, rows: { item: row, previous: s.rows }, customers: i }
    },
    observe: (s) => ({ metrics: { dishes: s.counts.length } }),
  }
}

/**
 * The feature matrix of n customers of the Indian buffet process ({@link indianBuffetAlgorithm}).
 *
 * @param key - The key.
 * @param alpha - The mass parameter.
 * @param n - The number of customers.
 * @param options - Tracing.
 * @returns A binary matrix `[n, dishes]` in order of first tasting (left-ordered).
 * @concept indian-buffet-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { indianBuffet } from 'omai-engine/stochastic'
 * indianBuffet(key(0), 2, 5).shape[0] // => 5
 * ```
 */
export function indianBuffet(key: Key, alpha: number, n: number, options: TraceOptions = {}): Tensor<'float64'> {
  positiveInteger(n, 'n', 'indianBuffet')
  const r = runAlgorithm(indianBuffetAlgorithm(alpha), { ...traceOf(options), key, maxSteps: n })
  const rows = historyArray(r.state.rows)
  const k = r.state.counts.length
  const out = new Float64Array(n * k)
  for (const [i, row] of rows.entries()) for (const [j, v] of row.entries()) out[i * k + j] = v
  return fromStorage(out, [n, k], 'float64')
}

/**
 * The expected number of dishes after n customers of the Indian buffet process, alpha H_n (H_n the harmonic number);
 * each customer takes Poisson(alpha) dishes.
 *
 * @param alpha - The mass parameter.
 * @param n - The number of customers.
 * @returns E K_n.
 * @concept indian-buffet-process
 * @example
 * ```ts
 * import { expectedDishes } from 'omai-engine/stochastic'
 * expectedDishes(2, 2) // => 3
 * ```
 */
export function expectedDishes(alpha: number, n: number): number {
  positiveInteger(n, 'n', 'expectedDishes')
  let h = 0
  for (let i = 1; i <= n; i++) h += 1 / i
  return alpha * h
}

/**
 * A draw from a beta process BP(c, alpha H), truncated to K atoms: for c = 1 by the stick-breaking construction of the
 * Indian buffet (Teh, Gorur and Ghahramani 2007), pi_k = prod_{j <= k} nu_j with nu_j ~ Beta(alpha, 1), decreasing;
 * otherwise by the finite approximation pi_k ~ Beta(c alpha / K, c (1 - alpha / K)). The weights are feature
 * probabilities (they need not sum to one).
 *
 * @param key - The key.
 * @param params - The mass alpha and the concentration c (default 1).
 * @param base - The base law of the atoms.
 * @param truncation - K.
 * @returns The atoms and weights.
 * @concept beta-process
 * @example
 * ```ts
 * import { uniform } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { betaProcess } from 'omai-engine/stochastic'
 * betaProcess(key(0), { mass: 2 }, uniform(0, 1), 30).weights.shape // => [30]
 * ```
 */
export function betaProcess(
  key: Key,
  params: { readonly mass: number; readonly concentration?: number },
  base: Distribution,
  truncation: number,
): DiscreteMeasure {
  positiveInteger(truncation, 'truncation', 'betaProcess')
  const alpha = params.mass
  const c = params.concentration ?? 1
  if (!(alpha > 0 && c > 0)) throw new OmaiError('value', 'betaProcess: mass and concentration must be positive')
  let weights: Value
  if (c === 1) {
    const nu = betaLaw(alpha, 1).rsample(foldIn(key, 0), [truncation])
    weights = cumprod(nu, 0)
  } else {
    if (!(alpha < truncation)) throw new OmaiError('value', 'betaProcess: the truncation must exceed the mass')
    weights = betaLaw((c * alpha) / truncation, c * (1 - alpha / truncation)).rsample(foldIn(key, 0), [truncation])
  }
  return { atoms: base.sample(foldIn(key, 1), [truncation]), weights }
}

/**
 * A draw from a gamma process (a completely random measure with Levy measure alpha x^{-1} e^{-x / scale} dx H), by the
 * finite approximation of K atoms with weights Gamma(alpha / K, scale): the total mass is exactly Gamma(alpha, scale)
 * and the normalised weights are Dirichlet(alpha / K, ...), which converges to DP(alpha, H) as K grows.
 *
 * @param key - The key.
 * @param params - The concentration alpha and the scale (default 1).
 * @param base - The base law of the atoms.
 * @param truncation - K.
 * @returns The atoms and (unnormalised) weights.
 * @concept gamma-process
 * @example
 * ```ts
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { gammaProcess } from 'omai-engine/stochastic'
 * gammaProcess(key(0), { alpha: 3 }, normal(0, 1), 100).weights.shape // => [100]
 * ```
 */
export function gammaProcess(
  key: Key,
  params: { readonly alpha: TensorLike; readonly scale?: TensorLike },
  base: Distribution,
  truncation: number,
): DiscreteMeasure {
  positiveInteger(truncation, 'truncation', 'gammaProcess')
  const [a, s] = promoteFloat(params.alpha, params.scale ?? 1) as [Value, Value]
  const weights = mul(gammaLaw(div(a, truncation), 1).rsample(foldIn(key, 0), [truncation]), s)
  return { atoms: base.sample(foldIn(key, 1), [truncation]), weights }
}

/**
 * The gamma process as a subordinator in time: G_t with independent Gamma(rate dt, scale) increments (mean rate scale t,
 * variance rate scale^2 t), simulated exactly on a grid by implicitly reparameterised gamma draws (the clock of the
 * variance gamma process).
 *
 * @param key - The key.
 * @param times - The grid.
 * @param params - The rate (shape per unit time), scale (default 1), paths and tracing.
 * @returns Increasing paths `[paths, times.length]` from 0.
 * @concept gamma-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { gammaSubordinator, timeGrid } from 'omai-engine/stochastic'
 * gammaSubordinator(key(0), timeGrid(1, 5), { rate: 2, paths: 3 }).values.shape // => [3, 6]
 * ```
 */
export function gammaSubordinator(
  key: Key,
  times: ArrayLike<number>,
  params: { readonly rate: TensorLike; readonly scale?: TensorLike; readonly paths?: number } & TraceOptions,
): Paths {
  const [rate, scale] = promoteFloat(params.rate, params.scale ?? 1) as [Value, Value]
  const paths = positiveInteger(params.paths ?? 1, 'paths', 'gammaSubordinator')
  return simulatePaths(
    key,
    {
      name: 'gamma-subordinator',
      config: { paths },
      times: Float64Array.from(times),
      x0: fromStorage(new Float64Array(paths), [paths], 'float64'),
      step: (x, c) => add(x, mul(gammaLaw(mul(rate, c.dt), 1).rsample(c.key, [paths]), scale)),
    },
    traceOf(params),
  )
}
