/**
 * Branching processes, populations in which each individual independently leaves a random number of offspring: the
 * Galton-Watson process with its generating function, moments and extinction probability (also simulated as a traced
 * algorithm), and continuous-time birth-death processes with their generator, stationary law, Kendall's closed forms
 * for the linear case, Gillespie simulation and the Yule pure-birth law.
 */

import { OmaiError } from '../base/index.ts'
import { categorical, exponential, foldIn, type Key, multinomial } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  div,
  exp,
  expm1,
  fromStorage,
  less,
  mul,
  neg,
  pow,
  promoteFloat,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { checkProbabilities, positiveInteger, primalNumbers, traceOf } from './_shared.ts'
import { type History, historyArray, type TraceOptions } from './paths.ts'

/** An offspring law on {0, 1, ..., K} as probabilities. */
export type OffspringPmf = TensorLike

function pmfOf(pmf: OffspringPmf, where: string): { p: Value; data: number[] } {
  const p = asFloat(asTensor(pmf))
  if (p.shape.length !== 1 || (p.shape[0] ?? 0) === 0) throw new OmaiError('shape', `${where}: pmf must be [K + 1]`)
  const data = primalNumbers(p)
  checkProbabilities(data, where, 1e-9)
  return { p, data }
}

/**
 * The probability generating function G(s) = sum_k p_k s^k of an offspring law (Horner's rule). Differentiable in s and
 * the probabilities.
 *
 * @param pmf - The offspring probabilities p_0, ..., p_K.
 * @param s - Points (usually in [0, 1]).
 * @returns G(s).
 * @concept generating-functions
 * @example
 * ```ts
 * import { offspringPgf } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(offspringPgf([0.25, 0.5, 0.25], 0.5)) // => 0.5625
 * ```
 */
export function offspringPgf(pmf: OffspringPmf, s: TensorLike): Value {
  const { p } = pmfOf(pmf, 'offspringPgf')
  const [x] = promoteFloat(s) as [Value]
  const k = p.shape[0] ?? 1
  let acc: Value = mul(x, 0)
  for (let i = k - 1; i >= 0; i--) acc = add(mul(acc, x), reshape(sliceAxis(p, 0, i, i + 1), []))
  return acc
}

function pgfNumbers(pk: readonly number[], s: number): number {
  let acc = 0
  for (let i = pk.length - 1; i >= 0; i--) acc = acc * s + (pk[i] ?? 0)
  return acc
}

function pgfDerivative(pk: readonly number[], s: number): number {
  let acc = 0
  for (let i = pk.length - 1; i >= 1; i--) acc = acc * s + i * (pk[i] ?? 0)
  return acc
}

/**
 * The mean m and variance sigma^2 of an offspring law (m <= 1: extinction is certain).
 *
 * @param pmf - The offspring probabilities.
 * @returns The mean and variance.
 * @concept branching-processes
 * @example
 * ```ts
 * import { offspringMoments } from 'omai-engine/stochastic'
 * offspringMoments([0.25, 0.25, 0.5]) // => { mean: 1.25, variance: 0.6875 }
 * ```
 */
export function offspringMoments(pmf: OffspringPmf): { readonly mean: number; readonly variance: number } {
  const { data } = pmfOf(pmf, 'offspringMoments')
  let m = 0
  let m2 = 0
  for (let k = 0; k < data.length; k++) {
    m += k * (data[k] ?? 0)
    m2 += k * k * (data[k] ?? 0)
  }
  return { mean: m, variance: m2 - m * m }
}

/**
 * The extinction probability of a Galton-Watson process started from one individual: the smallest root of G(s) = s in
 * [0, 1], found by iterating s <- G(s) from 0 (the probabilities P(Z_n = 0) increase to it) and polishing by Newton's
 * method. It is 1 when the mean offspring m <= 1 (unless every individual has exactly one child) and below 1 when
 * m > 1. Differentiable in the offspring probabilities by the implicit function theorem.
 *
 * @param pmf - The offspring probabilities.
 * @param options - `tol` (default 1e-14) and the iteration cap `maxIter` (default 100000).
 * @returns q.
 * @throws OmaiError with code `convergence` when the iteration does not settle within the cap.
 * @concept branching-processes
 * @example
 * ```ts
 * import { extinctionProbability } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * // p = (1/4, 1/4, 1/2): G(s) = s solves to q = 1/2
 * Math.round(item(extinctionProbability([0.25, 0.25, 0.5])) * 1e12) / 1e12 // => 0.5
 * item(extinctionProbability([0.5, 0, 0.5])) // => 1
 * ```
 */
export function extinctionProbability(
  pmf: OffspringPmf,
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): Value {
  const { p, data } = pmfOf(pmf, 'extinctionProbability')
  const tol = options.tol ?? 1e-14
  const cap = options.maxIter ?? 100000
  const { mean } = offspringMoments(data)
  let q: number
  if ((data[1] ?? 0) >= 1 - 1e-15) q = 0
  else if (mean <= 1 + 1e-12) q = 1
  else {
    q = 0
    let converged = false
    for (let i = 0; i < cap; i++) {
      const next = pgfNumbers(data, q)
      if (Math.abs(next - q) <= tol) {
        q = next
        converged = true
        break
      }
      q = next
    }
    // Newton polish on G(s) - s (G'(q) < 1 at the attracting root)
    for (let i = 0; i < 5; i++) {
      const f = pgfNumbers(data, q) - q
      const d = pgfDerivative(data, q) - 1
      if (d === 0) break
      q -= f / d
    }
    if (!converged && Math.abs(pgfNumbers(data, q) - q) > 1e-10) {
      throw new OmaiError('convergence', 'extinctionProbability: the iteration did not converge')
    }
  }
  if (q >= 1 || q <= 0) return add(mul(sum(p), 0), q)
  // One implicit Newton step with the traced pmf: the value stays q, the derivative is dG/dp / (1 - G'(q)).
  const g = offspringPgf(p, q)
  return add(q, div(sub(g, q), 1 - pgfDerivative(data, q)))
}

/**
 * The probabilities P(Z_n = 0) = G^n(0) that a Galton-Watson process started from one individual is extinct by
 * generation n, for n = 0, ..., generations. Differentiable in the probabilities.
 *
 * @param pmf - The offspring probabilities.
 * @param generations - The last generation.
 * @returns `[generations + 1]`, increasing to the extinction probability.
 * @concept branching-processes
 * @example
 * ```ts
 * import { extinctionByGeneration } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(extinctionByGeneration([0.5, 0, 0.5], 2)) // => [0, 0.5, 0.625]
 * ```
 */
export function extinctionByGeneration(pmf: OffspringPmf, generations: number): Value {
  const { p } = pmfOf(pmf, 'extinctionByGeneration')
  if (!(Number.isInteger(generations) && generations >= 0)) {
    throw new OmaiError('value', 'extinctionByGeneration: generations must be >= 0')
  }
  const out: Value[] = [mul(sum(p), 0)]
  for (let n = 0; n < generations; n++) out.push(offspringPgf(p, out[n] as Value))
  return stack(out, 0)
}

/**
 * The mean and variance of generation n of a Galton-Watson process started from z0 individuals:
 * E Z_n = z0 m^n and Var Z_n = z0 sigma^2 m^{n-1} (m^n - 1) / (m - 1) (z0 n sigma^2 when m = 1).
 *
 * @param pmf - The offspring probabilities.
 * @param n - The generation.
 * @param z0 - The initial population (default 1).
 * @returns The mean and variance.
 * @concept branching-processes
 * @example
 * ```ts
 * import { galtonWatsonMoments } from 'omai-engine/stochastic'
 * galtonWatsonMoments([0.25, 0.25, 0.5], 2) // => { mean: 1.5625, variance: 1.93359375 }
 * ```
 */
export function galtonWatsonMoments(
  pmf: OffspringPmf,
  n: number,
  z0 = 1,
): { readonly mean: number; readonly variance: number } {
  if (!(Number.isInteger(n) && n >= 0)) throw new OmaiError('value', 'galtonWatsonMoments: n must be >= 0')
  const { mean: m, variance: s2 } = offspringMoments(pmf)
  const mean = z0 * m ** n
  const variance = Math.abs(m - 1) < 1e-15 ? z0 * n * s2 : (z0 * s2 * m ** (n - 1) * (m ** n - 1)) / (m - 1)
  return { mean, variance }
}

/** The state of {@link galtonWatsonAlgorithm}: every path's population by generation. */
export interface GaltonWatsonState {
  readonly generation: number
  readonly sizes: readonly Float64Array[]
  readonly exploded: boolean
}

/**
 * A Galton-Watson branching process as a traced {@link Algorithm}: each generation every individual has an independent
 * number of children drawn from the offspring law (all at once, as multinomial counts), for many paths.
 *
 * @param pmf - The offspring probabilities.
 * @param options - Initial population (default 1), paths (default 1) and the population cap (default 1e7) beyond
 *   which a path is reported as exploded and frozen.
 * @returns The algorithm.
 * @concept branching-processes
 * @example
 * ```ts
 * import { galtonWatsonAlgorithm } from 'omai-engine/stochastic'
 * galtonWatsonAlgorithm([0.5, 0.5]).name // => 'galton-watson'
 * ```
 */
export function galtonWatsonAlgorithm(
  pmf: OffspringPmf,
  options: { readonly z0?: number; readonly paths?: number; readonly maxPopulation?: number } = {},
): Algorithm<GaltonWatsonState> {
  const { data } = pmfOf(pmf, 'galtonWatson')
  const z0 = options.z0 ?? 1
  if (!(Number.isInteger(z0) && z0 >= 0)) throw new OmaiError('value', 'galtonWatson: z0 must be an integer >= 0')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'galtonWatson')
  const cap = options.maxPopulation ?? 1e7
  return {
    name: 'galton-watson',
    config: { z0, paths, maxPopulation: cap },
    schema: channels({
      metrics: {
        mean: { description: 'mean population over paths' },
        extinct: { description: 'fraction of paths extinct' },
      },
    }),
    init: () => ({ generation: 0, sizes: [new Float64Array(paths).fill(z0)], exploded: false }),
    step: (s, ctx) => {
      const last = s.sizes[s.sizes.length - 1] ?? new Float64Array(paths)
      let exploded = s.exploded
      const next = Float64Array.from(last, (z, j) => {
        if (z === 0 || z > cap) return z
        const counts = multinomial(foldIn(ctx.key, j), z, data, 1).data
        let children = 0
        for (let k = 0; k < counts.length; k++) children += k * (counts[k] ?? 0)
        if (children > cap) exploded = true
        return children
      })
      return { generation: s.generation + 1, sizes: [...s.sizes, next], exploded }
    },
    observe: (s) => {
      const last = s.sizes[s.sizes.length - 1] ?? new Float64Array(0)
      return {
        metrics: {
          mean: last.reduce((a, b) => a + b, 0) / paths,
          extinct: last.filter((z) => z === 0).length / paths,
        },
      }
    },
  }
}

/**
 * Simulate a Galton-Watson process ({@link galtonWatsonAlgorithm}) for a number of generations.
 *
 * @param key - The key.
 * @param pmf - The offspring probabilities.
 * @param generations - The number of generations.
 * @param options - Initial population, paths, population cap and tracing.
 * @returns The population sizes `[paths, generations + 1]`.
 * @concept branching-processes
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { galtonWatson } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(galtonWatson(key(0), [0, 0, 1], 3)) // => [[1, 2, 4, 8]]
 * ```
 */
export function galtonWatson(
  key: Key,
  pmf: OffspringPmf,
  generations: number,
  options: { readonly z0?: number; readonly paths?: number; readonly maxPopulation?: number } & TraceOptions = {},
): Tensor<'float64'> {
  if (!(Number.isInteger(generations) && generations >= 0)) {
    throw new OmaiError('value', 'galtonWatson: generations must be >= 0')
  }
  const { z0, paths, maxPopulation, ...trace } = options
  const alg = galtonWatsonAlgorithm(pmf, {
    ...(z0 === undefined ? {} : { z0 }),
    ...(paths === undefined ? {} : { paths }),
    ...(maxPopulation === undefined ? {} : { maxPopulation }),
  })
  const r = runAlgorithm(alg, { ...traceOf(trace), key, maxSteps: generations })
  const p = r.state.sizes[0]?.length ?? 1
  const out = new Float64Array(p * (generations + 1))
  for (const [g, sizes] of r.state.sizes.entries())
    for (let j = 0; j < p; j++) out[j * (generations + 1) + g] = sizes[j] ?? 0
  return fromStorage(out, [p, generations + 1], 'float64')
}

/** Rates of a birth-death process: lambda_n and mu_n as functions of the population n. */
export interface BirthDeathRates {
  readonly birth: (n: number) => number
  readonly death: (n: number) => number
}

/**
 * The generator of a birth-death process truncated to {0, ..., N}: q_{n, n+1} = lambda_n, q_{n, n-1} = mu_n (no
 * births out of N), for use with the continuous-time chain functions.
 *
 * @param rates - Birth and death rates.
 * @param maxPopulation - The truncation N.
 * @returns Q `[N + 1, N + 1]`.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { birthDeathGenerator } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(birthDeathGenerator({ birth: () => 1, death: (n) => 2 * n }, 2)) // => [[-1, 1, 0], [2, -3, 1], [0, 4, -4]]
 * ```
 */
export function birthDeathGenerator(rates: BirthDeathRates, maxPopulation: number): Tensor<'float64'> {
  positiveInteger(maxPopulation, 'maxPopulation', 'birthDeathGenerator')
  const n = maxPopulation + 1
  const q = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const up = i < maxPopulation ? rates.birth(i) : 0
    const down = i > 0 ? rates.death(i) : 0
    if (!(up >= 0 && down >= 0)) throw new OmaiError('value', 'birthDeathGenerator: rates must be >= 0')
    if (i < maxPopulation) q[i * n + i + 1] = up
    if (i > 0) q[i * n + i - 1] = down
    q[i * n + i] = -(up + down)
  }
  return fromStorage(q, [n, n], 'float64')
}

/**
 * The stationary law of a birth-death process on {0, ..., N} in product form, pi_n proportional to
 * prod_{k=1}^{n} lambda_{k-1} / mu_k (detailed balance).
 *
 * @param rates - Birth and death rates (mu_n > 0 for n >= 1).
 * @param maxPopulation - The truncation N.
 * @returns pi `[N + 1]`.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { birthDeathStationary } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(birthDeathStationary({ birth: () => 1, death: () => 1 }, 3)) // => [0.25, 0.25, 0.25, 0.25]
 * ```
 */
export function birthDeathStationary(rates: BirthDeathRates, maxPopulation: number): Tensor<'float64'> {
  positiveInteger(maxPopulation, 'maxPopulation', 'birthDeathStationary')
  const logW = new Float64Array(maxPopulation + 1)
  for (let n = 1; n <= maxPopulation; n++) {
    const mu = rates.death(n)
    if (!(mu > 0)) throw new OmaiError('value', 'birthDeathStationary: death rates must be positive')
    logW[n] = (logW[n - 1] ?? 0) + Math.log(rates.birth(n - 1)) - Math.log(mu)
  }
  let top = Number.NEGATIVE_INFINITY
  for (const v of logW) top = Math.max(top, v)
  let total = 0
  for (const v of logW) total += Math.exp(v - top)
  return fromStorage(
    Float64Array.from(logW, (v) => Math.exp(v - top) / total),
    [maxPopulation + 1],
    'float64',
  )
}

/** Parameters of {@link linearBirthDeath}: per-capita birth and death rates. */
export interface LinearBirthDeathParams {
  readonly lambda: TensorLike
  readonly mu: TensorLike
}

/**
 * Closed forms of the linear birth-death process (Kendall 1948) with per-capita rates lambda and mu, started from n0:
 * E X_t = n0 e^{(lambda - mu) t}; the extinction probability by time t from one individual,
 * mu (e^{(lambda - mu) t} - 1) / (lambda e^{(lambda - mu) t} - mu) (lambda t / (1 + lambda t) when lambda = mu); and
 * the ultimate extinction probability min(1, mu / lambda)^{n0}. Differentiable in the rates and t.
 *
 * @param params - The rates.
 * @returns Functions of t and n0.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { linearBirthDeath } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * const bd = linearBirthDeath({ lambda: 2, mu: 1 })
 * item(bd.ultimateExtinction(1)) // => 0.5
 * Math.round(item(bd.extinctionBy(1e3)) * 1e12) / 1e12 // => 0.5
 * ```
 */
export function linearBirthDeath(params: LinearBirthDeathParams): {
  readonly mean: (t: TensorLike, n0?: number) => Value
  readonly extinctionBy: (t: TensorLike, n0?: number) => Value
  readonly ultimateExtinction: (n0?: number) => Value
  readonly rates: BirthDeathRates
} {
  const [lambda, mu] = promoteFloat(params.lambda, params.mu) as [Value, Value]
  const l = primalNumbers(lambda)[0] ?? 0
  const m = primalNumbers(mu)[0] ?? 0
  if (!(l >= 0 && m >= 0)) throw new OmaiError('value', 'linearBirthDeath: rates must be >= 0')
  const r = sub(lambda, mu)
  const one = (t: TensorLike): Value => {
    if (Math.abs(l - m) < 1e-14) {
      const lt = mul(lambda, t)
      return div(lt, add(lt, 1))
    }
    if (l < m) {
      const e = expm1(mul(r, t))
      return div(mul(mu, e), sub(mul(lambda, add(e, 1)), mu))
    }
    // lambda > mu: divide through by e^{(lambda - mu) t} so nothing overflows
    const decay = expm1(mul(r, neg(asTensor(t))))
    return div(mul(mu, neg(decay)), sub(lambda, mul(mu, add(decay, 1))))
  }
  return {
    mean: (t, n0 = 1) => mul(exp(mul(r, t)), n0),
    extinctionBy: (t, n0 = 1) => pow(one(t), n0),
    ultimateExtinction: (n0 = 1) => pow(where(less(mu, lambda), div(mu, lambda), 1), n0),
    rates: { birth: (n) => l * n, death: (n) => m * n },
  }
}

/** The state of {@link birthDeathAlgorithm}. */
export interface BirthDeathState {
  readonly t: number
  readonly n: number
  /** The events so far (time and population after it), newest first. */
  readonly events: History<{ readonly t: number; readonly n: number }>
  readonly finished: boolean
}

/**
 * A birth-death process on {0, 1, 2, ...} as a traced {@link Algorithm} (Gillespie): from population n wait an
 * exponential time with rate lambda_n + mu_n, then a birth with probability lambda_n / (lambda_n + mu_n), else a death,
 * until the horizon or a population with no events.
 *
 * @param rates - Birth and death rates.
 * @param n0 - The initial population.
 * @param horizon - The end time.
 * @returns The algorithm.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { birthDeathAlgorithm } from 'omai-engine/stochastic'
 * birthDeathAlgorithm({ birth: (n) => n, death: (n) => n }, 5, 1).name // => 'birth-death'
 * ```
 */
export function birthDeathAlgorithm(rates: BirthDeathRates, n0: number, horizon: number): Algorithm<BirthDeathState> {
  if (!(Number.isInteger(n0) && n0 >= 0)) throw new OmaiError('value', 'birthDeath: n0 must be an integer >= 0')
  if (!(horizon > 0)) throw new OmaiError('value', 'birthDeath: the horizon must be positive')
  return {
    name: 'birth-death',
    config: { n0, horizon },
    schema: channels({ metrics: { t: { description: 'time' }, n: { description: 'population' } } }),
    init: () => ({ t: 0, n: n0, events: { item: { t: 0, n: n0 }, previous: undefined }, finished: false }),
    step: (s, ctx) => {
      const up = rates.birth(s.n)
      const down = s.n > 0 ? rates.death(s.n) : 0
      if (!(up >= 0 && down >= 0)) throw new OmaiError('value', 'birthDeath: rates must be >= 0')
      const total = up + down
      if (total <= 0) return { ...s, finished: true }
      const t = s.t + (exponential(foldIn(ctx.key, 0), [1], total).data[0] ?? 0)
      if (t >= horizon) return { ...s, finished: true }
      const birth = (categorical(foldIn(ctx.key, 1), [up, down], [1]).data[0] ?? 0) === 0
      const n = birth ? s.n + 1 : s.n - 1
      return { t, n, events: { item: { t, n }, previous: s.events }, finished: false }
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => ({ metrics: { t: s.t, n: s.n } }),
  }
}

/**
 * Simulate a birth-death process ({@link birthDeathAlgorithm}).
 *
 * @param key - The key.
 * @param rates - Birth and death rates.
 * @param n0 - The initial population.
 * @param horizon - The end time.
 * @param options - The event cap (default 1e6) and tracing.
 * @returns The event times and the population after each event (starting with time 0 and n0).
 * @throws OmaiError with code `convergence` when the cap is reached before the horizon.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { simulateBirthDeath } from 'omai-engine/stochastic'
 * simulateBirthDeath(key(0), { birth: () => 0, death: (n) => n }, 3, 1e3).sizes.at(-1) // => 0
 * ```
 */
export function simulateBirthDeath(
  key: Key,
  rates: BirthDeathRates,
  n0: number,
  horizon: number,
  options: { readonly maxEvents?: number } & TraceOptions = {},
): { readonly times: Float64Array; readonly sizes: Int32Array } {
  const cap = options.maxEvents ?? 1e6
  const r = runAlgorithm(birthDeathAlgorithm(rates, n0, horizon), { ...traceOf(options), key, maxSteps: cap + 1 })
  if (!r.state.finished) throw new OmaiError('convergence', `simulateBirthDeath: more than ${cap} events`)
  const events = historyArray(r.state.events)
  return { times: Float64Array.from(events, (e) => e.t), sizes: Int32Array.from(events, (e) => e.n) }
}

/**
 * The Yule (pure birth) process law: started from one individual with per-capita birth rate lambda, X_t is geometric
 * on {1, 2, ...} with success probability e^{-lambda t}, P(X_t = k) = e^{-lambda t} (1 - e^{-lambda t})^{k-1}.
 * Differentiable in lambda and t.
 *
 * @param k - Population sizes (>= 1).
 * @param t - The time.
 * @param lambda - The birth rate.
 * @returns The probabilities.
 * @concept birth-death-processes
 * @example
 * ```ts
 * import { yulePmf } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(yulePmf(1, Math.LN2, 1)) // => 0.5
 * ```
 */
export function yulePmf(k: TensorLike, t: TensorLike, lambda: TensorLike): Value {
  const [kk, tt, l] = promoteFloat(k, t, lambda) as [Value, Value, Value]
  const p = exp(mul(mul(l, tt), -1))
  return mul(p, pow(sub(1, p), sub(kk, 1)))
}
