import { WhiteboxError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { type AliasTable, aliasSample, type Key, uniform } from '../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { uniforms } from './_mcmc.ts'

/** Options shared by the exact samplers: whether to return a record of each step (for teaching). */
export interface StepRecordOptions {
  /** Return the step records (default false). */
  readonly steps?: boolean
}

/** A sample with optional step records. */
export interface Recorded<S> {
  readonly samples: Tensor<'float64'>
  readonly steps?: readonly S[]
}

/** One draw of {@link inverseTransformSample}: the uniform and its image. */
export interface InverseTransformStep {
  readonly u: number
  readonly x: number
}

/**
 * Inverse-transform sampling: x = F^-1(u) with u ~ U(0, 1) has CDF F. Takes a quantile function or a distribution
 * (its `ppf`, vectorised over all the uniforms).
 *
 * @param key - The random key.
 * @param n - The number of draws.
 * @param quantile - F^-1, or a scalar distribution.
 * @param options - Whether to record the (u, x) pairs.
 * @returns The draws `[n]` and, with `steps`, each uniform and its image.
 * @throws WhiteboxError with code `value` for a negative or fractional n.
 * @concept inverse-transform-sampling
 * @example
 * ```ts
 * import { inverseTransformSample } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * // Exponential(1): F^-1(u) = -log(1 - u).
 * const r = inverseTransformSample(key(0), 3, (u) => -Math.log(1 - u), { steps: true })
 * r.steps?.every((s) => Math.abs(s.x + Math.log(1 - s.u)) < 1e-15) // => true
 * ```
 */
export function inverseTransformSample(
  key: Key,
  n: number,
  quantile: ((u: number) => number) | Distribution,
  options: StepRecordOptions = {},
): Recorded<InverseTransformStep> {
  checkN(n, 'inverseTransformSample')
  const u = uniform(key, [n]).data
  const x =
    typeof quantile === 'function'
      ? u.map((v) => quantile(v))
      : Float64Array.from(toFlat(quantile.ppf(fromStorage(Float64Array.from(u), [n], 'float64'))))
  return {
    samples: fromStorage(x, [n], 'float64'),
    ...(options.steps === true ? { steps: Array.from(u, (v, i) => ({ u: v, x: x[i] ?? Number.NaN })) } : {}),
  }
}

function checkN(n: number, name: string): void {
  if (!(Number.isInteger(n) && n >= 0)) throw new WhiteboxError('value', `${name}: n must be a non-negative integer`)
}

/** Options of {@link inverseTransformFromCdf}. */
export interface CdfInversionOptions extends StepRecordOptions {
  /** A bracket [low, high] containing the support's bulk (default: grown from [-1, 1] until it brackets u). */
  readonly bracket?: readonly [number, number]
  /** The absolute tolerance on x (default 1e-12). */
  readonly tol?: number
  /** The most bisection steps per draw (default 200). */
  readonly maxIter?: number
}

/** One draw of {@link inverseTransformFromCdf}: the uniform, the bisection iterations and the result. */
export interface CdfInversionStep {
  readonly u: number
  readonly iterations: number
  readonly x: number
}

/**
 * Inverse-transform sampling from a CDF alone: each u ~ U(0, 1) is inverted by bisection on F(x) = u, the bracket
 * first doubled outwards until it contains the root. Works for any continuous distribution whose CDF can be
 * computed; every draw costs about log2(width / tol) CDF evaluations, capped by `maxIter`.
 *
 * @param key - The random key.
 * @param n - The number of draws.
 * @param cdf - F, non-decreasing from 0 to 1.
 * @param options - Bracket, tolerance, iteration cap and step records.
 * @returns The draws `[n]` and, with `steps`, the uniform and iteration count of each.
 * @concept inverse-transform-sampling
 * @example
 * ```ts
 * import { inverseTransformFromCdf } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * // The logistic distribution: F(x) = 1 / (1 + e^-x), F^-1(u) = log(u / (1 - u)).
 * const r = inverseTransformFromCdf(key(0), 4, (x) => 1 / (1 + Math.exp(-x)), { steps: true })
 * r.steps?.every((s) => Math.abs(s.x - Math.log(s.u / (1 - s.u))) < 1e-9) // => true
 * ```
 */
export function inverseTransformFromCdf(
  key: Key,
  n: number,
  cdf: (x: number) => number,
  options: CdfInversionOptions = {},
): Recorded<CdfInversionStep> {
  checkN(n, 'inverseTransformFromCdf')
  const tol = options.tol ?? 1e-12
  const maxIter = options.maxIter ?? 200
  const u = uniform(key, [n]).data
  const steps: CdfInversionStep[] = []
  const x = u.map((v) => {
    let [a, b] = options.bracket ?? [-1, 1]
    let iterations = 0
    while (cdf(a) > v && iterations < maxIter) {
      a -= 2 * (b - a)
      iterations++
    }
    while (cdf(b) < v && iterations < maxIter) {
      b += 2 * (b - a)
      iterations++
    }
    while (b - a > tol && iterations < maxIter) {
      const m = 0.5 * (a + b)
      if (cdf(m) < v) a = m
      else b = m
      iterations++
    }
    const r = 0.5 * (a + b)
    if (options.steps === true) steps.push({ u: v, iterations, x: r })
    return r
  })
  return { samples: fromStorage(x, [n], 'float64'), ...(options.steps === true ? { steps } : {}) }
}

/** One draw of {@link discreteInverseTransform}: the uniform, the categories scanned and the result. */
export interface DiscreteInverseStep {
  readonly u: number
  readonly comparisons: number
  readonly index: number
}

/**
 * Discrete inverse-transform sampling by sequential search: the first index whose cumulative probability exceeds u,
 * the textbook method (expected comparisons 1 + E[index]; sort the categories by decreasing probability to speed it
 * up, or use the alias method for O(1) draws).
 *
 * @param key - The random key.
 * @param n - The number of draws.
 * @param probabilities - Non-negative weights (normalised here).
 * @param options - Whether to record each search.
 * @returns The indices `[n]` (as float64) and, with `steps`, each uniform and its search.
 * @throws WhiteboxError with code `value` for negative weights or weights summing to zero.
 * @concept inverse-transform-sampling
 * @example
 * ```ts
 * import { discreteInverseTransform } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * Array.from(discreteInverseTransform(key(0), 5, [0, 0, 1]).samples.data) // => [2, 2, 2, 2, 2]
 * ```
 */
export function discreteInverseTransform(
  key: Key,
  n: number,
  probabilities: readonly number[],
  options: StepRecordOptions = {},
): Recorded<DiscreteInverseStep> {
  checkN(n, 'discreteInverseTransform')
  const total = probabilities.reduce((a, b) => a + b, 0)
  if (!probabilities.every((p) => p >= 0) || !(total > 0)) {
    throw new WhiteboxError('value', 'discreteInverseTransform: weights must be non-negative with a positive sum')
  }
  const u = uniform(key, [n]).data
  const steps: DiscreteInverseStep[] = []
  const out = u.map((v) => {
    let c = 0
    let i = 0
    for (; i < probabilities.length - 1; i++) {
      c += (probabilities[i] ?? 0) / total
      if (v < c) break
    }
    if (options.steps === true) steps.push({ u: v, comparisons: i + 1, index: i })
    return i
  })
  return { samples: fromStorage(out, [n], 'float64'), ...(options.steps === true ? { steps } : {}) }
}

/** One pair of {@link boxMuller}. */
export interface BoxMullerStep {
  readonly u1: number
  readonly u2: number
  /** The radius sqrt(-2 log u1) and angle 2 pi u2. */
  readonly r: number
  readonly theta: number
  readonly z1: number
  readonly z2: number
}

/**
 * The Box-Muller transform: from independent u1, u2 ~ U(0, 1), z1 = r cos(theta) and z2 = r sin(theta) with
 * r = sqrt(-2 log u1) and theta = 2 pi u2 are independent standard normals (the polar form of a 2-D Gaussian: the
 * squared radius is exponential and the angle uniform). The engine's own normal sampler (random.normal) uses it.
 *
 * @param key - The random key.
 * @param n - The number of normals.
 * @param options - Whether to record each pair.
 * @returns The draws `[n]` and, with `steps`, each pair's uniforms, radius, angle and outputs.
 * @concept box-muller
 * @reference G. E. P. Box and M. E. Muller, A note on the generation of random normal deviates (Annals of
 *   Mathematical Statistics 1958).
 * @example
 * ```ts
 * import { boxMuller } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const r = boxMuller(key(0), 2, { steps: true })
 * const s = r.steps?.[0]
 * Math.abs((s?.z1 ?? 0) ** 2 + (s?.z2 ?? 0) ** 2 + 2 * Math.log(1 - (s?.u1 ?? 0))) < 1e-12 // => true
 * ```
 */
export function boxMuller(key: Key, n: number, options: StepRecordOptions = {}): Recorded<BoxMullerStep> {
  checkN(n, 'boxMuller')
  const pairs = Math.ceil(n / 2)
  const u = uniform(key, [2 * pairs]).data
  const out = new Float64Array(n)
  const steps: BoxMullerStep[] = []
  for (let i = 0; i < pairs; i++) {
    const u1 = u[2 * i] ?? 0
    const u2 = u[2 * i + 1] ?? 0
    // 1 - u1 lies in (0, 1], so the logarithm is finite.
    const r = Math.sqrt(-2 * Math.log(1 - u1))
    const theta = 2 * Math.PI * u2
    const z1 = r * Math.cos(theta)
    const z2 = r * Math.sin(theta)
    out[2 * i] = z1
    if (2 * i + 1 < n) out[2 * i + 1] = z2
    if (options.steps === true) steps.push({ u1, u2, r, theta, z1, z2 })
  }
  return { samples: fromStorage(out, [n], 'float64'), ...(options.steps === true ? { steps } : {}) }
}

/** One proposal of {@link marsagliaPolar}. */
export interface PolarStep {
  readonly v1: number
  readonly v2: number
  /** s = v1^2 + v2^2; accepted when 0 < s < 1. */
  readonly s: number
  readonly accepted: boolean
}

/**
 * Marsaglia's polar method: draw (v1, v2) uniform on the square [-1, 1]^2 until s = v1^2 + v2^2 lies in (0, 1),
 * then v1 sqrt(-2 log s / s) and v2 sqrt(-2 log s / s) are independent standard normals. It avoids trigonometric
 * functions at the cost of rejecting 1 - pi / 4 (about 21%) of the pairs.
 *
 * @param key - The random key.
 * @param n - The number of normals.
 * @param options - Whether to record every proposal (accepted or not).
 * @returns The draws `[n]` and, with `steps`, every proposed pair.
 * @concept box-muller
 * @reference G. Marsaglia and T. A. Bray, A convenient method for generating normal variables (SIAM Review 1964).
 * @example
 * ```ts
 * import { marsagliaPolar } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const r = marsagliaPolar(key(0), 2000, { steps: true })
 * const rate = (r.steps ?? []).filter((s) => s.accepted).length / (r.steps?.length ?? 1)
 * Math.abs(rate - Math.PI / 4) < 0.04 // => true
 * ```
 */
export function marsagliaPolar(key: Key, n: number, options: StepRecordOptions = {}): Recorded<PolarStep> {
  checkN(n, 'marsagliaPolar')
  const rand = uniforms(key)
  const out = new Float64Array(n)
  const steps: PolarStep[] = []
  let k = 0
  // Each pair is accepted with probability pi / 4; the cap only guards against a broken uniform stream.
  for (let tries = 0; k < n && tries < 100 * (n + 10); tries++) {
    const v1 = 2 * rand() - 1
    const v2 = 2 * rand() - 1
    const s = v1 * v1 + v2 * v2
    const accepted = s > 0 && s < 1
    if (options.steps === true) steps.push({ v1, v2, s, accepted })
    if (!accepted) continue
    const f = Math.sqrt((-2 * Math.log(s)) / s)
    out[k++] = v1 * f
    if (k < n) out[k++] = v2 * f
  }
  return { samples: fromStorage(out, [n], 'float64'), ...(options.steps === true ? { steps } : {}) }
}

/** One step of Vose's alias-table construction. */
export interface AliasStep {
  /** The under-full column filled, and the over-full column that donates its top. */
  readonly small: number
  readonly large: number
  /** The small column's probability, and the large column's scaled weight after donating. */
  readonly probability: number
  readonly remaining: number
}

/**
 * Vose's alias method with the construction recorded: scale the weights to mean 1, then repeatedly fill an
 * under-full column (scaled weight < 1) from an over-full one, which becomes its alias. Each column then holds at most
 * two categories, so a draw costs one uniform column and one biased coin. The table is the one `random.aliasTable`
 * builds, and the draws come from `random.aliasSample`.
 *
 * @param key - The random key.
 * @param n - The number of draws.
 * @param weights - Non-negative weights.
 * @param options - Whether to return the construction steps.
 * @returns The draws `[n]` (as float64), the table and, with `steps`, the construction.
 * @throws WhiteboxError with code `value` for negative weights or weights summing to zero.
 * @concept alias-method
 * @reference M. D. Vose, A linear algorithm for generating random numbers with a given distribution (IEEE TSE 1991).
 * @example
 * ```ts
 * import { aliasMethod } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const r = aliasMethod(key(0), 10, [1, 3], { steps: true })
 * Array.from(r.table.alias) // => [1, 1]
 * r.steps?.length // => 1
 * ```
 */
export function aliasMethod(
  key: Key,
  n: number,
  weights: readonly number[],
  options: StepRecordOptions = {},
): Recorded<AliasStep> & { readonly table: AliasTable } {
  checkN(n, 'aliasMethod')
  const k = weights.length
  const total = weights.reduce((a, b) => a + b, 0)
  if (k === 0 || !weights.every((w) => w >= 0) || !(total > 0)) {
    throw new WhiteboxError('value', 'aliasMethod: weights must be non-negative with a positive sum')
  }
  const scaled = weights.map((w) => (w * k) / total)
  const probability = new Float64Array(k)
  const alias = new Int32Array(k)
  const small: number[] = []
  const large: number[] = []
  scaled.forEach((s, i) => {
    ;(s < 1 ? small : large).push(i)
  })
  const steps: AliasStep[] = []
  while (small.length > 0 && large.length > 0) {
    const s = small.pop() ?? 0
    const l = large.pop() ?? 0
    probability[s] = scaled[s] ?? 0
    alias[s] = l
    scaled[l] = (scaled[l] ?? 0) + (scaled[s] ?? 0) - 1
    steps.push({ small: s, large: l, probability: probability[s] ?? 0, remaining: scaled[l] ?? 0 })
    ;((scaled[l] ?? 0) < 1 ? small : large).push(l)
  }
  for (const i of [...small, ...large]) {
    probability[i] = 1
    alias[i] = i
  }
  const table: AliasTable = { probability, alias }
  const draws = Float64Array.from(aliasSample(key, table, [n]).data)
  return { samples: fromStorage(draws, [n], 'float64'), table, ...(options.steps === true ? { steps } : {}) }
}
