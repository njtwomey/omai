import { jvp } from '../autodiff/index.ts'
import { kahanSum, WhiteboxError } from '../base/index.ts'
import { convolve, fftConvolve } from '../numerics/convolution/index.ts'
import { abs, mul, onesLike, promoteFloat, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

/** A finite probability space: outcomes with their probabilities (non-negative, summing to one). */
export interface ProbabilitySpace<T> {
  readonly outcomes: readonly T[]
  readonly probabilities: readonly number[]
}

/** An event: a predicate on outcomes. */
export type Event<T> = (outcome: T) => boolean

/** A discrete random variable: a number for each outcome. */
export type RandomVariable<T> = (outcome: T) => number

/**
 * A finite probability space from outcomes and weights (normalised to sum to one; equally likely when omitted).
 *
 * @param outcomes - The outcomes.
 * @param weights - Non-negative weights, one per outcome (default all equal).
 * @returns The space.
 * @throws WhiteboxError with code `value` for negative weights, a length mismatch or zero total.
 * @concept probability-spaces
 * @example
 * ```ts
 * import { probabilitySpace } from 'whitebox-engine/foundations'
 * probabilitySpace(['H', 'T'], [3, 1]).probabilities // => [0.75, 0.25]
 * ```
 */
export function probabilitySpace<T>(outcomes: readonly T[], weights?: readonly number[]): ProbabilitySpace<T> {
  const w = weights ?? outcomes.map(() => 1)
  if (w.length !== outcomes.length) throw new WhiteboxError('value', 'probabilitySpace: one weight per outcome')
  if (w.some((x) => !(x >= 0))) throw new WhiteboxError('value', 'probabilitySpace: weights must be non-negative')
  const total = kahanSum(w)
  if (!(total > 0)) throw new WhiteboxError('value', 'probabilitySpace: the weights sum to zero')
  return { outcomes: [...outcomes], probabilities: w.map((x) => x / total) }
}

/**
 * The product of two spaces: pairs of outcomes with multiplied probabilities (two independent experiments).
 *
 * @param a - The first space.
 * @param b - The second space.
 * @returns The space of pairs `[x, y]`.
 * @concept independence
 * @example
 * ```ts
 * import { probabilitySpace, productSpace } from 'whitebox-engine/foundations'
 * const coin = probabilitySpace(['H', 'T'])
 * productSpace(coin, coin).outcomes // => [['H', 'H'], ['H', 'T'], ['T', 'H'], ['T', 'T']]
 * ```
 */
export function productSpace<A, B>(a: ProbabilitySpace<A>, b: ProbabilitySpace<B>): ProbabilitySpace<[A, B]> {
  const outcomes: [A, B][] = []
  const probabilities: number[] = []
  for (const [i, x] of a.outcomes.entries()) {
    for (const [j, y] of b.outcomes.entries()) {
      outcomes.push([x, y])
      probabilities.push((a.probabilities[i] ?? 0) * (b.probabilities[j] ?? 0))
    }
  }
  return { outcomes, probabilities }
}

/**
 * The space of n independent repetitions of an experiment (outcomes are arrays of length n).
 *
 * @param space - The single experiment.
 * @param n - The number of repetitions.
 * @returns The product space.
 * @concept independence
 * @example
 * ```ts
 * import { powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * powerSpace(probabilitySpace([1, 2, 3, 4, 5, 6]), 2).outcomes.length // => 36
 * ```
 */
export function powerSpace<T>(space: ProbabilitySpace<T>, n: number): ProbabilitySpace<T[]> {
  let out: ProbabilitySpace<T[]> = { outcomes: [[]], probabilities: [1] }
  for (let k = 0; k < n; k++) {
    const next = productSpace(out, space)
    out = { outcomes: next.outcomes.map(([xs, x]) => [...xs, x]), probabilities: next.probabilities }
  }
  return out
}

/**
 * The probability of an event: the total probability of the outcomes in it.
 *
 * @param space - The space.
 * @param event - The event.
 * @returns P(event).
 * @concept events
 * @example
 * ```ts
 * import { eventProbability, probabilitySpace } from 'whitebox-engine/foundations'
 * eventProbability(probabilitySpace([1, 2, 3, 4, 5, 6]), (x) => x % 2 === 0) // => 0.5
 * ```
 */
export function eventProbability<T>(space: ProbabilitySpace<T>, event: Event<T>): number {
  return kahanSum(space.outcomes.map((o, i) => (event(o) ? (space.probabilities[i] ?? 0) : 0)))
}

/**
 * The conditional probability `P(A | B) = P(A and B) / P(B)`.
 *
 * @param space - The space.
 * @param a - The event of interest.
 * @param given - The conditioning event (positive probability).
 * @returns P(A | B).
 * @throws WhiteboxError with code `value` when P(B) = 0.
 * @concept conditional-probability
 * @example
 * ```ts
 * import { conditionalProbability, powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * const dice = powerSpace(probabilitySpace([1, 2, 3, 4, 5, 6]), 2)
 * conditionalProbability(dice, ([a, b]) => a + b === 8, ([a]) => a === 3) // => 0.16666666666666666
 * ```
 */
export function conditionalProbability<T>(space: ProbabilitySpace<T>, a: Event<T>, given: Event<T>): number {
  const pb = eventProbability(space, given)
  if (!(pb > 0)) throw new WhiteboxError('value', 'conditionalProbability: the conditioning event has probability 0')
  return eventProbability(space, (o) => a(o) && given(o)) / pb
}

/**
 * Condition a space on an event: keep the outcomes in it and renormalise (the new space of `P(. | B)`).
 *
 * @param space - The space.
 * @param given - The event (positive probability).
 * @returns The conditional space.
 * @concept conditional-probability
 * @example
 * ```ts
 * import { conditionSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * conditionSpace(probabilitySpace([1, 2, 3, 4]), (x) => x > 2).probabilities // => [0.5, 0.5]
 * ```
 */
export function conditionSpace<T>(space: ProbabilitySpace<T>, given: Event<T>): ProbabilitySpace<T> {
  const keep = space.outcomes.map((o) => given(o))
  return probabilitySpace(
    space.outcomes.filter((_, i) => keep[i]),
    space.probabilities.filter((_, i) => keep[i]),
  )
}

/** Bayes' rule over a set of hypotheses, with the intermediate quantities. */
export interface BayesUpdate {
  readonly prior: number[]
  readonly likelihood: number[]
  /** `prior_i * likelihood_i`. */
  readonly joint: number[]
  /** `P(data) = sum_i prior_i likelihood_i` (the law of total probability). */
  readonly evidence: number
  readonly posterior: number[]
}

/**
 * Bayes' rule over mutually exclusive, exhaustive hypotheses: `P(H_i | D) = P(D | H_i) P(H_i) / sum_j P(D | H_j) P(H_j)`,
 * with the joint probabilities and the evidence shown.
 *
 * @param prior - Prior probabilities (normalised if they do not sum to one).
 * @param likelihood - `P(D | H_i)` for each hypothesis.
 * @returns The update.
 * @throws WhiteboxError with code `value` when the evidence is zero or the lengths differ.
 * @concept bayes-rule
 * @example
 * ```ts
 * import { bayesRule } from 'whitebox-engine/foundations'
 * // Two urns, equally likely; a red ball has probability 0.3 from the first and 0.8 from the second.
 * bayesRule([0.5, 0.5], [0.3, 0.8]).posterior // => [0.2727272727272727, 0.7272727272727273]
 * ```
 */
export function bayesRule(prior: readonly number[], likelihood: readonly number[]): BayesUpdate {
  if (prior.length !== likelihood.length) throw new WhiteboxError('value', 'bayesRule: one likelihood per hypothesis')
  const total = kahanSum(prior)
  const p = prior.map((x) => x / total)
  const joint = p.map((x, i) => x * (likelihood[i] ?? 0))
  const evidence = kahanSum(joint)
  if (!(evidence > 0)) throw new WhiteboxError('value', 'bayesRule: the data have probability 0 under every hypothesis')
  return { prior: p, likelihood: [...likelihood], joint, evidence, posterior: joint.map((x) => x / evidence) }
}

/** The quantities of a diagnostic test. */
export interface DiagnosticTest {
  /** P(disease | positive), the positive predictive value. */
  readonly positivePredictive: number
  /** P(no disease | negative), the negative predictive value. */
  readonly negativePredictive: number
  /** P(positive). */
  readonly positiveRate: number
  /** The likelihood ratio of a positive result, sensitivity / (1 - specificity). */
  readonly positiveLikelihoodRatio: number
}

/**
 * Bayes' rule for a diagnostic test (the base-rate fallacy calculator): from the prevalence, the sensitivity
 * `P(+ | disease)` and the specificity `P(- | healthy)`, the probability of disease given a result.
 *
 * @param options - Prevalence, sensitivity and specificity, each in [0, 1].
 * @returns The predictive values, the positive rate and the likelihood ratio.
 * @concept bayes-rule
 * @example
 * ```ts
 * import { diagnosticTest } from 'whitebox-engine/foundations'
 * // A 99% accurate test for a 1-in-100 condition: a positive means 50%.
 * diagnosticTest({ prevalence: 0.01, sensitivity: 0.99, specificity: 0.99 }).positivePredictive // => 0.5
 * ```
 */
export function diagnosticTest(options: {
  readonly prevalence: number
  readonly sensitivity: number
  readonly specificity: number
}): DiagnosticTest {
  const { prevalence: p, sensitivity: se, specificity: sp } = options
  for (const v of [p, se, sp])
    if (!(v >= 0 && v <= 1)) throw new WhiteboxError('value', 'diagnosticTest: probabilities in [0, 1]')
  const positive = bayesRule([p, 1 - p], [se, 1 - sp])
  const negative = bayesRule([p, 1 - p], [1 - se, sp])
  return {
    positivePredictive: positive.posterior[0] ?? Number.NaN,
    negativePredictive: negative.posterior[1] ?? Number.NaN,
    positiveRate: positive.evidence,
    positiveLikelihoodRatio: se / (1 - sp),
  }
}

/**
 * Whether two events are independent, `P(A and B) = P(A) P(B)`, with the three probabilities.
 *
 * @param space - The space.
 * @param a - An event.
 * @param b - Another event.
 * @param tol - Absolute tolerance (default 1e-12).
 * @returns The verdict and P(A), P(B), P(A and B).
 * @concept independence
 * @example
 * ```ts
 * import { independent, powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * const dice = powerSpace(probabilitySpace([1, 2, 3, 4, 5, 6]), 2)
 * independent(dice, ([a]) => a === 1, ([a, b]) => a + b === 7).independent // => true
 * independent(dice, ([a]) => a === 1, ([a, b]) => a + b === 6).independent // => false
 * ```
 */
export function independent<T>(
  space: ProbabilitySpace<T>,
  a: Event<T>,
  b: Event<T>,
  tol = 1e-12,
): { readonly independent: boolean; readonly pA: number; readonly pB: number; readonly pAB: number } {
  const pA = eventProbability(space, a)
  const pB = eventProbability(space, b)
  const pAB = eventProbability(space, (o) => a(o) && b(o))
  return { independent: Math.abs(pAB - pA * pB) <= tol, pA, pB, pAB }
}

/**
 * Whether events are mutually independent: the product rule holds for every subfamily of two or more (pairwise
 * independence is not enough).
 *
 * @param space - The space.
 * @param events - The events (at most 16).
 * @param tol - Absolute tolerance (default 1e-12).
 * @returns `true` when every subfamily factorises.
 * @concept independence
 * @example
 * ```ts
 * import { mutuallyIndependent, powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * // Two fair coins: "first is H", "second is H", "they agree" are pairwise but not mutually independent.
 * const coins = powerSpace(probabilitySpace(['H', 'T']), 2)
 * mutuallyIndependent(coins, [([a]) => a === 'H', ([, b]) => b === 'H', ([a, b]) => a === b]) // => false
 * ```
 */
export function mutuallyIndependent<T>(space: ProbabilitySpace<T>, events: readonly Event<T>[], tol = 1e-12): boolean {
  if (events.length > 16) throw new WhiteboxError('value', 'mutuallyIndependent: at most 16 events')
  const p = events.map((e) => eventProbability(space, e))
  for (let mask = 1; mask < 1 << events.length; mask++) {
    const ix = events.map((_, i) => i).filter((i) => mask & (1 << i))
    if (ix.length < 2) continue
    const joint = eventProbability(space, (o) => ix.every((i) => (events[i] as Event<T>)(o)))
    const product = ix.reduce((acc, i) => acc * (p[i] ?? 0), 1)
    if (Math.abs(joint - product) > tol) return false
  }
  return true
}

/** A probability mass function on finitely many real values (ascending, distinct). */
export interface Pmf {
  readonly values: number[]
  readonly probabilities: number[]
}

/**
 * The distribution (pmf) of a random variable on a finite space: the pushforward `P(X = x) = P({o : X(o) = x})`.
 *
 * @param space - The space.
 * @param x - The random variable.
 * @returns The values (ascending) and their probabilities.
 * @concept random-variables
 * @example
 * ```ts
 * import { distributionOf, powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * const coins = powerSpace(probabilitySpace([0, 1]), 3)
 * distributionOf(coins, (o) => o[0] + o[1] + o[2]) // => { values: [0, 1, 2, 3], probabilities: [0.125, 0.375, 0.375, 0.125] }
 * ```
 */
export function distributionOf<T>(space: ProbabilitySpace<T>, x: RandomVariable<T>): Pmf {
  const totals = new Map<number, number[]>()
  for (const [i, o] of space.outcomes.entries()) {
    const v = x(o)
    const list = totals.get(v) ?? []
    list.push(space.probabilities[i] ?? 0)
    totals.set(v, list)
  }
  const values = [...totals.keys()].sort((a, b) => a - b)
  return { values, probabilities: values.map((v) => kahanSum(totals.get(v) ?? [])) }
}

/**
 * The expectation `E[X] = sum_o X(o) P(o)` of a random variable on a finite space (or of a pmf's values).
 *
 * @param space - The space, or a pmf.
 * @param x - The random variable (omit for a pmf).
 * @returns E[X].
 * @concept expectation
 * @example
 * ```ts
 * import { expectation, probabilitySpace } from 'whitebox-engine/foundations'
 * expectation(probabilitySpace([1, 2, 3, 4, 5, 6]), (o) => o) // => 3.5
 * ```
 */
export function expectation<T>(space: ProbabilitySpace<T> | Pmf, x?: RandomVariable<T>): number {
  if ('values' in space) return kahanSum(space.values.map((v, i) => v * (space.probabilities[i] ?? 0)))
  const f = x ?? ((o: T) => o as unknown as number)
  return kahanSum(space.outcomes.map((o, i) => f(o) * (space.probabilities[i] ?? 0)))
}

/**
 * The variance `Var X = E[(X - E X)^2]` of a random variable on a finite space (or of a pmf).
 *
 * @param space - The space, or a pmf.
 * @param x - The random variable (omit for a pmf).
 * @returns Var X.
 * @concept variance
 * @example
 * ```ts
 * import { probabilitySpace, varianceOf } from 'whitebox-engine/foundations'
 * varianceOf(probabilitySpace([1, 2, 3, 4, 5, 6]), (o) => o) // => 2.9166666666666665
 * ```
 */
export function varianceOf<T>(space: ProbabilitySpace<T> | Pmf, x?: RandomVariable<T>): number {
  if ('values' in space) {
    const m = expectation(space)
    return kahanSum(space.values.map((v, i) => (v - m) ** 2 * (space.probabilities[i] ?? 0)))
  }
  const f = x ?? ((o: T) => o as unknown as number)
  const m = expectation(space, f)
  return kahanSum(space.outcomes.map((o, i) => (f(o) - m) ** 2 * (space.probabilities[i] ?? 0)))
}

/**
 * The covariance `E[(X - E X)(Y - E Y)]` of two random variables on a finite space.
 *
 * @param space - The space.
 * @param x - A random variable.
 * @param y - Another random variable.
 * @returns Cov(X, Y).
 * @concept covariance
 * @example
 * ```ts
 * import { covarianceOf, powerSpace, probabilitySpace } from 'whitebox-engine/foundations'
 * const dice = powerSpace(probabilitySpace([1, 2, 3, 4, 5, 6]), 2)
 * covarianceOf(dice, ([a]) => a, ([a, b]) => a + b) // => 2.916666666666667
 * ```
 */
export function covarianceOf<T>(space: ProbabilitySpace<T>, x: RandomVariable<T>, y: RandomVariable<T>): number {
  const mx = expectation(space, x)
  const my = expectation(space, y)
  return kahanSum(space.outcomes.map((o, i) => (x(o) - mx) * (y(o) - my) * (space.probabilities[i] ?? 0)))
}

/**
 * Transform a discrete distribution: the pmf of `g(X)`, merging values that g sends to the same place.
 *
 * @param pmf - The distribution of X.
 * @param g - The transformation.
 * @returns The distribution of g(X).
 * @concept transformations-of-random-variables
 * @example
 * ```ts
 * import { transformPmf } from 'whitebox-engine/foundations'
 * transformPmf({ values: [-1, 0, 1], probabilities: [0.25, 0.5, 0.25] }, (x) => x * x) // => { values: [0, 1], probabilities: [0.5, 0.5] }
 * ```
 */
export function transformPmf(pmf: Pmf, g: (x: number) => number): Pmf {
  return distributionOf({ outcomes: pmf.values, probabilities: pmf.probabilities }, g)
}

/**
 * The density of `Y = g(X)` for a strictly monotone, differentiable g, by the change-of-variables formula
 * `f_Y(y) = f_X(g^-1(y)) |d g^-1 / dy|`, the derivative by forward-mode differentiation of the inverse.
 *
 * @param density - The density of X, vectorised over a value of points.
 * @param inverse - `g^-1`, vectorised.
 * @returns The density of Y (zero where the inverse leaves the support, as the density says).
 * @concept transformations-of-random-variables
 * @example
 * ```ts
 * import { transformDensity } from 'whitebox-engine/foundations'
 * import { exp, log, mul, neg, toArray, type Value } from 'whitebox-engine/tensor'
 * // If X ~ Exp(1) then Y = X / 2 ~ Exp(2): f_Y(y) = 2 e^(-2y).
 * const fY = transformDensity((x: Value) => exp(neg(x)), (y: Value) => mul(y, 2))
 * toArray(fY([0.5])) // => [0.7357588823428847]
 * ```
 */
export function transformDensity(density: (x: Value) => Value, inverse: (y: Value) => Value): (y: TensorLike) => Value {
  return (y) => {
    const [yv] = promoteFloat(y) as [Value]
    const { out, tangent } = jvp(inverse, yv, onesLike(yv))
    return mul(density(out), abs(tangent))
  }
}

/**
 * The distribution of the sum of two independent integer-valued random variables, by convolution of their pmfs
 * `P(X + Y = s) = sum_k P(X = k) P(Y = s - k)`. Values need not start at zero or be contiguous.
 *
 * @param a - The pmf of X (integer values).
 * @param b - The pmf of Y (integer values).
 * @returns The pmf of X + Y.
 * @throws WhiteboxError with code `value` for non-integer values.
 * @concept convolution-of-distributions
 * @example
 * ```ts
 * import { sumPmf } from 'whitebox-engine/foundations'
 * const die = { values: [1, 2, 3, 4, 5, 6], probabilities: [1, 1, 1, 1, 1, 1].map((x) => x / 6) }
 * sumPmf(die, die).probabilities[5] // => 0.16666666666666666
 * ```
 */
export function sumPmf(a: Pmf, b: Pmf): Pmf {
  const dense = (p: Pmf): { start: number; probs: number[] } => {
    if (p.values.some((v) => !Number.isInteger(v))) throw new WhiteboxError('value', 'sumPmf: integer values only')
    const start = p.values[0] ?? 0
    const end = p.values[p.values.length - 1] ?? 0
    const probs = Array.from({ length: end - start + 1 }, () => 0)
    for (const [i, v] of p.values.entries()) probs[v - start] = (probs[v - start] ?? 0) + (p.probabilities[i] ?? 0)
    return { start, probs }
  }
  const da = dense(a)
  const db = dense(b)
  const c = toFlat(convolve(da.probs, db.probs))
  const values: number[] = []
  const probabilities: number[] = []
  for (const [k, p] of c.entries()) {
    if (p <= 0) continue
    values.push(da.start + db.start + k)
    probabilities.push(p)
  }
  return { values, probabilities }
}

/**
 * The density of the sum of two independent continuous random variables on a grid, `f_{X+Y}(s) = int f_X(t)
 * f_Y(s - t) dt`, by FFT convolution of the densities sampled with spacing h (the trapezoid rule: accurate to O(h^2) for
 * smooth densities, exact at a jump at the ends of the range).
 *
 * @param fx - The density of X, vectorised.
 * @param fy - The density of Y, vectorised.
 * @param range - The interval `[lo, hi]` holding (nearly) all the mass of each variable.
 * @param n - Grid points (default 801).
 * @returns The grid of sums and the density there.
 * @concept convolution-of-distributions
 * @example
 * ```ts
 * import { sumDensity } from 'whitebox-engine/foundations'
 * import { div, exp, mul, neg, type Value } from 'whitebox-engine/tensor'
 * // The sum of two standard normals is N(0, 2), with density 1 / (2 sqrt(pi)) at 0.
 * const phi = (x: Value) => div(exp(neg(mul(mul(x, x), 0.5))), Math.sqrt(2 * Math.PI))
 * const s = sumDensity(phi, phi, [-8, 8], 801)
 * Math.round(s.density[800] * 1e8) / 1e8 // => 0.28209479
 * ```
 */
export function sumDensity(
  fx: (x: Value) => Value,
  fy: (x: Value) => Value,
  range: readonly [number, number],
  n = 801,
): { readonly s: number[]; readonly density: number[] } {
  const [lo, hi] = range
  const h = (hi - lo) / (n - 1)
  const xs = Array.from({ length: n }, (_, i) => lo + i * h)
  const [grid] = promoteFloat(xs) as [Value]
  // Trapezoid weights: halve the end samples.
  const trap = (v: number[]): number[] => v.map((x, i) => (i === 0 || i === n - 1 ? x / 2 : x))
  const a = trap(toFlat(fx(grid)))
  const b = trap(toFlat(fy(grid)))
  // FFT convolution: O(n log n), and accurate to rounding (relative to the peak) for densities.
  const c = toFlat(fftConvolve(a, b)).map((v) => v * h)
  return { s: c.map((_, k) => 2 * lo + k * h), density: c }
}

/**
 * The distribution of the total of n fair dice with the given number of sides, by repeated convolution.
 *
 * @param n - How many dice.
 * @param sides - Faces per die (default 6).
 * @returns The pmf of the total.
 * @concept convolution-of-distributions
 * @example
 * ```ts
 * import { diceSum } from 'whitebox-engine/foundations'
 * diceSum(3).probabilities[7] // => 0.125
 * ```
 */
export function diceSum(n: number, sides = 6): Pmf {
  const die: Pmf = {
    values: Array.from({ length: sides }, (_, i) => i + 1),
    probabilities: Array.from({ length: sides }, () => 1 / sides),
  }
  let total: Pmf = { values: [0], probabilities: [1] }
  for (let k = 0; k < n; k++) total = sumPmf(total, die)
  return total
}
