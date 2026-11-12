/**
 * Minka's clutter problem, the example that opens his thesis on expectation propagation: one unknown mean `x` with a
 * broad Gaussian prior, and observations that are each, with probability `1 - weight`, a draw from `N(x, 1)` and
 * otherwise clutter from a much wider `N(0, clutterVariance)` centred on nothing. The likelihood of a point is a
 * mixture, so the posterior is a lumpy density (a main mode where the good points agree, shoulders where a point
 * might have been clutter) and every Gaussian approximation has to choose what to be wrong about: expectation
 * propagation (alpha = 1) covers all of the mass, the variational direction (alpha toward 0) sits on the main mode.
 * The module makes the problem (a seeded generator that keeps its truth), its factor graph for `alphaEpMethod` and
 * `epMethod` (each point a `numericOperator` factor on the one variable), the exact posterior on a fine grid, and the
 * two Kullback-Leibler divergences of a Gaussian from it, so a sweep of alpha can be judged against the truth.
 */

import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type CompiledGraph, compileGraph } from '../core/graph.ts'
import { gaussianPrior } from '../core/priors.ts'
import { shiftedSum } from '../numerics/_moments.ts'
import { numericOperator } from '../operators/likelihoods/numeric.ts'
import { defaultOperators } from '../operators/registry.ts'

/** A clutter problem: the observations, and what generated them. */
export interface ClutterProblem {
  /** The observations. */
  readonly ys: readonly number[]
  /** The mean the good points were drawn around (the quantity to infer). */
  readonly truth: number
  /** The probability that a point is clutter. */
  readonly weight: number
  /** The variance of the clutter. */
  readonly clutterVariance: number
  /** The variance of the Gaussian prior on the mean. */
  readonly priorVariance: number
  /** Which points were clutter (kept from the generator, for display). */
  readonly isClutter: readonly boolean[]
}

/** What {@link clutterProblem} takes besides the key. */
export interface ClutterOptions {
  /** The number of points (default 20). */
  readonly n?: number
  /** The mean of the good points (default 2). */
  readonly truth?: number
  /** The probability that a point is clutter (default 0.5). */
  readonly weight?: number
  /** The variance of the clutter (default 10). */
  readonly clutterVariance?: number
  /** The variance of the prior on the mean (default 100). */
  readonly priorVariance?: number
}

/**
 * A clutter problem from a key: each of `n` points is clutter with probability `weight` (a draw from `N(0, clutterVariance)`)
 * and otherwise a draw from `N(truth, 1)`.
 *
 * @param key - The random key.
 * @param options - Size, truth, clutter weight and variances.
 * @returns The problem.
 * @concept expectation-propagation
 * @index dataset Minka's clutter problem: a mean with outliers, whose posterior is lumpy; seeded, with its truth.
 * @example
 * ```ts
 * import { clutterProblem } from 'omai-engine/message-passing'
 * import { key } from 'omai-engine/random'
 * const p = clutterProblem(key(1), { n: 10 })
 * [p.ys.length, p.truth, p.isClutter.length] // => [10, 2, 10]
 * ```
 */
export function clutterProblem(key: Key, options: ClutterOptions = {}): ClutterProblem {
  const n = options.n ?? 20
  const truth = options.truth ?? 2
  const weight = options.weight ?? 0.5
  const clutterVariance = options.clutterVariance ?? 10
  const priorVariance = options.priorVariance ?? 100
  const u = toFlat(uniform(foldIn(key, 0), [n]))
  const good = toFlat(normal(foldIn(key, 1), [n], { loc: truth }))
  const bad = toFlat(normal(foldIn(key, 2), [n], { scale: Math.sqrt(clutterVariance) }))
  const isClutter = Array.from({ length: n }, (_, i) => (u[i] ?? 1) < weight)
  const ys = Array.from({ length: n }, (_, i) => (isClutter[i] === true ? (bad[i] ?? 0) : (good[i] ?? 0)))
  return { ys, truth, weight, clutterVariance, priorVariance, isClutter }
}

const density = (x: number, mean: number, variance: number): number =>
  Math.exp((-0.5 * (x - mean) ** 2) / variance) / Math.sqrt(2 * Math.PI * variance)

/** log of the likelihood of one point at `x`: the mixture of a good draw and clutter. */
const logLikelihood = (y: number, x: number, weight: number, clutterVariance: number): number =>
  Math.log((1 - weight) * density(y, x, 1) + weight * density(y, 0, clutterVariance))

/**
 * The factor graph of a clutter problem: one Gaussian variable `x`, its Gaussian prior, and a numeric-operator factor
 * for each point (its log density the mixture), ready for `epMethod`, `alphaEpMethod` and the rest of the engine.
 *
 * @param problem - The problem.
 * @returns The compiled graph.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { clutterGraph, clutterProblem } from 'omai-engine/message-passing'
 * import { key } from 'omai-engine/random'
 * clutterGraph(clutterProblem(key(1), { n: 5 })).factors.length // => 6
 * ```
 */
export function clutterGraph(problem: ClutterProblem): CompiledGraph {
  const clutter = numericOperator({
    name: 'clutter',
    arity: 1,
    description: 'a point that is a draw around x with probability 1 - w and wide clutter otherwise',
    logDensity: ([x], p) => logLikelihood(p['y'] as number, x as number, p['w'] as number, p['a'] as number),
  })
  return compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [
        gaussianPrior('x', 0, problem.priorVariance),
        ...problem.ys.map((y) => ({
          operator: 'clutter',
          args: ['x'],
          params: { y, w: problem.weight, a: problem.clutterVariance },
        })),
      ],
    },
    { ...defaultOperators(), clutter },
  )
}

/** The exact posterior of a clutter problem on a grid. */
export interface ClutterPosterior {
  /** The grid, ascending. */
  readonly x: readonly number[]
  /** The posterior density on it. */
  readonly density: readonly number[]
  readonly mean: number
  readonly sd: number
  /** log p(ys): the log evidence. */
  readonly logEvidence: number
  /** The Kullback-Leibler divergences of a Gaussian from the posterior, in both directions. */
  divergences(mean: number, variance: number): { readonly forward: number; readonly reverse: number }
}

/**
 * The posterior of a clutter problem by quadrature on a fine grid (the trapezoid rule on 4001 points over the points'
 * range widened by 8): its density, mean, sd and evidence, and the divergences of a Gaussian from it.
 *
 * @param problem - The problem.
 * @returns The posterior.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { clutterPosterior, clutterProblem } from 'omai-engine/message-passing'
 * import { key } from 'omai-engine/random'
 * const post = clutterPosterior(clutterProblem(key(1), { n: 8 }))
 * Math.abs(post.mean - 2) < 2 // => true
 * ```
 */
export function clutterPosterior(problem: ClutterProblem): ClutterPosterior {
  const points = 4001
  const lo = Math.min(0, ...problem.ys) - 8
  const hi = Math.max(0, ...problem.ys) + 8
  const h = (hi - lo) / (points - 1)
  const x = Array.from({ length: points }, (_, i) => lo + i * h)
  const logp = x.map(
    (v) =>
      Math.log(density(v, 0, problem.priorVariance)) +
      problem.ys.reduce((s, y) => s + logLikelihood(y, v, problem.weight, problem.clutterVariance), 0),
  )
  const { top, weights } = shiftedSum(logp)
  const raw = Array.from(weights)
  const trapezoid = (f: (i: number) => number): number => {
    let s = 0
    for (let i = 0; i < points; i++) s += f(i) * (i === 0 || i === points - 1 ? 0.5 : 1)
    return s * h
  }
  const z = trapezoid((i) => raw[i] ?? 0)
  const dens = raw.map((r) => r / z)
  const mean = trapezoid((i) => (x[i] ?? 0) * (dens[i] ?? 0))
  const variance = trapezoid((i) => ((x[i] ?? 0) - mean) ** 2 * (dens[i] ?? 0))
  const tiny = 1e-300
  return {
    x,
    density: dens,
    mean,
    sd: Math.sqrt(variance),
    logEvidence: top + Math.log(z),
    divergences: (m, v) => ({
      forward: trapezoid(
        (i) => (dens[i] ?? 0) * (Math.log((dens[i] ?? 0) + tiny) - Math.log(density(x[i] ?? 0, m, v) + tiny)),
      ),
      reverse: trapezoid((i) => {
        const q = density(x[i] ?? 0, m, v)
        return q * (Math.log(q + tiny) - Math.log((dens[i] ?? 0) + tiny))
      }),
    }),
  }
}
