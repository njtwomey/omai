/**
 * Structure scores for score-based structure learning, pgmpy style: decomposable local scores of a variable given its
 * parents for discrete data (log-likelihood, BIC, AIC, K2, BDeu and BDs) and for linear-Gaussian data, cached scorers
 * for search, whole-structure scores and structure priors.
 */

import { MdmlError } from '../../base/index.ts'
import { lstsq } from '../../numerics/linalg/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { item, toArray } from '../../tensor/index.ts'
import type { DirectedModel } from '../graphs.ts'
import { type Coded, code, type DataRow, familyCounts, numericColumns } from './_data.ts'

/** The decomposable scores of discrete structures (pgmpy's structure scores). */
export type DiscreteScore = 'logLikelihood' | 'bic' | 'aic' | 'k2' | 'bdeu' | 'bds'

/** The scores of linear-Gaussian structures. */
export type GaussianScore = 'logLikelihoodGauss' | 'bicGauss' | 'aicGauss'

/** Options of the score functions. */
export interface ScoreOptions {
  /** The states of each discrete variable (default: the observed values, sorted). */
  readonly states?: Readonly<Record<string, readonly string[]>>
  /** The equivalent sample size of BDeu and BDs (default 10, as pgmpy). */
  readonly equivalentSampleSize?: number
}

const lg = (x: number): number => item(lgamma(x))

/** Per-family statistics shared by the discrete scores. */
function discreteLocal(
  d: Coded,
  variable: string,
  parents: readonly string[],
  score: DiscreteScore,
  ess: number,
): number {
  const counts = familyCounts(d, variable, parents)
  const r = d.states[variable]?.length ?? 1
  const q = counts.length
  const n = d.n
  const ll = (): number => {
    let s = 0
    for (const row of counts) {
      const nj = row.reduce((a, b) => a + b, 0)
      for (const c of row) if (c > 0) s += c * Math.log(c / nj)
    }
    return s
  }
  switch (score) {
    case 'logLikelihood':
      return ll()
    case 'bic':
      return ll() - 0.5 * Math.log(n) * q * (r - 1)
    case 'aic':
      return ll() - q * (r - 1)
    case 'k2': {
      let s = q * lg(r)
      for (const row of counts) {
        const nj = row.reduce((a, b) => a + b, 0)
        for (const c of row) s += lg(c + 1)
        s -= lg(nj + r)
      }
      return s
    }
    case 'bdeu':
    case 'bds': {
      const observed = counts.filter((row) => row.some((c) => c > 0))
      const qObs = Math.max(observed.length, 1)
      const beta = ess / (q * r)
      const alpha = score === 'bdeu' ? ess / q : ess / qObs
      // Unobserved configurations contribute r log G(beta) - log G(alpha) each (pgmpy's adjustment terms).
      let s = (q - observed.length) * (r * lg(beta) - lg(alpha))
      for (const row of observed) {
        const nj = row.reduce((a, b) => a + b, 0)
        for (const c of row) s += lg(c + beta)
        s -= lg(nj + alpha)
      }
      return s + (score === 'bdeu' ? q : observed.length) * lg(alpha) - q * r * lg(beta)
    }
    default:
      throw new MdmlError('value', `unknown score ${String(score)}`)
  }
}

/**
 * The local score of a variable given a parent set, for discrete data: the log-likelihood at the maximum-likelihood
 * parameters, BIC (log-likelihood minus log(N)/2 per free parameter), AIC (minus one per free parameter), or the
 * Bayesian Dirichlet marginal likelihoods K2, BDeu (equivalent sample size spread uniformly) and BDs (spread over the
 * observed parent configurations), as pgmpy's structure scores.
 *
 * @param data - Discrete rows.
 * @param variable - The variable.
 * @param parents - Its parents.
 * @param score - The score.
 * @param options - States and the equivalent sample size.
 * @returns The local score (higher is better).
 * @concept structure-learning
 * @example
 * ```ts
 * import { localScore } from 'mdml-engine/pgm/learning'
 * const data = [{ a: 0, b: 0 }, { a: 1, b: 1 }, { a: 1, b: 1 }, { a: 0, b: 1 }]
 * localScore(data, 'b', ['a'], 'logLikelihood') > localScore(data, 'b', [], 'logLikelihood') // => true
 * ```
 */
export function localScore(
  data: readonly DataRow[],
  variable: string,
  parents: readonly string[],
  score: DiscreteScore,
  options: ScoreOptions = {},
): number {
  const d = code(data, [variable, ...parents], options.states)
  return discreteLocal(d, variable, parents, score, options.equivalentSampleSize ?? 10)
}

/**
 * A cached local scorer for discrete data, as the search algorithms use it: a function of (variable, parents).
 *
 * @param data - Discrete rows.
 * @param variables - The variables that may appear.
 * @param score - The score.
 * @param options - States and the equivalent sample size.
 * @returns The local score function (memoised by family).
 * @concept structure-learning
 * @example
 * ```ts
 * import { discreteScorer, localScore } from 'mdml-engine/pgm/learning'
 * const data = [{ a: 0, b: 0 }, { a: 1, b: 1 }, { a: 1, b: 0 }]
 * discreteScorer(data, ['a', 'b'], 'k2')('b', ['a']) === localScore(data, 'b', ['a'], 'k2') // => true
 * ```
 */
export function discreteScorer(
  data: readonly DataRow[],
  variables: readonly string[],
  score: DiscreteScore,
  options: ScoreOptions = {},
): (variable: string, parents: readonly string[]) => number {
  const d = code(data, variables, options.states)
  const cache = new Map<string, number>()
  const ess = options.equivalentSampleSize ?? 10
  return (variable, parents) => {
    const key = `${variable}|${parents.join(',')}`
    let v = cache.get(key)
    if (v === undefined) {
      v = discreteLocal(d, variable, parents, score, ess)
      cache.set(key, v)
    }
    return v
  }
}

/**
 * The local score of a continuous variable given its parents under a linear-Gaussian model: the maximised Gaussian
 * log-likelihood of the least-squares regression on the parents (with intercept), and its BIC and AIC penalties for
 * (parents + 2) parameters, as pgmpy's LogLikelihoodGauss, BICGauss and AICGauss.
 *
 * @param data - Rows of numbers.
 * @param variable - The variable.
 * @param parents - Its parents.
 * @param score - The score (default `bicGauss`).
 * @returns The local score.
 * @concept structure-learning
 * @example
 * ```ts
 * import { gaussianLocalScore } from 'mdml-engine/pgm/learning'
 * const data = [0, 1, 2, 3, 4].map((x) => ({ x, y: 2 * x + (x % 2 === 0 ? 0.1 : -0.1) }))
 * gaussianLocalScore(data, 'y', ['x']) > gaussianLocalScore(data, 'y', []) // => true
 * ```
 */
export function gaussianLocalScore(
  data: readonly DataRow[],
  variable: string,
  parents: readonly string[],
  score: GaussianScore = 'bicGauss',
): number {
  return gaussianScorer(data, [variable, ...parents], score)(variable, parents)
}

/**
 * A cached local scorer for continuous data under linear-Gaussian models: a function of (variable, parents).
 *
 * @param data - Rows of numbers.
 * @param variables - The variables that may appear.
 * @param score - The score.
 * @returns The local score function (memoised by family).
 * @concept structure-learning
 * @example
 * ```ts
 * import { gaussianScorer } from 'mdml-engine/pgm/learning'
 * const data = [0, 1, 2, 3].map((x) => ({ x, y: x * x }))
 * Number.isFinite(gaussianScorer(data, ['x', 'y'], 'aicGauss')('y', ['x'])) // => true
 * ```
 */
export function gaussianScorer(
  data: readonly DataRow[],
  variables: readonly string[],
  score: GaussianScore,
): (variable: string, parents: readonly string[]) => number {
  const cols = numericColumns(data, variables)
  const n = data.length
  const col = (v: string): Float64Array => cols[variables.indexOf(v)] ?? new Float64Array(n)
  const cache = new Map<string, number>()
  return (variable, parents) => {
    const key = `${variable}|${parents.join(',')}`
    const hit = cache.get(key)
    if (hit !== undefined) return hit
    const y = col(variable)
    const design = Array.from({ length: n }, (_, i) => [1, ...parents.map((p) => col(p)[i] ?? 0)])
    const beta = toArray(lstsq(design, Array.from(y))) as number[]
    let rss = 0
    for (let i = 0; i < n; i++)
      rss += ((y[i] ?? 0) - (design[i] ?? []).reduce((s, x, k) => s + x * (beta[k] ?? 0), 0)) ** 2
    const ll = (-n / 2) * (Math.log((2 * Math.PI * rss) / n) + 1)
    const k = parents.length + 2
    const v = score === 'logLikelihoodGauss' ? ll : score === 'bicGauss' ? ll - (k / 2) * Math.log(n) : ll - k
    cache.set(key, v)
    return v
  }
}

/**
 * The score of a whole structure: the sum of its local scores (decomposability), for discrete or linear-Gaussian
 * scores.
 *
 * @param structure - The DAG.
 * @param data - The rows.
 * @param score - The score.
 * @param options - States and the equivalent sample size.
 * @returns The structure score.
 * @concept structure-learning
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { structureScore } from 'mdml-engine/pgm/learning'
 * const data = [{ a: 0, b: 0 }, { a: 1, b: 1 }, { a: 1, b: 1 }, { a: 0, b: 0 }]
 * structureScore(dag([['a', 'b']]), data, 'bic') > structureScore(dag([], ['a', 'b']), data, 'bic') // => true
 * ```
 */
export function structureScore(
  structure: DirectedModel,
  data: readonly DataRow[],
  score: DiscreteScore | GaussianScore,
  options: ScoreOptions = {},
): number {
  const names = structure.nodes.map((n) => n.name)
  const local =
    score === 'logLikelihoodGauss' || score === 'bicGauss' || score === 'aicGauss'
      ? gaussianScorer(data, names, score)
      : discreteScorer(data, names, score, options)
  const edges = structure.nodes.reduce((k, n) => k + n.parents.length, 0)
  return structure.nodes.reduce((s, n) => s + local(n.name, n.parents), 0) + structurePrior(score, edges, names.length)
}

/**
 * The log structure prior of a score: zero except for BDs, which uses the marginal uniform prior of Scutari (2016),
 * -(edges + n(n - 1)/2) log 2 (as pgmpy).
 *
 * @param score - The score.
 * @param edges - Number of edges.
 * @param nodes - Number of nodes.
 * @returns The log prior.
 * @concept structure-learning
 * @example
 * ```ts
 * import { structurePrior } from 'mdml-engine/pgm/learning'
 * [structurePrior('bic', 2, 3), structurePrior('bds', 0, 2) === -Math.LN2] // => [0, true]
 * ```
 */
export function structurePrior(score: DiscreteScore | GaussianScore, edges: number, nodes: number): number {
  return score === 'bds' ? -(edges + (nodes * (nodes - 1)) / 2) * Math.LN2 : 0
}
