/**
 * Learning the parameters of Bayes networks from data: state counts, maximum likelihood and Bayesian (Dirichlet prior)
 * estimates of conditional probability tables, the data log-likelihood, expectation maximisation for missing values and
 * latent variables (a traced algorithm), and least-squares fitting of linear-Gaussian networks.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { lstsq } from '../../numerics/linalg/index.ts'
import { dirichlet, foldIn, type Key } from '../../random/index.ts'
import { toArray } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { factorValues, permuteFactor } from '../factor.ts'
import type { DirectedModel } from '../graphs.ts'
import { calibrate, junctionTreeQuery } from '../inference/junction-tree.ts'
import { type BayesNet, type BayesNode, bayesNet } from '../models/bayes-net.ts'
import { type LinearGaussianBayesNet, linearGaussianBayesNet } from '../models/linear-gaussian.ts'
import { type Coded, code, type DataRow, familyCounts, numericColumns } from './_data.ts'

export type { DataRow }

/** Options shared by the parameter estimators. */
export interface ParameterOptions {
  /** The states of each variable (default: the observed values, sorted). */
  readonly states?: Readonly<Record<string, readonly string[]>>
  /** One weight per row (default 1). */
  readonly weights?: readonly number[]
}

/**
 * Weighted counts of a variable's states under each joint configuration of its parents (pgmpy's `state_counts`): rows
 * are parent configurations in row-major order (the first parent slowest), columns the variable's states. Rows with a
 * missing value in the family are skipped.
 *
 * @param data - The rows.
 * @param variable - The variable.
 * @param parents - Its parents.
 * @param options - States and row weights.
 * @returns The counts and the states used.
 * @concept parameter-learning
 * @example
 * ```ts
 * import { stateCounts } from 'mdml-engine/pgm/learning'
 * stateCounts([{ a: 'x', b: 'u' }, { a: 'x', b: 'v' }, { a: 'y', b: 'v' }], 'b', ['a']).counts // => [[1, 1], [0, 1]]
 * ```
 */
export function stateCounts(
  data: readonly DataRow[],
  variable: string,
  parents: readonly string[],
  options: ParameterOptions = {},
): { readonly counts: number[][]; readonly states: Readonly<Record<string, readonly string[]>> } {
  const d = code(data, [variable, ...parents], options.states, options.weights)
  return { counts: familyCounts(d, variable, parents), states: d.states }
}

/** CPT rows from counts plus pseudo-counts; a row with no mass becomes uniform (as pgmpy). */
function normaliseRows(counts: readonly (readonly number[])[]): number[] {
  return counts.flatMap((row) => {
    const total = row.reduce((a, b) => a + b, 0)
    return total > 0 ? row.map((v) => v / total) : row.map(() => 1 / row.length)
  })
}

function nodesFrom(
  structure: DirectedModel,
  d: Coded,
  cpt: (node: string, parents: readonly string[]) => number[],
): BayesNet {
  return bayesNet(
    structure.nodes.map(
      (n): BayesNode => ({
        name: n.name,
        states: d.states[n.name] ?? [],
        parents: n.parents,
        cpt: cpt(n.name, n.parents),
      }),
    ),
  )
}

/**
 * Maximum-likelihood parameters of a Bayes network from complete data: each CPT row is the relative frequency of the
 * node's states under that parent configuration (unobserved configurations get uniform rows, as pgmpy's
 * MaximumLikelihoodEstimator).
 *
 * @param structure - The DAG (a Dag or a Bayes network, whose parents order the CPTs).
 * @param data - The rows (values are state names).
 * @param options - States and row weights.
 * @returns The fitted network.
 * @concept parameter-learning
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { maximumLikelihood } from 'mdml-engine/pgm/learning'
 * const bn = maximumLikelihood(dag([['a', 'b']]), [{ a: 'x', b: 'u' }, { a: 'x', b: 'v' }, { a: 'y', b: 'v' }])
 * bn.nodes.map((n) => n.cpt) // => [[2 / 3, 1 / 3], [0.5, 0.5, 0, 1]]
 * ```
 */
export function maximumLikelihood(
  structure: DirectedModel,
  data: readonly DataRow[],
  options: ParameterOptions = {},
): BayesNet {
  const d = code(
    data,
    structure.nodes.map((n) => n.name),
    options.states,
    options.weights,
  )
  return nodesFrom(structure, d, (v, parents) => normaliseRows(familyCounts(d, v, parents)))
}

/** Options of {@link bayesianEstimate}. */
export interface BayesianEstimateOptions extends ParameterOptions {
  /**
   * The Dirichlet prior: `BDeu` (equivalent sample size spread evenly over every cell, default), `K2` (one pseudo-count
   * per cell) or `dirichlet` (the given pseudo-counts).
   */
  readonly prior?: 'BDeu' | 'K2' | 'dirichlet'
  /** The equivalent sample size of BDeu (default 5). */
  readonly equivalentSampleSize?: number
  /** Pseudo-counts for `dirichlet`: one number for every cell, or per variable a [configuration][state] table. */
  readonly pseudoCounts?: number | Readonly<Record<string, readonly (readonly number[])[]>>
}

/**
 * Bayesian (posterior-mean) parameter estimates with a Dirichlet prior: counts plus pseudo-counts, normalised per
 * parent configuration (pgmpy's BayesianEstimator with BDeu, K2 or explicit Dirichlet priors).
 *
 * @param structure - The DAG.
 * @param data - The rows.
 * @param options - Prior, equivalent sample size or pseudo-counts, states and weights.
 * @returns The fitted network.
 * @throws MdmlError with code `shape` for pseudo-count tables of the wrong size.
 * @concept parameter-learning
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { bayesianEstimate } from 'mdml-engine/pgm/learning'
 * bayesianEstimate(dag([], ['a']), [{ a: 'x' }, { a: 'x' }, { a: 'y' }], { prior: 'K2' }).nodes[0]?.cpt // => [0.6, 0.4]
 * ```
 */
export function bayesianEstimate(
  structure: DirectedModel,
  data: readonly DataRow[],
  options: BayesianEstimateOptions = {},
): BayesNet {
  const d = code(
    data,
    structure.nodes.map((n) => n.name),
    options.states,
    options.weights,
  )
  const prior = options.prior ?? 'BDeu'
  return nodesFrom(structure, d, (v, parents) => {
    const counts = familyCounts(d, v, parents)
    const r = d.states[v]?.length ?? 1
    const q = counts.length
    const pseudo = (j: number, k: number): number => {
      if (prior === 'K2') return 1
      if (prior === 'BDeu') return (options.equivalentSampleSize ?? 5) / (q * r)
      const pc = options.pseudoCounts
      if (pc === undefined) return 0
      if (typeof pc === 'number') return pc
      const table = pc[v]
      if (table === undefined) return 0
      if (table.length !== q || table.some((row) => row.length !== r))
        throw new MdmlError('shape', `bayesianEstimate: pseudo-counts for ${v} must be ${q} x ${r}`)
      return table[j]?.[k] ?? 0
    }
    return normaliseRows(counts.map((row, j) => row.map((c, k) => c + pseudo(j, k))))
  })
}

/**
 * The log-likelihood of data under a Bayes network: the sum over rows of log P(observed values), marginalising any
 * missing values by exact inference.
 *
 * @param bn - The network.
 * @param data - The rows.
 * @returns The log-likelihood (-Infinity when a row is impossible).
 * @concept parameter-learning
 * @example
 * ```ts
 * import { bayesNet } from 'mdml-engine/pgm'
 * import { dataLogLikelihood } from 'mdml-engine/pgm/learning'
 * const bn = bayesNet([{ name: 'a', states: ['x', 'y'], parents: [], cpt: [0.25, 0.75] }])
 * Math.abs(dataLogLikelihood(bn, [{ a: 'y' }, { a: null }]) - Math.log(0.75)) < 1e-12 // => true
 * ```
 */
export function dataLogLikelihood(bn: BayesNet, data: readonly DataRow[]): number {
  const names = bn.nodes.map((n) => n.name)
  const d = code(data, names, Object.fromEntries(bn.nodes.map((n) => [n.name, n.states])))
  let total = 0
  for (const { evidence, weight } of patterns(d)) total += weight * calibrate(bn, { evidence }).logPartition
  return total
}

/** Distinct observation patterns of coded data with their total weights. */
function patterns(d: Coded): { evidence: Record<string, number>; weight: number }[] {
  const byKey = new Map<string, { evidence: Record<string, number>; weight: number }>()
  for (let i = 0; i < d.n; i++) {
    const evidence: Record<string, number> = {}
    for (const v of d.names) {
      const s = d.codes[v]?.[i] ?? -1
      if (s >= 0) evidence[v] = s
    }
    const key = d.names.map((v) => d.codes[v]?.[i] ?? -1).join(',')
    const entry = byKey.get(key)
    if (entry === undefined) byKey.set(key, { evidence, weight: d.weights[i] ?? 1 })
    else entry.weight += d.weights[i] ?? 1
  }
  return [...byKey.values()]
}

/** Options of {@link emAlgorithm} and {@link expectationMaximisation}. */
export interface EmOptions extends ParameterOptions {
  /** The states of latent variables (variables that never appear in the data), by name. */
  readonly latent?: Readonly<Record<string, readonly string[]>>
  /** Starting parameters (a network with this structure and states); otherwise random CPTs drawn with `key`. */
  readonly init?: BayesNet
  readonly key?: Key
  /** Dirichlet pseudo-counts added in every M-step (MAP-EM), one per cell (default 0). */
  readonly pseudoCount?: number
  /** Stop when no CPT entry changes by more than this (default 1e-8; 0 runs the full iteration budget). */
  readonly tol?: number
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of an EM run. */
export interface EmState {
  readonly network: BayesNet
  /** log-likelihood of the data under the parameters of the previous iteration (the E-step's). */
  readonly logLikelihood: number
  readonly change: number
}

/**
 * Expectation maximisation for Bayes-network parameters with missing values and latent variables, as an
 * {@link Algorithm} (Dempster, Laird and Rubin 1977; Lauritzen 1995): the E-step calibrates a junction tree for each
 * distinct pattern of observed values and accumulates the expected family counts; the M-step normalises them (plus
 * optional pseudo-counts). The log-likelihood never decreases and is recorded each iteration.
 *
 * @param structure - The DAG (latent variables included).
 * @param data - The rows; missing values are null, latent variables absent.
 * @param options - Latent states, initial parameters or a key (default: the run's key) for random ones, pseudo-counts
 * and the stopping rule.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a variable without states.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { emAlgorithm } from 'mdml-engine/pgm/learning'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = [{ a: 'x', b: 'u' }, { a: 'x', b: null }, { a: 'y', b: 'v' }]
 * runAlgorithm(emAlgorithm(dag([['a', 'b']]), data, { key: key(0) }), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function emAlgorithm(
  structure: DirectedModel,
  data: readonly DataRow[],
  options: EmOptions = {},
): Algorithm<EmState> {
  const names = structure.nodes.map((n) => n.name)
  const latent = options.latent ?? {}
  const observedNames = names.filter((v) => !(v in latent))
  const coded = code(data, observedNames, options.states, options.weights)
  const states: Record<string, readonly string[]> = { ...coded.states, ...latent }
  for (const v of names) if (states[v] === undefined) throw new MdmlError('value', `em: no states for ${v}`)
  const pats = patterns(coded)
  const tol = options.tol ?? 1e-8
  const pseudo = options.pseudoCount ?? 0
  const initial = (k: Key): BayesNet => {
    if (options.init !== undefined) return options.init
    return bayesNet(
      structure.nodes.map((n, i) => {
        const r = states[n.name]?.length ?? 1
        const q = n.parents.reduce((a, p) => a * (states[p]?.length ?? 1), 1)
        const draws = toArray(dirichlet(foldIn(k, i), new Array<number>(r).fill(1), q)) as number[][]
        return { name: n.name, states: states[n.name] ?? [], parents: n.parents, cpt: draws.flat() }
      }),
    )
  }
  const em = (bn: BayesNet): { network: BayesNet; logLikelihood: number } => {
    const expected = bn.nodes.map((n) => {
      const r = n.states.length
      const q = n.parents.reduce((a, p) => a * (states[p]?.length ?? 1), 1)
      return Array.from({ length: q }, () => new Array<number>(r).fill(pseudo))
    })
    let ll = 0
    for (const { evidence, weight } of pats) {
      const tree = calibrate(bn, { evidence })
      ll += weight * tree.logPartition
      for (const [ni, n] of bn.nodes.entries()) {
        const family = [...n.parents, n.name]
        const free = family.filter((v) => !(v in evidence))
        const table = expected[ni] as number[][]
        const card = family.map((v) => states[v]?.length ?? 1)
        if (free.length === 0) {
          let j = 0
          for (const p of n.parents) j = j * (states[p]?.length ?? 1) + (evidence[p] ?? 0)
          ;(table[j] as number[])[evidence[n.name] ?? 0] = (table[j]?.[evidence[n.name] ?? 0] ?? 0) + weight
          continue
        }
        const post = factorValues(permuteFactor(junctionTreeQuery(tree, free), free))
        const size = free.reduce((a, v) => a * (states[v]?.length ?? 1), 1)
        for (let e = 0; e < size; e++) {
          // Decode the free assignment and complete it with the evidence.
          let rest = e
          const values: Record<string, number> = { ...evidence }
          for (let k = free.length - 1; k >= 0; k--) {
            const c = states[free[k] ?? '']?.length ?? 1
            values[free[k] ?? ''] = rest % c
            rest = Math.floor(rest / c)
          }
          let j = 0
          for (const [m, p] of n.parents.entries()) j = j * (card[m] ?? 1) + (values[p] ?? 0)
          const x = values[n.name] ?? 0
          ;(table[j] as number[])[x] = (table[j]?.[x] ?? 0) + weight * (post[e] ?? 0)
        }
      }
    }
    const network = bayesNet(bn.nodes.map((n, ni) => ({ ...n, cpt: normaliseRows(expected[ni] ?? []) })))
    return { network, logLikelihood: ll }
  }
  return {
    name: 'expectationMaximisation',
    config: { variables: names, latent: Object.keys(latent), tol, pseudoCount: pseudo } as Json,
    schema: channels({
      metrics: {
        logLikelihood: { description: 'log-likelihood of the observed data', direction: 'higher' },
        change: { description: 'largest change of a CPT entry', direction: 'lower' },
      },
    }),
    init: (k) => ({
      network: initial(options.key ?? k),
      logLikelihood: Number.NEGATIVE_INFINITY,
      change: Number.POSITIVE_INFINITY,
    }),
    step: (s) => {
      const { network, logLikelihood } = em(s.network)
      let change = 0
      for (const [i, n] of network.nodes.entries())
        for (const [k, v] of n.cpt.entries()) change = Math.max(change, Math.abs(v - (s.network.nodes[i]?.cpt[k] ?? 0)))
      return { network, logLikelihood, change }
    },
    done: (s, t) => (t > 0 && s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        logLikelihood: Number.isFinite(s.logLikelihood) ? s.logLikelihood : -1,
        change: Number.isFinite(s.change) ? s.change : -1,
      },
    }),
  }
}

/**
 * Fit Bayes-network parameters by expectation maximisation ({@link emAlgorithm}).
 *
 * @param structure - The DAG (latent variables included).
 * @param data - The rows (missing values null; latent variables absent).
 * @param options - Latent states, initial parameters or key, pseudo-counts, stopping rule and tracer.
 * @returns The fitted network, the final log-likelihood of the data, and convergence information.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { expectationMaximisation } from 'mdml-engine/pgm/learning'
 * import { key } from 'mdml-engine/random'
 * const data = [{ a: 'x', b: 'u' }, { a: 'x', b: null }, { a: 'y', b: 'v' }]
 * expectationMaximisation(dag([['a', 'b']]), data, { key: key(0) }).network.nodes[0]?.cpt.map((p) => Math.round(p * 1e6) / 1e6) // => [0.666667, 0.333333]
 * ```
 */
export function expectationMaximisation(
  structure: DirectedModel,
  data: readonly DataRow[],
  options: EmOptions = {},
): {
  readonly network: BayesNet
  readonly logLikelihood: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
} {
  const result = runAlgorithm(emAlgorithm(structure, data, options), {
    maxSteps: options.maxIter ?? 100,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return {
    network: result.state.network,
    logLikelihood: dataLogLikelihood(result.state.network, data),
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}

/**
 * Fit a linear-Gaussian network by least squares (pgmpy's LinearGaussianBayesianNetwork.fit): each node regressed on
 * its parents with an intercept; the noise variance from the residuals, unbiased (dividing by n - 1 - parents, the
 * default) or maximum likelihood (dividing by n).
 *
 * @param structure - The DAG.
 * @param data - Rows of numbers.
 * @param options - `variance`: `unbiased` (default) or `mle`.
 * @returns The fitted network.
 * @concept parameter-learning
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { fitLinearGaussian } from 'mdml-engine/pgm/learning'
 * const rows = [0, 1, 2, 3].map((x) => ({ x, y: 1 + 2 * x }))
 * fitLinearGaussian(dag([['x', 'y']]), rows).nodes.map((n) => Math.round(n.intercept * 1e9) / 1e9) // => [1.5, 1]
 * ```
 */
export function fitLinearGaussian(
  structure: DirectedModel,
  data: readonly DataRow[],
  options: { readonly variance?: 'unbiased' | 'mle' } = {},
): LinearGaussianBayesNet {
  const names = structure.nodes.map((n) => n.name)
  const cols = numericColumns(data, names)
  const n = data.length
  const column = (v: string): Float64Array => cols[names.indexOf(v)] ?? new Float64Array(n)
  return linearGaussianBayesNet(
    structure.nodes.map((node) => {
      const y = column(node.name)
      const design = Array.from({ length: n }, (_, i) => [1, ...node.parents.map((p) => column(p)[i] ?? 0)])
      const beta =
        node.parents.length === 0
          ? [y.reduce((a, b) => a + b, 0) / n]
          : (toArray(lstsq(design, Array.from(y))) as number[])
      let rss = 0
      for (let i = 0; i < n; i++) {
        const fit = (design[i] ?? []).reduce((s, x, k) => s + x * (beta[k] ?? 0), 0)
        rss += ((y[i] ?? 0) - fit) ** 2
      }
      const ddof = options.variance === 'mle' ? 0 : 1 + node.parents.length
      return {
        name: node.name,
        parents: node.parents,
        weights: beta.slice(1),
        intercept: beta[0] ?? 0,
        variance: Math.max(rss / (n - ddof), Number.MIN_VALUE),
      }
    }),
  )
}
