/**
 * Graph estimators that work one node at a time: Meinshausen and Buhlmann's neighbourhood selection (a lasso of each
 * variable on all the others; an edge where the coefficients are non-zero, joined by an "and" or an "or" rule) and
 * CLIME (Cai, Liu and Luo's constrained l1 minimisation for inverse matrix estimation: each column of the precision
 * matrix is the sparsest vector that nearly inverts the sample covariance, a linear programme). Both are traced
 * `Algorithm`s with a step per node or column, and both reuse the library's solvers (the lasso by coordinate descent,
 * the simplex method). Neither needs a likelihood, and neighbourhood selection needs no positive definiteness.
 */

import { OmaiError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { lassoCoordinateDescent, linprog } from '../optim/index.ts'
import { asTensor, item, type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { precisionGraph } from './evaluation.ts'

/** How the two regressions of a pair of nodes are joined into one edge. */
export type NeighbourhoodRule = 'and' | 'or'

/**
 * Meinshausen and Buhlmann's asymptotically justified lasso penalty, `Phi^-1(1 - level / (2 p^2)) / sqrt(n)`, in
 * scikit-learn's `Lasso` convention (`(1 / 2n) |y - Xw|^2 + penalty |w|_1`) for standardised variables: it keeps the
 * estimated graph free of edges between different connectivity components with probability at least `1 - level`.
 *
 * @param n - The number of observations.
 * @param p - The number of variables.
 * @param level - The probability of falsely joining connectivity components (default 0.05).
 * @returns The penalty.
 * @example
 * ```ts
 * import { meinshausenBuhlmannPenalty } from 'omai-engine/covariance'
 * meinshausenBuhlmannPenalty(100, 10, 0.05) // => 0.34807564043462424
 * ```
 */
export function meinshausenBuhlmannPenalty(n: number, p: number, level = 0.05): number {
  return item(ndtri(1 - level / (2 * p * p))) / Math.sqrt(n)
}

/** Options of {@link neighbourhoodSelectionAlgorithm}. */
export interface NeighbourhoodOptions {
  /** The lasso penalty in scikit-learn's convention (see {@link meinshausenBuhlmannPenalty}; the default). */
  readonly penalty?: number
  /** Standardise the variables (centre, unit variance) before the regressions (default true; false only centres). */
  readonly standardise?: boolean
  /** The duality-gap tolerance of the coordinate descent (default 1e-4, as scikit-learn). */
  readonly tol?: number
  /** The coordinate-descent sweep cap per node (default 1000). */
  readonly maxIter?: number
}

/** The state of {@link neighbourhoodSelectionAlgorithm}: the regressions done so far. */
export interface NeighbourhoodState {
  /** The number of nodes regressed. */
  readonly node: number
  /** `coefficients[a][b]`: the coefficient of variable `b` in the lasso of variable `a` (zero until `a` is done). */
  readonly coefficients: number[][]
}

/** The result of {@link neighbourhoodSelection}. */
export interface Neighbourhoods {
  /** `coefficients[a][b]`: the coefficient of `b` in the lasso of `a`; the diagonal is zero. */
  readonly coefficients: number[][]
  /** The edges of the chosen rule `[p][p]`. */
  readonly graph: boolean[][]
  /** The graph in which a pair is joined when both regressions select it (a subgraph of `or`). */
  readonly and: boolean[][]
  /** The graph in which a pair is joined when either regression selects it. */
  readonly or: boolean[][]
  /** The lasso penalty used. */
  readonly penalty: number
}

function standardised(x: TensorLike, standardise: boolean): { z: number[][]; n: number; p: number } {
  const rows = toArray(asTensor(x)) as number[][]
  const n = rows.length
  const p = rows[0]?.length ?? 0
  if (n < 3 || p < 2) throw new OmaiError('shape', 'neighbourhoodSelection: need at least 3 rows and 2 columns')
  const mean = Array.from({ length: p }, (_, j) => rows.reduce((s, r) => s + (r[j] ?? 0), 0) / n)
  const sd = Array.from({ length: p }, (_, j) =>
    standardise ? Math.sqrt(rows.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / n) || 1 : 1,
  )
  return { z: rows.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] ?? 1))), n, p }
}

/**
 * Neighbourhood selection (Meinshausen and Buhlmann 2006) as a traced {@link Algorithm}: each step regresses one
 * variable on all the others by the lasso (the library's coordinate descent) and records the coefficients, the
 * number of selected neighbours so far and the node.
 *
 * @param x - Observations `[n, p]`.
 * @param options - The penalty (default {@link meinshausenBuhlmannPenalty}), standardisation and solver tolerances.
 * @returns The algorithm; its final state holds every node's coefficients.
 * @example
 * ```ts
 * import { neighbourhoodSelectionAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 1], [2, 2], [3, 3], [4, 5], [5, 4]]
 * runAlgorithm(neighbourhoodSelectionAlgorithm(x, { penalty: 0.01 }), { maxSteps: 2 }).state.node // => 2
 * ```
 */
export function neighbourhoodSelectionAlgorithm(
  x: TensorLike,
  options: NeighbourhoodOptions = {},
): Algorithm<NeighbourhoodState> {
  const { z, n, p } = standardised(x, options.standardise ?? true)
  const penalty = options.penalty ?? meinshausenBuhlmannPenalty(n, p)
  if (!(penalty >= 0)) throw new OmaiError('value', 'neighbourhoodSelection: the penalty must be non-negative')
  return {
    name: 'neighbourhood-selection',
    config: { penalty, standardise: options.standardise ?? true },
    schema: channels({
      metrics: {
        neighbours: { description: 'the number of non-zero coefficients over the nodes regressed so far' },
      },
      tensors: {
        coefficients: {
          role: 'param',
          description: 'the lasso coefficients of each node on the others',
          shape: ['p', 'p'],
        },
      },
    }),
    init: () => ({ node: 0, coefficients: Array.from({ length: p }, () => new Array<number>(p).fill(0)) }),
    step: (st) => {
      const a = st.node
      const others = Array.from({ length: p }, (_, j) => j).filter((j) => j !== a)
      const design = z.map((r) => others.map((j) => r[j] ?? 0))
      const y = z.map((r) => r[a] ?? 0)
      const fit = lassoCoordinateDescent(design, y, n * penalty, {
        tol: options.tol ?? 1e-4,
        maxIterations: options.maxIter ?? 1000,
      })
      const coef = toArray(fit.coef) as number[]
      const coefficients = st.coefficients.map((r) => [...r])
      for (const [k, j] of others.entries()) (coefficients[a] as number[])[j] = coef[k] ?? 0
      return { node: a + 1, coefficients }
    },
    done: (st) => (st.node >= p ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { neighbours: st.coefficients.reduce((s, r) => s + r.filter((v) => v !== 0).length, 0) },
      tensors: { coefficients: { role: 'param', value: st.coefficients.flat(), shape: [p, p] } },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as NeighbourhoodState,
  }
}

/**
 * Neighbourhood selection (Meinshausen and Buhlmann): the lasso of every variable on all the others, whose non-zero
 * coefficients are the node's neighbours. The two regressions of a pair can disagree, so the graph is the pairs
 * selected by both (`'and'`) or by either (`'or'`, the default): the "and" graph is always a subgraph of the "or" graph.
 *
 * @param x - Observations `[n, p]`.
 * @param options - The penalty, the `rule` that sets `graph` (default `'or'`), standardisation and tolerances.
 * @returns The coefficients, the chosen graph, both graphs and the penalty.
 * @index function Meinshausen-Buhlmann neighbourhood selection: a lasso per node, "and" or "or" graph.
 * @example
 * ```ts
 * import { neighbourhoodSelection } from 'omai-engine/covariance'
 * const x = [[1, 1, 5], [2, 2, 1], [3, 3, 4], [4, 5, 2], [5, 4, 3], [6, 6, 2]]
 * neighbourhoodSelection(x, { penalty: 0.1 }).or[0]?.[1] // => true
 * ```
 */
export function neighbourhoodSelection(
  x: TensorLike,
  options: NeighbourhoodOptions & { readonly rule?: NeighbourhoodRule } = {},
): Neighbourhoods {
  const algorithm = neighbourhoodSelectionAlgorithm(x, options)
  const p = (toArray(asTensor(x)) as number[][])[0]?.length ?? 0
  const out = runAlgorithm(algorithm, { maxSteps: p })
  const coefficients = out.state.coefficients
  const selected = coefficients.map((r) => r.map((v) => v !== 0))
  const and = selected.map((r, a) => r.map((v, b) => v && selected[b]?.[a] === true))
  const or = selected.map((r, a) => r.map((v, b) => v || selected[b]?.[a] === true))
  const penalty = (algorithm.config as { penalty: number }).penalty
  return { coefficients, graph: (options.rule ?? 'or') === 'and' ? and : or, and, or, penalty }
}

/** Options of {@link climeAlgorithm}. */
export interface ClimeOptions {
  /** The constraint level `lambda >= 0`: `|S b - e_j|_inf <= lambda` for each column (default 0.1; the usual order is `sqrt(log p / n)`). */
  readonly lambda?: number
}

/** The state of {@link climeAlgorithm}. */
export interface ClimeState {
  /** The number of columns solved. */
  readonly column: number
  /** `columns[j]`: the solution `b` of column `j` (zeros until `j` is done), before symmetrisation. */
  readonly columns: number[][]
}

/**
 * CLIME (Cai, Liu and Luo 2011) as a traced {@link Algorithm}: step `j` solves the linear programme
 * `min |b|_1 subject to |S b - e_j|_inf <= lambda` (with `b = u - v`, `u, v >= 0`) by the simplex method of `optim`.
 *
 * @param s - The sample covariance (or correlation) `[p][p]`.
 * @param options - The constraint level `lambda` (default 0.1).
 * @returns The algorithm; its final state holds every column.
 * @throws OmaiError with code `convergence` (from a step) when a column's programme is infeasible: `lambda` is too small.
 * @example
 * ```ts
 * import { climeAlgorithm } from 'omai-engine/covariance'
 * climeAlgorithm([[1, 0.5], [0.5, 1]], { lambda: 0.1 }).name // => 'clime'
 * ```
 */
export function climeAlgorithm(s: readonly (readonly number[])[], options: ClimeOptions = {}): Algorithm<ClimeState> {
  const p = s.length
  const lambda = options.lambda ?? 0.1
  if (!(lambda >= 0)) throw new OmaiError('value', 'clime: lambda must be non-negative')
  if (s.some((r) => r.length !== p)) throw new OmaiError('shape', 'clime: the covariance must be square')
  const aUb = [...s.map((r) => [...r, ...r.map((v) => -v)]), ...s.map((r) => [...r.map((v) => -v), ...r])]
  return {
    name: 'clime',
    config: { lambda },
    schema: channels({
      metrics: {
        nonzeros: { description: 'the number of non-zero entries over the columns solved so far' },
        objective: { description: 'the l1 norm of the column just solved', direction: 'lower' },
      },
      tensors: { columns: { role: 'param', description: 'the columns of the CLIME estimate', shape: ['p', 'p'] } },
    }),
    init: () => ({ column: 0, columns: Array.from({ length: p }, () => new Array<number>(p).fill(0)) }),
    step: (st) => {
      const j = st.column
      const e = Array.from({ length: p }, (_, i) => (i === j ? 1 : 0))
      const res = linprog({
        c: new Array<number>(2 * p).fill(1),
        aUb,
        bUb: [...e.map((v) => lambda + v), ...e.map((v) => lambda - v)],
      })
      if (res.status !== 'optimal')
        throw new OmaiError('convergence', `clime: column ${j} is ${res.status}; raise lambda`)
      const sol = toArray(res.x) as number[]
      const columns = st.columns.map((r) => [...r])
      columns[j] = Array.from({ length: p }, (_, i) => (sol[i] ?? 0) - (sol[p + i] ?? 0))
      return { column: j + 1, columns }
    },
    done: (st) => (st.column >= p ? 'converged' : undefined),
    observe: (st) => ({
      metrics: {
        nonzeros: st.columns.reduce((a, r) => a + r.filter((v) => Math.abs(v) > 1e-10).length, 0),
        ...(st.column > 0 ? { objective: (st.columns[st.column - 1] ?? []).reduce((a, v) => a + Math.abs(v), 0) } : {}),
      },
      tensors: { columns: { role: 'param', value: st.columns.flat(), shape: [p, p] } },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as ClimeState,
  }
}

/**
 * CLIME: a sparse precision matrix from a sample covariance by `p` linear programmes (one per column), symmetrised by
 * keeping, of each pair of entries, the one of smaller magnitude. With `lambda = 0` and an invertible `s` it is the
 * inverse of `s`; larger `lambda` gives sparser estimates. Unlike the graphical lasso the estimate is not
 * guaranteed positive definite, and no likelihood is involved.
 *
 * @param s - The sample covariance (or correlation) `[p][p]`.
 * @param options - The constraint level `lambda`.
 * @returns The symmetrised precision estimate `[p][p]`, its graph and the unsymmetrised columns.
 * @index function CLIME precision estimate: one l1-minimising linear programme per column.
 * @example
 * ```ts
 * import { clime } from 'omai-engine/covariance'
 * clime([[2, 0], [0, 4]], { lambda: 0 }).precision // => [[0.5, 0], [0, 0.25]]
 * ```
 */
export function clime(
  s: TensorLike,
  options: ClimeOptions = {},
): { readonly precision: number[][]; readonly graph: boolean[][]; readonly columns: number[][] } {
  const rows = toArray(asTensor(s)) as number[][]
  const p = rows.length
  const columns = runAlgorithm(climeAlgorithm(rows, options), { maxSteps: p }).state.columns
  // columns[j][i] is entry (i, j) of the asymmetric estimate.
  const precision = Array.from({ length: p }, (_, a) =>
    Array.from({ length: p }, (__, b) => {
      const u = columns[b]?.[a] ?? 0
      const v = columns[a]?.[b] ?? 0
      return Math.abs(u) <= Math.abs(v) ? u : v
    }),
  )
  return { precision, graph: precisionGraph(precision, 1e-8), columns }
}
