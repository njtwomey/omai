/**
 * Hierarchical and grouped time series and the reconciliation of their forecasts (Hyndman et al.; Wickramasuriya et
 * al.'s MinT): the summing matrix S of a tree or of groupings of the bottom series, and the linear combinations
 * y~ = S G y^ that make base forecasts coherent, with G from the bottom-up rule, ordinary and weighted least squares
 * (structural or variance weights), or the minimum-trace solution with a sample or shrunk covariance of the base
 * forecast errors. Top-down splits by historical proportions are provided too.
 */

import { MdmlError } from '../base/index.ts'
import { identity, invert, type Matrix, multiply, transpose } from './_matrix.ts'

/** A node of a hierarchy: a name and its children (a leaf has none). */
export interface HierarchyNode {
  readonly name: string
  readonly children?: readonly HierarchyNode[]
}

/** The summing matrix of a hierarchy and the names of its rows (all series, top first) and columns (bottom series). */
export interface SummingMatrix {
  readonly S: Matrix
  readonly names: readonly string[]
  readonly bottom: readonly string[]
}

/**
 * The summing matrix S of a tree, `[all series, bottom series]`: the rows are the nodes in breadth-first order (the
 * root first), the columns the leaves left to right, and S[i][j] = 1 when leaf j lies under node i.
 *
 * @param root - The tree.
 * @returns S with the node and leaf names.
 * @example
 * ```ts
 * import { treeSummingMatrix } from 'mdml-engine/timeseries'
 * treeSummingMatrix({ name: 'total', children: [{ name: 'a' }, { name: 'b' }] }).S // => [[1, 1], [1, 0], [0, 1]]
 * ```
 */
export function treeSummingMatrix(root: HierarchyNode): SummingMatrix {
  const leaves: string[] = []
  const collect = (n: HierarchyNode): void => {
    if ((n.children ?? []).length === 0) leaves.push(n.name)
    else for (const c of n.children ?? []) collect(c)
  }
  collect(root)
  const underOf = (n: HierarchyNode): Set<string> => {
    if ((n.children ?? []).length === 0) return new Set([n.name])
    const s = new Set<string>()
    for (const c of n.children ?? []) for (const x of underOf(c)) s.add(x)
    return s
  }
  const names: string[] = []
  const S: Matrix = []
  let level: HierarchyNode[] = [root]
  while (level.length > 0) {
    for (const n of level) {
      names.push(n.name)
      const under = underOf(n)
      S.push(leaves.map((l) => (under.has(l) ? 1 : 0)))
    }
    level = level.flatMap((n) => [...(n.children ?? [])])
  }
  return { S, names, bottom: leaves }
}

/**
 * The summing matrix of grouped series: each grouping labels every bottom series, and each distinct label becomes an
 * aggregate (as does the total, first), with the bottom series last.
 *
 * @param groupings - One label array per grouping, each with one label per bottom series.
 * @param bottom - Names of the bottom series (default `b0, b1, ...`).
 * @returns S with names `total`, `<grouping>:<label>` and the bottom names.
 * @example
 * ```ts
 * import { groupedSummingMatrix } from 'mdml-engine/timeseries'
 * // Four bottom series in two crossed groupings.
 * groupedSummingMatrix([['x', 'x', 'y', 'y'], ['p', 'q', 'p', 'q']]).S.length // => 1 + 2 + 2 + 4
 * ```
 */
export function groupedSummingMatrix(
  groupings: readonly (readonly string[])[],
  bottom?: readonly string[],
): SummingMatrix {
  const n = groupings[0]?.length ?? 0
  if (groupings.some((g) => g.length !== n))
    throw new MdmlError('shape', 'groupedSummingMatrix: every grouping needs one label per bottom series')
  const bottomNames = bottom ?? Array.from({ length: n }, (_, i) => `b${i}`)
  const S: Matrix = [new Array<number>(n).fill(1)]
  const names = ['total']
  groupings.forEach((g, gi) => {
    for (const label of [...new Set(g)]) {
      names.push(`g${gi}:${label}`)
      S.push(g.map((l) => (l === label ? 1 : 0)))
    }
  })
  for (let i = 0; i < n; i++) {
    names.push(bottomNames[i] ?? `b${i}`)
    S.push(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  }
  return { S, names, bottom: bottomNames }
}

/** The methods of {@link reconcile}. */
export type ReconcileMethod = 'bottom-up' | 'ols' | 'wls-structural' | 'wls-variance' | 'mint-sample' | 'mint-shrink'

/** The result of a reconciliation. */
export interface Reconciliation {
  /** The coherent forecasts for all series, `[horizon][series]`. */
  readonly reconciled: Matrix
  /** The bottom-level forecasts, `[horizon][bottom series]`. */
  readonly bottom: Matrix
  /** The matrix G mapping base forecasts to bottom-level forecasts, `[bottom series][series]`. */
  readonly G: Matrix
  /** The covariance W of the base forecast errors that the method assumed (identity for OLS and bottom-up). */
  readonly W: Matrix
}

/**
 * The covariance of base forecast errors from residuals `[T][series]` shrunk towards its diagonal (Schafer and
 * Strimmer; as `fable`'s `shrink`): lambda sum of the off-diagonal sampling variances over the squared off-diagonal
 * correlations, clipped to [0, 1], with the uncentred sample covariance E'E / T.
 *
 * @param residuals - Residuals, one row per time.
 * @returns The shrunk covariance and the shrinkage intensity.
 * @example
 * ```ts
 * import { shrinkCovariance } from 'mdml-engine/timeseries'
 * const r = [[1, 2], [-1, -1.5], [0.5, 1], [-0.5, -1.2], [1.2, 2.2], [-1, -2]]
 * shrinkCovariance(r).lambda >= 0 // => true
 * ```
 */
export function shrinkCovariance(residuals: readonly (readonly number[])[]): {
  readonly covariance: Matrix
  readonly lambda: number
} {
  const n = residuals.length
  const k = residuals[0]?.length ?? 0
  const sample: Matrix = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => residuals.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0) / n),
  )
  const scale = sample.map((r, i) => Math.sqrt(r[i] ?? 1) || 1)
  const xs = residuals.map((r) => r.map((v, i) => v / (scale[i] ?? 1)))
  let varianceSum = 0
  let squaredSum = 0
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      if (i === j) continue
      let sq = 0
      let sum = 0
      for (const r of xs) {
        const p = (r[i] ?? 0) * (r[j] ?? 0)
        sq += p * p
        sum += p
      }
      varianceSum += (sq - (sum * sum) / n) / (n * (n - 1))
      squaredSum += (sum / n) ** 2
    }
  const lambda = squaredSum === 0 ? 1 : Math.max(0, Math.min(1, varianceSum / squaredSum))
  return {
    covariance: sample.map((r, i) => r.map((v, j) => (i === j ? v : (1 - lambda) * v))),
    lambda,
  }
}

/** The matrix G = (S' W^-1 S)^-1 S' W^-1 for a covariance (or weight) matrix W. */
function generalisedLeastSquares(S: Matrix, W: Matrix): Matrix {
  const Wi = invert(W)
  const St = transpose(S)
  return multiply(invert(multiply(St, multiply(Wi, S))), multiply(St, Wi))
}

/**
 * Reconcile base forecasts so that they add up along the hierarchy: y~ = S G y^ with G chosen by the method (bottom-up
 * keeps the bottom forecasts; `ols` projects onto the coherent subspace; `wls-structural` weights by the number of
 * bottom series under each node; `wls-variance` by the residual variances; `mint-sample` and `mint-shrink` use the
 * full covariance of the base errors, the minimum-trace solution of Wickramasuriya, Athanasopoulos and Hyndman).
 *
 * @param base - Base forecasts, `[horizon][series]`, the series ordered as the rows of S.
 * @param S - The summing matrix, `[series][bottom series]`.
 * @param method - The reconciliation method (default `ols`).
 * @param residuals - In-sample base residuals `[T][series]`, needed by the variance and MinT methods.
 * @returns The coherent forecasts, the bottom-level ones, G and the assumed covariance.
 * @throws MdmlError with code `value` when a method needs residuals and none were given.
 * @example
 * ```ts
 * import { reconcile } from 'mdml-engine/timeseries'
 * // Total 10 against parts 4 and 5: the least-squares projection spreads the discrepancy of 1.
 * const S = [[1, 1], [1, 0], [0, 1]]
 * reconcile([[10, 4, 5]], S, 'ols').bottom[0]?.map((v) => Math.round(v * 1e9) / 1e9) // => [4.333333333, 5.333333333]
 * ```
 */
export function reconcile(
  base: readonly (readonly number[])[],
  S: readonly (readonly number[])[],
  method: ReconcileMethod = 'ols',
  residuals?: readonly (readonly number[])[],
): Reconciliation {
  const n = S.length
  const nb = S[0]?.length ?? 0
  if (base.some((r) => r.length !== n)) throw new MdmlError('shape', `reconcile: base forecasts need ${n} series`)
  const Sm = S.map((r) => [...r])
  let W: Matrix = identity(n)
  let G: Matrix
  if (method === 'bottom-up') {
    G = Array.from({ length: nb }, (_, i) => Array.from({ length: n }, (_, j) => (j === n - nb + i ? 1 : 0)))
  } else {
    if (method === 'wls-structural') W = W.map((r, i) => r.map((v) => v * (Sm[i] ?? []).reduce((a, b) => a + b, 0)))
    else if (method !== 'ols') {
      if (residuals === undefined || (residuals[0]?.length ?? 0) !== n)
        throw new MdmlError('value', `reconcile: ${method} needs in-sample residuals of all ${n} series`)
      const T = residuals.length
      if (method === 'wls-variance')
        W = identity(n).map((r, i) => r.map((v) => v * (residuals.reduce((s, e) => s + (e[i] ?? 0) ** 2, 0) / T)))
      else if (method === 'mint-sample')
        W = Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => residuals.reduce((s, e) => s + (e[i] ?? 0) * (e[j] ?? 0), 0) / T),
        )
      else W = shrinkCovariance(residuals).covariance
    }
    G = generalisedLeastSquares(Sm, W)
  }
  const SG = multiply(Sm, G)
  const reconciled = base.map((f) =>
    multiply(
      SG,
      f.map((v) => [v]),
    ).map((r) => r[0] ?? 0),
  )
  const bottom = base.map((f) =>
    multiply(
      G,
      f.map((v) => [v]),
    ).map((r) => r[0] ?? 0),
  )
  return { reconciled, bottom, G, W }
}

/**
 * Top-down proportions of the bottom series within the total, from their history: `average-proportions` averages the
 * share of the total at each time, `proportion-averages` is the share of the average (Gross and Sohl's methods A and
 * B; for a forecast of the total y, the bottom forecasts are the proportions times y).
 *
 * @param bottom - Historical bottom series, `[T][bottom series]`.
 * @param method - `average-proportions` (default) or `proportion-averages`.
 * @returns The proportions, summing to one.
 * @example
 * ```ts
 * import { topDownProportions } from 'mdml-engine/timeseries'
 * topDownProportions([[1, 3], [2, 2]], 'proportion-averages') // => [0.375, 0.625]
 * ```
 */
export function topDownProportions(
  bottom: readonly (readonly number[])[],
  method: 'average-proportions' | 'proportion-averages' = 'average-proportions',
): number[] {
  const nb = bottom[0]?.length ?? 0
  const T = bottom.length
  if (method === 'proportion-averages') {
    const means = Array.from({ length: nb }, (_, j) => bottom.reduce((s, r) => s + (r[j] ?? 0), 0) / T)
    const total = means.reduce((a, b) => a + b, 0)
    return means.map((m) => m / total)
  }
  return Array.from(
    { length: nb },
    (_, j) => bottom.reduce((s, r) => s + (r[j] ?? 0) / r.reduce((a, b) => a + b, 0), 0) / T,
  )
}

/**
 * Whether forecasts are coherent: every aggregate equals the sum of the bottom series beneath it.
 *
 * @param forecasts - Forecasts `[horizon][series]`.
 * @param S - The summing matrix.
 * @param tol - Absolute tolerance (default 1e-9).
 * @returns True when S times the bottom block reproduces every row.
 * @example
 * ```ts
 * import { isCoherent } from 'mdml-engine/timeseries'
 * isCoherent([[9, 4, 5]], [[1, 1], [1, 0], [0, 1]]) // => true
 * ```
 */
export function isCoherent(
  forecasts: readonly (readonly number[])[],
  S: readonly (readonly number[])[],
  tol = 1e-9,
): boolean {
  const nb = S[0]?.length ?? 0
  return forecasts.every((f) => {
    const b = f.slice(f.length - nb)
    return S.every((row, i) => Math.abs(row.reduce((s, v, j) => s + v * (b[j] ?? 0), 0) - (f[i] ?? 0)) <= tol)
  })
}
