/**
 * Finding all Nash equilibria of two-player games, as in nashpy: support enumeration, which tries every pair of
 * equal-size supports, and vertex enumeration, which walks the vertices of the best-response polytopes, plus removal of
 * repeated equilibria.
 */

import { MdmlError, maxOf, minOf } from '../../base/index.ts'
import { combinations } from '../../foundations/index.ts'
import { dgesv } from '../../numerics/linalg/index.ts'
import { type Equilibrium, expectedPayoffs, type NormalFormGame, payoffMatrices } from './game.ts'

/** Options for {@link supportEnumeration} and {@link vertexEnumeration}. */
export interface EnumerationOptions {
  /** Tolerance for probabilities, best-response checks and tight constraints (default 1e-10). */
  readonly tol?: number
}

/** Solve a square system, or `undefined` when it is singular or the solution is not finite. */
function solveOrUndefined(a: readonly number[], b: readonly number[], n: number): Float64Array | undefined {
  try {
    const x = dgesv(a, b, n)
    return x.every(Number.isFinite) ? x : undefined
  } catch {
    return undefined
  }
}

/**
 * The mixed strategy on `support` that makes the opponent indifferent across its actions `rows`, where `value(i, j)`
 * is the opponent's payoff for its action i against action j: solves `sum_j value(i, j) p_j = v` for i in `rows` and
 * `sum_j p_j = 1` (square when the supports have equal size).
 */
function indifference(
  value: (i: number, j: number) => number,
  rows: readonly number[],
  support: readonly number[],
): { p: Float64Array; v: number } | undefined {
  const k = support.length
  const a: number[] = []
  for (const i of rows) {
    for (const j of support) a.push(value(i, j))
    a.push(-1)
  }
  for (let t = 0; t < k; t++) a.push(1)
  a.push(0)
  const b = [...new Array<number>(rows.length).fill(0), 1]
  const sol = solveOrUndefined(a, b, k + 1)
  if (sol === undefined) return undefined
  return { p: sol.subarray(0, k), v: sol[k] ?? 0 }
}

function equilibrium(game: NormalFormGame, x: readonly number[], y: readonly number[]): Equilibrium {
  return { strategies: [x, y], payoffs: expectedPayoffs(game, [x, y]) }
}

/**
 * All Nash equilibria of a non-degenerate two-player game by support enumeration (nashpy's `support_enumeration`,
 * Nisan et al. 2007 algorithm 3.4): for every pair of supports of equal size, solve the indifference conditions of both
 * players and keep the solutions that are probability vectors on those supports and mutual best responses. Degenerate
 * games can have equilibria with supports of unequal sizes, or continua of equilibria; {@link vertexEnumeration} finds
 * their extreme points.
 *
 * @param game - A two-player game.
 * @param options - The tolerance.
 * @returns The equilibria, by support size and then supports in lexicographic order.
 * @concept nash-equilibrium
 * @reference Nisan, Roughgarden, Tardos and Vazirani (2007), Algorithmic Game Theory, chapter 3.
 * @index function All Nash equilibria of a small game by support enumeration.
 * @example
 * ```ts
 * import { battleOfTheSexes, supportEnumeration } from 'mdml-engine/games'
 * const rows = supportEnumeration(battleOfTheSexes()).map((e) => e.strategies[0])
 * rows.map((x) => x.map((p) => Math.round(p * 100) / 100)) // => [[1, 0], [0, 1], [0.6, 0.4]]
 * ```
 */
export function supportEnumeration(game: NormalFormGame, options: EnumerationOptions = {}): Equilibrium[] {
  const tol = options.tol ?? 1e-10
  const [a, b] = payoffMatrices(game)
  const [m, n] = game.shape as [number, number]
  const rowsAll = Array.from({ length: m }, (_, i) => i)
  const colsAll = Array.from({ length: n }, (_, j) => j)
  const out: Equilibrium[] = []
  for (let k = 1; k <= Math.min(m, n); k++) {
    for (const rows of combinations(rowsAll, k)) {
      for (const cols of combinations(colsAll, k)) {
        // y on `cols` makes the row player indifferent over `rows`; x on `rows` does the same for the column player.
        const ys = indifference((i, j) => a[i]?.[j] ?? 0, rows, cols)
        const xs = indifference((j, i) => b[i]?.[j] ?? 0, cols, rows)
        if (ys === undefined || xs === undefined) continue
        if (ys.p.some((p) => p <= tol) || xs.p.some((p) => p <= tol)) continue
        const y = new Array<number>(n).fill(0)
        for (const [t, j] of cols.entries()) y[j] = ys.p[t] ?? 0
        const x = new Array<number>(m).fill(0)
        for (const [t, i] of rows.entries()) x[i] = xs.p[t] ?? 0
        const rowValues = a.map((r) => r.reduce((s, v, j) => s + v * (y[j] ?? 0), 0))
        const colValues = colsAll.map((j) => rowsAll.reduce((s, i) => s + (b[i]?.[j] ?? 0) * (x[i] ?? 0), 0))
        const scale = 1 + Math.abs(ys.v) + Math.abs(xs.v)
        if (maxOf(rowValues) > ys.v + tol * scale || maxOf(colValues) > xs.v + tol * scale) continue
        out.push(equilibrium(game, x, y))
      }
    }
  }
  return out
}

interface Vertex {
  readonly point: Float64Array
  readonly labels: ReadonlySet<number>
}

/**
 * The non-zero vertices of `{z >= 0, M z <= 1}` (M positive, `rows x dim`) with their labels: label `zeroLabel(t)`
 * for each `z_t = 0` and `tightLabel(r)` for each tight row `r`.
 */
function polytopeVertices(
  mat: readonly (readonly number[])[],
  dim: number,
  zeroLabel: (t: number) => number,
  tightLabel: (r: number) => number,
  tol: number,
): Vertex[] {
  const rows = mat.length
  const constraints = Array.from({ length: rows + dim }, (_, c) => c)
  const found: Vertex[] = []
  for (const tight of combinations(constraints, dim)) {
    // Constraint c < dim is z_c = 0; constraint dim + r is row r of M z = 1.
    const a: number[] = []
    const rhs: number[] = []
    for (const c of tight) {
      if (c < dim) {
        for (let t = 0; t < dim; t++) a.push(t === c ? 1 : 0)
        rhs.push(0)
      } else {
        for (let t = 0; t < dim; t++) a.push(mat[c - dim]?.[t] ?? 0)
        rhs.push(1)
      }
    }
    const z = solveOrUndefined(a, rhs, dim)
    if (z === undefined || z.every((v) => Math.abs(v) <= tol)) continue
    if (z.some((v) => v < -tol)) continue
    const products = mat.map((r) => r.reduce((s, v, t) => s + v * (z[t] ?? 0), 0))
    if (products.some((v) => v > 1 + tol)) continue
    if (found.some((f) => f.point.every((v, t) => Math.abs(v - (z[t] ?? 0)) <= 1e3 * tol))) continue
    const labels = new Set<number>()
    for (let t = 0; t < dim; t++) if (Math.abs(z[t] ?? 0) <= tol) labels.add(zeroLabel(t))
    for (let r = 0; r < rows; r++) if (Math.abs((products[r] ?? 0) - 1) <= tol) labels.add(tightLabel(r))
    found.push({ point: z.map((v) => Math.max(0, v)), labels })
  }
  return found
}

/**
 * The extreme Nash equilibria of a two-player game by vertex enumeration (nashpy's `vertex_enumeration`, Nisan et al.
 * 2007 algorithm 3.5): after shifting payoffs to be positive, enumerate the vertices of the best-response polytopes
 * `P = {x >= 0, B^T x <= 1}` and `Q = {y >= 0, A y <= 1}` and keep the fully labelled pairs, normalised. Every
 * equilibrium of a degenerate game is a convex combination of these.
 *
 * @param game - A two-player game.
 * @param options - The tolerance for tight constraints.
 * @returns The equilibria (the order follows the vertex enumeration).
 * @concept nash-equilibrium
 * @reference von Stengel (2007), Equilibrium computation for two-player games in strategic and extensive form.
 * @example
 * ```ts
 * import { matchingPennies, vertexEnumeration } from 'mdml-engine/games'
 * const round = (s) => s.map((p) => Math.round(p * 1e9) / 1e9)
 * vertexEnumeration(matchingPennies()).map((e) => e.strategies.map(round)) // => [[[0.5, 0.5], [0.5, 0.5]]]
 * ```
 */
export function vertexEnumeration(game: NormalFormGame, options: EnumerationOptions = {}): Equilibrium[] {
  const tol = options.tol ?? 1e-10
  const [a, b] = payoffMatrices(game)
  const [m, n] = game.shape as [number, number]
  const shift = (mat: number[][]): number[][] => {
    const lo = minOf(mat.flat())
    return mat.map((r) => r.map((v) => v - lo + 1))
  }
  const ap = shift(a)
  const bp = shift(b)
  const bt = Array.from({ length: n }, (_, j) => Array.from({ length: m }, (_, i) => bp[i]?.[j] ?? 0))
  // P: x_i = 0 is label i, (B^T x)_j = 1 is label m + j. Q: (A y)_i = 1 is label i, y_j = 0 is label m + j.
  const pv = polytopeVertices(
    bt,
    m,
    (i) => i,
    (j) => m + j,
    tol,
  )
  const qv = polytopeVertices(
    ap,
    n,
    (j) => m + j,
    (i) => i,
    tol,
  )
  const out: Equilibrium[] = []
  for (const p of pv) {
    for (const q of qv) {
      let full = true
      for (let l = 0; l < m + n && full; l++) if (!p.labels.has(l) && !q.labels.has(l)) full = false
      if (!full) continue
      let sx = 0
      for (const v of p.point) sx += v
      let sy = 0
      for (const v of q.point) sy += v
      out.push(
        equilibrium(
          game,
          Array.from(p.point, (v) => v / sx),
          Array.from(q.point, (v) => v / sy),
        ),
      )
    }
  }
  return out
}

/**
 * Remove equilibria that repeat (every probability within `tol`), keeping the first of each.
 *
 * @param equilibria - Equilibria, for example from several initial labels of Lemke-Howson.
 * @param tol - Absolute tolerance per probability (default 1e-8).
 * @returns The distinct equilibria in their first order.
 * @throws MdmlError with code `shape` when the equilibria are of different games.
 * @example
 * ```ts
 * import { distinctEquilibria } from 'mdml-engine/games'
 * const e = { strategies: [[1, 0], [0, 1]], payoffs: [0, 0] }
 * distinctEquilibria([e, e]).length // => 1
 * ```
 */
export function distinctEquilibria(equilibria: readonly Equilibrium[], tol = 1e-8): Equilibrium[] {
  const out: Equilibrium[] = []
  for (const e of equilibria) {
    const same = out.some((o) => {
      if (o.strategies.length !== e.strategies.length)
        throw new MdmlError('shape', 'distinctEquilibria: equilibria of different games')
      return o.strategies.every((s, p) => s.every((v, a) => Math.abs(v - (e.strategies[p]?.[a] ?? Number.NaN)) <= tol))
    })
    if (!same) out.push(e)
  }
  return out
}
