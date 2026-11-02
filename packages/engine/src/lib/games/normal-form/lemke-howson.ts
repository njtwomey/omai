/**
 * The Lemke-Howson algorithm, which finds a Nash equilibrium of a two-player game by complementary pivoting along a
 * path of best-response polytope vertices. A traced version that records every pivot, the solver as in nashpy, and the
 * run from every initial label.
 */

import { MdmlError, minOf } from '../../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../../trace/index.ts'
import { type Equilibrium, expectedPayoffs, type GameRunOptions, type NormalFormGame, payoffMatrices } from './game.ts'

/** Options for {@link lemkeHowson}. */
export interface LemkeHowsonOptions extends GameRunOptions {
  /** The label dropped first: a row action `0..m-1` or a column action `m..m+n-1` (default 0). */
  readonly initialLabel?: number
  /** Stop with status `'iteration limit'` after this many pivots (default 1000). */
  readonly maxPivots?: number
}

/** One complementary pivot: the polytope pivoted in, the label entering the basis and the label leaving it. */
export interface Pivot {
  readonly polytope: 'row' | 'column'
  readonly entering: number
  readonly leaving: number
}

/** The state of a traced Lemke-Howson run. */
export interface LemkeHowsonState {
  /** Tableaux of the row player's polytope P (n rows) and the column player's Q (m rows), with right-hand sides last. */
  readonly tableaux: readonly [Float64Array, Float64Array]
  /** The basic label of every row of each tableau. */
  readonly bases: readonly [readonly number[], readonly number[]]
  /** The label to bring into the basis next, and which polytope it enters. */
  readonly entering: number
  readonly polytope: 0 | 1
  readonly path: readonly Pivot[]
  readonly status: 'running' | 'equilibrium' | 'iteration limit'
}

/** The outcome of {@link lemkeHowson}. */
export interface LemkeHowsonResult extends Equilibrium {
  readonly initialLabel: number
  readonly pivots: number
  readonly path: readonly Pivot[]
  readonly status: 'equilibrium' | 'iteration limit'
}

const SCHEMA: TraceSchema = channels({
  metrics: { pivots: { description: 'complementary pivots so far' } },
  tensors: {
    row: { role: 'state', description: "the row player's normalised basic solution", shape: ['m'] },
    column: { role: 'state', description: "the column player's normalised basic solution", shape: ['n'] },
  },
  messages: {
    pivot: { description: 'a pivot', payload: { polytope: 'row or column', entering: 'label', leaving: 'label' } },
  },
})

// Tableau layout: one row per constraint, one column per label 0..m+n-1, then the right-hand side. In P (rows j of
// B^T x + s = 1) the columns of labels i < m hold x_i and those of labels m + j the slacks; in Q (rows i of
// r + A y = 1) the columns of labels i < m hold the slacks r_i and those of labels m + j hold y_j.
function initialTableaux(game: NormalFormGame): {
  tableaux: [Float64Array, Float64Array]
  bases: [number[], number[]]
} {
  const [a, b] = payoffMatrices(game)
  const [m, n] = game.shape as [number, number]
  const w = m + n + 1
  const lo = (mat: number[][]): number => minOf(mat.flat())
  const sa = 1 - lo(a)
  const sb = 1 - lo(b)
  const p = new Float64Array(n * w)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) p[j * w + i] = (b[i]?.[j] ?? 0) + sb
    p[j * w + m + j] = 1
    p[j * w + m + n] = 1
  }
  const q = new Float64Array(m * w)
  for (let i = 0; i < m; i++) {
    q[i * w + i] = 1
    for (let j = 0; j < n; j++) q[i * w + m + j] = (a[i]?.[j] ?? 0) + sa
    q[i * w + m + n] = 1
  }
  return {
    tableaux: [p, q],
    bases: [Array.from({ length: n }, (_, j) => m + j), Array.from({ length: m }, (_, i) => i)],
  }
}

/**
 * The pivot row for `column` by the lexicographic minimum-ratio rule: the smallest `rhs / t[column]` over rows with a
 * positive entry, ties broken by the rows of the columns that formed the initial basis (the perturbed problem), which
 * keeps the path well defined on degenerate games.
 */
function pivotRow(t: Float64Array, rows: number, w: number, column: number, initial: readonly number[]): number {
  let best = -1
  const eps = 1e-12
  for (let r = 0; r < rows; r++) {
    const c = t[r * w + column] ?? 0
    if (c <= eps) continue
    if (best < 0) {
      best = r
      continue
    }
    const cb = t[best * w + column] ?? 1
    const keys = [w - 1, ...initial]
    for (const k of keys) {
      const x = (t[r * w + k] ?? 0) / c
      const y = (t[best * w + k] ?? 0) / cb
      const scale = 1e-11 * Math.max(1, Math.abs(x), Math.abs(y))
      if (x < y - scale) {
        best = r
        break
      }
      if (x > y + scale) break
    }
  }
  return best
}

function pivot(t: Float64Array, rows: number, w: number, row: number, column: number): void {
  const pv = t[row * w + column] ?? 1
  for (let k = 0; k < w; k++) t[row * w + k] = (t[row * w + k] ?? 0) / pv
  for (let r = 0; r < rows; r++) {
    if (r === row) continue
    const f = t[r * w + column] ?? 0
    if (f === 0) continue
    for (let k = 0; k < w; k++) t[r * w + k] = (t[r * w + k] ?? 0) - f * (t[row * w + k] ?? 0)
  }
}

function strategies(game: NormalFormGame, state: LemkeHowsonState): [number[], number[]] {
  const [m, n] = game.shape as [number, number]
  const w = m + n + 1
  const x = new Array<number>(m).fill(0)
  const y = new Array<number>(n).fill(0)
  for (const [r, l] of state.bases[0].entries()) if (l < m) x[l] = state.tableaux[0][r * w + m + n] ?? 0
  for (const [r, l] of state.bases[1].entries()) if (l >= m) y[l - m] = state.tableaux[1][r * w + m + n] ?? 0
  const normalise = (v: number[]): number[] => {
    let s = 0
    for (const p of v) s += p
    return s > 0 ? v.map((p) => p / s) : v
  }
  return [normalise(x), normalise(y)]
}

/**
 * Lemke-Howson as a traced algorithm: each step is one complementary pivot, from the artificial equilibrium (the
 * origin of both polytopes) along the path that drops `initialLabel`, until that label is picked up again.
 *
 * @param game - A two-player game.
 * @param options - The initial label.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a game with other than two players or a label out of range.
 * @concept lemke-howson
 * @example
 * ```ts
 * import { lemkeHowsonAlgorithm, matchingPennies } from 'mdml-engine/games'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(lemkeHowsonAlgorithm(matchingPennies()), { maxSteps: 10 }).state.status // => 'equilibrium'
 * ```
 */
export function lemkeHowsonAlgorithm(
  game: NormalFormGame,
  options: Pick<LemkeHowsonOptions, 'initialLabel'> = {},
): Algorithm<LemkeHowsonState> {
  if (game.shape.length !== 2) throw new MdmlError('value', 'lemkeHowson: a two-player game is needed')
  const [m, n] = game.shape as [number, number]
  const k = options.initialLabel ?? 0
  if (!Number.isInteger(k) || k < 0 || k >= m + n)
    throw new MdmlError('value', `lemkeHowson: the initial label must be an integer in [0, ${m + n})`)
  const w = m + n + 1
  const initialColumns: [number[], number[]] = [
    Array.from({ length: n }, (_, j) => m + j),
    Array.from({ length: m }, (_, i) => i),
  ]
  return {
    name: 'lemke-howson',
    config: { initialLabel: k, shape: [m, n] },
    schema: SCHEMA,
    init: () => {
      const { tableaux, bases } = initialTableaux(game)
      // A row label is non-basic in P (x_k = 0), a column label in Q (y_j = 0): pivot it in there first.
      return { tableaux, bases, entering: k, polytope: k < m ? 0 : 1, path: [], status: 'running' }
    },
    step: (s) => {
      const p = s.polytope
      const t = Float64Array.from(s.tableaux[p])
      const rows = p === 0 ? n : m
      const r = pivotRow(t, rows, w, s.entering, initialColumns[p])
      if (r < 0) throw new MdmlError('convergence', 'lemkeHowson: unbounded ray (the payoffs should be positive)')
      pivot(t, rows, w, r, s.entering)
      const basis = [...s.bases[p]]
      const leaving = basis[r] ?? -1
      basis[r] = s.entering
      const tableaux: [Float64Array, Float64Array] = p === 0 ? [t, s.tableaux[1]] : [s.tableaux[0], t]
      const bases: [readonly number[], readonly number[]] = p === 0 ? [basis, s.bases[1]] : [s.bases[0], basis]
      const path = [
        ...s.path,
        { polytope: p === 0 ? ('row' as const) : ('column' as const), entering: s.entering, leaving },
      ]
      return {
        tableaux,
        bases,
        entering: leaving,
        polytope: p === 0 ? 1 : 0,
        path,
        status: leaving === k ? 'equilibrium' : 'running',
      }
    },
    done: (s) => (s.status === 'equilibrium' ? 'converged' : undefined),
    observe: (s) => {
      const [x, y] = strategies(game, s)
      const last = s.path[s.path.length - 1]
      return {
        metrics: { pivots: s.path.length },
        tensors: { row: { role: 'state', value: x }, column: { role: 'state', value: y } },
        ...(last === undefined
          ? {}
          : {
              messages: [
                { kind: 'pivot', payload: { polytope: last.polytope, entering: last.entering, leaving: last.leaving } },
              ],
            }),
      }
    },
  }
}

/**
 * One Nash equilibrium of a two-player game by the Lemke-Howson algorithm (nashpy's `lemke_howson`, Nisan et al.
 * 2007 algorithm 3.6) with lexicographic pivoting so that degenerate games terminate. Different initial labels can
 * reach different equilibria; every equilibrium it finds has odd index.
 *
 * @param game - A two-player game.
 * @param options - The initial label, a pivot cap and tracing.
 * @returns The equilibrium, its payoffs, the pivots taken and the status.
 * @throws MdmlError with code `value` for a game with other than two players or a label out of range.
 * @concept lemke-howson
 * @reference Lemke and Howson (1964), Equilibrium points of bimatrix games; von Stengel (2007).
 * @index function Nash equilibrium of a bimatrix game by Lemke-Howson.
 * @example
 * ```ts
 * import { battleOfTheSexes, lemkeHowson } from 'mdml-engine/games'
 * lemkeHowson(battleOfTheSexes(), { initialLabel: 0 }).strategies // => [[1, 0], [1, 0]]
 * ```
 */
export function lemkeHowson(game: NormalFormGame, options: LemkeHowsonOptions = {}): LemkeHowsonResult {
  const { maxPivots = 1000, initialLabel = 0, ...run } = options
  const { state } = runAlgorithm(lemkeHowsonAlgorithm(game, { initialLabel }), { ...run, maxSteps: maxPivots })
  const strategy = strategies(game, state)
  return {
    strategies: strategy,
    payoffs: expectedPayoffs(game, strategy),
    initialLabel,
    pivots: state.path.length,
    path: state.path,
    status: state.status === 'equilibrium' ? 'equilibrium' : 'iteration limit',
  }
}

/**
 * Lemke-Howson from every initial label `0..m+n-1` (nashpy's `lemke_howson_enumeration`): one equilibrium per label,
 * repeats included (pass the result to `distinctEquilibria` for the set). Not every equilibrium need be reached.
 *
 * @param game - A two-player game.
 * @param options - A pivot cap per run and tracing.
 * @returns One result per initial label, in label order.
 * @concept lemke-howson
 * @example
 * ```ts
 * import { lemkeHowsonEnumeration, rockPaperScissors } from 'mdml-engine/games'
 * lemkeHowsonEnumeration(rockPaperScissors()).length // => 6
 * ```
 */
export function lemkeHowsonEnumeration(
  game: NormalFormGame,
  options: Omit<LemkeHowsonOptions, 'initialLabel'> = {},
): LemkeHowsonResult[] {
  const labels = (game.shape[0] ?? 0) + (game.shape[1] ?? 0)
  return Array.from({ length: labels }, (_, initialLabel) => lemkeHowson(game, { ...options, initialLabel }))
}
