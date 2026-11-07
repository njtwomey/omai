/**
 * The simplex algorithm behind `linprogSimplex`: the two-phase method on a dense tableau as a traced algorithm, one
 * pivot per step, recording the basis and the vertices visited so the walk along the feasible polytope can be shown.
 */

import { maxOf } from '../base/index.ts'
import { ddot, dgesv } from '../numerics/linalg/index.ts'
import { type Algorithm, runStatusOf } from '../trace/index.ts'
import { basisDuals, LP_SCHEMA, lpConfig } from './_lp.ts'
import {
  fromStandard,
  type LinprogOptions,
  type LpPathPoint,
  type StandardForm,
  type StandardSolution,
} from './lp-standard-form.ts'

/** The state of a {@link simplexAlgorithm} run: the dense tableau, its basis and the vertices visited. */
export interface SimplexState {
  /** 1 while finding a feasible vertex (minimising the artificial variables), 2 while optimising. */
  readonly phase: 1 | 2
  /** The tableau, (rows + 1) x (cols + 1) row-major: constraint rows, then reduced costs; the last column is the
   * right-hand side (minus the objective in the last row). */
  readonly tableau: Float64Array
  readonly rows: number
  /** Columns (standard-form columns, plus artificial ones in phase 1). */
  readonly cols: number
  /** The basic column of each row. */
  readonly basis: Int32Array
  /** The standard-form row of each tableau row (redundant equality rows are dropped after phase 1). */
  readonly rowIds: Int32Array
  readonly iterations: number
  /** How many times the tableau has been recomputed from the original matrix (re-inversions). */
  readonly reinversions: number
  readonly path: readonly LpPathPoint[]
  /** The last pivot. */
  readonly pivot?: { readonly entering: number; readonly leaving: number; readonly ratio: number }
  /** The current solution; its status is final once `finished`. */
  readonly solution: StandardSolution
  readonly finished: boolean
}

function pivotOn(t: Float64Array, rows: number, width: number, r: number, e: number): Float64Array {
  const out = Float64Array.from(t)
  const p = out[r * width + e] ?? 1
  for (let k = 0; k < width; k++) out[r * width + k] = (out[r * width + k] ?? 0) / p
  for (let i = 0; i <= rows; i++) {
    if (i === r) continue
    const f = out[i * width + e] ?? 0
    if (f === 0) continue
    for (let k = 0; k < width; k++) out[i * width + k] = (out[i * width + k] ?? 0) - f * (out[r * width + k] ?? 0)
    out[i * width + e] = 0
  }
  return out
}

function point(form: StandardForm, s: { tableau: Float64Array; rows: number; cols: number; basis: Int32Array }) {
  const N = form.a.cols
  const width = s.cols + 1
  const z = new Float64Array(N)
  for (let r = 0; r < s.rows; r++) {
    const k = s.basis[r] ?? -1
    if (k >= 0 && k < N) z[k] = Math.max(0, s.tableau[r * width + s.cols] ?? 0)
  }
  return z
}

/**
 * The two-phase simplex method on a dense tableau, as a traced algorithm: every step is one pivot (or the switch from
 * phase 1 to phase 2). Phase 1 minimises the sum of artificial variables from the slack-and-artificial basis to find a
 * feasible vertex (or prove infeasibility), drives remaining artificial variables out and drops redundant rows; phase 2
 * optimises the cost. A pivot must exceed the tolerance and also `pivotFloor` (default 1e-7) times the largest
 * entry of its column (a tiny pivot on a degenerate program is rounding noise and ends on a singular basis). The
 * entering column is by default the one of the most negative reduced cost (Dantzig's rule, the quick one: a handful of
 * pivots where Bland's takes hundreds), with Bland's rule (smallest eligible index for entering and leaving, which
 * cannot cycle in exact arithmetic) taking over while the last pivots, more than there are rows of them, all left the
 * vertex where it was; `pivotRule: 'bland'` uses Bland's rule throughout, the slow path that is easy to follow. Updating
 * the tableau pivot by pivot accumulates rounding error, so before every decision (optimal, infeasible, end of phase 1), and every 50 pivots, the tableau is recomputed from the
 * original matrix and the basis (re-inversion, one LU solve); if the clean tableau disagrees (a negative reduced cost,
 * or a basic variable below zero), pivoting continues, with dual-simplex pivots to restore feasibility. A basis that
 * has become singular, or 8 re-inversions without agreement, end with status `numerical difficulties` (`linprog` then
 * re-solves by the interior-point method). On badly conditioned near-tied programs phase 1 can still stall and
 * report `infeasible`; `linprog` confirms such verdicts with the interior-point method. Records the objective, phase,
 * infeasibility, every pivot and the vertex.
 *
 * @param form - A standard form (from `standardForm`).
 * @param options - Pivot rule and tolerance.
 * @returns The algorithm.
 * @concept simplex-method
 * @example
 * ```ts
 * import { simplexAlgorithm, standardForm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const form = standardForm({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * runAlgorithm(simplexAlgorithm(form), { maxSteps: 50 }).state.path.map((p) => p.x) // => [[0, 0], [0, 2], [3, 1]]
 * ```
 */
export function simplexAlgorithm(form: StandardForm, options: LinprogOptions = {}): Algorithm<SimplexState> {
  const tol = options.tol ?? 1e-9
  const rule = options.pivotRule ?? 'dantzig'
  const N = form.a.cols
  const m = form.a.rows
  const bScale = 1 + maxOf(Array.from(form.b, Math.abs))
  const feasTol = tol * bScale
  const MAX_REINVERSIONS = 8
  const REINVERT_EVERY = 50
  const pivotRelative = options.pivotFloor ?? 1e-7
  // The run of pivots that left the vertex where it was (ratio 0) at the end of the path, counted up to `limit`: a
  // degenerate vertex is where Dantzig's rule can cycle, so a run longer than the number of rows hands the choice of
  // the entering column to Bland's rule until a pivot moves the vertex.
  const degenerateRun = (path: readonly LpPathPoint[], limit: number): number => {
    const last = path[path.length - 1]
    let run = 0
    for (let i = path.length - 2; i >= 0 && run <= limit; i--) {
      const x = path[i]?.x
      if (last === undefined || x === undefined || x.some((v, k) => v !== last.x[k])) break
      run += 1
    }
    return run
  }
  // Artificial column N + k belongs to row artRow[k] (the rows init cannot start from a slack).
  const artRow: number[] = []
  for (let r = 0; r < m; r++) {
    const sl = form.slackOf[r] ?? -1
    if (!(sl >= 0 && (form.rows[r]?.sign ?? 1) === 1)) artRow.push(r)
  }
  const column = (j: number, row: number): number =>
    j < N ? (form.a.data[row * N + j] ?? 0) : artRow[j - N] === row ? 1 : 0
  const costOf = (phase: 1 | 2, j: number): number => (phase === 2 ? (form.c[j] ?? 0) : j >= N ? 1 : 0)
  type Core = Omit<SimplexState, 'solution' | 'finished'>
  // The tableau recomputed from the original matrix: B^-1 [A | b] for the basis B, and reduced costs c - c_B B^-1 A.
  const reinvert = (s: Core): Core | undefined => {
    const rows = s.rows
    const width = s.cols + 1
    const basisMatrix = new Float64Array(rows * rows)
    const rhs = new Float64Array(rows * width)
    for (let i = 0; i < rows; i++) {
      const row = s.rowIds[i] ?? 0
      for (let k = 0; k < rows; k++) basisMatrix[i * rows + k] = column(s.basis[k] ?? 0, row)
      for (let j = 0; j < s.cols; j++) rhs[i * width + j] = column(j, row)
      rhs[i * width + s.cols] = form.b[row] ?? 0
    }
    let x: Float64Array
    try {
      x = dgesv(basisMatrix, rhs, rows, width)
    } catch {
      return undefined
    }
    const t = new Float64Array((rows + 1) * width)
    t.set(x)
    for (let j = 0; j <= s.cols; j++) {
      let d = j < s.cols ? costOf(s.phase, j) : 0
      for (let i = 0; i < rows; i++) d -= costOf(s.phase, s.basis[i] ?? 0) * (x[i * width + j] ?? 0)
      t[rows * width + j] = d
    }
    for (let i = 0; i < rows; i++) t[i * width + (s.basis[i] ?? 0)] = 1
    return { ...s, tableau: t, reinversions: s.reinversions + 1 }
  }
  // The row of the most negative basic value beyond the feasibility tolerance, or -1.
  const infeasibleRow = (s: Core): number => {
    const width = s.cols + 1
    let r = -1
    let worst = -feasTol
    for (let i = 0; i < s.rows; i++) {
      const v = s.tableau[i * width + s.cols] ?? 0
      if (v < worst) {
        worst = v
        r = i
      }
    }
    return r
  }
  const vertex = (
    s: SimplexState | Omit<SimplexState, 'path' | 'solution' | 'finished' | 'iterations'>,
  ): LpPathPoint => {
    const x = fromStandard(form, point(form, s))
    return { phase: s.phase, x: Array.from(x), objective: ddot(form.original.c, x) }
  }
  const solutionOf = (
    s: Omit<SimplexState, 'solution' | 'finished'>,
    status: StandardSolution['status'],
  ): StandardSolution => {
    const z = point(form, s)
    if (status !== 'optimal' || s.phase !== 2)
      return { status, z, y: new Float64Array(m), iterations: s.iterations, path: s.path }
    const width = s.cols + 1
    const rowsDuals = basisDuals(form, s.basis, s.rowIds)
    const y = new Float64Array(m)
    for (const [i, r] of s.rowIds.entries()) y[r] = rowsDuals[i] ?? 0
    const reducedCosts = Float64Array.from({ length: N }, (_, k) => s.tableau[s.rows * width + k] ?? 0)
    return {
      status,
      z,
      y,
      iterations: s.iterations,
      path: s.path,
      basis: Int32Array.from(s.basis),
      rowIds: Int32Array.from(s.rowIds),
      tableau: s.tableau.slice(0, s.rows * width),
      reducedCosts,
    }
  }
  const finish = (s: Omit<SimplexState, 'solution' | 'finished'>, status: StandardSolution['status']): SimplexState => {
    const { pivot: _last, ...rest } = s
    return { ...rest, solution: solutionOf(s, status), finished: true }
  }
  const ongoing = (s: Omit<SimplexState, 'solution' | 'finished'>): SimplexState => ({
    ...s,
    solution: solutionOf(s, 'iteration limit'),
    finished: false,
  })

  // Phase 2 from a phase-1 tableau: drive artificial variables out, drop redundant rows and artificial columns, price out.
  const toPhase2 = (s: Omit<SimplexState, 'solution' | 'finished'>): Omit<SimplexState, 'solution' | 'finished'> => {
    let t = s.tableau
    let width = s.cols + 1
    const basis = Int32Array.from(s.basis)
    const keep: number[] = []
    for (let r = 0; r < s.rows; r++) {
      if ((basis[r] ?? 0) < N) {
        keep.push(r)
        continue
      }
      let e = -1
      for (let j = 0; j < N; j++) {
        if (Math.abs(t[r * width + j] ?? 0) > tol) {
          e = j
          break
        }
      }
      if (e >= 0) {
        t = pivotOn(t, s.rows, width, r, e)
        basis[r] = e
        keep.push(r)
      }
    }
    const rows = keep.length
    const newWidth = N + 1
    const next = new Float64Array((rows + 1) * newWidth)
    const nb = new Int32Array(rows)
    const ids = new Int32Array(rows)
    for (const [i, r] of keep.entries()) {
      for (let j = 0; j < N; j++) next[i * newWidth + j] = t[r * width + j] ?? 0
      next[i * newWidth + N] = t[r * width + s.cols] ?? 0
      nb[i] = basis[r] ?? 0
      ids[i] = s.rowIds[r] ?? 0
    }
    width = newWidth
    let obj = 0
    for (let j = 0; j < N; j++) {
      let d = form.c[j] ?? 0
      for (let i = 0; i < rows; i++) d -= (form.c[nb[i] ?? 0] ?? 0) * (next[i * width + j] ?? 0)
      next[rows * width + j] = d
    }
    for (let i = 0; i < rows; i++) obj += (form.c[nb[i] ?? 0] ?? 0) * (next[i * width + N] ?? 0)
    next[rows * width + N] = -obj
    return { ...s, phase: 2, tableau: next, rows, cols: N, basis: nb, rowIds: ids }
  }

  return {
    name: 'simplex',
    config: lpConfig(options),
    schema: LP_SCHEMA,
    init: () => {
      const basis = new Int32Array(m)
      let nArt = 0
      for (let r = 0; r < m; r++) {
        const s = form.slackOf[r] ?? -1
        if (s >= 0 && (form.rows[r]?.sign ?? 1) === 1) basis[r] = s
        else basis[r] = N + nArt++
      }
      const cols = N + nArt
      const width = cols + 1
      const t = new Float64Array((m + 1) * width)
      for (let r = 0; r < m; r++) {
        for (let j = 0; j < N; j++) t[r * width + j] = form.a.data[r * N + j] ?? 0
        if ((basis[r] ?? 0) >= N) t[r * width + (basis[r] ?? 0)] = 1
        t[r * width + cols] = form.b[r] ?? 0
      }
      // Phase-1 reduced costs: minus the sum of the rows with an artificial basic variable.
      for (let r = 0; r < m; r++) {
        if ((basis[r] ?? 0) < N) continue
        for (let j = 0; j < N; j++) t[m * width + j] = (t[m * width + j] ?? 0) - (t[r * width + j] ?? 0)
        t[m * width + cols] = (t[m * width + cols] ?? 0) - (t[r * width + cols] ?? 0)
      }
      const start = {
        phase: 1 as const,
        tableau: t,
        rows: m,
        cols,
        basis,
        rowIds: Int32Array.from({ length: m }, (_, i) => i),
        iterations: 0,
        reinversions: 0,
      }
      const base = nArt === 0 ? toPhase2(start as Omit<SimplexState, 'solution' | 'finished'>) : start
      return ongoing({ ...base, path: [vertex(base as SimplexState)] } as Omit<SimplexState, 'solution' | 'finished'>)
    },
    step: (current) => {
      if (current.finished) return current
      let s: Core = current
      if (s.iterations > 0 && s.iterations % REINVERT_EVERY === 0 && s.reinversions * REINVERT_EVERY < s.iterations)
        s = reinvert(s) ?? s
      const width = s.cols + 1
      const obj = s.rows * width
      // A basic value below zero (exposed by re-inversion): a dual-simplex pivot on its row restores feasibility.
      const bad = infeasibleRow(s)
      if (bad >= 0) {
        let e = -1
        let best = Infinity
        for (let j = 0; j < s.cols; j++) {
          const a = s.tableau[bad * width + j] ?? 0
          if (a >= -tol) continue
          const q = Math.max(0, s.tableau[obj + j] ?? 0) / -a
          if (q < best) {
            best = q
            e = j
          }
        }
        // The row reads (sum of non-negative variables times non-negative entries) = a negative value.
        if (e < 0) return finish(s, 'infeasible')
        const tableau = pivotOn(s.tableau, s.rows, width, bad, e)
        const basis = Int32Array.from(s.basis)
        const leaving = basis[bad] ?? -1
        basis[bad] = e
        const next = {
          ...s,
          tableau,
          basis,
          iterations: s.iterations + 1,
          pivot: { entering: e, leaving, ratio: best },
        }
        return ongoing({ ...next, path: [...s.path, vertex(next)] })
      }
      let e = -1
      let best = -tol
      const bland = rule === 'bland' || degenerateRun(s.path, s.rows) > s.rows
      for (let j = 0; j < s.cols; j++) {
        const d = s.tableau[obj + j] ?? 0
        if (d < best) {
          e = j
          if (bland) break
          best = d
        }
      }
      if (e < 0) {
        // Decide only on a clean tableau: if re-inversion shows more work, keep pivoting from it.
        if (s.reinversions >= MAX_REINVERSIONS) return finish(s, 'numerical difficulties')
        const clean = reinvert(s)
        // A singular basis means rounding has corrupted the tableau: no decision drawn from it is trustworthy.
        if (clean === undefined) return finish(s, 'numerical difficulties')
        {
          const cleanObj = clean.rows * width
          let stillOptimal = infeasibleRow(clean) < 0
          for (let j = 0; j < clean.cols && stillOptimal; j++)
            if ((clean.tableau[cleanObj + j] ?? 0) < -tol) stillOptimal = false
          if (!stillOptimal) return ongoing({ ...clean, path: s.path })
          s = clean
        }
        if (s.phase === 1) {
          const infeasibility = -(s.tableau[obj + s.cols] ?? 0)
          if (infeasibility > feasTol) return finish(s, 'infeasible')
          const { pivot: _last, ...p2 } = toPhase2(s)
          return ongoing({ ...p2, path: [...s.path, vertex(p2 as SimplexState)] })
        }
        return finish(s, 'optimal')
      }
      // A pivot must be large beside the rest of its column, not only above the absolute tolerance: on degenerate
      // programs (right-hand sides of zero, so every ratio ties) a tiny pivot is rounding noise, or a real entry that
      // multiplies the tableau's error by its inverse, and either leaves a singular basis (fast-check found an
      // equilibrium program on nearly tied payoffs where the simplex stopped with 'numerical difficulties' for both
      // rules and the interior point wrongly said 'unbounded'). A column with no acceptable pivot is re-confirmed on a
      // clean tableau below.
      let r = -1
      let ratio = Infinity
      let colMax = 0
      for (let i = 0; i < s.rows; i++) colMax = Math.max(colMax, Math.abs(s.tableau[i * width + e] ?? 0))
      const pivotFloor = Math.max(tol, pivotRelative * colMax)
      for (let i = 0; i < s.rows; i++) {
        const a = s.tableau[i * width + e] ?? 0
        if (a <= pivotFloor) continue
        const q = (s.tableau[i * width + s.cols] ?? 0) / a
        if (
          r < 0 ||
          q < ratio - 1e-12 * Math.max(1, Math.abs(ratio)) ||
          (Math.abs(q - ratio) <= 1e-12 * Math.max(1, Math.abs(ratio)) && (s.basis[i] ?? 0) < (s.basis[r] ?? 0))
        ) {
          ratio = q
          r = i
        }
      }
      if (r < 0) {
        // Confirm on a clean tableau: a column with no positive entry may be rounding, not a ray.
        if (s.reinversions >= MAX_REINVERSIONS) return finish(s, 'numerical difficulties')
        const clean = reinvert(s)
        if (clean === undefined) return finish(s, 'numerical difficulties')
        let ray = (clean.tableau[clean.rows * width + e] ?? 0) < -tol
        for (let i = 0; i < clean.rows && ray; i++) if ((clean.tableau[i * width + e] ?? 0) > tol) ray = false
        return ray ? finish(clean, 'unbounded') : ongoing({ ...clean, path: s.path })
      }
      const tableau = pivotOn(s.tableau, s.rows, width, r, e)
      const basis = Int32Array.from(s.basis)
      const leaving = basis[r] ?? -1
      basis[r] = e
      const next = { ...s, tableau, basis, iterations: s.iterations + 1, pivot: { entering: e, leaving, ratio } }
      return ongoing({ ...next, path: [...s.path, vertex(next)] })
    },
    done: (s) => (s.finished ? runStatusOf(s.solution.status) : undefined),
    observe: (s) => {
      const width = s.cols + 1
      const last = s.path[s.path.length - 1]
      const value = -(s.tableau[s.rows * width + s.cols] ?? 0)
      return {
        metrics: {
          phase: s.phase,
          ...(s.phase === 1 ? { infeasibility: value } : {}),
          ...(last === undefined ? {} : { objective: last.objective }),
        },
        ...(last === undefined ? {} : { tensors: { x: { role: 'param' as const, value: last.x } } }),
        ...(s.pivot === undefined
          ? {}
          : {
              messages: [
                {
                  kind: 'pivot',
                  payload: {
                    entering: s.pivot.entering,
                    leaving: s.pivot.leaving,
                    ratio: Number.isFinite(s.pivot.ratio) ? s.pivot.ratio : null,
                  },
                },
              ],
            }),
      }
    },
  }
}
