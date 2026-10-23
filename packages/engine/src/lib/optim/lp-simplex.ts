import { maxOf } from '../base/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import type { Algorithm } from '../trace/index.ts'
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
 * optimises the cost. Bland's rule (smallest eligible index for entering and leaving) prevents cycling; Dantzig's rule
 * (most negative reduced cost) is available. Records the objective, phase, infeasibility, every pivot and the vertex.
 *
 * @param form - A standard form (from `standardForm`).
 * @param options - Pivot rule and tolerance.
 * @returns The algorithm.
 * @concept simplex-method
 * @example
 * ```ts
 * import { simplexAlgorithm, standardForm } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const form = standardForm({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * runAlgorithm(simplexAlgorithm(form), { maxSteps: 50 }).state.path.map((p) => p.x) // => [[0, 0], [4, 0], [3, 1]]
 * ```
 */
export function simplexAlgorithm(form: StandardForm, options: LinprogOptions = {}): Algorithm<SimplexState> {
  const tol = options.tol ?? 1e-9
  const rule = options.pivotRule ?? 'bland'
  const N = form.a.cols
  const m = form.a.rows
  const bScale = 1 + maxOf(Array.from(form.b, Math.abs))
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
      }
      const base = nArt === 0 ? toPhase2(start as Omit<SimplexState, 'solution' | 'finished'>) : start
      return ongoing({ ...base, path: [vertex(base as SimplexState)] } as Omit<SimplexState, 'solution' | 'finished'>)
    },
    step: (s) => {
      if (s.finished) return s
      const width = s.cols + 1
      const obj = s.rows * width
      let e = -1
      let best = -tol
      for (let j = 0; j < s.cols; j++) {
        const d = s.tableau[obj + j] ?? 0
        if (d < best) {
          e = j
          if (rule === 'bland') break
          best = d
        }
      }
      if (e < 0) {
        if (s.phase === 1) {
          const infeasibility = -(s.tableau[obj + s.cols] ?? 0)
          if (infeasibility > tol * bScale) return finish(s, 'infeasible')
          const { pivot: _last, ...p2 } = toPhase2(s)
          return ongoing({ ...p2, path: [...s.path, vertex(p2 as SimplexState)] })
        }
        return finish(s, 'optimal')
      }
      let r = -1
      let ratio = Infinity
      for (let i = 0; i < s.rows; i++) {
        const a = s.tableau[i * width + e] ?? 0
        if (a <= tol) continue
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
      if (r < 0) return finish(s, 'unbounded')
      const tableau = pivotOn(s.tableau, s.rows, width, r, e)
      const basis = Int32Array.from(s.basis)
      const leaving = basis[r] ?? -1
      basis[r] = e
      const next = { ...s, tableau, basis, iterations: s.iterations + 1, pivot: { entering: e, leaving, ratio } }
      return ongoing({ ...next, path: [...s.path, vertex(next)] })
    },
    done: (s) => (s.finished ? (s.solution.status === 'optimal' ? 'converged' : 'failed') : undefined),
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
