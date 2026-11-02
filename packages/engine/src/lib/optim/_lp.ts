// Plumbing shared by the linear-programming files (not exported from the barrel).
import type { Json } from '../base/index.ts'
import { channels, type TraceSchema } from '../trace/index.ts'
import { solveDenseTransposed } from './_programming.ts'
import type { LinprogOptions, StandardForm } from './lp-standard-form.ts'

/** Duals of the standard rows from an optimal basis: B^T y = c_B. */
export function basisDuals(form: StandardForm, basis: Int32Array, rowIds: Int32Array): Float64Array {
  const bmat = basisMatrix(form, basis, rowIds)
  const m = basis.length
  const cb = Float64Array.from(basis, (k) => form.c[k] ?? 0)
  return solveDenseTransposed(bmat, cb, m)
}

/** The basis matrix B (the basic columns of the kept rows), m x m row-major. */
export function basisMatrix(form: StandardForm, basis: Int32Array, rowIds: Int32Array): Float64Array {
  const m = basis.length
  const N = form.a.cols
  const bmat = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    const r = rowIds[i] ?? 0
    for (let k = 0; k < m; k++) bmat[i * m + k] = form.a.data[r * N + (basis[k] ?? 0)] ?? 0
  }
  return bmat
}

/** The trace schema of the linear-programming algorithms. */
export const LP_SCHEMA: TraceSchema = channels({
  metrics: {
    objective: { description: 'objective value of the current point', direction: 'lower' },
    phase: { description: 'simplex phase (1 finds a feasible vertex, 2 optimises)' },
    infeasibility: { description: 'phase-1 objective: total artificial infeasibility' },
    primalResidual: { description: 'relative primal residual (interior point)' },
    dualResidual: { description: 'relative dual residual (interior point)' },
    gap: { description: 'relative duality gap (interior point)' },
  },
  tensors: { x: { role: 'param', description: 'current point in the original variables' } },
  messages: { pivot: { description: 'a simplex pivot: entering and leaving columns, ratio' } },
})

/** JSON configuration of a run. */
export function lpConfig(options: LinprogOptions): Json {
  return { tol: options.tol ?? null, pivotRule: options.pivotRule ?? 'bland', pivotFloor: options.pivotFloor ?? 1e-7 }
}
