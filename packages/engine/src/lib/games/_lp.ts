// Linear programs of the games domain: optim.linprog, with a fallback for programs the simplex mis-solves.
import {
  type LinearProgram,
  type LinprogOptions,
  type LinprogResult,
  linprog,
  linprogInteriorPoint,
} from '../optim/index.ts'
import { toFlat } from '../tensor/index.ts'

/** Whether a result's constraint residuals (`con = b_eq - A_eq x`, `slack = b_ub - A_ub x`) show a feasible point. */
function feasible(r: LinprogResult, tol: number): boolean {
  for (const v of toFlat(r.con)) if (!(Math.abs(v) <= tol)) return false
  for (const v of toFlat(r.slack)) if (!(v >= -tol)) return false
  return true
}

/**
 * Solve with the requested method (default the simplex); when none was requested and the simplex does not report an
 * optimum, or reports one whose point violates the constraints by more than 1e-7, retry with the interior-point
 * method. The equilibrium, maxmin and excess programs here are well posed, but payoff differences of very different
 * sizes (near ties) strain the simplex's pivot tolerances.
 */
export function solveLp(problem: LinearProgram, options: LinprogOptions = {}): LinprogResult {
  const r = linprog(problem, options)
  if (options.method !== undefined || (r.success && feasible(r, 1e-7))) return r
  const ip = linprogInteriorPoint(problem, options)
  return ip.success || !r.success ? ip : r
}
