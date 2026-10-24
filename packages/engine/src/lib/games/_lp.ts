// Linear programs of the games domain: optim.linprog, with a fallback for programs known to be feasible and bounded.
import {
  type LinearProgram,
  type LinprogOptions,
  type LinprogResult,
  linprog,
  linprogInteriorPoint,
} from '../optim/index.ts'

/**
 * Solve with the requested method (default the simplex); when none was requested and the simplex does not report an
 * optimum, retry with the interior-point method. Every program here (maximin, equilibrium polytopes, dominance,
 * excess minimisation) is feasible and bounded by construction, so a failure is the simplex's tolerances meeting payoff
 * differences of very different sizes, which the interior-point method tolerates.
 */
export function solveLp(problem: LinearProgram, options: LinprogOptions = {}): LinprogResult {
  const r = linprog(problem, options)
  if (r.success || options.method !== undefined) return r
  return linprogInteriorPoint(problem, options)
}
