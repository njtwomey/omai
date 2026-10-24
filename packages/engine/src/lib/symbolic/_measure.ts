// Expression size measures used to choose between rewrites (a version of SymPy's `count_ops`), and a numeric
// equivalence check at fixed sample points used to guard heuristic rewrites.
import { isNumeric } from './_canon.ts'
import { children } from './_node.ts'
import { type C, evalComplex, freeNames } from './_numeric.ts'
import type { Expr } from './expr.ts'

/** Number of operations: one per argument joined by + or *, per power, function, division and negation. @internal */
export function countOps(e: Expr): number {
  switch (e.kind) {
    case 'num':
      return e.d !== 1n ? 1 + (e.n < 0n ? 1 : 0) : e.n < 0n ? 1 : 0
    case 'float':
      return e.value < 0 ? 1 : 0
    case 'sym':
    case 'const':
      return 0
    case 'add': {
      let s = e.args.length - 1
      for (const a of e.args) s += countOps(a)
      return s
    }
    case 'mul': {
      let s = e.args.length - 1
      for (const a of e.args) s += countOps(a)
      return s
    }
    case 'pow':
      return 1 + countOps(e.base) + (isNumeric(e.exp) && e.exp.kind === 'num' && e.exp.n === -1n ? 0 : countOps(e.exp))
    default: {
      let s = 1
      for (const c of children(e)) s += countOps(c)
      return s
    }
  }
}

/** Ops first, then printed length, to break ties. @internal */
export function complexity(e: Expr): number {
  return countOps(e) * 1000 + Math.min(999, e.key.length / 4)
}

const SAMPLES = [0.37, -0.81, 1.23, 2.07, -1.61, 0.53]

/**
 * Whether two expressions agree numerically at a few fixed complex-safe sample points (each free symbol gets its
 * own offset). Points where either side cannot be evaluated or is not finite are skipped; at least two must agree.
 * @internal
 */
export function numericallyEqual(a: Expr, b: Expr, tol = 1e-8): boolean {
  return numericCompare(a, b, tol) === 'equal'
}

/**
 * Compare two expressions at the sample points: equal, different, or unknown when too few points can be evaluated
 * (functions without numeric implementations, such as erf).
 * @internal
 */
export function numericCompare(a: Expr, b: Expr, tol = 1e-8): 'equal' | 'different' | 'unknown' {
  const names = [...new Set([...freeNames(a), ...freeNames(b)])].sort()
  let agreed = 0
  for (let s = 0; s < SAMPLES.length; s++) {
    const env = new Map<string, C>()
    names.forEach((n, i) => {
      env.set(n, [(SAMPLES[(s + i) % SAMPLES.length] as number) + 0.1 * i, 0])
    })
    const va = evalComplex(a, env)
    const vb = evalComplex(b, env)
    if (va === undefined || vb === undefined) continue
    if (![va[0], va[1], vb[0], vb[1]].every(Number.isFinite)) continue
    const scale = Math.max(1, Math.hypot(va[0], va[1]), Math.hypot(vb[0], vb[1]))
    if (Math.hypot(va[0] - vb[0], va[1] - vb[1]) > tol * scale) return 'different'
    agreed++
  }
  return agreed >= 2 || (agreed >= 1 && names.length === 0) ? 'equal' : 'unknown'
}
