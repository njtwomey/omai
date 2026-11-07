// The trust-region subproblem of COBYLA (Powell's TRSTLP strategy; not exported from the barrel).
import { daxpy, ddot, dnrm2, solve } from '../../numerics/linalg/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { rethrowUnless } from '../_catch.ts'

/**
 * Minimise a linear function q.z subject to rows.z >= rhs and |z[0..ballDims)| <= radius, from a feasible z, by an
 * active-set method that follows the projected steepest descent direction and stops on the ball (Powell's TRSTLP
 * strategy).
 */
function linearInBall(
  z0: Float64Array,
  q: Float64Array,
  rows: readonly Float64Array[],
  rhs: readonly number[],
  ballDims: number,
  radius: number,
): Float64Array {
  let z: Float64Array = Float64Array.from(z0)
  const nz = z.length
  const active: number[] = []
  const tight = 1e-12 * Math.max(1, radius)
  for (const [k, r] of rows.entries()) if (Math.abs(ddot(r, z) - (rhs[k] ?? 0)) <= tight) active.push(k)
  const qn = dnrm2(q)
  for (let iter = 0; iter < 4 * (nz + rows.length) + 10; iter++) {
    // Project -q onto the null space of the active rows, dropping rows with negative multipliers.
    let s: Float64Array = Float64Array.from(q, (v) => -v)
    for (;;) {
      if (active.length === 0) break
      const k = active.length
      const gram = new Float64Array(k * k)
      const aq = new Float64Array(k)
      let trace = 0
      for (let i = 0; i < k; i++) {
        const ri = rows[active[i] ?? 0] as Float64Array
        aq[i] = ddot(ri, q)
        for (let j = 0; j < k; j++) gram[i * k + j] = ddot(ri, rows[active[j] ?? 0] as Float64Array)
        trace += gram[i * k + i] ?? 0
      }
      for (let i = 0; i < k; i++) gram[i * k + i] = (gram[i * k + i] ?? 0) + 1e-13 * Math.max(trace, 1e-300)
      let lambda: number[]
      try {
        lambda = toFlat(solve(fromStorage(gram, [k, k], 'float64'), fromStorage(aq, [k], 'float64')))
      } catch (error) {
        rethrowUnless(error, 'value', 'singular')
        active.pop()
        continue
      }
      let worst = -1
      let worstValue = 0
      for (const [i, l] of lambda.entries()) {
        if (l < worstValue) {
          worstValue = l
          worst = i
        }
      }
      if (worst >= 0) {
        active.splice(worst, 1)
        continue
      }
      s = Float64Array.from(q, (v) => -v)
      for (const [i, l] of lambda.entries()) s = daxpy(l, rows[active[i] ?? 0] as Float64Array, s)
      break
    }
    if (dnrm2(s) <= 1e-12 * Math.max(qn, 1e-300)) break
    // The longest step along s that keeps the inactive constraints and the ball.
    let t = Number.POSITIVE_INFINITY
    let blocking = -1
    let zs = 0
    let ss = 0
    let zz = 0
    for (let i = 0; i < ballDims; i++) {
      zs += (z[i] ?? 0) * (s[i] ?? 0)
      ss += (s[i] ?? 0) ** 2
      zz += (z[i] ?? 0) ** 2
    }
    if (ss > 0) t = (-zs + Math.sqrt(Math.max(0, zs * zs + ss * (radius * radius - zz)))) / ss
    for (const [k, r] of rows.entries()) {
      if (active.includes(k)) continue
      const rs = ddot(r, s)
      if (rs >= 0) continue
      const tk = Math.max(0, ((rhs[k] ?? 0) - ddot(r, z)) / rs)
      if (tk < t) {
        t = tk
        blocking = k
      }
    }
    if (!Number.isFinite(t)) break
    z = daxpy(t, s, z)
    if (blocking < 0) break
    active.push(blocking)
  }
  return z
}

/** The trust-region step of COBYLA: least violation of the linearised constraints first, then the linearised objective. */
export function trustStep(g: Float64Array, a: readonly Float64Array[], c0: Float64Array, rho: number): Float64Array {
  const n = g.length
  const m = a.length
  let resmax0 = 0
  for (const v of c0) resmax0 = Math.max(resmax0, -v)
  let d: Float64Array = new Float64Array(n)
  let level = 0
  if (resmax0 > 0) {
    // Phase 1: minimise t subject to a_k.d + t >= -c_k, t >= 0, |d| <= rho.
    const rows = a.map((ak) => Float64Array.from([...ak, 1]))
    rows.push(Float64Array.from({ length: n + 1 }, (_, i) => (i === n ? 1 : 0)))
    const rhs = [...Array.from(c0, (v) => -v), 0]
    const q = Float64Array.from({ length: n + 1 }, (_, i) => (i === n ? 1 : 0))
    const z = linearInBall(Float64Array.from([...d, resmax0]), q, rows, rhs, n, rho)
    d = z.slice(0, n)
    level = Math.max(0, z[n] ?? 0)
  }
  if (m === 0) {
    const gn = dnrm2(g)
    return gn > 0 ? Float64Array.from(g, (v) => (-rho * v) / gn) : d
  }
  // Phase 2: minimise g.d keeping the violation at its least value.
  const rhs = Array.from(c0, (v) => -v - level)
  return linearInBall(d, g, a, rhs, n, rho)
}
