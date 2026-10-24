// The largest eigenpairs of a symmetric tridiagonal matrix by Sturm bisection and inverse iteration, for the DPSS
// tapers (the eigenproblem of Percival and Walden, 1993, as scipy.signal.windows.dpss sets it up).
import { ddot, dnrm2 } from '../../numerics/linalg/index.ts'

/** The number of eigenvalues below x of the tridiagonal matrix with diagonal d and off-diagonal e (e[i] couples i, i+1). */
function sturmCount(d: Float64Array, e: Float64Array, x: number, tiny: number): number {
  let count = 0
  let q = (d[0] ?? 0) - x
  if (q < 0) count++
  for (let i = 1; i < d.length; i++) {
    const denom = Math.abs(q) < tiny ? (q < 0 ? -tiny : tiny) : q
    q = (d[i] ?? 0) - x - ((e[i - 1] ?? 0) * (e[i - 1] ?? 0)) / denom
    if (q < 0) count++
  }
  return count
}

/** The idx-th smallest eigenvalue (0-based) by bisection between Gershgorin bounds. */
function eigenvalue(d: Float64Array, e: Float64Array, idx: number, lo0: number, hi0: number, tiny: number): number {
  let lo = lo0
  let hi = hi0
  for (let it = 0; it < 200; it++) {
    const mid = 0.5 * (lo + hi)
    if (mid === lo || mid === hi) break
    if (sturmCount(d, e, mid, tiny) > idx) hi = mid
    else lo = mid
  }
  return 0.5 * (lo + hi)
}

/** Solve (T - shift I) y = b for tridiagonal T by Gaussian elimination with partial pivoting (as LAPACK dgtsv). */
function shiftedSolve(d: Float64Array, e: Float64Array, shift: number, b: Float64Array, tiny: number): Float64Array {
  const n = d.length
  // Rows hold up to three non-zeros after pivoting: diagonal, first and second super-diagonals.
  const dl = Float64Array.from({ length: n - 1 }, (_, i) => e[i] ?? 0)
  const dd = Float64Array.from(d, (v) => v - shift)
  const du = Float64Array.from({ length: n - 1 }, (_, i) => e[i] ?? 0)
  const du2 = new Float64Array(Math.max(n - 2, 0))
  const y = Float64Array.from(b)
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(dd[i] ?? 0) >= Math.abs(dl[i] ?? 0)) {
      if (Math.abs(dd[i] ?? 0) < tiny) dd[i] = tiny
      const f = (dl[i] ?? 0) / (dd[i] ?? 1)
      dd[i + 1] = (dd[i + 1] ?? 0) - f * (du[i] ?? 0)
      y[i + 1] = (y[i + 1] ?? 0) - f * (y[i] ?? 0)
      if (i < n - 2) du2[i] = 0
    } else {
      const f = (dd[i] ?? 0) / (dl[i] ?? 1)
      dd[i] = dl[i] ?? 0
      const t = dd[i + 1] ?? 0
      dd[i + 1] = (du[i] ?? 0) - f * t
      du[i] = t
      if (i < n - 2) {
        du2[i] = du[i + 1] ?? 0
        du[i + 1] = -f * (du[i + 1] ?? 0)
      }
      const yi = y[i] ?? 0
      y[i] = y[i + 1] ?? 0
      y[i + 1] = yi - f * (y[i + 1] ?? 0)
    }
  }
  if (Math.abs(dd[n - 1] ?? 0) < tiny) dd[n - 1] = tiny
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i] ?? 0
    if (i + 1 < n) s -= (du[i] ?? 0) * (x[i + 1] ?? 0)
    if (i + 2 < n) s -= (du2[i] ?? 0) * (x[i + 2] ?? 0)
    x[i] = s / (dd[i] ?? 1)
  }
  return x
}

/**
 * The first K DPSS tapers of length M with half-bandwidth NW: unit-norm eigenvectors of the tridiagonal matrix with
 * diagonal ((M - 1 - 2t) / 2)^2 cos(2 pi W) and off-diagonal t (M - t) / 2, by decreasing eigenvalue, with scipy's
 * signs (symmetric tapers have a positive sum; antisymmetric ones start with a positive lobe).
 */
export function dpssTapers(m: number, nw: number, k: number): Float64Array[] {
  const w = nw / m
  const d = Float64Array.from({ length: m }, (_, t) => ((m - 1 - 2 * t) / 2) ** 2 * Math.cos(2 * Math.PI * w))
  const e = Float64Array.from({ length: m - 1 }, (_, i) => ((i + 1) * (m - i - 1)) / 2)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let i = 0; i < m; i++) {
    const r = Math.abs(e[i - 1] ?? 0) + Math.abs(e[i] ?? 0)
    lo = Math.min(lo, (d[i] ?? 0) - r)
    hi = Math.max(hi, (d[i] ?? 0) + r)
  }
  const scale = Math.max(Math.abs(lo), Math.abs(hi), 1)
  const tiny = Number.EPSILON * scale
  const out: Float64Array[] = []
  for (let j = 0; j < k; j++) {
    const lambda = eigenvalue(d, e, m - 1 - j, lo, hi, tiny * 1e-3)
    // Inverse iteration from a start with both symmetric and antisymmetric parts, orthogonal to earlier tapers.
    let v: Float64Array = Float64Array.from({ length: m }, (_, i) => 1 + Math.sin(0.7 * i + 0.3))
    for (let it = 0; it < 4; it++) {
      v = shiftedSolve(d, e, lambda, v, tiny)
      for (const u of out) {
        const c = ddot(u, v)
        for (let i = 0; i < m; i++) v[i] = (v[i] ?? 0) - c * (u[i] ?? 0)
      }
      const nrm = dnrm2(v)
      for (let i = 0; i < m; i++) v[i] = (v[i] ?? 0) / nrm
    }
    if (j % 2 === 0) {
      let s = 0
      for (const x of v) s += x
      if (s < 0) for (let i = 0; i < m; i++) v[i] = -(v[i] ?? 0)
    } else {
      const thresh = Math.max(1e-7, 1 / m)
      for (const x of v) {
        if (x * x > thresh) {
          if (x < 0) for (let i = 0; i < m; i++) v[i] = -(v[i] ?? 0)
          break
        }
      }
    }
    out.push(v)
  }
  return out
}
