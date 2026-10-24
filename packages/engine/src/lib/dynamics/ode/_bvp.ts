// The collocation machinery of scipy's solve_bvp (Kierzenka and Shampine's fourth-order Lobatto IIIA method with
// residual control), on row-major [n, m] arrays of numbers.
import { type SparseLu, sparseLu, sparseLuSolve } from './_sparse-lu.ts'

const EPS = Number.EPSILON

/** Vectorised field on numbers: x [m], y [n * m] row-major, p [k] -> f [n * m]. */
export type NumField = (x: Float64Array, y: Float64Array, p: Float64Array) => Float64Array
/** Boundary residual on numbers: ya [n], yb [n], p [k] -> [n + k]. */
export type NumBc = (ya: Float64Array, yb: Float64Array, p: Float64Array) => Float64Array

/** Column j of a row-major [n, m] array. */
export function column(y: Float64Array, n: number, m: number, j: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = y[i * m + j] ?? 0
  return out
}

/** The collocation residuals, midpoint states and field values (scipy's collocation_fun). */
export function collocation(
  fun: NumField,
  y: Float64Array,
  p: Float64Array,
  x: Float64Array,
  h: Float64Array,
  n: number,
): { colRes: Float64Array; yMid: Float64Array; f: Float64Array; fMid: Float64Array } {
  const m = x.length
  const f = fun(x, y, p)
  const yMid = new Float64Array(n * (m - 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m - 1; j++) {
      yMid[i * (m - 1) + j] =
        0.5 * ((y[i * m + j + 1] ?? 0) + (y[i * m + j] ?? 0)) -
        0.125 * (h[j] ?? 0) * ((f[i * m + j + 1] ?? 0) - (f[i * m + j] ?? 0))
    }
  }
  const xMid = Float64Array.from({ length: m - 1 }, (_, j) => (x[j] ?? 0) + 0.5 * (h[j] ?? 0))
  const fMid = fun(xMid, yMid, p)
  const colRes = new Float64Array(n * (m - 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m - 1; j++) {
      colRes[i * (m - 1) + j] =
        (y[i * m + j + 1] ?? 0) -
        (y[i * m + j] ?? 0) -
        ((h[j] ?? 0) / 6) * ((f[i * m + j] ?? 0) + (f[i * m + j + 1] ?? 0) + 4 * (fMid[i * (m - 1) + j] ?? 0))
    }
  }
  return { colRes, yMid, f, fMid }
}

/** Finite-difference df/dy [n][n][m] and df/dp [n][k][m] (scipy's estimate_fun_jac). */
function funJac(
  fun: NumField,
  x: Float64Array,
  y: Float64Array,
  p: Float64Array,
  f0: Float64Array,
  n: number,
): { dy: Float64Array; dp: Float64Array } {
  const m = x.length
  const k = p.length
  const dy = new Float64Array(n * n * m)
  for (let i = 0; i < n; i++) {
    const yNew = Float64Array.from(y)
    const hi = new Float64Array(m)
    for (let j = 0; j < m; j++) {
      const v = y[i * m + j] ?? 0
      yNew[i * m + j] = v + Math.sqrt(EPS) * (1 + Math.abs(v))
      hi[j] = (yNew[i * m + j] ?? 0) - v
    }
    const fNew = fun(x, yNew, p)
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < m; j++)
        dy[(r * n + i) * m + j] = ((fNew[r * m + j] ?? 0) - (f0[r * m + j] ?? 0)) / (hi[j] ?? 1)
    }
  }
  const dp = new Float64Array(n * k * m)
  for (let i = 0; i < k; i++) {
    const pNew = Float64Array.from(p)
    pNew[i] = (p[i] ?? 0) + Math.sqrt(EPS) * (1 + Math.abs(p[i] ?? 0))
    const hi = (pNew[i] ?? 0) - (p[i] ?? 0)
    const fNew = fun(x, y, pNew)
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < m; j++) dp[(r * k + i) * m + j] = ((fNew[r * m + j] ?? 0) - (f0[r * m + j] ?? 0)) / hi
    }
  }
  return { dy, dp }
}

/** Finite-difference boundary Jacobians as [n + k][n], [n + k][n], [n + k][k] (scipy's estimate_bc_jac). */
function bcJac(
  bc: NumBc,
  ya: Float64Array,
  yb: Float64Array,
  p: Float64Array,
  bc0: Float64Array,
): { a: Float64Array; b: Float64Array; p: Float64Array } {
  const n = ya.length
  const k = p.length
  const rows = n + k
  const da = new Float64Array(rows * n)
  const db = new Float64Array(rows * n)
  const dp = new Float64Array(rows * k)
  for (let i = 0; i < n; i++) {
    const a2 = Float64Array.from(ya)
    a2[i] = (ya[i] ?? 0) + Math.sqrt(EPS) * (1 + Math.abs(ya[i] ?? 0))
    const ha = (a2[i] ?? 0) - (ya[i] ?? 0)
    const ra = bc(a2, yb, p)
    const b2 = Float64Array.from(yb)
    b2[i] = (yb[i] ?? 0) + Math.sqrt(EPS) * (1 + Math.abs(yb[i] ?? 0))
    const hb = (b2[i] ?? 0) - (yb[i] ?? 0)
    const rb = bc(ya, b2, p)
    for (let r = 0; r < rows; r++) {
      da[r * n + i] = ((ra[r] ?? 0) - (bc0[r] ?? 0)) / ha
      db[r * n + i] = ((rb[r] ?? 0) - (bc0[r] ?? 0)) / hb
    }
  }
  for (let i = 0; i < k; i++) {
    const p2 = Float64Array.from(p)
    p2[i] = (p[i] ?? 0) + Math.sqrt(EPS) * (1 + Math.abs(p[i] ?? 0))
    const hp = (p2[i] ?? 0) - (p[i] ?? 0)
    const rp = bc(ya, yb, p2)
    for (let r = 0; r < rows; r++) dp[r * k + i] = ((rp[r] ?? 0) - (bc0[r] ?? 0)) / hp
  }
  return { a: da, b: db, p: dp }
}

/** The global collocation Jacobian (scipy's construct_global_jac), factorised. */
function globalJacobian(
  fun: NumField,
  bc: NumBc,
  x: Float64Array,
  h: Float64Array,
  y: Float64Array,
  p: Float64Array,
  yMid: Float64Array,
  f: Float64Array,
  fMid: Float64Array,
  bc0: Float64Array,
  n: number,
): SparseLu {
  const m = x.length
  const k = p.length
  const xMid = Float64Array.from({ length: m - 1 }, (_, j) => (x[j] ?? 0) + 0.5 * (h[j] ?? 0))
  const J = funJac(fun, x, y, p, f, n)
  const Jm = funJac(fun, xMid, yMid, p, fMid, n)
  const B = bcJac(bc, column(y, n, m, 0), column(y, n, m, m - 1), p, bc0)
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  const push = (r: number, c: number, v: number): void => {
    rows.push(r)
    cols.push(c)
    vals.push(v)
  }
  const at = (arr: Float64Array, r: number, c: number, j: number, width: number, mm: number): number =>
    arr[(r * width + c) * mm + j] ?? 0
  for (let j = 0; j < m - 1; j++) {
    const hj = h[j] ?? 0
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let t0 = 0
        let t1 = 0
        for (let c = 0; c < n; c++) {
          t0 += at(Jm.dy, a, c, j, n, m - 1) * at(J.dy, c, b, j, n, m)
          t1 += at(Jm.dy, a, c, j, n, m - 1) * at(J.dy, c, b, j + 1, n, m)
        }
        const d0 =
          (a === b ? -1 : 0) -
          (hj / 6) * (at(J.dy, a, b, j, n, m) + 2 * at(Jm.dy, a, b, j, n, m - 1)) -
          ((hj * hj) / 12) * t0
        const d1 =
          (a === b ? 1 : 0) -
          (hj / 6) * (at(J.dy, a, b, j + 1, n, m) + 2 * at(Jm.dy, a, b, j, n, m - 1)) +
          ((hj * hj) / 12) * t1
        push(j * n + a, j * n + b, d0)
        push(j * n + a, (j + 1) * n + b, d1)
      }
      for (let q = 0; q < k; q++) {
        let t = 0
        for (let c = 0; c < n; c++)
          t += at(Jm.dy, a, c, j, n, m - 1) * (at(J.dp, c, q, j, k, m) - at(J.dp, c, q, j + 1, k, m))
        const mid = at(Jm.dp, a, q, j, k, m - 1) + 0.125 * hj * t
        push(j * n + a, m * n + q, (-hj / 6) * (at(J.dp, a, q, j, k, m) + at(J.dp, a, q, j + 1, k, m) + 4 * mid))
      }
    }
  }
  for (let r = 0; r < n + k; r++) {
    for (let c = 0; c < n; c++) {
      push((m - 1) * n + r, c, B.a[r * n + c] ?? 0)
      push((m - 1) * n + r, (m - 1) * n + c, B.b[r * n + c] ?? 0)
    }
    for (let q = 0; q < k; q++) push((m - 1) * n + r, m * n + q, B.p[r * k + q] ?? 0)
  }
  return sparseLu(m * n + k, rows, cols, vals)
}

function residualVector(colRes: Float64Array, bcRes: Float64Array, n: number, m: number): Float64Array {
  const out = new Float64Array(n * (m - 1) + bcRes.length)
  for (let j = 0; j < m - 1; j++) for (let i = 0; i < n; i++) out[j * n + i] = colRes[i * (m - 1) + j] ?? 0
  out.set(bcRes, n * (m - 1))
  return out
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0)
  return s
}

/** Damped Newton on the collocation system (scipy's solve_newton). */
export function solveNewton(
  fun: NumField,
  bc: NumBc,
  x: Float64Array,
  h: Float64Array,
  yIn: Float64Array,
  pIn: Float64Array,
  B: Float64Array | undefined,
  bvpTol: number,
  bcTol: number,
  n: number,
): { y: Float64Array; p: Float64Array; singular: boolean } {
  const m = x.length
  const tolR = Float64Array.from(h, (hj) => (2 / 3) * hj * 5e-2 * bvpTol)
  const maxNjev = 4
  const maxIter = 8
  const sigma = 0.2
  const tau = 0.5
  const nTrial = 4
  let y = yIn
  let p = pIn
  let { colRes, yMid, f, fMid } = collocation(fun, y, p, x, h, n)
  let bcRes = bc(column(y, n, m, 0), column(y, n, m, m - 1), p)
  let res = residualVector(colRes, bcRes, n, m)
  let njev = 0
  let singular = false
  let recompute = true
  let lu: SparseLu | undefined
  let step: Float64Array = new Float64Array(0)
  let cost = 0
  for (let iteration = 0; iteration < maxIter; iteration++) {
    if (recompute) {
      lu = globalJacobian(fun, bc, x, h, y, p, yMid, f, fMid, bcRes, n)
      njev += 1
      if (lu.singular) {
        singular = true
        break
      }
      step = sparseLuSolve(lu, res)
      cost = dot(step, step)
    }
    if (lu === undefined) break
    let alpha = 1
    let yNew = y
    let pNew = p
    let stepNew: Float64Array = step
    let costNew = cost
    for (let trial = 0; trial <= nTrial; trial++) {
      yNew = new Float64Array(n * m)
      for (let j = 0; j < m; j++)
        for (let i = 0; i < n; i++) yNew[i * m + j] = (y[i * m + j] ?? 0) - alpha * (step[j * n + i] ?? 0)
      if (B !== undefined) {
        const y0 = column(yNew, n, m, 0)
        for (let i = 0; i < n; i++) {
          let s = 0
          for (let c = 0; c < n; c++) s += (B[i * n + c] ?? 0) * (y0[c] ?? 0)
          yNew[i * m] = s
        }
      }
      pNew = Float64Array.from(p, (v, q) => v - alpha * (step[m * n + q] ?? 0))
      ;({ colRes, yMid, f, fMid } = collocation(fun, yNew, pNew, x, h, n))
      bcRes = bc(column(yNew, n, m, 0), column(yNew, n, m, m - 1), pNew)
      res = residualVector(colRes, bcRes, n, m)
      stepNew = sparseLuSolve(lu, res)
      costNew = dot(stepNew, stepNew)
      if (costNew < (1 - 2 * alpha * sigma) * cost) break
      if (trial < nTrial) alpha *= tau
    }
    y = yNew
    p = pNew
    if (njev === maxNjev) break
    let ok = true
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j < m - 1; j++) {
        if (!(Math.abs(colRes[i * (m - 1) + j] ?? 0) < (tolR[j] ?? 0) * (1 + Math.abs(fMid[i * (m - 1) + j] ?? 0)))) {
          ok = false
          break
        }
      }
    }
    if (ok && bcRes.every((v) => Math.abs(v) < bcTol)) break
    if (alpha === 1) {
      step = stepNew
      cost = costNew
      recompute = false
    } else recompute = true
  }
  return { y, p, singular }
}

/** A cubic Hermite spline through (x_j, y_j) with slopes yp_j (scipy's create_spline), with derivatives. */
export interface Spline {
  readonly x: Float64Array
  readonly n: number
  /** Value (nu = 0) or first derivative (nu = 1) at one point. */
  at(t: number, nu?: 0 | 1): Float64Array
}

/** Build the solution spline from the nodes, states and slopes. */
export function createSpline(y: Float64Array, yp: Float64Array, x: Float64Array, h: Float64Array, n: number): Spline {
  const m = x.length
  const c = new Float64Array(4 * n * (m - 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m - 1; j++) {
      const hj = h[j] ?? 1
      const slope = ((y[i * m + j + 1] ?? 0) - (y[i * m + j] ?? 0)) / hj
      const t = ((yp[i * m + j] ?? 0) + (yp[i * m + j + 1] ?? 0) - 2 * slope) / hj
      const base = (i * (m - 1) + j) * 4
      c[base] = t / hj
      c[base + 1] = (slope - (yp[i * m + j] ?? 0)) / hj - t
      c[base + 2] = yp[i * m + j] ?? 0
      c[base + 3] = y[i * m + j] ?? 0
    }
  }
  const interval = (t: number): number => {
    if (t < (x[0] ?? 0)) return 0
    let lo = 0
    let hi = m - 2
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if ((x[mid] ?? 0) <= t) lo = mid
      else hi = mid - 1
    }
    return lo
  }
  return {
    x,
    n,
    at: (t, nu = 0) => {
      const j = interval(t)
      const s = t - (x[j] ?? 0)
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const base = (i * (m - 1) + j) * 4
        const [c0, c1, c2, c3] = [c[base] ?? 0, c[base + 1] ?? 0, c[base + 2] ?? 0, c[base + 3] ?? 0]
        out[i] = nu === 0 ? ((c0 * s + c1) * s + c2) * s + c3 : (3 * c0 * s + 2 * c1) * s + c2
      }
      return out
    },
  }
}

/** The RMS relative residuals of each interval by Lobatto quadrature (scipy's estimate_rms_residuals). */
export function rmsResiduals(
  fun: NumField,
  sol: Spline,
  x: Float64Array,
  h: Float64Array,
  p: Float64Array,
  rMid: Float64Array,
  fMid: Float64Array,
  n: number,
): Float64Array {
  const m = x.length
  const s = (j: number): number => 0.5 * (h[j] ?? 0) * Math.sqrt(3 / 7)
  const x1 = Float64Array.from({ length: m - 1 }, (_, j) => (x[j] ?? 0) + 0.5 * (h[j] ?? 0) + s(j))
  const x2 = Float64Array.from({ length: m - 1 }, (_, j) => (x[j] ?? 0) + 0.5 * (h[j] ?? 0) - s(j))
  const pack = (xs: Float64Array, nu: 0 | 1): Float64Array => {
    const out = new Float64Array(n * (m - 1))
    for (let j = 0; j < m - 1; j++) {
      const v = sol.at(xs[j] ?? 0, nu)
      for (let i = 0; i < n; i++) out[i * (m - 1) + j] = v[i] ?? 0
    }
    return out
  }
  const y1 = pack(x1, 0)
  const y2 = pack(x2, 0)
  const y1p = pack(x1, 1)
  const y2p = pack(x2, 1)
  const f1 = fun(x1, y1, p)
  const f2 = fun(x2, y2, p)
  const out = new Float64Array(m - 1)
  for (let j = 0; j < m - 1; j++) {
    let r1 = 0
    let r2 = 0
    let rm = 0
    for (let i = 0; i < n; i++) {
      const q = i * (m - 1) + j
      const a = ((y1p[q] ?? 0) - (f1[q] ?? 0)) / (1 + Math.abs(f1[q] ?? 0))
      const b = ((y2p[q] ?? 0) - (f2[q] ?? 0)) / (1 + Math.abs(f2[q] ?? 0))
      const c = (rMid[q] ?? 0) / (1 + Math.abs(fMid[q] ?? 0))
      r1 += a * a
      r2 += b * b
      rm += c * c
    }
    out[j] = Math.sqrt(0.5 * ((32 / 45) * rm + (49 / 90) * (r1 + r2)))
  }
  return out
}

/** Insert one node in the middle of the intervals of `one` and two nodes in those of `two` (scipy's modify_mesh). */
export function modifyMesh(x: Float64Array, one: readonly number[], two: readonly number[]): Float64Array {
  const out = [...x]
  for (const j of one) out.push(0.5 * ((x[j] ?? 0) + (x[j + 1] ?? 0)))
  for (const j of two) {
    out.push((2 * (x[j] ?? 0) + (x[j + 1] ?? 0)) / 3)
    out.push(((x[j] ?? 0) + 2 * (x[j + 1] ?? 0)) / 3)
  }
  return Float64Array.from(out.sort((a, b) => a - b))
}
