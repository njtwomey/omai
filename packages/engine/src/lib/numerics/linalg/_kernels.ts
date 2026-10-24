// Dense kernels on single row-major matrices (Float64Array): the textbook algorithms, as in Golub and Van Loan.
// The primitives in this folder apply them matrix by matrix over batch axes.
import { WhiteboxError } from '../../base/index.ts'
import { dnrm2 } from './blas.ts'

const EPS = 2.220446049250313e-16

/** (A + A^T) / 2 of an n x n matrix: the factorisations of symmetric matrices read A through this. */
function symmetrised(a: Float64Array, n: number): Float64Array {
  const s = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) s[i * n + j] = 0.5 * ((a[i * n + j] ?? 0) + (a[j * n + i] ?? 0))
  return s
}

/**
 * Cholesky factor L (lower, A = L L^T) of the symmetric part of an n x n matrix (identical to reading the lower triangle
 * when A is symmetric, and consistent with the derivative rule, which treats A as symmetric); throws when not positive
 * definite.
 */
export function choleskyKernel(input: Float64Array, n: number): Float64Array {
  const a = symmetrised(input, n)
  const l = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j] ?? 0
    for (let k = 0; k < j; k++) d -= (l[j * n + k] ?? 0) ** 2
    if (!(d > 0)) throw new WhiteboxError('value', 'cholesky: matrix is not positive definite')
    const ljj = Math.sqrt(d)
    l[j * n + j] = ljj
    for (let i = j + 1; i < n; i++) {
      let s = a[i * n + j] ?? 0
      for (let k = 0; k < j; k++) s -= (l[i * n + k] ?? 0) * (l[j * n + k] ?? 0)
      l[i * n + j] = s / ljj
    }
  }
  return l
}

/** Options of a triangular solve: which triangle of A is used, whether to solve with A^T, and a unit diagonal. */
export interface TriangularOptions {
  readonly lower: boolean
  readonly transpose: boolean
  readonly unitDiagonal: boolean
}

/** X with op(T) X = B, T the given triangle of an n x n matrix, B n x k. */
export function triangularKernel(
  t: Float64Array,
  b: Float64Array,
  n: number,
  k: number,
  o: TriangularOptions,
): Float64Array {
  const x = Float64Array.from(b)
  const at = (i: number, j: number): number => (o.transpose ? (t[j * n + i] ?? 0) : (t[i * n + j] ?? 0))
  // Solving with the transpose of a lower triangle is an upper-triangular solve, and vice versa.
  const forward = o.lower !== o.transpose
  for (let c = 0; c < k; c++) {
    for (let step = 0; step < n; step++) {
      const i = forward ? step : n - 1 - step
      let s = x[i * k + c] ?? 0
      if (forward) for (let j = 0; j < i; j++) s -= at(i, j) * (x[j * k + c] ?? 0)
      else for (let j = i + 1; j < n; j++) s -= at(i, j) * (x[j * k + c] ?? 0)
      x[i * k + c] = o.unitDiagonal ? s : s / at(i, i)
    }
  }
  return x
}

/** LU with partial pivoting: P A = L U, returned packed (unit L below the diagonal, U on and above) with the row order. */
export function luKernel(
  a: Float64Array,
  n: number,
): { lu: Float64Array; perm: Int32Array; piv: Int32Array; sign: number } {
  const lu = Float64Array.from(a)
  const perm = Int32Array.from({ length: n }, (_, i) => i)
  // LAPACK's pivot indices: row j was interchanged with row piv[j] at step j.
  const piv = Int32Array.from({ length: n }, (_, i) => i)
  let sign = 1
  for (let j = 0; j < n; j++) {
    let p = j
    let best = Math.abs(lu[j * n + j] ?? 0)
    for (let i = j + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + j] ?? 0)
      if (v > best) {
        best = v
        p = i
      }
    }
    piv[j] = p
    if (p !== j) {
      for (let c = 0; c < n; c++) {
        const tmp = lu[j * n + c] ?? 0
        lu[j * n + c] = lu[p * n + c] ?? 0
        lu[p * n + c] = tmp
      }
      const tp = perm[j] ?? 0
      perm[j] = perm[p] ?? 0
      perm[p] = tp
      sign = -sign
    }
    const pivot = lu[j * n + j] ?? 0
    if (pivot === 0) continue
    for (let i = j + 1; i < n; i++) {
      const f = (lu[i * n + j] ?? 0) / pivot
      lu[i * n + j] = f
      for (let c = j + 1; c < n; c++) lu[i * n + c] = (lu[i * n + c] ?? 0) - f * (lu[j * n + c] ?? 0)
    }
  }
  return { lu, perm, piv, sign }
}

/** X with A X = B (A n x n, B n x k), by LU with partial pivoting; throws for an exactly singular matrix. */
export function solveKernel(a: Float64Array, b: Float64Array, n: number, k: number): Float64Array {
  const { lu, perm } = luKernel(a, n)
  for (let i = 0; i < n; i++) {
    if ((lu[i * n + i] ?? 0) === 0) throw new WhiteboxError('value', 'solve: matrix is singular')
  }
  const pb = new Float64Array(n * k)
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) pb[i * k + c] = b[(perm[i] ?? 0) * k + c] ?? 0
  const y = triangularKernel(lu, pb, n, k, { lower: true, transpose: false, unitDiagonal: true })
  return triangularKernel(lu, y, n, k, { lower: false, transpose: false, unitDiagonal: false })
}

/** Sign and log |det| from LU (sign 0 and -inf for a singular matrix). */
export function slogdetKernel(a: Float64Array, n: number): { sign: number; logabsdet: number } {
  const { lu, sign } = luKernel(a, n)
  let s = sign
  let log = 0
  for (let i = 0; i < n; i++) {
    const d = lu[i * n + i] ?? 0
    if (d === 0) return { sign: 0, logabsdet: Number.NEGATIVE_INFINITY }
    if (d < 0) s = -s
    log += Math.log(Math.abs(d))
  }
  return { sign: s, logabsdet: log }
}

/** Reduced QR by Householder reflections: Q (m x k, orthonormal columns) and R (k x n, upper), k = min(m, n). */
export function qrKernel(a: Float64Array, m: number, n: number): { q: Float64Array; r: Float64Array } {
  const k = Math.min(m, n)
  const work = Float64Array.from(a)
  const vs: Float64Array[] = []
  for (let j = 0; j < k; j++) {
    const v = new Float64Array(m - j)
    // Norms by the scaled dnrm2, so that tiny columns do not underflow when squared.
    const column = Float64Array.from({ length: m - j }, (_, i) => work[(i + j) * n + j] ?? 0)
    const norm = dnrm2(column)
    const x0 = work[j * n + j] ?? 0
    // As LAPACK's dlarfg: no reflection when the entries below the diagonal are exactly zero (this keeps the sign of
    // R_jj); tiny nonzero entries are still eliminated.
    const below = dnrm2(column.subarray(1))
    if (norm === 0 || below === 0) {
      vs.push(v)
      continue
    }
    // alpha = -sign(x0) |x| (LAPACK's choice, which avoids cancellation in x0 - alpha). The reflector is formed from the
    // column scaled by a power of two to unit size (exact), so that subnormal columns keep their direction.
    const alpha = x0 >= 0 ? -norm : norm
    // Two power-of-two factors, as 2^k alone overflows for subnormal norms.
    const k = -Math.round(Math.log2(norm))
    const s1 = 2 ** Math.trunc(k / 2)
    const s2 = 2 ** (k - Math.trunc(k / 2))
    for (let i = j; i < m; i++) v[i - j] = (work[i * n + j] ?? 0) * s1 * s2
    v[0] = (v[0] ?? 0) - alpha * s1 * s2
    const vn = dnrm2(v)
    for (let i = 0; i < v.length; i++) v[i] = (v[i] ?? 0) / vn
    for (let c = j; c < n; c++) {
      let d = 0
      for (let i = j; i < m; i++) d += (v[i - j] ?? 0) * (work[i * n + c] ?? 0)
      for (let i = j; i < m; i++) work[i * n + c] = (work[i * n + c] ?? 0) - 2 * (v[i - j] ?? 0) * d
    }
    vs.push(v)
  }
  const r = new Float64Array(k * n)
  for (let i = 0; i < k; i++) for (let c = i; c < n; c++) r[i * n + c] = work[i * n + c] ?? 0
  // Q = H_0 H_1 ... H_{k-1} applied to the first k columns of the identity.
  const q = new Float64Array(m * k)
  for (let i = 0; i < k; i++) q[i * k + i] = 1
  for (let j = k - 1; j >= 0; j--) {
    const v = vs[j] as Float64Array
    for (let c = 0; c < k; c++) {
      let d = 0
      for (let i = j; i < m; i++) d += (v[i - j] ?? 0) * (q[i * k + c] ?? 0)
      for (let i = j; i < m; i++) q[i * k + c] = (q[i * k + c] ?? 0) - 2 * (v[i - j] ?? 0) * d
    }
  }
  return { q, r }
}

/** Flip each column so that its entry of largest magnitude is positive (a fixed sign convention for vectors). */
function normaliseColumns(v: Float64Array, rows: number, cols: number, also?: Float64Array, alsoRows?: number): void {
  for (let c = 0; c < cols; c++) {
    let best = 0
    for (let i = 0; i < rows; i++) if (Math.abs(v[i * cols + c] ?? 0) > Math.abs(best)) best = v[i * cols + c] ?? 0
    if (best < 0) {
      for (let i = 0; i < rows; i++) v[i * cols + c] = -(v[i * cols + c] ?? 0)
      if (also !== undefined && alsoRows !== undefined) {
        for (let i = 0; i < alsoRows; i++) also[i * cols + c] = -(also[i * cols + c] ?? 0)
      }
    }
  }
}

/**
 * Symmetric eigendecomposition by the cyclic Jacobi method, of the symmetric part of A: eigenvalues ascending,
 * eigenvectors as columns, each with its largest-magnitude entry positive.
 */
export function eighKernel(
  a: Float64Array,
  n: number,
  onRotation?: (p: number, q: number, c: number, s: number) => void,
  onSweep?: (offDiagonalNorm: number) => void,
): { w: Float64Array; v: Float64Array } {
  const s = symmetrised(a, n)
  const v = new Float64Array(n * n)
  for (let i = 0; i < n; i++) v[i * n + i] = 1
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    let total = 0
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const e = (s[i * n + j] ?? 0) ** 2
        total += e
        if (i !== j) off += e
      }
    }
    onSweep?.(Math.sqrt(off))
    if (off <= EPS * EPS * total) break
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = s[p * n + q] ?? 0
        if (apq === 0) continue
        const app = s[p * n + p] ?? 0
        const aqq = s[q * n + q] ?? 0
        const theta = (aqq - app) / (2 * apq)
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const sn = t * c
        onRotation?.(p, q, c, sn)
        for (let r = 0; r < n; r++) {
          const arp = s[r * n + p] ?? 0
          const arq = s[r * n + q] ?? 0
          s[r * n + p] = c * arp - sn * arq
          s[r * n + q] = sn * arp + c * arq
        }
        for (let r = 0; r < n; r++) {
          const apr = s[p * n + r] ?? 0
          const aqr = s[q * n + r] ?? 0
          s[p * n + r] = c * apr - sn * aqr
          s[q * n + r] = sn * apr + c * aqr
        }
        for (let r = 0; r < n; r++) {
          const vrp = v[r * n + p] ?? 0
          const vrq = v[r * n + q] ?? 0
          v[r * n + p] = c * vrp - sn * vrq
          v[r * n + q] = sn * vrp + c * vrq
        }
      }
    }
  }
  return sortedEigen(
    Float64Array.from({ length: n }, (_, i) => s[i * n + i] ?? 0),
    v,
    n,
  )
}

/** Eigenpairs in ascending order with the sign convention (largest-magnitude entry of each vector positive). */
function sortedEigen(values: Float64Array, vectors: Float64Array, n: number): { w: Float64Array; v: Float64Array } {
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => (values[i] ?? 0) - (values[j] ?? 0))
  const w = Float64Array.from(order, (i) => values[i] ?? 0)
  const sorted = new Float64Array(n * n)
  order.forEach((src, dst) => {
    for (let r = 0; r < n; r++) sorted[r * n + dst] = vectors[r * n + src] ?? 0
  })
  normaliseColumns(sorted, n, n)
  return { w, v: sorted }
}

/**
 * Symmetric eigendecomposition by Householder reduction to tridiagonal form followed by the implicit-shift QL
 * algorithm (EISPACK tred2 and tql2; Numerical Recipes tred2 and tqli): O(n^3) with a small constant, the LAPACK-style
 * route.
 */
export function eighQlKernel(input: Float64Array, n: number): { w: Float64Array; v: Float64Array } {
  const a = symmetrised(input, n)
  const at = (i: number, j: number): number => a[i * n + j] ?? 0
  const set = (i: number, j: number, x: number): void => {
    a[i * n + j] = x
  }
  const d = new Float64Array(n)
  const e = new Float64Array(n)
  // tred2: A = Q T Q^T, Q accumulated in a.
  for (let i = n - 1; i >= 1; i--) {
    const l = i - 1
    let h = 0
    if (l > 0) {
      let scale = 0
      for (let k = 0; k <= l; k++) scale += Math.abs(at(i, k))
      if (scale === 0) e[i] = at(i, l)
      else {
        for (let k = 0; k <= l; k++) {
          set(i, k, at(i, k) / scale)
          h += at(i, k) ** 2
        }
        let f = at(i, l)
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h)
        e[i] = scale * g
        h -= f * g
        set(i, l, f - g)
        f = 0
        for (let j = 0; j <= l; j++) {
          set(j, i, at(i, j) / h)
          g = 0
          for (let k = 0; k <= j; k++) g += at(j, k) * at(i, k)
          for (let k = j + 1; k <= l; k++) g += at(k, j) * at(i, k)
          e[j] = g / h
          f += (e[j] ?? 0) * at(i, j)
        }
        const hh = f / (h + h)
        for (let j = 0; j <= l; j++) {
          f = at(i, j)
          g = (e[j] ?? 0) - hh * f
          e[j] = g
          for (let k = 0; k <= j; k++) set(j, k, at(j, k) - (f * (e[k] ?? 0) + g * at(i, k)))
        }
      }
    } else e[i] = at(i, l)
    d[i] = h
  }
  d[0] = 0
  e[0] = 0
  for (let i = 0; i < n; i++) {
    const l = i - 1
    if (d[i] !== 0) {
      for (let j = 0; j <= l; j++) {
        let g = 0
        for (let k = 0; k <= l; k++) g += at(i, k) * at(k, j)
        for (let k = 0; k <= l; k++) set(k, j, at(k, j) - g * at(k, i))
      }
    }
    d[i] = at(i, i)
    set(i, i, 1)
    for (let j = 0; j <= l; j++) {
      set(j, i, 0)
      set(i, j, 0)
    }
  }
  // tql2: implicit-shift QL on the tridiagonal (d, e), rotations applied to the columns of a.
  for (let i = 1; i < n; i++) e[i - 1] = e[i] ?? 0
  if (n > 0) e[n - 1] = 0
  // Deflate when an off-diagonal is negligible relative to its neighbours (Numerical Recipes' tqli) or to the norm of
  // the tridiagonal (EISPACK's tql2, LAPACK): without the absolute test a cluster of eigenvalues at rounding level (a
  // numerically rank-deficient kernel matrix) never meets the relative one and the iteration fails to converge.
  let norm = 0
  for (let i = 0; i < n; i++) norm = Math.max(norm, Math.abs(d[i] ?? 0) + Math.abs(e[i] ?? 0))
  const floor = EPS * norm
  for (let l = 0; l < n; l++) {
    let iterations = 0
    let m: number
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(d[m] ?? 0) + Math.abs(d[m + 1] ?? 0)
        const off = Math.abs(e[m] ?? 0)
        if (off <= EPS * dd || off <= floor) break
      }
      if (m !== l) {
        if (iterations++ === 60) throw new WhiteboxError('convergence', 'eigh: QL iteration did not converge')
        let g = ((d[l + 1] ?? 0) - (d[l] ?? 0)) / (2 * (e[l] ?? 0))
        let r = Math.hypot(g, 1)
        g = (d[m] ?? 0) - (d[l] ?? 0) + (e[l] ?? 0) / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)))
        let s2 = 1
        let c = 1
        let p = 0
        let underflow = false
        let i: number
        for (i = m - 1; i >= l; i--) {
          const f = s2 * (e[i] ?? 0)
          const b = c * (e[i] ?? 0)
          r = Math.hypot(f, g)
          e[i + 1] = r
          if (r === 0) {
            d[i + 1] = (d[i + 1] ?? 0) - p
            e[m] = 0
            underflow = true
            break
          }
          s2 = f / r
          c = g / r
          g = (d[i + 1] ?? 0) - p
          r = ((d[i] ?? 0) - g) * s2 + 2 * c * b
          p = s2 * r
          d[i + 1] = g + p
          g = c * r - b
          for (let k = 0; k < n; k++) {
            const zf = at(k, i + 1)
            set(k, i + 1, s2 * at(k, i) + c * zf)
            set(k, i, c * at(k, i) - s2 * zf)
          }
        }
        if (underflow && i >= l) continue
        d[l] = (d[l] ?? 0) - p
        e[l] = g
        e[m] = 0
      }
    } while (m !== l)
  }
  return sortedEigen(d, a, n)
}

/** Complete the columns j >= rank of an m x k matrix with orthonormal columns to an orthonormal set (Gram-Schmidt). */
function completeColumns(u: Float64Array, m: number, k: number, filled: readonly boolean[]): void {
  let candidate = 0
  for (let c = 0; c < k; c++) {
    if (filled[c]) continue
    while (candidate < m) {
      const x = new Float64Array(m)
      x[candidate++] = 1
      for (let d = 0; d < k; d++) {
        if (!filled[d]) continue
        let dot = 0
        for (let i = 0; i < m; i++) dot += (u[i * k + d] ?? 0) * (x[i] ?? 0)
        for (let i = 0; i < m; i++) x[i] = (x[i] ?? 0) - dot * (u[i * k + d] ?? 0)
      }
      let norm = 0
      for (const e of x) norm += e * e
      norm = Math.sqrt(norm)
      if (norm > 1e-8) {
        for (let i = 0; i < m; i++) u[i * k + c] = (x[i] ?? 0) / norm
        ;(filled as boolean[])[c] = true
        break
      }
    }
  }
}

/** Reduced SVD of a tall or square matrix (m >= n) by one-sided Jacobi: U (m x n), s (n, descending), V (n x n). */
function svdTall(a: Float64Array, m: number, n: number): { u: Float64Array; s: Float64Array; v: Float64Array } {
  const u = Float64Array.from(a)
  const v = new Float64Array(n * n)
  for (let i = 0; i < n; i++) v[i * n + i] = 1
  for (let sweep = 0; sweep < 100; sweep++) {
    let rotated = false
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          const up = u[i * n + p] ?? 0
          const uq = u[i * n + q] ?? 0
          alpha += up * up
          beta += uq * uq
          gamma += up * uq
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const sn = c * t
        for (let i = 0; i < m; i++) {
          const up = u[i * n + p] ?? 0
          const uq = u[i * n + q] ?? 0
          u[i * n + p] = c * up - sn * uq
          u[i * n + q] = sn * up + c * uq
        }
        for (let i = 0; i < n; i++) {
          const vp = v[i * n + p] ?? 0
          const vq = v[i * n + q] ?? 0
          v[i * n + p] = c * vp - sn * vq
          v[i * n + q] = sn * vp + c * vq
        }
      }
    }
    if (!rotated) break
  }
  const norms = Array.from({ length: n }, (_, c) => {
    let s = 0
    for (let i = 0; i < m; i++) s += (u[i * n + c] ?? 0) ** 2
    return Math.sqrt(s)
  })
  const order = norms.map((_, i) => i).sort((i, j) => (norms[j] ?? 0) - (norms[i] ?? 0))
  const s = Float64Array.from(order, (i) => norms[i] ?? 0)
  const largest = s[0] ?? 0
  const uo = new Float64Array(m * n)
  const vo = new Float64Array(n * n)
  const filled: boolean[] = []
  order.forEach((src, dst) => {
    const sv = norms[src] ?? 0
    const keep = sv > largest * EPS * Math.max(m, n) && sv > 0
    filled.push(keep)
    for (let i = 0; i < m; i++) uo[i * n + dst] = keep ? (u[i * n + src] ?? 0) / sv : 0
    for (let i = 0; i < n; i++) vo[i * n + dst] = v[i * n + src] ?? 0
  })
  completeColumns(uo, m, n, filled)
  normaliseColumns(uo, m, n, vo, n)
  return { u: uo, s, v: vo }
}

/**
 * Reduced SVD of a tall or square matrix (m >= n) by Golub-Kahan bidiagonalisation and the implicit-shift QR iteration
 * on the bidiagonal (Golub and Reinsch 1970; the Numerical Recipes svdcmp form).
 */
function svdGolubKahan(
  input: Float64Array,
  m: number,
  n: number,
): { u: Float64Array; s: Float64Array; v: Float64Array } {
  const a = Float64Array.from(input)
  const v = new Float64Array(n * n)
  const w = new Float64Array(n)
  const rv1 = new Float64Array(n)
  const A = (i: number, j: number): number => a[i * n + j] ?? 0
  const setA = (i: number, j: number, x: number): void => {
    a[i * n + j] = x
  }
  const V = (i: number, j: number): number => v[i * n + j] ?? 0
  const setV = (i: number, j: number, x: number): void => {
    v[i * n + j] = x
  }
  const sign = (x: number, y: number): number => (y >= 0 ? Math.abs(x) : -Math.abs(x))
  let g = 0
  let scale = 0
  let anorm = 0
  let l = 0
  // Householder bidiagonalisation.
  for (let i = 0; i < n; i++) {
    l = i + 1
    rv1[i] = scale * g
    g = 0
    let s = 0
    scale = 0
    if (i < m) {
      for (let k = i; k < m; k++) scale += Math.abs(A(k, i))
      if (scale !== 0) {
        for (let k = i; k < m; k++) {
          setA(k, i, A(k, i) / scale)
          s += A(k, i) ** 2
        }
        const f = A(i, i)
        g = -sign(Math.sqrt(s), f)
        const h = f * g - s
        setA(i, i, f - g)
        for (let j = l; j < n; j++) {
          let t = 0
          for (let k = i; k < m; k++) t += A(k, i) * A(k, j)
          const ff = t / h
          for (let k = i; k < m; k++) setA(k, j, A(k, j) + ff * A(k, i))
        }
        for (let k = i; k < m; k++) setA(k, i, A(k, i) * scale)
      }
    }
    w[i] = scale * g
    g = 0
    s = 0
    scale = 0
    if (i < m && i !== n - 1) {
      for (let k = l; k < n; k++) scale += Math.abs(A(i, k))
      if (scale !== 0) {
        for (let k = l; k < n; k++) {
          setA(i, k, A(i, k) / scale)
          s += A(i, k) ** 2
        }
        const f = A(i, l)
        g = -sign(Math.sqrt(s), f)
        const h = f * g - s
        setA(i, l, f - g)
        for (let k = l; k < n; k++) rv1[k] = A(i, k) / h
        for (let j = l; j < m; j++) {
          let t = 0
          for (let k = l; k < n; k++) t += A(j, k) * A(i, k)
          for (let k = l; k < n; k++) setA(j, k, A(j, k) + t * (rv1[k] ?? 0))
        }
        for (let k = l; k < n; k++) setA(i, k, A(i, k) * scale)
      }
    }
    anorm = Math.max(anorm, Math.abs(w[i] ?? 0) + Math.abs(rv1[i] ?? 0))
  }
  // Accumulate the right-hand transformations.
  for (let i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g !== 0) {
        for (let j = l; j < n; j++) setV(j, i, A(i, j) / A(i, l) / g)
        for (let j = l; j < n; j++) {
          let t = 0
          for (let k = l; k < n; k++) t += A(i, k) * V(k, j)
          for (let k = l; k < n; k++) setV(k, j, V(k, j) + t * V(k, i))
        }
      }
      for (let j = l; j < n; j++) {
        setV(i, j, 0)
        setV(j, i, 0)
      }
    }
    setV(i, i, 1)
    g = rv1[i] ?? 0
    l = i
  }
  // Accumulate the left-hand transformations.
  for (let i = Math.min(m, n) - 1; i >= 0; i--) {
    l = i + 1
    g = w[i] ?? 0
    for (let j = l; j < n; j++) setA(i, j, 0)
    if (g !== 0) {
      g = 1 / g
      for (let j = l; j < n; j++) {
        let t = 0
        for (let k = l; k < m; k++) t += A(k, i) * A(k, j)
        const f = (t / A(i, i)) * g
        for (let k = i; k < m; k++) setA(k, j, A(k, j) + f * A(k, i))
      }
      for (let j = i; j < m; j++) setA(j, i, A(j, i) * g)
    } else for (let j = i; j < m; j++) setA(j, i, 0)
    setA(i, i, A(i, i) + 1)
  }
  // Diagonalise the bidiagonal by implicit-shift QR.
  for (let k = n - 1; k >= 0; k--) {
    for (let its = 1; its <= 75; its++) {
      let flag = true
      let nm = 0
      for (l = k; l >= 0; l--) {
        nm = l - 1
        if (Math.abs(rv1[l] ?? 0) + anorm === anorm) {
          flag = false
          break
        }
        if (Math.abs(w[nm] ?? 0) + anorm === anorm) break
      }
      if (flag) {
        let c = 0
        let s = 1
        for (let i = l; i <= k; i++) {
          const f = s * (rv1[i] ?? 0)
          rv1[i] = c * (rv1[i] ?? 0)
          if (Math.abs(f) + anorm === anorm) break
          const gg = w[i] ?? 0
          let h = Math.hypot(f, gg)
          w[i] = h
          h = 1 / h
          c = gg * h
          s = -f * h
          for (let j = 0; j < m; j++) {
            const y = A(j, nm)
            const z = A(j, i)
            setA(j, nm, y * c + z * s)
            setA(j, i, z * c - y * s)
          }
        }
      }
      let z = w[k] ?? 0
      if (l === k) {
        if (z < 0) {
          w[k] = -z
          for (let j = 0; j < n; j++) setV(j, k, -V(j, k))
        }
        break
      }
      if (its === 75) throw new WhiteboxError('convergence', 'svd: Golub-Kahan iteration did not converge')
      let x = w[l] ?? 0
      nm = k - 1
      let y = w[nm] ?? 0
      g = rv1[nm] ?? 0
      let h = rv1[k] ?? 0
      let f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
      g = Math.hypot(f, 1)
      f = ((x - z) * (x + z) + h * (y / (f + sign(g, f)) - h)) / x
      let c = 1
      let s = 1
      for (let j = l; j <= nm; j++) {
        const i = j + 1
        g = rv1[i] ?? 0
        y = w[i] ?? 0
        h = s * g
        g = c * g
        z = Math.hypot(f, h)
        rv1[j] = z
        c = f / z
        s = h / z
        f = x * c + g * s
        g = g * c - x * s
        h = y * s
        y *= c
        for (let jj = 0; jj < n; jj++) {
          const xv = V(jj, j)
          const zv = V(jj, i)
          setV(jj, j, xv * c + zv * s)
          setV(jj, i, zv * c - xv * s)
        }
        z = Math.hypot(f, h)
        w[j] = z
        if (z !== 0) {
          z = 1 / z
          c = f * z
          s = h * z
        }
        f = c * g + s * y
        x = c * y - s * g
        for (let jj = 0; jj < m; jj++) {
          const ya = A(jj, j)
          const za = A(jj, i)
          setA(jj, j, ya * c + za * s)
          setA(jj, i, za * c - ya * s)
        }
      }
      rv1[l] = 0
      rv1[k] = f
      w[k] = x
    }
  }
  // Sort descending, with the sign convention and orthonormal completion of the null columns.
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => (w[j] ?? 0) - (w[i] ?? 0))
  const sv = Float64Array.from(order, (i) => w[i] ?? 0)
  const largest = sv[0] ?? 0
  const uo = new Float64Array(m * n)
  const vo = new Float64Array(n * n)
  const filled: boolean[] = []
  order.forEach((src, dst) => {
    const keep = (w[src] ?? 0) > largest * EPS * Math.max(m, n) && (w[src] ?? 0) > 0
    filled.push(keep)
    for (let i = 0; i < m; i++) uo[i * n + dst] = keep ? A(i, src) : 0
    for (let i = 0; i < n; i++) vo[i * n + dst] = V(i, src)
  })
  completeColumns(uo, m, n, filled)
  normaliseColumns(uo, m, n, vo, n)
  return { u: uo, s: sv, v: vo }
}

/** Reduced SVD A = U diag(s) V^T: U (m x k), s (k, descending), V (n x k), k = min(m, n). */
export function svdKernel(
  a: Float64Array,
  m: number,
  n: number,
  method: 'jacobi' | 'golub-kahan' = 'jacobi',
): { u: Float64Array; s: Float64Array; v: Float64Array } {
  const tall = method === 'jacobi' ? svdTall : svdGolubKahan
  if (m >= n) return tall(a, m, n)
  // A^T = U' S V'^T, so A = V' S U'^T.
  const at = new Float64Array(n * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) at[j * m + i] = a[i * n + j] ?? 0
  const t = tall(at, n, m)
  // The same sign convention as the tall case: each column of U has its largest entry positive.
  normaliseColumns(t.v, m, m, t.u, n)
  return { u: t.v, s: t.s, v: t.u }
}
