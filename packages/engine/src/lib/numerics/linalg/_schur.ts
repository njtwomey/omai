// Hessenberg reduction and the real Schur form by Francis double-shift QR (Golub and Van Loan, algorithms 7.4.2, 7.5.1
// and 7.5.2), on single row-major matrices.
import { MdmlError } from '../../base/index.ts'

const EPS = Number.EPSILON

/** A Householder vector v (unit norm) with (I - 2 v v^T) x = -sign(x0) |x| e1, or null when x needs no reflection. */
function reflector(x: readonly number[]): number[] | null {
  let norm = 0
  for (const e of x) norm += e * e
  norm = Math.sqrt(norm)
  const x0 = x[0] ?? 0
  // Skip only when the trailing entries are exactly zero: a tiny but nonzero rotation still matters (it is what moves
  // the bulge once a subdiagonal entry is small but not yet negligible).
  if (norm === 0 || x.slice(1).every((e) => e === 0)) return null
  const v = [...x]
  v[0] = x0 + (x0 >= 0 ? norm : -norm)
  let vn = 0
  for (const e of v) vn += e * e
  vn = Math.sqrt(vn)
  return v.map((e) => e / vn)
}

/** Apply I - 2 v v^T from the left to rows r0.. of columns c0..c1 of an n x n matrix. */
function reflectRows(a: Float64Array, n: number, v: readonly number[], r0: number, c0: number, c1: number): void {
  for (let c = c0; c <= c1; c++) {
    let d = 0
    for (let k = 0; k < v.length; k++) d += (v[k] ?? 0) * (a[(r0 + k) * n + c] ?? 0)
    for (let k = 0; k < v.length; k++) a[(r0 + k) * n + c] = (a[(r0 + k) * n + c] ?? 0) - 2 * (v[k] ?? 0) * d
  }
}

/** Apply I - 2 v v^T from the right to columns c0.. of rows r0..r1 of an n x n matrix. */
function reflectCols(a: Float64Array, n: number, v: readonly number[], c0: number, r0: number, r1: number): void {
  for (let r = r0; r <= r1; r++) {
    let d = 0
    for (let k = 0; k < v.length; k++) d += (v[k] ?? 0) * (a[r * n + c0 + k] ?? 0)
    for (let k = 0; k < v.length; k++) a[r * n + c0 + k] = (a[r * n + c0 + k] ?? 0) - 2 * (v[k] ?? 0) * d
  }
}

/** Upper Hessenberg form A = Q H Q^T by Householder reflections. */
export function hessenbergKernel(a: Float64Array, n: number): { h: Float64Array; q: Float64Array } {
  const h = Float64Array.from(a)
  const q = new Float64Array(n * n)
  for (let i = 0; i < n; i++) q[i * n + i] = 1
  for (let k = 0; k < n - 2; k++) {
    const x = Array.from({ length: n - k - 1 }, (_, i) => h[(k + 1 + i) * n + k] ?? 0)
    const v = reflector(x)
    if (v === null) continue
    reflectRows(h, n, v, k + 1, 0, n - 1)
    reflectCols(h, n, v, k + 1, 0, n - 1)
    reflectCols(q, n, v, k + 1, 0, n - 1)
  }
  for (let i = 2; i < n; i++) for (let j = 0; j < i - 1; j++) h[i * n + j] = 0
  return { h, q }
}

/**
 * The real Schur form A = Z T Z^T: T quasi-upper-triangular (1 x 1 blocks for real eigenvalues, 2 x 2 blocks for complex
 * pairs), Z orthogonal.
 */
export function schurKernel(a: Float64Array, n: number): { t: Float64Array; z: Float64Array } {
  const { h, q: z } = hessenbergKernel(a, n)
  let hi = n - 1
  let iterations = 0
  while (hi > 0) {
    // Deflate negligible subdiagonal entries, and find the start of the active unreduced block.
    let lo = hi
    while (lo > 0) {
      const s = Math.abs(h[(lo - 1) * n + lo - 1] ?? 0) + Math.abs(h[lo * n + lo] ?? 0)
      if (Math.abs(h[lo * n + lo - 1] ?? 0) <= EPS * (s === 0 ? 1 : s)) {
        h[lo * n + lo - 1] = 0
        break
      }
      lo -= 1
    }
    if (lo === hi) {
      hi -= 1
      iterations = 0
      continue
    }
    if (lo === hi - 1) {
      hi -= 2
      iterations = 0
      continue
    }
    iterations += 1
    if (iterations > 60 * n) throw new MdmlError('convergence', 'schur: QR iteration did not converge')
    francisStep(h, z, n, lo, hi, iterations)
  }
  return { t: h, z }
}

/** One Francis double-shift QR step on the active block lo..hi, with exceptional shifts every tenth iteration. */
function francisStep(h: Float64Array, z: Float64Array, n: number, lo: number, hi: number, iterations: number): void {
  const at = (i: number, j: number): number => h[i * n + j] ?? 0
  // The shifts are the roots of (lambda - sx)(lambda - sy) - sw: the trailing 2 x 2 block's eigenvalues, or every
  // tenth iteration EISPACK's exceptional shift about the trailing diagonal entry, which breaks cycles.
  let sx: number
  let sy: number
  let sw: number
  if (iterations % 10 === 0) {
    const e = Math.abs(at(hi, hi - 1)) + Math.abs(at(hi - 1, hi - 2))
    sx = at(hi, hi) + 0.75 * e
    sy = sx
    sw = -0.4375 * e * e
  } else {
    sx = at(hi, hi)
    sy = at(hi - 1, hi - 1)
    sw = at(hi, hi - 1) * at(hi - 1, hi)
  }
  // The first column of (H - sx)(H - sy) - sw, divided by h[lo+1][lo] and formed from differences (EISPACK hqr), so
  // that a shift equal to the diagonal (repeated eigenvalues) does not cancel the column into rounding noise.
  const zz = at(lo, lo)
  const r0 = sx - zz
  const s0 = sy - zz
  let x = (r0 * s0 - sw) / at(lo + 1, lo) + at(lo, lo + 1)
  let y = at(lo + 1, lo + 1) - zz - r0 - s0
  let w = at(lo + 2, lo + 1)
  for (let k = lo; k <= hi - 2; k++) {
    const v = reflector([x, y, w])
    if (v !== null) {
      reflectRows(h, n, v, k, Math.max(lo, k - 1), n - 1)
      reflectCols(h, n, v, k, 0, Math.min(k + 3, hi))
      reflectCols(z, n, v, k, 0, n - 1)
    }
    x = at(k + 1, k)
    y = at(k + 2, k)
    if (k < hi - 2) w = at(k + 3, k)
  }
  const v = reflector([x, y])
  if (v !== null) {
    reflectRows(h, n, v, hi - 1, hi - 2, n - 1)
    reflectCols(h, n, v, hi - 1, 0, hi)
    reflectCols(z, n, v, hi - 1, 0, n - 1)
  }
  // Bulge chasing leaves rounding residue below the subdiagonal; clear it.
  for (let i = lo + 2; i <= hi; i++) for (let j = lo; j < i - 1; j++) h[i * n + j] = 0
}

/** The eigenvalues of a real Schur form, in its diagonal order. */
export function schurEigenvalues(t: Float64Array, n: number): { re: number[]; im: number[] } {
  const re: number[] = []
  const im: number[] = []
  let i = 0
  while (i < n) {
    if (i === n - 1 || (t[(i + 1) * n + i] ?? 0) === 0) {
      re.push(t[i * n + i] ?? 0)
      im.push(0)
      i += 1
      continue
    }
    const a = t[i * n + i] ?? 0
    const b = t[i * n + i + 1] ?? 0
    const c = t[(i + 1) * n + i] ?? 0
    const d = t[(i + 1) * n + i + 1] ?? 0
    const mean = (a + d) / 2
    const disc = ((a - d) / 2) ** 2 + b * c
    if (disc >= 0) {
      // A 2 x 2 block with real eigenvalues (left undeflated by rounding).
      const r = Math.sqrt(disc)
      re.push(mean + r, mean - r)
      im.push(0, 0)
    } else {
      const r = Math.sqrt(-disc)
      re.push(mean, mean)
      im.push(r, -r)
    }
    i += 2
  }
  return { re, im }
}

/**
 * An eigenvector of A for eigenvalue (re + i im), by two steps of inverse iteration on the real 2n x 2n embedding of
 * (A - lambda I) v = b, normalised to unit length with its largest-magnitude component real and positive.
 */
export function eigenvectorKernel(
  a: Float64Array,
  n: number,
  re: number,
  im: number,
  solve: (m: Float64Array, b: Float64Array, size: number) => Float64Array,
): { re: Float64Array; im: Float64Array } {
  const size = 2 * n
  const m = new Float64Array(size * size)
  // A slightly perturbed shift keeps the system nonsingular while making the eigenvector dominant.
  const scale = Math.max(1, Math.abs(re), Math.abs(im))
  const sr = re + scale * 1e-10
  const si = im
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = (a[i * n + j] ?? 0) - (i === j ? sr : 0)
      m[i * size + j] = v
      m[(n + i) * size + n + j] = v
    }
    m[i * size + n + i] = si
    m[(n + i) * size + i] = -si
  }
  let b = new Float64Array(size)
  for (let i = 0; i < n; i++) b[i] = 1 / Math.sqrt(n) + i * 1e-3
  for (let iter = 0; iter < 3; iter++) {
    const x = solve(m, b, size)
    let norm = 0
    for (const v of x) norm += v * v
    norm = Math.sqrt(norm)
    b = x.map((v) => v / norm)
  }
  // Rotate the phase so that the largest component is real and positive.
  let best = 0
  let bestMag = -1
  for (let i = 0; i < n; i++) {
    const mag = (b[i] ?? 0) ** 2 + (b[n + i] ?? 0) ** 2
    if (mag > bestMag) {
      bestMag = mag
      best = i
    }
  }
  const pr = (b[best] ?? 0) / Math.sqrt(bestMag)
  const pi = -(b[n + best] ?? 0) / Math.sqrt(bestMag)
  const outRe = new Float64Array(n)
  const outIm = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const xr = b[i] ?? 0
    const xi = b[n + i] ?? 0
    outRe[i] = xr * pr - xi * pi
    outIm[i] = xr * pi + xi * pr
  }
  if (im === 0) outIm.fill(0)
  return { re: outRe, im: outIm }
}
