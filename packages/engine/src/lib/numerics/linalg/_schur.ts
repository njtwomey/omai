// Hessenberg reduction and the real Schur form by Francis double-shift QR (Golub and Van Loan, algorithms 7.4.2, 7.5.1
// and 7.5.2), on single row-major matrices.
import { OmaiError } from '../../base/index.ts'

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
    if (iterations > 60 * n) throw new OmaiError('convergence', 'schur: QR iteration did not converge')
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
 * (A - lambda I) v = b, normalised to unit length with its largest-magnitude component real and positive. O(n^3) per eigenvalue, so `eig` uses
 * {@link schurEigenvectors}; this is kept as the reference that one is tested against.
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

/**
 * Solve (T - sigma I) y = w in place for a real quasi-upper-triangular T and a complex shift sigma = (sr, si), by block
 * back-substitution (1 x 1 blocks by a complex division, 2 x 2 blocks by Cramer's rule). `starts` lists the first row of
 * every diagonal block; a pivot smaller than `tiny` is replaced by `tiny`, which keeps an exactly singular shift finite.
 */
function solveShifted(
  t: Float64Array,
  n: number,
  starts: readonly number[],
  sr: number,
  si: number,
  tiny: number,
  wr: Float64Array,
  wi: Float64Array,
): void {
  for (let b = starts.length - 1; b >= 0; b--) {
    const p = starts[b] ?? 0
    const end = b === starts.length - 1 ? n : (starts[b + 1] ?? n)
    // Subtract the already solved unknowns below the block from its right-hand side.
    for (let i = p; i < end; i++) {
      let sRe = 0
      let sIm = 0
      for (let j = end; j < n; j++) {
        const tij = t[i * n + j] ?? 0
        sRe += tij * (wr[j] ?? 0)
        sIm += tij * (wi[j] ?? 0)
      }
      wr[i] = (wr[i] ?? 0) - sRe
      wi[i] = (wi[i] ?? 0) - sIm
    }
    if (end - p === 1) {
      divide(wr, wi, p, (t[p * n + p] ?? 0) - sr, -si, tiny)
      continue
    }
    // 2 x 2 block [[a - sigma, b], [c, d - sigma]]: det = (a - sigma)(d - sigma) - b c.
    const a = (t[p * n + p] ?? 0) - sr
    const bb = t[p * n + p + 1] ?? 0
    const c = t[(p + 1) * n + p] ?? 0
    const d = (t[(p + 1) * n + p + 1] ?? 0) - sr
    // (a + i(-si)) (d + i(-si)) = a d - si^2 - i si (a + d).
    let detRe = a * d - si * si - bb * c
    let detIm = -si * (a + d)
    if (Math.hypot(detRe, detIm) < Math.max(tiny * tiny, 1e-300)) {
      detRe = Math.max(tiny * tiny, 1e-300)
      detIm = 0
    }
    const r1Re = wr[p] ?? 0
    const r1Im = wi[p] ?? 0
    const r2Re = wr[p + 1] ?? 0
    const r2Im = wi[p + 1] ?? 0
    // y1 = ((d - sigma) r1 - b r2) / det, y2 = ((a - sigma) r2 - c r1) / det, with d - sigma = (d, -si).
    wr[p] = d * r1Re + si * r1Im - bb * r2Re
    wi[p] = d * r1Im - si * r1Re - bb * r2Im
    wr[p + 1] = a * r2Re + si * r2Im - c * r1Re
    wi[p + 1] = a * r2Im - si * r2Re - c * r1Im
    divide(wr, wi, p, detRe, detIm, 0)
    divide(wr, wi, p + 1, detRe, detIm, 0)
  }
}

/** w[i] = w[i] / (pRe + i pIm) in place, with a divisor of modulus below `tiny` replaced by `tiny`. */
function divide(wr: Float64Array, wi: Float64Array, i: number, pRe: number, pIm: number, tiny: number): void {
  let dr = pRe
  let di = pIm
  if (Math.hypot(dr, di) < tiny) {
    dr = tiny
    di = 0
  }
  const x = wr[i] ?? 0
  const y = wi[i] ?? 0
  // Smith's algorithm, which avoids overflow in the squared modulus.
  if (Math.abs(dr) >= Math.abs(di)) {
    const q = di / dr
    const den = dr + di * q
    wr[i] = (x + y * q) / den
    wi[i] = (y - x * q) / den
  } else {
    const q = dr / di
    const den = dr * q + di
    wr[i] = (x * q + y) / den
    wi[i] = (y * q - x) / den
  }
}

/**
 * The eigenvectors of A = Z T Z^T for the eigenvalues `re + i im` of its real Schur form, as the columns of two n x n
 * row-major matrices (real and imaginary parts): the same three steps of shifted inverse iteration, with the same
 * shift, start vector, unit length and phase (largest component real and positive) as {@link eigenvectorKernel}, but
 * carried out on the quasi-triangular T (a block back-substitution, O(n^2) per step) instead of a dense 2n x 2n LU
 * (O(n^3) per eigenvalue), then rotated back by Z. O(n^3) in all.
 */
export function schurEigenvectors(
  t: Float64Array,
  z: Float64Array,
  n: number,
  re: readonly number[],
  im: readonly number[],
): { re: Float64Array; im: Float64Array } {
  // The diagonal blocks, found as schurEigenvalues finds them.
  const starts: number[] = []
  for (let i = 0; i < n; ) {
    starts.push(i)
    i += i === n - 1 || (t[(i + 1) * n + i] ?? 0) === 0 ? 1 : 2
  }
  let normT = 0
  for (const e of t) normT = Math.max(normT, Math.abs(e))
  const tiny = Math.max(EPS * normT, 1e-300)
  // Z^T b for the start vector b_i = 1 / sqrt(n) + 1e-3 i, which is the same for every eigenvalue.
  const c0 = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const bi = 1 / Math.sqrt(n) + i * 1e-3
    for (let j = 0; j < n; j++) c0[j] = (c0[j] ?? 0) + (z[i * n + j] ?? 0) * bi
  }
  const outRe = new Float64Array(n * n)
  const outIm = new Float64Array(n * n)
  const wr = new Float64Array(n)
  const wi = new Float64Array(n)
  const vr = new Float64Array(n)
  const vi = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const lr = re[k] ?? 0
    const li = im[k] ?? 0
    const sr = lr + Math.max(1, Math.abs(lr), Math.abs(li)) * 1e-10
    wr.set(c0)
    wi.fill(0)
    for (let iter = 0; iter < 3; iter++) {
      solveShifted(t, n, starts, sr, li, tiny, wr, wi)
      let norm = 0
      for (let i = 0; i < n; i++) norm += (wr[i] ?? 0) ** 2 + (wi[i] ?? 0) ** 2
      norm = Math.sqrt(norm)
      for (let i = 0; i < n; i++) {
        wr[i] = (wr[i] ?? 0) / norm
        wi[i] = (wi[i] ?? 0) / norm
      }
    }
    // Back to the original basis: v = Z w.
    for (let i = 0; i < n; i++) {
      let sRe = 0
      let sIm = 0
      for (let j = 0; j < n; j++) {
        const zij = z[i * n + j] ?? 0
        sRe += zij * (wr[j] ?? 0)
        sIm += zij * (wi[j] ?? 0)
      }
      vr[i] = sRe
      vi[i] = sIm
    }
    // Rotate the phase so that the largest component is real and positive.
    let best = 0
    let bestMag = -1
    for (let i = 0; i < n; i++) {
      const mag = (vr[i] ?? 0) ** 2 + (vi[i] ?? 0) ** 2
      if (mag > bestMag) {
        bestMag = mag
        best = i
      }
    }
    const pr = (vr[best] ?? 0) / Math.sqrt(bestMag)
    const pi = -(vi[best] ?? 0) / Math.sqrt(bestMag)
    for (let i = 0; i < n; i++) {
      const xr = vr[i] ?? 0
      const xi = vi[i] ?? 0
      outRe[i * n + k] = xr * pr - xi * pi
      outIm[i * n + k] = li === 0 ? 0 : xr * pi + xi * pr
    }
  }
  return { re: outRe, im: outIm }
}
