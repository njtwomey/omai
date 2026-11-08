/**
 * The plain-number kernels behind the power-basis polynomial functions of `power.ts`: trimming, derivative, integral,
 * sum, difference, product, division, the companion matrix, roots and the polynomial from roots, on arrays of
 * coefficients (lowest degree first) that the public functions read once at entry and wrap once at exit. `rational.ts`
 * and `orthogonal.ts` call them directly, so a polynomial does not become a tensor and back between two steps.
 */

import { OmaiError } from '../../base/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { eigvals } from '../linalg/index.ts'

/** Drop trailing (highest-degree) coefficients of absolute value at most `tol`, keeping at least one. */
export function trimKernel(c: readonly number[], tol = 0): number[] {
  let n = c.length
  while (n > 1 && Math.abs(c[n - 1] ?? 0) <= tol) n--
  return c.slice(0, n)
}

/** The m-th derivative, each differentiation scaled by `scl`. */
export function derKernel(c: readonly number[], m: number, scl: number): number[] {
  let out = [...c]
  for (let r = 0; r < m; r++) {
    if (out.length <= 1) return [0]
    out = out.slice(1).map((v, i) => v * (i + 1) * scl)
  }
  return out
}

function horner(c: readonly number[], x: number): number {
  let acc = 0
  for (let i = c.length - 1; i >= 0; i--) acc = acc * x + (c[i] ?? 0)
  return acc
}

/** The m-th integral with constants `k`, lower bound `lbnd` and scale `scl`. */
export function intKernel(c: readonly number[], m: number, k: readonly number[], lbnd: number, scl: number): number[] {
  let out = [...c]
  for (let r = 0; r < m; r++) {
    const next = [0, ...out.map((v, i) => (v * scl) / (i + 1))]
    const at = horner(next, lbnd)
    next[0] = (next[0] ?? 0) + (k[r] ?? 0) - at
    out = next
  }
  return out
}

/** The sum, trimmed of trailing zeros. */
export function addKernel(a: readonly number[], b: readonly number[]): number[] {
  return trimKernel(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) + (b[i] ?? 0)))
}

/** The difference a - b, trimmed of trailing zeros. */
export function subKernel(a: readonly number[], b: readonly number[]): number[] {
  return trimKernel(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) - (b[i] ?? 0)))
}

/** The product: the convolution of the coefficients. */
export function mulKernel(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => {
    b.forEach((y, j) => {
      out[i + j] = (out[i + j] ?? 0) + x * y
    })
  })
  return out
}

/** Division with remainder, a = q b + r with deg r < deg b. */
export function divKernel(a: readonly number[], b: readonly number[]): { quotient: number[]; remainder: number[] } {
  const d = trimKernel(b)
  const lead = d[d.length - 1] ?? 0
  if (lead === 0) throw new OmaiError('value', 'polydiv: division by the zero polynomial')
  const r = [...a]
  const nq = a.length - d.length + 1
  if (nq <= 0) return { quotient: [0], remainder: trimKernel(r) }
  const q = new Array<number>(nq).fill(0)
  for (let i = nq - 1; i >= 0; i--) {
    const f = (r[i + d.length - 1] ?? 0) / lead
    q[i] = f
    d.forEach((v, j) => {
      r[i + j] = (r[i + j] ?? 0) - f * v
    })
  }
  return { quotient: q, remainder: trimKernel(r.slice(0, Math.max(1, d.length - 1))) }
}

/** The row-major `[n, n]` companion matrix of a polynomial of degree n at least 1. */
export function companionKernel(c: readonly number[]): { readonly data: Float64Array; readonly n: number } {
  const p = trimKernel(c)
  const n = p.length - 1
  if (n < 1) throw new OmaiError('value', 'companion: degree at least 1 is needed')
  const lead = p[n] ?? 1
  const m = new Float64Array(n * n)
  for (let i = 1; i < n; i++) m[i * n + i - 1] = 1
  for (let i = 0; i < n; i++) m[i * n + n - 1] = -(p[i] ?? 0) / lead
  return { data: m, n }
}

/** The roots as parallel real and imaginary parts, sorted by real then imaginary part. */
export function rootsKernel(c: readonly number[]): { re: number[]; im: number[] } {
  const p = trimKernel(c)
  const n = p.length - 1
  if (n < 1) return { re: [], im: [] }
  if (n === 1) return { re: [-(p[0] ?? 0) / (p[1] ?? 1)], im: [0] }
  const { data } = companionKernel(p)
  const { re, im } = eigvals(fromStorage(data, [n, n], 'float64'))
  const imag = toFlat(im)
  const pairs = toFlat(re).map((r, i) => [r, imag[i] ?? 0] as const)
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return { re: pairs.map((q) => q[0]), im: pairs.map((q) => q[1]) }
}

/** The monic polynomial with roots `re + i im`, multiplied out in complex arithmetic (the real parts are returned). */
export function fromRootsKernel(re: readonly number[], im: readonly number[]): number[] {
  let pr = [1]
  let pi = [0]
  re.forEach((r, k) => {
    const s = im[k] ?? 0
    const nr = new Array<number>(pr.length + 1).fill(0)
    const ni = new Array<number>(pr.length + 1).fill(0)
    pr.forEach((a, i) => {
      const b = pi[i] ?? 0
      nr[i + 1] = (nr[i + 1] ?? 0) + a
      ni[i + 1] = (ni[i + 1] ?? 0) + b
      nr[i] = (nr[i] ?? 0) - (a * r - b * s)
      ni[i] = (ni[i] ?? 0) - (a * s + b * r)
    })
    pr = nr
    pi = ni
  })
  return pr
}
