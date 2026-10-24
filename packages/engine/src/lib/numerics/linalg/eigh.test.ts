import { describe, expect, it } from 'vitest'
import { fromStorage, toArray, toFlat } from '../../tensor/index.ts'
import { eigh } from './eigh.ts'

/**
 * The diffusion-map operator of points on an S curve: a Gaussian kernel at the median squared distance, density
 * normalised (alpha 1) and made symmetric. Most of its eigenvalues sit at rounding level, the cluster on which the
 * QL iteration's relative deflation test alone never succeeds (it failed from about 250 rows).
 */
function diffusionOperator(n: number): Float64Array {
  const xs = Array.from({ length: n }, (_, i) => {
    const t = 3 * Math.PI * ((i + 0.5) / n - 0.5)
    const h = 2 * ((i * 0.6180339887498949) % 1)
    return [Math.sin(t), h, Math.sign(t) * (Math.cos(t) - 1)] as const
  })
  const d2 = (i: number, j: number): number => {
    const a = xs[i] ?? [0, 0, 0]
    const b = xs[j] ?? [0, 0, 0]
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
  }
  const all: number[] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) all.push(d2(i, j))
  all.sort((a, b) => a - b)
  const eps = all[Math.floor(all.length / 2)] ?? 1
  const k = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) k[i * n + j] = Math.exp(-d2(i, j) / eps)
  const rowSums = (): number[] =>
    Array.from({ length: n }, (_, i) => {
      let s = 0
      for (let j = 0; j < n; j++) s += k[i * n + j] ?? 0
      return s
    })
  const q = rowSums()
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) k[i * n + j] = (k[i * n + j] ?? 0) / ((q[i] ?? 1) * (q[j] ?? 1))
  const deg = rowSums()
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) k[i * n + j] = (k[i * n + j] ?? 0) / Math.sqrt((deg[i] ?? 1) * (deg[j] ?? 1))
  return k
}

describe('eigh on a numerically rank-deficient matrix', () => {
  const n = 260
  const a = diffusionOperator(n)
  let cached: { readonly w: number[]; readonly v: number[][] } | undefined
  const solved = (): { readonly w: number[]; readonly v: number[][] } => {
    if (cached === undefined) {
      const { eigenvalues, eigenvectors } = eigh(fromStorage(a, [n, n], 'float64'))
      cached = { w: toFlat(eigenvalues), v: toArray(eigenvectors) as number[][] }
    }
    return cached
  }

  it('converges, with the Markov operator’s top eigenvalue of one and ascending eigenvalues', () => {
    const { w } = solved()
    expect(w[n - 1]).toBeCloseTo(1, 12)
    for (let i = 1; i < n; i++) expect(w[i] ?? 0).toBeGreaterThanOrEqual(w[i - 1] ?? 0)
    expect(w.filter((x) => Math.abs(x) < 1e-12).length).toBeGreaterThan(n / 4)
  })

  it('has a small residual and orthonormal eigenvectors', () => {
    const { w, v } = solved()
    let residual = 0
    let gram = 0
    for (let c = 0; c < n; c += 13) {
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (a[i * n + j] ?? 0) * (v[j]?.[c] ?? 0)
        residual = Math.max(residual, Math.abs(s - (w[c] ?? 0) * (v[i]?.[c] ?? 0)))
      }
      for (let d = 0; d < n; d += 7) {
        let s = 0
        for (let i = 0; i < n; i++) s += (v[i]?.[c] ?? 0) * (v[i]?.[d] ?? 0)
        gram = Math.max(gram, Math.abs(s - (c === d ? 1 : 0)))
      }
    }
    expect(residual).toBeLessThan(1e-13)
    expect(gram).toBeLessThan(1e-12)
  })
})
