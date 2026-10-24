// Helpers shared by the linear decompositions (private to the area).
import type { MatrixView } from '../../lib/learn/index.ts'
import { matrixView } from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { fromStorage, matmul, type Tensor, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'

/** Column means of a matrix view. */
export function columnMeans(m: MatrixView): number[] {
  return Array.from({ length: m.d }, (_, j) => {
    let s = 0
    for (let i = 0; i < m.n; i++) s += m.data[i * m.d + j] ?? 0
    return s / m.n
  })
}

/** The rows minus a location, flat. */
export function centred(m: MatrixView, mean: readonly number[]): Float64Array {
  const out = new Float64Array(m.n * m.d)
  for (let i = 0; i < m.n; i++)
    for (let j = 0; j < m.d; j++) out[i * m.d + j] = (m.data[i * m.d + j] ?? 0) - (mean[j] ?? 0)
  return out
}

/** A thin SVD of a flat [n, d] matrix as plain arrays: u [n][k], s [k], vt [k][d]; signs as scikit-learn's PCA. */
export function thinSvd(flat: Float64Array, n: number, d: number): { u: number[][]; s: number[]; vt: number[][] } {
  const r = linalg.svd(fromStorage(flat, [n, d], 'float64'))
  const k = Math.min(n, d)
  const u = toFlat(r.u)
  const s = toFlat(r.s)
  const vt = toFlat(r.vh)
  const U = Array.from({ length: n }, (_, i) => Array.from({ length: k }, (_, c) => u[i * k + c] ?? 0))
  const V = Array.from({ length: k }, (_, c) => Array.from({ length: d }, (_, j) => vt[c * d + j] ?? 0))
  flipSigns(U, V)
  return { u: U, s: s.slice(0, k), vt: V }
}

/** scikit-learn's svd_flip with the decision on V: each component's largest-magnitude loading is positive. */
export function flipSigns(u: number[][] | undefined, vt: number[][]): void {
  for (const [c, row] of vt.entries()) {
    let best = 0
    for (let j = 1; j < row.length; j++) if (Math.abs(row[j] ?? 0) > Math.abs(row[best] ?? 0)) best = j
    if ((row[best] ?? 0) < 0) {
      vt[c] = row.map((v) => -v)
      if (u !== undefined) for (const ur of u) ur[c] = -(ur[c] ?? 0)
    }
  }
}

/** Scores (x - mean) W^T for components W [k][d], optionally divided per component. */
export function project(
  x: TensorLike,
  mean: readonly number[],
  components: readonly (readonly number[])[],
  scale?: readonly number[],
): Tensor {
  const d = mean.length
  const m = matrixView(x, 'transform', { nFeatures: d })
  const k = components.length
  const out = new Float64Array(m.n * k)
  for (let i = 0; i < m.n; i++)
    for (let c = 0; c < k; c++) {
      let s = 0
      const w = components[c] ?? []
      for (let j = 0; j < d; j++) s += ((m.data[i * d + j] ?? 0) - (mean[j] ?? 0)) * (w[j] ?? 0)
      out[i * k + c] = s / (scale?.[c] ?? 1)
    }
  return fromStorage(out, [m.n, k], 'float64')
}

/** Reconstructions Z W + mean, optionally scaling the scores back per component. */
export function reconstruct(
  z: TensorLike,
  mean: readonly number[],
  components: readonly (readonly number[])[],
  scale?: readonly number[],
): Tensor {
  const k = components.length
  const d = mean.length
  const m = matrixView(z, 'inverseTransform', { nFeatures: k })
  const out = new Float64Array(m.n * d)
  for (let i = 0; i < m.n; i++)
    for (let j = 0; j < d; j++) {
      let s = mean[j] ?? 0
      for (let c = 0; c < k; c++) s += (m.data[i * k + c] ?? 0) * (scale?.[c] ?? 1) * (components[c]?.[j] ?? 0)
      out[i * d + j] = s
    }
  return fromStorage(out, [m.n, d], 'float64')
}

/** Rows of a tensor as plain arrays. */
export function rowsOf(t: Tensor): number[][] {
  const [n = 0, k = 0] = t.shape
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * k, (i + 1) * k))
}

/** Mean squared reconstruction error per entry with the first r components, r = 1..k, from singular values. */
export function rankErrors(s: readonly number[], totalSquares: number, n: number, d: number, k: number): number[] {
  let kept = 0
  return Array.from({ length: k }, (_, r) => {
    kept += (s[r] ?? 0) ** 2
    return Math.max(0, totalSquares - kept) / (n * d)
  })
}

/** A matrix product of row arrays, through the tensor matmul. */
export function mm(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] {
  if (a.length === 0 || b.length === 0) return a.map(() => [])
  return toArray(matmul(a as number[][], b as number[][])) as number[][]
}

/** The transpose of row arrays. */
export function tr(a: readonly (readonly number[])[]): number[][] {
  return (a[0] ?? []).map((_, j) => a.map((r) => r[j] ?? 0))
}

/** The Moore-Penrose pseudo-inverse of row arrays (numerics/linalg). */
export function pinvRows(a: readonly (readonly number[])[]): number[][] {
  return toArray(linalg.pinv(a as number[][])) as number[][]
}
