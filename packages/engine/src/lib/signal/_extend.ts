// Extension of rows `[batch, n]` beyond their ends, shared by upfirdn and the wavelet transforms (the modes of
// PyWavelets, which scipy.signal.upfirdn adopted). Built from gathers and arithmetic, so differentiable.
import { add, concat, mul, sliceAxis, sub, take, type Value, zeros } from '../tensor/index.ts'
import { constant, indexTensor } from './_shared.ts'

/** An extension mode: pywt's names (and scipy's upfirdn's), plus 'zero'. */
export type ExtensionMode =
  | 'zero'
  | 'constant'
  | 'symmetric'
  | 'reflect'
  | 'edge'
  | 'wrap'
  | 'antisymmetric'
  | 'antireflect'
  | 'smooth'
  | 'line'

/** Extend rows by p samples at both ends under an upfirdn mode. */
export function extendRows(rows: Value, p: number, mode: ExtensionMode): Value {
  const n = rows.shape[1] ?? 0
  const idx = (f: (i: number) => number): Value => take(rows, indexTensor(Array.from({ length: p }, (_, i) => f(i))), 1)
  const first = sliceAxis(rows, 1, 0, 1)
  const last = sliceAxis(rows, 1, n - 1, n)
  const mirror = (k: number, period: number): number => {
    const j = ((k % period) + period) % period
    return j
  }
  switch (mode) {
    case 'edge':
      return concat([idx(() => 0), rows, idx(() => n - 1)], 1)
    case 'wrap':
      return concat([idx((i) => mirror(n - p + i, n)), rows, idx((i) => mirror(i, n))], 1)
    case 'symmetric':
    case 'antisymmetric': {
      const sym = (k: number): number => {
        const j = mirror(k, 2 * n)
        return j < n ? j : 2 * n - 1 - j
      }
      const left = idx((i) => sym(-p + i))
      const right = idx((i) => sym(n + i))
      if (mode === 'symmetric') return concat([left, rows, right], 1)
      return concat([mul(left, -1), rows, mul(right, -1)], 1)
    }
    case 'reflect':
    case 'antireflect': {
      const ref = (k: number): number => {
        if (n === 1) return 0
        const j = mirror(k, 2 * (n - 1))
        return j < n ? j : 2 * (n - 1) - j
      }
      const left = idx((i) => ref(-p + i))
      const right = idx((i) => ref(n + i))
      if (mode === 'reflect') return concat([left, rows, right], 1)
      return concat([sub(mul(first, 2), left), rows, sub(mul(last, 2), right)], 1)
    }
    case 'smooth':
    case 'line': {
      const second = sliceAxis(rows, 1, Math.min(1, n - 1), Math.min(2, n))
      const penult = sliceAxis(rows, 1, Math.max(n - 2, 0), Math.max(n - 1, 1))
      const slopeL = mode === 'line' ? mul(sub(last, first), 1 / Math.max(n - 1, 1)) : sub(second, first)
      const slopeR = mode === 'line' ? slopeL : sub(last, penult)
      const steps = (k: number[]): Value => constant(k, [1, k.length])
      const left = add(first, mul(slopeL, steps(Array.from({ length: p }, (_, i) => i - p))))
      const right = add(last, mul(slopeR, steps(Array.from({ length: p }, (_, i) => i + 1))))
      return concat([left, rows, right], 1)
    }
    case 'zero':
      return concat([zeros([rows.shape[0] ?? 1, p]), rows, zeros([rows.shape[0] ?? 1, p])], 1)
    default:
      return rows
  }
}
