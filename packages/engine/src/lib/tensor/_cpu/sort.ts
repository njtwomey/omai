import type { Tensor } from '../tensor.ts'
import { lanes, wrap } from './_util.ts'

// NaN sorts last in either direction, as in numpy.
function compareValues(a: number, b: number, descending: boolean): number {
  const an = Number.isNaN(a)
  const bn = Number.isNaN(b)
  if (an || bn) return an === bn ? 0 : an ? 1 : -1
  return descending ? b - a : a - b
}

export function argsort(x: Tensor, axis: number, descending: boolean): Tensor<'int32'> {
  const { outer, n, inner } = lanes(x.shape, axis)
  const out = new Int32Array(x.data.length)
  const order = new Array<number>(n)
  const values = new Array<number>(n)
  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < inner; k++) {
      for (let i = 0; i < n; i++) {
        order[i] = i
        values[i] = x.data[(o * n + i) * inner + k] ?? 0
      }
      // Array.prototype.sort is stable, so equal values keep their original order.
      order.sort((i, j) => compareValues(values[i] ?? 0, values[j] ?? 0, descending))
      for (let i = 0; i < n; i++) out[(o * n + i) * inner + k] = order[i] ?? 0
    }
  }
  return wrap('int32', x.shape, out)
}
