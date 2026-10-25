// Keyed splits of learn data into parts (train, calibration, test), shared by the trust methods.
import { WhiteboxError } from '../../lib/base/index.ts'
import { type LearnData, matrixView, takeData } from '../../lib/learn/index.ts'
import { type Key, permutation } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'

/** The rows of a keyed shuffle cut into consecutive parts of the given fractions (the last takes the rest). */
export function splitRows(n: number, fractions: readonly number[], key: Key): number[][] {
  const order = Array.from(toFlat(permutation(key, n)))
  const out: number[][] = []
  let start = 0
  for (const [i, f] of fractions.entries()) {
    const end = i === fractions.length - 1 ? n : Math.min(n, start + Math.round(f * n))
    out.push(order.slice(start, end).sort((a, b) => a - b))
    start = end
  }
  if (out.some((p) => p.length === 0)) throw new WhiteboxError('value', `trust: ${n} rows are too few to split`)
  return out
}

/** The data cut into parts by {@link splitRows}. */
export function splitData<D extends LearnData>(data: D, fractions: readonly number[], key: Key): D[] {
  const n = matrixView(data.x, 'splitData').n
  return splitRows(n, fractions, key).map((rows) => takeData(data, rows))
}

/** Column `j` of a matrix. */
export function columnOf(x: LearnData['x'], j: number): number[] {
  const m = matrixView(x, 'columnOf')
  return Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0)
}
