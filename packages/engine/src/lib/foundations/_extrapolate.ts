// Limits of sampled sequences by Richardson extrapolation, shared by calculus.ts (limits) and series.ts (tests).
import { richardson } from '../numerics/approximation/index.ts'

/**
 * Extrapolate values `v_i = g(h0 / 2^i)` to `h -> 0`, assuming `g(h) = L + c1 h + c2 h^2 + ...`: the Richardson
 * tableau with step ratio 2, reading off the entry that changed least from its neighbour above.
 *
 * @param values - Samples at halving steps (at least two).
 * @returns The estimate and the size of the change it rests on (a rough error estimate).
 */
export function extrapolateToZero(values: readonly number[]): { readonly value: number; readonly change: number } {
  const { table } = richardson(values, { ratio: 2, order: 1 })
  let value = values[values.length - 1] ?? Number.NaN
  let change = Infinity
  for (let r = 1; r < table.length; r++) {
    const row = table[r] ?? []
    for (let c = 0; c < row.length - 1; c++) {
      const err = Math.abs((row[c] ?? 0) - (table[r - 1]?.[c] ?? 0))
      if (err < change && Number.isFinite(row[c] ?? Number.NaN)) {
        change = err
        value = row[c + 1] ?? value
      }
    }
  }
  return { value, change }
}
