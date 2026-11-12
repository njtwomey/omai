import type { Dataset } from 'omai-engine/data'
import { add, arange, asTensor, div, fromStorage, mul, toFlat } from 'omai-engine/tensor'

/** A one-dimensional regression problem as plain arrays, with the true curve when the data carries it. */
export interface CurveData {
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly lo: number
  readonly hi: number
  /** The generator's true regression function at the given points, when it has one. */
  readonly truthAt?: (at: readonly number[]) => number[]
}

/**
 * A one-dimensional regression dataset as arrays (`x` with one column), or `undefined` for anything else.
 *
 * @param data - The dataset.
 * @param truth - The generator's truth, when it has one (its `regressionFunction` becomes `truthAt`).
 * @returns The curve data.
 */
export function curveData(
  data: Dataset,
  truth?: { readonly regressionFunction?: (x: never) => unknown },
): CurveData | undefined {
  if (data.kind !== 'regression') return undefined
  const x = asTensor(data.x)
  if ((x.shape[1] ?? 1) !== 1) return undefined
  const xs = Array.from(toFlat(x))
  const ys = Array.from(toFlat(asTensor(data.y)))
  const f = truth?.regressionFunction
  return {
    x: xs,
    y: ys,
    lo: Math.min(...xs),
    hi: Math.max(...xs),
    ...(f === undefined
      ? {}
      : {
          truthAt: (at: readonly number[]): number[] =>
            Array.from(
              toFlat(asTensor(f(fromStorage(Float64Array.from(at), [at.length, 1], 'float64') as never) as never)),
            ),
        }),
  }
}

/** `count` points spread evenly over `[lo, hi]`. */
export function gridOf(lo: number, hi: number, count: number): number[] {
  return toFlat(add(div(mul(arange(count), hi - lo), count - 1), lo))
}
