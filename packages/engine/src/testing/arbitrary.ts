// Test helpers shared by property tests: fast-check generators of tensors and a wrapper that runs checkPrimitive.
// Not published (the testing folder is excluded from the build and from barrels).
import fc from 'fast-check'
import { checkPrimitive } from '../lib/autodiff/index.ts'
import { fromStorage, sizeOf, type Tensor, type Value } from '../lib/tensor/index.ts'

/** A float64 tensor of a given shape with values in `[lo, hi]`. */
export function tensorOf(shape: readonly number[], lo = -2, hi = 2): fc.Arbitrary<Tensor<'float64'>> {
  const n = sizeOf(shape)
  return fc
    .array(fc.double({ min: lo, max: hi, noNaN: true, noDefaultInfinity: true }), { minLength: n, maxLength: n })
    .map((values) => fromStorage(Float64Array.from(values), [...shape], 'float64'))
}

/** Values in `[lo, hi]` with magnitude at least `gap` (keeps away from kinks at zero). */
export function awayFromZero(shape: readonly number[], gap = 0.1, hi = 2): fc.Arbitrary<Tensor<'float64'>> {
  return tensorOf(shape, gap, hi).chain((t) =>
    fc.array(fc.boolean(), { minLength: t.data.length, maxLength: t.data.length }).map((signs) =>
      fromStorage(
        Float64Array.from(t.data, (v, i) => (signs[i] ? v : -v)),
        t.shape,
        'float64',
      ),
    ),
  )
}

/**
 * Values in `[lo, hi]` made pairwise distinct (a 0.1 step per element is added, then any two values closer than `DISTINCT_GAP`
 * are pushed apart), for functions that are only subdifferentiable at ties (max, min, sort, top-k, the infinity norm) or at zero
 * spread (standard deviation). The gap keeps a finite-difference step from straddling a near tie: two values 9e-4 apart made the
 * check of the infinity norm fail on CI on 2026-11-13.
 */
export function distinctTensor(shape: readonly number[], lo = -2, hi = 2): fc.Arbitrary<Tensor<'float64'>> {
  return tensorOf(shape, lo, hi).map((t) => {
    const data = Float64Array.from(t.data, (v, i) => v + 0.1 * i)
    const order = Array.from(data.keys()).sort((a, b) => (data[a] ?? 0) - (data[b] ?? 0))
    for (let k = 1; k < order.length; k++) {
      const previous = data[order[k - 1] as number] ?? 0
      if ((data[order[k] as number] ?? 0) - previous < DISTINCT_GAP) data[order[k] as number] = previous + DISTINCT_GAP
    }
    return fromStorage(data, t.shape, 'float64')
  })
}

/** The least distance between two values of `distinctTensor`. */
const DISTINCT_GAP = 0.05

/** Shapes used for elementwise checks. */
export const unaryShapes: readonly (readonly number[])[] = [[], [4], [2, 3]]

/** Pairs of broadcast-compatible shapes used for binary checks. */
export const binaryShapes: readonly (readonly [readonly number[], readonly number[]])[] = [
  [[3], [3]],
  [[2, 3], [3]],
  [
    [2, 1],
    [1, 3],
  ],
  [[], [2, 2]],
  [[2, 3], []],
]

/** Assert that a function passes the derivative conformance check at the given inputs. */
export function expectConformant(
  f: (...xs: Value[]) => Value,
  inputs: readonly Value[],
  options: Parameters<typeof checkPrimitive>[2] = {},
): void {
  const report = checkPrimitive(f, inputs, options)
  if (!report.passed) {
    throw new Error(`derivative check failed:\n  ${report.failures.slice(0, 6).join('\n  ')}`)
  }
}

/** Number of random cases per property (kept small: each case runs several finite-difference sweeps). */
export const RUNS = 20
