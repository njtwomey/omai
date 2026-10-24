import { describe, expect, it } from 'vitest'
import { bind1, definePrimitive, matmul, mul, sin, sum, type Tensor, tensor, type Value } from '../tensor/index.ts'
import { checkPrimitive, dotProductTest, gradCheck } from './check.ts'

const twice = (batch: 'right' | 'wrong' | 'none', vjpScale = 2) =>
  definePrimitive<null>({
    name: `twice-${batch}`,
    abstractEval: ([x]) => [x as NonNullable<typeof x>],
    impl: ([x]) => [(x as Tensor).backend.map(x as Tensor, (v) => 2 * v, 'float64')],
    jvp: (_xs, [t]) => [t === null || t === undefined ? null : mul(t, 2)],
    vjp: ([g]) => [g === null || g === undefined ? null : mul(g, vjpScale)],
    ...(batch === 'none'
      ? {}
      : {
          batch: (xs: readonly Value[], ds: readonly (number | null)[]) => ({
            outputs: [mul(xs[0] as Value, batch === 'right' ? 2 : 3)],
            dims: [ds[0] ?? 0],
          }),
        }),
  })

describe('checkPrimitive', () => {
  it('passes correct rules, with or without a batch rule', () => {
    for (const kind of ['right', 'none'] as const) {
      const p = twice(kind)
      expect(
        checkPrimitive(
          (x) => bind1(p, [x], null),
          [
            tensor([
              [1, 2],
              [3, 4],
            ]),
          ],
        ).failures,
      ).toEqual([])
    }
  })

  it('catches a batch rule that disagrees with the loop', () => {
    const p = twice('wrong')
    const report = checkPrimitive((x) => bind1(p, [x], null), [tensor([1, 2])])
    expect(report.passed).toBe(false)
    expect(report.failures.some((f) => f.startsWith('vmap'))).toBe(true)
    expect(checkPrimitive((x) => bind1(p, [x], null), [tensor([1, 2])], { vmap: false }).passed).toBe(true)
  })

  it('catches a vjp that is not the transpose of the jvp', () => {
    const p = twice('right', 3)
    expect(checkPrimitive((x) => bind1(p, [x], null), [tensor([1, 2])]).passed).toBe(false)
    expect(dotProductTest((x) => bind1(p, [x], null), tensor([1, 2])).passed).toBe(false)
  })
})

describe('dotProductTest', () => {
  it('holds to rounding error for composite functions', () => {
    const a = tensor([
      [1, 2, 0],
      [0.5, -1, 3],
    ])
    const report = dotProductTest((x: Value) => sin(matmul(a, x)), tensor([0.3, -0.2, 0.9]))
    expect(report.passed).toBe(true)
    expect(report.maxRelError).toBeLessThan(1e-12)
  })

  it('agrees with gradCheck on a scalar function', () => {
    const f = (x: Value): Value => sum(mul(sin(x), x))
    expect(dotProductTest(f, tensor([0.1, 0.7])).passed).toBe(true)
    expect(gradCheck(f, tensor([0.1, 0.7])).passed).toBe(true)
  })
})
