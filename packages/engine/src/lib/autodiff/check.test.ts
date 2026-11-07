import { describe, expect, it } from 'vitest'
import { bind1, cos, definePrimitive, matmul, mul, sin, sum, type Tensor, tensor, type Value } from '../tensor/index.ts'
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

describe('checkPrimitive at second order', () => {
  // sin with a vjp and jvp that are right at first order: through operations (`cos`, differentiable again), through a
  // kernel (`backend.map`, a dead end for the next derivative), or with the wrong sign of the second derivative.
  const sine = (rule: 'ops' | 'kernel' | 'wrong') => {
    const dsin = (x: Value): Value =>
      rule === 'kernel' ? (x as Tensor).backend.map(x as Tensor, Math.cos, 'float64') : cos(x)
    return definePrimitive<null>({
      name: `sine-${rule}`,
      abstractEval: ([x]) => [x as NonNullable<typeof x>],
      impl: ([x]) => [(x as Tensor).backend.map(x as Tensor, Math.sin, 'float64')],
      jvp: ([x], [t]) => [t === null || t === undefined ? null : mul(t, dsin(x as Value))],
      vjp: ([g], [x]) => [
        g === null || g === undefined ? null : mul(g, rule === 'wrong' ? mul(cos(x as Value), -1) : dsin(x as Value)),
      ],
    })
  }
  const x = tensor([0.3, -0.8, 1.4])

  it('accepts rules written with operations and catches a rule that is only right at first order', () => {
    const run = (rule: 'ops' | 'kernel' | 'wrong', order: 1 | 2) =>
      checkPrimitive((v) => bind1(sine(rule), [v], null), [x], { order, vmap: false }).failures
    expect(run('ops', 2)).toEqual([])
    // the kernel rule has a correct gradient, so only the second-order check sees it
    expect(run('kernel', 1)).toEqual([])
    expect(run('kernel', 2).map((f) => f.split(' threw')[0])).toEqual(['second order input 0'])
    // the wrong rule is caught at first order too; the second-order failures name the mode
    expect(run('wrong', 2).some((f) => f.startsWith('hvp'))).toBe(true)
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
