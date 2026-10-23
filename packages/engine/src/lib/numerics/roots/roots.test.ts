import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad, jacobian } from '../../autodiff/index.ts'
import {
  add,
  concat,
  cos,
  exp,
  item,
  mul,
  pow,
  sliceAxis,
  square,
  sub,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import * as R from './index.ts'

type Scalar = (x: number) => number
const FUNCTIONS: Readonly<Record<string, readonly [Scalar, Scalar, Scalar, readonly [number, number], number]>> = {
  cubic: [(x) => x ** 3 - x - 1, (x) => 3 * x ** 2 - 1, (x) => 6 * x, [1, 2], 1.5],
  cos: [Math.cos, (x) => -Math.sin(x), (x) => -Math.cos(x), [0, 2], 1],
  exp: [(x) => Math.exp(x) - 5, Math.exp, Math.exp, [0, 3], 1],
  kepler: [(x) => x - 0.8 * Math.sin(x) - 1.2, (x) => 1 - 0.8 * Math.cos(x), (x) => 0.8 * Math.sin(x), [0, 4], 1.2],
  flat: [(x) => (x - 1.5) ** 3 + 1e-3 * (x - 1.5), (x) => 3 * (x - 1.5) ** 2 + 1e-3, (x) => 6 * (x - 1.5), [0, 3], 2.5],
}

const system = (v: Value): Value => {
  const x = sliceAxis(v, 0, 0, 1)
  const y = sliceAxis(v, 0, 1, 2)
  return concat([sub(add(square(x), square(y)), 4), sub(add(exp(x), y), 1)])
}

describe('root finding against scipy.optimize', () => {
  for (const c of loadFixture('numerics/roots/roots').cases) {
    it(c.name, () => {
      if (c.fn === 'fsolve' || c.fn === 'broyden') {
        const r = (c.fn === 'fsolve' ? R.fsolve : R.broyden)(system, decode(c.inputs[0]) as Tensor)
        expect(r.converged).toBe(true)
        expectMatches(r.x, c.output, c.rtol, c.atol, c.name)
        return
      }
      if (c.fn === 'fixedPoint') {
        for (const method of ['del2', 'iteration', 'anderson'] as const) {
          expectMatches(
            R.fixedPoint((x) => cosine(x), decode(c.inputs[0]) as Tensor, { method, xtol: 1e-13 }).x,
            c.output,
            1e-9,
            1e-12,
            `${c.name} ${method}`,
          )
        }
        return
      }
      const [f, fp, fpp, [a, b], x0] = FUNCTIONS[c.params['fn_name'] as string] as (typeof FUNCTIONS)[string]
      const r =
        c.fn === 'bisect'
          ? R.bisect(f, a, b)
          : c.fn === 'brentq' || c.fn === 'exact'
            ? R.brentq(f, a, b, { xtol: 1e-15 })
            : c.fn === 'ridder'
              ? R.ridder(f, a, b)
              : c.fn === 'toms748'
                ? R.toms748(f, a, b)
                : c.fn === 'newton'
                  ? R.newton(f, x0, { fprime: fp })
                  : c.fn === 'secant'
                    ? R.newton(f, x0)
                    : R.halley(f, x0, fp, fpp)
      expect(r.converged).toBe(true)
      expectMatches(r.root, c.output, c.rtol, c.atol, c.name)
    })
  }
})

const cosine = (x: Value): Value => cos(x)

describe('convergence behaviour', () => {
  it('Brent and TOMS 748 use fewer evaluations than bisection', () => {
    const f = (x: number) => x ** 3 - x - 1
    const b = R.bisect(f, 1, 2).functionCalls
    expect(R.brentq(f, 1, 2).functionCalls).toBeLessThan(b / 3)
    expect(R.toms748(f, 1, 2).functionCalls).toBeLessThan(b / 3)
  })

  it('Newton converges quadratically: the error roughly squares each step', () => {
    const errors = R.newton((x) => x * x - 2, 1, { fprime: (x) => 2 * x, tol: 1e-15 })
      .history.map((s) => Math.abs(s.x - Math.SQRT2))
      .filter((e) => e > 1e-13)
    for (let i = 1; i < errors.length; i++) expect(errors[i] ?? 1).toBeLessThan(2 * (errors[i - 1] ?? 0) ** 2 + 1e-15)
  })

  it('reject brackets without a sign change', () => {
    expect(() => R.brentq((x) => x * x + 1, -1, 1)).toThrow(/different signs/)
  })
})

describe('differentiable roots', () => {
  it('scalar roots differentiate by the implicit function theorem', () => {
    // x^3 + theta x - 1 = 0: dx/dtheta = -x / (3x^2 + theta).
    const rootAt = (t: Value): Value =>
      R.scalarRootOf(
        (x, th) => sub(add(pow(x, 3), mul(th, x)), 1),
        t,
        (g) => R.brentq(g, 0, 2).root,
      )
    const theta = 0.7
    const x = item(rootAt(tensor(theta)))
    expect(item(grad(rootAt)(tensor(theta)) as Value)).toBeCloseTo(-x / (3 * x * x + theta), 10)
  })

  it('system roots differentiate by the implicit function theorem', () => {
    // A x = theta with A = [[2, 1], [1, 3]]: dx/dtheta = A^{-1}.
    const f = (x: Value, th: Value): Value => {
      const x0 = sliceAxis(x, 0, 0, 1)
      const x1 = sliceAxis(x, 0, 1, 2)
      return sub(concat([add(mul(x0, 2), x1), add(x0, mul(x1, 3))]), th)
    }
    const j = toFlat(jacobian((t: Value) => R.rootOf(f, t, [0, 0]))(tensor([1, 2])))
    ;[0.6, -0.2, -0.2, 0.4].forEach((v, i) => {
      expect(j[i] ?? 0).toBeCloseTo(v, 10)
    })
  })
})

describe('continuation', () => {
  it('pseudo-arclength continuation passes the fold of x^2 + lambda = 0', () => {
    const r = R.arcLengthContinuation((x, l) => add(square(x), l), [-1], -1, { ds: 0.1, steps: 30 })
    expect(r.folds).toHaveLength(1)
    // Every point lies on the curve, and the path reaches the other branch.
    expect(r.points.every((p) => Math.abs((p.x[0] ?? 0) ** 2 + p.lambda) < 1e-9)).toBe(true)
    expect((r.points.at(-1)?.x[0] ?? 0) > 0).toBe(true)
  })

  it('homotopy reaches the root', () => {
    const r = R.homotopy((x) => sub(pow(x, 3), tensor([8])), [10])
    expect(toFlat(r.x)[0]).toBeCloseTo(2, 10)
    expect(r.path).toHaveLength(21)
  })
})
