import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { solve } from '../../numerics/linalg/index.ts'
import {
  abs,
  add,
  fromStorage,
  matmul,
  mul,
  reshape,
  square,
  sub,
  sum,
  swapaxes,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../../trace/index.ts'
import * as O from '../index.ts'

type Out = { readonly x: unknown; readonly fun: number; readonly y?: unknown }
const fixture = {
  cases: ['optim/minimize/convex', 'optim/minimize/frank_wolfe'].flatMap((f) => loadFixture(f).cases),
}
const byName = (name: string) => {
  const c = fixture.cases.find((c) => c.name === name)
  if (c === undefined) throw new Error(`missing case ${name}`)
  return { ...c, inputs: c.inputs.map((v) => decode(v) as Tensor), out: c.output as Out }
}

const mv = (a: Value, x: unknown): Value => reshape(matmul(a, reshape(x as Value, [-1, 1])), [-1])
const leastSquares = (a: Tensor, b: Tensor) => (x: unknown) => mul(sum(square(sub(mv(a, x), b))), 0.5)
const lipschitz = (a: Tensor): number => {
  // |A|_2^2 bounded by the Frobenius norm squared.
  let s = 0
  for (const v of a.data as Float64Array) s += v * v
  return s
}

describe('subgradient method against cvxpy', () => {
  it('l1 regression with the Polyak step', () => {
    const c = byName('l1Regression/0')
    const [a, b] = c.inputs as [Tensor, Tensor]
    const f = (x: unknown) => sum(abs(sub(mv(a, x), b)))
    const r = O.subgradientMethod(f, [0, 0, 0, 0], {
      stepRule: 'polyak',
      fStar: c.out.fun,
      tol: 1e-11,
      maxIterations: 50000,
    })
    expect(r.fun - c.out.fun).toBeLessThan(1e-9)
    expectMatches(r.x as Value, c.out.x, 1e-6, 1e-6, c.name)
  })

  it('the diminishing rule decreases the best value and honours a projection', () => {
    const rec = recorder()
    const f = (x: unknown) => sum(abs(sub(x as Value, tensor([2, -3]))))
    const r = O.subgradientMethod(f, [0, 0], {
      stepRule: 'diminishing',
      stepSize: 0.5,
      maxIterations: 400,
      project: (x) => O.projectBox(x, -1, 1),
      tracer: rec,
      policy: tracePolicy({}),
    })
    expect(toFlat(r.x as Value).map((v) => Math.round(v * 1e2) / 1e2)).toEqual([1, -1])
    const best = metricSeries(rec.run(), 'best').values
    expect(best.every((v, i) => i === 0 || v <= (best[i - 1] ?? Infinity))).toBe(true)
    for (const [rule, size] of [
      ['constant', 0.05],
      ['constantLength', 0.05],
      ['squareSummable', 2],
    ] as const) {
      expect(O.subgradientMethod(f, [0, 0], { stepRule: rule, stepSize: size, maxIterations: 300 }).fun).toBeLessThan(
        0.2,
      )
    }
    expect(() => O.subgradientMethod(f, [0, 0], { stepRule: 'polyak' })).toThrow()
  })
})

describe('mirror descent and Frank-Wolfe against cvxpy', () => {
  it('exponentiated gradient reaches an interior simplex solution', () => {
    const c = byName('simplexLeastSquares/interior')
    const [a, b] = c.inputs as [Tensor, Tensor]
    const r = O.mirrorDescent(leastSquares(a, b), new Array(5).fill(0.2), {
      stepRule: 'constant',
      stepSize: 1 / lipschitz(a),
      tol: 1e-15,
      maxIterations: 20000,
    })
    expectMatches(r.x as Value, c.out.x, c.rtol, c.atol, c.name)
    // The weighted average converges too, more slowly.
    const avg = O.mirrorDescent(leastSquares(a, b), new Array(5).fill(0.2), {
      stepRule: 'constant',
      stepSize: 1 / lipschitz(a),
      maxIterations: 3000,
      output: 'average',
    })
    expect(Math.abs(avg.fun - c.out.fun)).toBeLessThan(1e-3)
    // The Euclidean mirror with a simplex projection is projected gradient descent.
    const pg = O.mirrorDescent(leastSquares(a, b), new Array(5).fill(0.2), {
      mirror: O.euclideanMirror((x) =>
        toFlat(O.projectSimplex(fromStorage(Float64Array.from(x), [x.length], 'float64'))),
      ),
      stepRule: 'constant',
      stepSize: 1 / lipschitz(a),
      tol: 1e-15,
      maxIterations: 20000,
    })
    expectMatches(pg.x as Value, c.out.x, c.rtol, c.atol, c.name)
  })

  for (const name of ['simplexLeastSquares/interior', 'simplexLeastSquares/face']) {
    for (const variant of ['away', 'pairwise'] as const) {
      it(`${name} by Frank-Wolfe with ${variant} steps`, () => {
        const c = byName(name)
        const [a, b] = c.inputs as [Tensor, Tensor]
        const r = O.frankWolfe(leastSquares(a, b), [1, 0, 0, 0, 0], O.simplexLmo(), {
          variant,
          gapTol: 1e-10,
          maxIterations: 20000,
        })
        expect(r.converged).toBe(true)
        expectMatches(r.x as Value, c.out.x, c.rtol, c.atol, c.name)
        expect(r.fun - c.out.fun).toBeLessThan(1e-9)
      })
    }
  }

  it('the L1 ball and the box by away steps', () => {
    const c = byName('l1BallLeastSquares/0')
    const [a, b] = c.inputs as [Tensor, Tensor]
    const r = O.frankWolfe(leastSquares(a, b), [0, 0, 0, 0, 0], O.l1BallLmo(1), {
      variant: 'away',
      gapTol: 1e-10,
      maxIterations: 20000,
    })
    expectMatches(r.x as Value, c.out.x, c.rtol, c.atol, c.name)
    const d = byName('boxLeastSquares/0')
    const [a2, b2] = d.inputs as [Tensor, Tensor]
    const box = O.frankWolfe(leastSquares(a2, b2), [0, 0, 0, 0, 0], O.boxLmo(-0.5, 0.5), {
      variant: 'away',
      gapTol: 1e-10,
      maxIterations: 20000,
    })
    expectMatches(box.x as Value, d.out.x, d.rtol, d.atol, d.name)
  })

  it('the nuclear ball: Frank-Wolfe approaches the exact projection', () => {
    const c = byName('nuclearLeastSquares/0')
    const [m] = c.inputs as [Tensor]
    expectMatches(O.projectNuclearBall(m, 1.5), c.out.x, c.rtol, c.atol, c.name)
    const rec = recorder()
    const f = (x: unknown) => mul(sum(square(sub(reshape(x as Value, [3, 4]), m))), 0.5)
    const r = O.frankWolfe(f, new Array(12).fill(0), O.nuclearBallLmo(1.5, 3, 4), {
      maxIterations: 400,
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    expect(r.fun - c.out.fun).toBeLessThan(5e-3)
    const gaps = metricSeries(rec.run(), 'gap').values.filter(Number.isFinite)
    expect((gaps.at(-1) ?? Infinity) < (gaps[0] ?? 0)).toBe(true)
    expect(messages(rec.run(), 'step').length).toBeGreaterThan(0)
    // The standard step 2 / (k + 2) over the L2 ball.
    const ball = O.frankWolfe((x) => sum(square(sub(x as Value, tensor([3, 4])))), [0, 0], O.l2BallLmo(1), {
      variant: 'standard',
      maxIterations: 2000,
    })
    expect(toFlat(ball.x as Value).map((v) => Math.round(v * 1e2) / 1e2)).toEqual([0.6, 0.8])
    expect(() => O.frankWolfe((x) => sum(square(x as Value)), [0.5, 0], O.l2BallLmo(1), { variant: 'away' })).toThrow()
  })
})

describe('dual methods against cvxpy', () => {
  it('dual ascent and the method of multipliers on an equality-constrained QP', () => {
    const c = byName('equalityQp/0')
    const [p, q, a, b] = c.inputs as [Tensor, Tensor, Tensor, Tensor]
    const f = (x: unknown) => add(mul(sum(mul(x as Value, mv(p, x))), 0.5), sum(mul(q, x as Value)))
    const at = (y: Float64Array): Value =>
      reshape(matmul(reshape(fromStorage(y, [y.length], 'float64'), [1, -1]), a), [-1])
    const plain = O.dualAscent(f, a, b, new Float64Array(6), {
      stepSize: 0.2,
      tol: 1e-11,
      maxIterations: 20000,
      argmin: (y) => toFlat(solve(p, sub(mul(q, -1), at(y)))),
    })
    expect(plain.converged).toBe(true)
    expectMatches(plain.x as Value, c.out.x, c.rtol, c.atol, c.name)
    expectMatches(plain.multipliers, c.out.y, c.rtol, c.atol, c.name)
    const rho = 5
    const ata = matmul(swapaxes(a, 0, 1), a)
    const augmented = O.dualAscent(f, a, b, new Float64Array(6), {
      augmented: rho,
      tol: 1e-11,
      argmin: (y) =>
        toFlat(
          solve(
            add(p, mul(ata, rho)),
            add(sub(mul(q, -1), at(y)), mul(reshape(matmul(swapaxes(a, 0, 1), reshape(b, [-1, 1])), [-1]), rho)),
          ),
        ),
    })
    expect(augmented.converged).toBe(true)
    expect(augmented.steps).toBeLessThan(plain.steps)
    expectMatches(augmented.multipliers, c.out.y, c.rtol, c.atol, c.name)
  })

  it('dual decomposition of a separable problem, in closed form and numerically', () => {
    const c = byName('separable/0')
    const [w, cc] = c.inputs as [Tensor, Tensor]
    const blocks = Array.from({ length: 4 }, (_, i) => {
      const wi = (w.data as Float64Array)[i] ?? 1
      const ci = (cc.data as Float64Array)[i] ?? 0
      return {
        f: (x: unknown) => mul(sum(square(sub(x as Value, ci))), wi / 2),
        a: [[1]],
        x0: [0],
        argmin: (g: Float64Array) => [ci - (g[0] ?? 0) / wi],
      }
    })
    const closed = O.dualDecomposition(blocks, [1], { stepSize: 0.3, tol: 1e-12, maxIterations: 5000 })
    expectMatches(closed.x as Value, c.out.x, c.rtol, c.atol, c.name)
    const numeric = O.dualDecomposition(
      blocks.map(({ argmin: _a, ...b }) => b),
      [1],
      {
        stepSize: 0.3,
        tol: 1e-10,
        maxIterations: 5000,
        solve: (lagrangian, start) => toFlat(O.fista(lagrangian, (x) => x, start, { tol: 1e-14 }).x as Value),
      },
    )
    expectMatches(numeric.x as Value, c.out.x, 1e-5, 1e-6, c.name)
  })
})
