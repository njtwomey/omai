import { describe, expect, it } from 'vitest'
import { problem, rosen } from '../../../testing/optim-problems.ts'
import { add, mul, square, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { messages, metricSeries, recorder, tensorSeries, tracePolicy } from '../../trace/index.ts'
import * as O from '../index.ts'

const round = (xs: readonly number[], d = 6): number[] => xs.map((v) => Math.round(v * 10 ** d) / 10 ** d)

describe('BFGS', () => {
  it('traces the objective, line-search trials and curvature pairs', () => {
    const rec = recorder()
    const r = O.bfgs(rosen, [-1.2, 1], {
      tracer: rec,
      policy: tracePolicy({ messages: true, tensors: true, roles: ['param'], detail: 'full' }),
    })
    expect(r.converged).toBe(true)
    const run = rec.run()
    const loss = metricSeries(run, 'loss').values
    expect(loss).toHaveLength(r.steps + 1)
    expect((loss.at(-1) ?? 1) < 1e-10).toBe(true)
    const ls = messages(run, 'lineSearch')
    expect(ls).toHaveLength(r.steps)
    const trials = (ls[0]?.payload as { trials: number[][] } | undefined)?.trials ?? []
    expect(trials.length).toBeGreaterThan(0)
    const pairs = messages(run, 'curvature')
    expect(pairs.every((m) => (m.payload as { sy: number }).sy > 0)).toBe(true)
    expect(tensorSeries(run, 'x')[0]?.values).toEqual([-1.2, 1])
  })
  it('returns the inverse Hessian, near the true one at a quadratic minimum', () => {
    const r = O.bfgs(problem('spdQuadratic'), [1, 1, 1], { gradTol: 1e-10 })
    const H = toFlat(r.hessInv)
    // A^-1 A = I for the quadratic's A.
    const A = [4, 1, 0.5, 1, 3, -0.5, 0.5, -0.5, 2]
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0
        for (let k = 0; k < 3; k++) s += (H[i * 3 + k] ?? 0) * (A[k * 3 + j] ?? 0)
        expect(Math.abs(s - (i === j ? 1 : 0))).toBeLessThan(0.05)
      }
  })
  it('works on parameter trees and with damping', () => {
    const f = (p: unknown): Value => {
      const { w, b } = p as { w: Value; b: Value }
      return add(sum(square(sub(w, tensor([1, -2])))), square(sub(b, 3)))
    }
    for (const damped of [false, true]) {
      const r = O.bfgs(f, { w: [0, 0], b: 0 }, { damped })
      const x = r.x as { w: Value; b: Value }
      expect(round(toFlat(x.w))).toEqual([1, -2])
      expect(round(toFlat(x.b))).toEqual([3])
    }
  })
  it('is the default method of minimize, and methods take their own options', () => {
    const r = O.minimize(rosen, [-1.2, 1])
    expect(r.converged).toBe(true)
    expect(round(toFlat(r.x as Value), 4)).toEqual([1, 1])
    const r2 = O.minimize(rosen, [-1.2, 1], { method: O.lbfgs, memory: 3, gradTol: 1e-8 })
    expect(r2.converged).toBe(true)
    const r3 = O.minimize(rosen, [-1.2, 1], { method: O.lbfgsb, bounds: { upper: 0.5 } })
    expect(round(toFlat(r3.x as Value), 4)).toEqual([0.5, 0.25])
  })
})

describe('L-BFGS and L-BFGS-B', () => {
  it('two-loop recursion equals the BFGS inverse built from the same pairs', () => {
    const s = [new Float64Array([1, 0.2]), new Float64Array([-0.3, 0.8])]
    const y = [new Float64Array([2, 0.1]), new Float64Array([-0.2, 1.5])]
    const gamma = 1
    let H: Float64Array = new Float64Array([gamma, 0, 0, gamma])
    for (let i = 0; i < 2; i++) H = O.bfgsInverseUpdate(H, s[i] as Float64Array, y[i] as Float64Array)
    const q = [0.7, -1.1]
    const direct = [(H[0] ?? 0) * 0.7 + (H[1] ?? 0) * -1.1, (H[2] ?? 0) * 0.7 + (H[3] ?? 0) * -1.1]
    expect(round(Array.from(O.twoLoopRecursion(s, y, q, gamma)), 12)).toEqual(round(direct, 12))
  })
  it('projects the start into the box and fixes variables with equal bounds', () => {
    const f = (x: Value): Value => sum(square(sub(x, tensor([3, 3, 3]))))
    const r = O.lbfgsb(f, [10, -10, 0], {
      bounds: [
        [0, 1],
        [-1, 5],
        [2, 2],
      ],
    })
    expect(r.converged).toBe(true)
    expect(round(toFlat(r.x as Value))).toEqual([1, 3, 2])
  })
  it('records the Cauchy point and free variables', () => {
    const rec = recorder()
    O.lbfgsb(rosen, [-1.2, 1], {
      bounds: [
        [-2, 0.5],
        [0, 2],
      ],
      tracer: rec,
    })
    const free = metricSeries(rec.run(), 'free').values
    expect(free.at(-1)).toBe(1)
  })
  it('projected gradient norm vanishes at a bounded minimum', () => {
    const box = O.boxOf({ lower: [0, 0] }, 2)
    expect(O.projectedGradientNorm([0, 2], [5, 0], box)).toBe(0)
    expect(O.projectedGradientNorm([0, 2], [-5, 0], box)).toBe(5)
  })
})

describe('OWL-QN', () => {
  it('leaves unpenalised coordinates free and zeros the rest', () => {
    const a = tensor([2, 0.3, -1, 0.2])
    const f = (x: Value): Value => mul(sum(square(sub(x, a))), 0.5)
    const r = O.owlqn(f, [0, 0, 0, 0], { l1: [0.5, 0.5, 0.5, 0], gradTol: 1e-10 })
    expect(round(toFlat(r.x as Value))).toEqual([1.5, 0, -0.5, 0.2])
    expect(toFlat(r.x as Value)[1]).toBe(0)
  })
  it('pseudo-gradient is zero exactly at the optimum', () => {
    const g = O.pseudoGradient([1.5, 0, -0.5], [-0.5, 0.3, 0.5], 0.5)
    expect(Array.from(g)).toEqual([0, 0, 0])
  })
  it('traces nonzeros along the path', () => {
    const rec = recorder()
    const f = (x: Value): Value => mul(sum(square(sub(x, tensor([2, 0.3, -1])))), 0.5)
    O.owlqn(f, [0, 0, 0], { l1: 0.5, tracer: rec })
    expect(metricSeries(rec.run(), 'nonzeros').values.at(-1)).toBe(2)
  })
  it('rejects bad weights', () => {
    expect(() => O.owlqn(rosen, [0, 0], { l1: -1 })).toThrow()
    expect(() => O.owlqn(rosen, [0, 0], { l1: [1, 2, 3] })).toThrow()
  })
})

describe('SR1 trust region', () => {
  it('handles indefinite curvature and converges on the standard problems', () => {
    for (const name of ['rosen', 'beale', 'himmelblau', 'wood']) {
      const r = O.sr1(
        problem(name),
        name === 'himmelblau' ? [0, 0] : name === 'wood' ? [-3, -1, -3, -1] : name === 'beale' ? [1, 1] : [-1.2, 1],
        {
          gradTol: 1e-8,
          maxIterations: 2000,
        },
      )
      expect(r.converged).toBe(true)
      expect(r.fun).toBeLessThan(1e-10)
      expect(r.nhev).toBe(0)
    }
  })
  it('the BFGS Hessian model also drives the trust region', () => {
    const obj = O.flatObjective(rosen, [-1.2, 1])
    expect(O.bfgsHessian().init(obj, obj.x0).matrix).toHaveLength(4)
    const r = O.trustRegion(rosen, [-1.2, 1], { gradTol: 1e-8 })
    expect(r.converged).toBe(true)
  })
})
