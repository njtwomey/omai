import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { problem } from '../../../testing/problems.ts'
import { add, concat, mul, square, sub, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as O from '../index.ts'

describe('first-order methods against scipy.optimize.minimize', () => {
  for (const c of loadFixture('optim/minimize/gradient').cases) {
    it(c.name, () => {
      const f = problem(c.params['problem'] as string)
      const x0 = decode(c.inputs[0]) as Tensor
      const r =
        c.fn === 'conjugateGradient'
          ? O.conjugateGradient(f, x0, { gradTol: 1e-8 })
          : c.fn === 'gradientDescent'
            ? O.gradientDescent(f, x0, { gradTol: 1e-9, step: 'bb1', maxIterations: 20000 })
            : O.coordinateDescent(f, x0, { gradTol: 1e-9, ftol: 0, maxIterations: 5000 })
      expect(r.converged).toBe(true)
      expectMatches(concat([r.x as Value, tensor([r.fun])]), c.output, c.rtol, c.atol, c.name)
      const nit = c.params['nit'] as number | undefined
      // Wood's function amplifies rounding differences, so iteration counts may drift on long runs.
      if (nit !== undefined) expect(Math.abs(r.steps - nit)).toBeLessThanOrEqual(Math.max(2, 0.15 * nit))
    })
  }
})

describe('first-order methods', () => {
  const quadratic = problem('quadratic')
  it('every beta rule converges on a quadratic in about n steps', () => {
    for (const beta of ['polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel', 'dai-yuan', 'hager-zhang'] as const) {
      const r = O.conjugateGradient(quadratic, [1, 1, 1], { beta, gradTol: 1e-10 })
      expect(r.converged).toBe(true)
      expect(r.fun).toBeCloseTo(-1.0506329113924051, 10)
    }
    const restarted = O.conjugateGradient(problem('rosen'), [-1.2, 1], { restart: 2, gradTol: 1e-6 })
    expect(restarted.converged).toBe(true)
  })
  it('gradient descent: fixed steps, Barzilai-Borwein and line searches agree', () => {
    for (const step of ['fixed', 'bb1', 'bb2', 'line-search'] as const) {
      const r = O.gradientDescent(quadratic, [1, 1, 1], { step, learningRate: 0.2, gradTol: 1e-9 })
      expect(r.converged).toBe(true)
      expect(r.fun).toBeCloseTo(-1.0506329113924051, 10)
    }
  })
  it('gradient descent with a fixed step diverges past 2 / L', () => {
    const r = O.gradientDescent((x: Value) => sum(square(x)), [1], {
      step: 'fixed',
      learningRate: 1.5,
      maxIterations: 2000,
    })
    expect(r.status).toBe('diverged')
  })
  it('coordinate descent orders and updates, reproducible random sweeps', () => {
    const sep = (x: Value): Value => sum(add(mul(square(sub(x, tensor([1, 2, 3]))), tensor([1, 2, 3])), 0))
    for (const order of ['cyclic', 'random', 'greedy'] as const) {
      for (const update of ['exact', 'newton', 'gradient'] as const) {
        const r = O.coordinateDescent(sep, [0, 0, 0], { order, update, learningRate: 0.1, gradTol: 1e-8, seed: 3 })
        expect(r.converged).toBe(true)
        expect(toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4)).toEqual([1, 2, 3])
      }
    }
    const a = O.coordinateDescent(quadratic, [1, 1, 1], { order: 'random', seed: 5, maxIterations: 3 })
    const b = O.coordinateDescent(quadratic, [1, 1, 1], { order: 'random', seed: 5, maxIterations: 3 })
    expect(toFlat(a.x as Value)).toEqual(toFlat(b.x as Value))
  })
  it('traces the conjugacy coefficient and line searches', () => {
    const rec = recorder()
    O.conjugateGradient(problem('rosen'), [-1.2, 1], { tracer: rec })
    expect(metricSeries(rec.run(), 'beta').values.length).toBeGreaterThan(5)
  })
})
