import { describe, expect, it } from 'vitest'
import { square, sub, toFlat, type Value } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import * as O from './index.ts'

// f(x) = mean_i (x - i)^2 over i = 0..4, minimised at x = 2.
const problem: O.FiniteSum = { n: 5, term: (x, i) => square(sub(x as Value, i)) }

describe('variance-reduced methods', () => {
  it('trace a decreasing objective per epoch and converge', () => {
    for (const method of [O.sag, O.saga, O.svrg]) {
      const rec = recorder()
      const r = method(problem, [10], { lipschitz: 2, tol: 1e-12, tracer: rec })
      expect(r.converged).toBe(true)
      expect(toFlat(r.x as Value)[0]).toBeCloseTo(2, 9)
      expect(r.gradNorm).toBeLessThan(1e-8)
      const loss = metricSeries(rec.run(), 'loss').values
      expect(loss[0]).toBeGreaterThan(loss.at(-1) ?? Infinity)
      expect(loss.at(-1)).toBeCloseTo(2, 9)
    }
  })

  it('are reproducible from the seed and need a step size', () => {
    const a = O.saga(problem, [10], { lipschitz: 2, maxEpochs: 3, seed: 4 })
    const b = O.saga(problem, [10], { lipschitz: 2, maxEpochs: 3, seed: 4 })
    const c = O.saga(problem, [10], { lipschitz: 2, maxEpochs: 3, seed: 5 })
    expect(toFlat(a.x as Value)).toEqual(toFlat(b.x as Value))
    expect(toFlat(a.x as Value)).not.toEqual(toFlat(c.x as Value))
    expect(a.status).toBe('completed')
    expect(() => O.sag(problem, [0])).toThrow(/learningRate or lipschitz/)
  })

  it('apply proximal steps', () => {
    // mean (x - i)^2 + 3 |x|: the minimiser shrinks 2 towards 0 by 3 / 2.
    const prox: O.ProximalStep = (x, step) =>
      Float64Array.from(x, (v) => Math.sign(v) * Math.max(Math.abs(v) - 3 * step, 0))
    for (const method of [O.saga, O.svrg]) {
      const r = method(problem, [1], { lipschitz: 2, prox, tol: 1e-12 })
      expect(toFlat(r.x as Value)[0]).toBeCloseTo(0.5, 8)
    }
  })
})
