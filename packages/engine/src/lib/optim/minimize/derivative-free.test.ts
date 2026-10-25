import { describe, expect, it } from 'vitest'
import { beale, himmelblau, quadratic, rosen } from '../../../testing/optim-problems.ts'
import { isMdmlError } from '../../base/index.ts'
import { add, concat, reshape, sliceAxis, square, sub, sum, toFlat, type Value } from '../../tensor/index.ts'
import { messages, metricSeries, policies, recorder, tensorSeries } from '../../trace/index.ts'
import { cobyla } from './cobyla.ts'
import { nelderMead, nelderMeadAlgorithm } from './derivative-free.ts'
import { minimize } from './minimizer.ts'
import { compassSearch, hookeJeeves } from './pattern-search.ts'
import { powell } from './powell.ts'

const flat = (x: unknown): number[] => toFlat(x as Value)
const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a).toHaveLength(b.length)
  for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] ?? 0))).toBeLessThan(tol)
}
const v = (x: unknown): Value => x as Value

describe('Nelder-Mead', () => {
  it('works on parameter trees and through minimize', () => {
    const f = (p: unknown): Value => {
      const { a, b } = p as { a: Value; b: Value }
      return add(sum(square(sub(a, [1, 2]))), square(sub(b, 3)))
    }
    const r = minimize(f, { a: [0, 0], b: 0 }, { method: nelderMead, xatol: 1e-9, fatol: 1e-14 })
    expect(r.converged).toBe(true)
    const x = r.x as { a: Value; b: Value }
    close([...flat(x.a), ...flat(x.b)], [1, 2, 3], 1e-7)
    expect(r.grad).toBeUndefined()
    expect(Number.isNaN(r.gradNorm)).toBe(true)
  })

  it('accepts an initial simplex and reports its moves and simplex in the trace', () => {
    const rec = recorder()
    const r = nelderMead((x) => rosen(v(x)), [-1.2, 1], {
      initialSimplex: [
        [-1.2, 1],
        [-1, 1],
        [-1.2, 1.2],
      ],
      tracer: rec,
      policy: { ...policies.debug, detail: 'full' },
    })
    expect(r.converged).toBe(true)
    close(flat(r.x), [1, 1], 1e-3)
    const run = rec.run()
    const moves = new Set(messages(run, 'move').map((m) => (m.payload as { operation: string }).operation))
    expect(moves.has('reflect')).toBe(true)
    expect(moves.has('contract inside') || moves.has('contract outside')).toBe(true)
    expect(tensorSeries(run, 'simplex')[0]?.shape).toEqual([3, 2])
    const size = metricSeries(run, 'simplexSize').values
    expect((size.at(-1) ?? 1) < (size[0] ?? 0)).toBe(true)
  })

  it('stops at the evaluation budget and keeps points inside bounds', () => {
    const r = nelderMead((x) => rosen(v(x)), [-1.2, 1], { maxEvaluations: 30 })
    expect(r.converged).toBe(false)
    expect(r.status).toBe('failed')
    expect(r.nfev).toBe(30)
    const b = nelderMead((x) => himmelblau(v(x)), [0, 0], {
      bounds: [
        [-1, 2.5],
        [-1, 1.5],
      ],
    })
    const xb = flat(b.x)
    expect(xb[0]).toBeLessThanOrEqual(2.5)
    expect(xb[1]).toBeLessThanOrEqual(1.5)
  })

  it('rejects a malformed simplex and bounds', () => {
    const f = (x: unknown): Value => sum(square(v(x)))
    expect(() => nelderMeadAlgorithm(f, [1, 2], { initialSimplex: [[0, 0]] })).toThrow()
    const bad = (): unknown =>
      nelderMead(f, [1, 2], {
        bounds: [
          [1, 0],
          [0, 1],
        ],
      })
    expect(bad).toThrow()
    try {
      bad()
    } catch (e) {
      expect(isMdmlError(e, 'value')).toBe(true)
    }
  })
})

describe('Powell', () => {
  it('minimises a quadratic exactly along conjugate directions', () => {
    const r = powell((x) => quadratic(v(x)), [0, 0, 0], { xtol: 1e-10, ftol: 1e-14 })
    expect(r.converged).toBe(true)
    // A x = b with A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3]
    close(flat(r.x), [2 / 9, 1 / 9, 13 / 9], 1e-7)
  })

  it('takes initial directions and records the direction set', () => {
    const rec = recorder()
    const r = powell((x) => beale(v(x)), [1, 1], {
      directions: [
        [1, 1],
        [1, -1],
      ],
      xtol: 1e-8,
      ftol: 1e-12,
      tracer: rec,
      policy: { ...policies.debug, detail: 'full' },
    })
    close(flat(r.x), [3, 0.5], 1e-5)
    expect(tensorSeries(rec.run(), 'directions')[0]?.values).toEqual([1, 1, 1, -1])
  })

  it('stays inside bounds and honours the evaluation budget', () => {
    const r = powell((x) => sum(square(sub(v(x), [3, -3]))), [0, 0], {
      bounds: [
        [-1, 1],
        [-2, 2],
      ],
    })
    close(flat(r.x), [1, -2], 1e-4)
    const capped = powell((x) => rosen(v(x)), [-1.2, 1], { maxEvaluations: 25 })
    expect(capped.status).toBe('failed')
    expect(capped.nfev).toBeLessThanOrEqual(25)
  })
})

describe('pattern searches', () => {
  it('Hooke-Jeeves and compass search find the minimum of Rosenbrock and Himmelblau', () => {
    const hj = hookeJeeves((x) => rosen(v(x)), [-1.2, 1], { stepTol: 1e-8, maxEvaluations: 20000 })
    expect(hj.converged).toBe(true)
    close(flat(hj.x), [1, 1], 1e-3)
    const cs = compassSearch((x) => himmelblau(v(x)), [0, 0], { stepTol: 1e-9 })
    expect(cs.converged).toBe(true)
    close(flat(cs.x), [3, 2], 1e-6)
    const minimal = compassSearch((x) => quadratic(v(x)), [0, 0, 0], {
      poll: 'minimal',
      stepTol: 1e-9,
      maxEvaluations: 20000,
    })
    close(flat(minimal.x), [2 / 9, 1 / 9, 13 / 9], 1e-6)
    const greedy = compassSearch((x) => quadratic(v(x)), [0, 0, 0], { opportunistic: false, stepTol: 1e-9 })
    close(flat(greedy.x), [2 / 9, 1 / 9, 13 / 9], 1e-6)
  })

  it('respect bounds and report mesh and moves', () => {
    const rec = recorder()
    const r = compassSearch((x) => sum(square(sub(v(x), [3, 3]))), [0, 0], {
      bounds: [
        [-1, 1],
        [-1, 2],
      ],
      tracer: rec,
    })
    close(flat(r.x), [1, 2], 1e-5)
    const mesh = metricSeries(rec.run(), 'mesh').values
    expect((mesh.at(-1) ?? 1) < 1e-5).toBe(true)
    const hj = hookeJeeves((x) => sum(square(sub(v(x), [3, 3]))), [0, 0], {
      bounds: [
        [-1, 1],
        [-1, 2],
      ],
    })
    close(flat(hj.x), [1, 2], 1e-5)
    expect(() =>
      hookeJeeves((x) => sum(v(x)), [5, 0], {
        bounds: [
          [-1, 1],
          [-1, 1],
        ],
      }),
    ).toThrow()
  })
})

describe('COBYLA', () => {
  it('solves a linearly constrained quadratic with bounds', () => {
    // minimise (x - 2)^2 + (y - 1)^2 subject to x + y <= 2, 0 <= x <= 1.5: the answer is (1.5, 0.5)
    const r = cobyla((x) => sum(square(sub(v(x), [2, 1]))), [0, 0], {
      constraints: [(x) => sub(2, sum(v(x)))],
      bounds: [
        [0, 1.5],
        [-Infinity, Infinity],
      ],
      rhoend: 1e-9,
    })
    expect(r.converged).toBe(true)
    close(flat(r.x), [1.5, 0.5], 1e-7)
  })

  it('accepts vector-valued constraints and traces radius, penalty and violation', () => {
    const rec = recorder()
    const r = cobyla((x) => sum(v(x)), [1, 1], {
      // x^2 + y^2 <= 1 as a vector constraint together with y >= -0.5
      constraints: [(x) => concat([reshape(sub(1, sum(square(v(x)))), [1]), add(sliceAxis(v(x), 0, 1, 2), 0.5)])],
      rhobeg: 0.5,
      rhoend: 1e-8,
      tracer: rec,
      policy: policies.debug,
    })
    expect(r.converged).toBe(true)
    // the minimum of x + y on the unit disc cut by y >= -0.5: x = -sqrt(3)/2, y = -0.5
    close(flat(r.x), [-Math.sqrt(3) / 2, -0.5], 1e-6)
    const run = rec.run()
    const rho = metricSeries(run, 'rho').values
    expect(rho.at(-1)).toBe(1e-8)
    expect(metricSeries(run, 'maxcv').values.at(-1)).toBeLessThan(1e-8)
    const actions = new Set(messages(run, 'action').map((m) => (m.payload as { action: string }).action))
    expect(actions.has('trust step')).toBe(true)
    expect(actions.has('reduce radius')).toBe(true)
  })

  it('reports infeasibility and the evaluation budget', () => {
    const r = cobyla((x) => sum(square(v(x))), [0], { constraints: [(x) => sub(v(x), 1), (x) => sub(-1, v(x))] })
    expect(r.converged).toBe(false)
    const capped = cobyla((x) => rosen(v(x)), [0, 0], { maxEvaluations: 20 })
    expect(capped.status).toBe('failed')
    expect(capped.nfev).toBeLessThanOrEqual(21)
  })
})
