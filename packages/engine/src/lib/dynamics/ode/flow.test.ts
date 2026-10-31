import { describe, expect, it } from 'vitest'
import { mul, neg, stack, sum, tensor, toFlat } from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import { MIDPOINT, RK4 } from './explicit.ts'
import { flowAlgorithm } from './flow.ts'
import { separableField, separableHamiltonian } from './symplectic.ts'

const sumAt = (y: ReturnType<typeof tensor>, i: number) => sum(mul(y, tensor(i === 0 ? [1, 0] : [0, 1])))

describe('flowAlgorithm', () => {
  // y'' = -y as a first-order field (y0' = y1, y1' = -y0): from (1, 0) the solution is (cos t, -sin t).
  const field = (_t: number, y: ReturnType<typeof tensor>) => stack([sumAt(y, 1), neg(sumAt(y, 0))])

  const error = (dt: number, method = RK4): number => {
    const steps = Math.round(2 / dt)
    const s = runAlgorithm(flowAlgorithm(field, [1, 0], { dt, steps, method }), { maxSteps: steps + 1 }).state
    return Math.hypot((s.y[0] as number) - Math.cos(2), (s.y[1] as number) + Math.sin(2))
  }

  it('integrates at the order of its method: halving the step divides the error by 2^order', () => {
    expect(error(0.1) / error(0.05)).toBeGreaterThan(14)
    expect(error(0.1) / error(0.05)).toBeLessThan(18)
    expect(error(0.1, MIDPOINT) / error(0.05, MIDPOINT)).toBeGreaterThan(3.5)
    expect(error(0.1, MIDPOINT) / error(0.05, MIDPOINT)).toBeLessThan(4.5)
  })

  it('calls the observer after each step with the new time and state, and stops when asked', () => {
    const rec = recorder()
    const alg = flowAlgorithm<'x/t' | 'x/energy'>(field, [1, 0], {
      dt: 0.1,
      steps: 100,
      observe: (t, y, ctx) => {
        ctx.log('x/t', t)
        ctx.log('x/energy', 0.5 * ((y[0] as number) ** 2 + (y[1] as number) ** 2))
      },
      stop: (t) => (t > 0.45 ? 'stopped' : undefined),
    })
    const { steps, status, state } = runAlgorithm(alg, { maxSteps: 200, tracer: rec })
    expect([steps, status]).toEqual([5, 'stopped'])
    expect(state.t).toBeCloseTo(0.5, 12)
    expect(metricSeries(rec.run(), 'x/t').values).toEqual([0.1, 0.2, 0.30000000000000004, 0.4, 0.5])
    expect(Math.abs((metricSeries(rec.run(), 'x/energy').values.at(-1) as number) - 0.5)).toBeLessThan(1e-6)
  })

  it('serialises, so a run resumes from a checkpoint to the same state', () => {
    const alg = flowAlgorithm(field, [1, 0.3], { dt: 0.05, steps: 40 })
    const straight = runAlgorithm(alg, { maxSteps: 100 }).state
    const half = runAlgorithm(alg, { maxSteps: 13 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.y)).toEqual(Array.from(straight.y))
    expect(s.step).toBe(40)
  })

  it('refuses a non-positive step', () => {
    expect(() => flowAlgorithm(field, [1, 0], { dt: 0, steps: 1 })).toThrow()
  })
})

describe('separableField', () => {
  it('is the Hamiltonian vector field on stacked positions and momenta', () => {
    const f = separableField(separableHamiltonian((q) => mul(sum(mul(q, q)), 0.5)))
    // H = (q^2 + p^2) / 2: dq = p, dp = -q.
    expect(
      toFlat(
        f(
          0,
          tensor([
            [1, 2],
            [3, 4],
          ]),
        ),
      ),
    ).toEqual([3, 4, -1, -2])
  })
})
