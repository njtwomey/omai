import { describe, expect, it } from 'vitest'
import { add, concat, cos, exp, sliceAxis, square, sub, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import * as R from './index.ts'

const system = (v: Value): Value => {
  const x = sliceAxis(v, 0, 0, 1)
  const y = sliceAxis(v, 0, 1, 2)
  return concat([sub(add(square(x), square(y)), 4), sub(add(exp(x), y), 1)])
}

const brief = (r: R.SystemResult): { x: number[]; it: number; calls: number; c: boolean; n: number } => ({
  x: Array.from(toFlat(r.x)),
  it: r.iterations,
  calls: r.functionCalls,
  c: r.converged,
  n: r.history.length,
})

describe('the system solvers give the numbers they gave before, exactly', () => {
  // Taken from the previous implementation.
  it('fsolve and broyden', () => {
    expect(brief(R.fsolve(system, [1, -1]))).toEqual({
      x: [1.0041687384746592, -1.72963728702587],
      it: 5,
      calls: 6,
      c: true,
      n: 6,
    })
    expect(brief(R.broyden(system, [1, -1]))).toEqual({
      x: [1.0041687384739901, -1.7296372870235],
      it: 7,
      calls: 8,
      c: true,
      n: 8,
    })
    expect(brief(R.fsolve(system, [3, 3], { maxIterations: 2 }))).toEqual({
      x: [0.7901139123368031, 2.2483798835735707],
      it: 2,
      calls: 3,
      c: false,
      n: 3,
    })
    expect(R.broyden(system, [3, 3], { maxIterations: 3 }).history[3]).toBe(3.3550550757175084)
    expect(R.fsolve(system, [1, -1]).history.at(-1)).toBe(0)
    expect(R.broyden(system, [1, -1]).history.at(-1)).toBe(9.557639587076201e-12)
  })

  it('fixedPoint in its three methods, and the error when it does not converge', () => {
    expect(brief(R.fixedPoint((x) => cos(x), [1]))).toEqual({
      x: [0.7390851332151607],
      it: 4,
      calls: 12,
      c: true,
      n: 4,
    })
    expect(brief(R.fixedPoint((x) => cos(x), [1], { method: 'iteration', xtol: 1e-10 }))).toEqual({
      x: [0.7390851331949863],
      it: 59,
      calls: 118,
      c: true,
      n: 59,
    })
    expect(brief(R.fixedPoint((x) => cos(x), [1], { method: 'anderson', xtol: 1e-12 }))).toEqual({
      x: [0.7390851332151607],
      it: 9,
      calls: 18,
      c: true,
      n: 9,
    })
    expect(brief(R.root(system, [1, -1], { method: 'anderson' })).it).toBe(14)
    expect(() => R.fixedPoint((x) => cos(x), [1], { method: 'iteration', maxIterations: 3 })).toThrow(
      'fixedPoint: no convergence after 3 iterations',
    )
  })
})

const algorithms = {
  fsolve: () => R.fsolveAlgorithm(system, [1, -1]),
  broyden: () => R.broydenAlgorithm(system, [1, -1]),
} as const

describe('the system algorithms: state, paths, invariants', () => {
  for (const [name, make] of Object.entries(algorithms)) {
    it(`${name}: a state that is JSON resumes to the same end`, () => {
      const alg = make()
      const whole = runAlgorithm(alg, { maxSteps: 50 })
      const half = runAlgorithm(alg, { maxSteps: 2 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 48, start: { state: stored, step: 2, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(whole.status).toBe('converged')
    })

    it(`${name}: logs the iterate, residual norm, step length and Jacobian age`, () => {
      const rec = recorder()
      runAlgorithm(make(), {
        maxSteps: 4,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
      })
      const metric = (p: string): number[] =>
        rec.events.flatMap((e) => (e.type === 'metric' && e.name === p ? [e.value] : []))
      expect(metric('system/residual')).toHaveLength(4)
      expect(metric('system/step-length').every((t) => t > 0 && t <= 1)).toBe(true)
      expect(metric('system/jacobian-age')).toEqual(name === 'broyden' ? [1, 2, 3, 4] : [0, 0, 0, 0])
      expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'system/x')).toHaveLength(4)
    })
  }

  it('the line search makes the residual norm non-increasing', () => {
    for (const make of Object.values(algorithms)) {
      const h = runAlgorithm(make(), { maxSteps: 50 }).state.history
      for (let i = 1; i < h.length; i++) expect(h[i] as number).toBeLessThanOrEqual((h[i - 1] as number) * (1 + 1e-12))
    }
  })

  it('fixed point: a state that is JSON resumes (Anderson keeps its memory), in every method', () => {
    for (const method of ['del2', 'iteration', 'anderson'] as const) {
      const alg = R.fixedPointAlgorithm((x) => cos(x), [1], { method, xtol: 1e-10 })
      const whole = runAlgorithm(alg, { maxSteps: 200 })
      const half = runAlgorithm(alg, { maxSteps: 3 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 197, start: { state: stored, step: 3, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(whole.status).toBe('converged')
      expect(Math.abs((whole.state.x[0] as number) - 0.7390851332151607)).toBeLessThan(1e-8)
    }
    expect(Object.keys(R.fixedPointContract)).toContain('fixed-point/residual')
  })
})
