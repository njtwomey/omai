import { describe, expect, it } from 'vitest'
import { abs, exp, sin, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { approximationContract } from './approximation-contract.ts'
import { chebyshevResolutions, minimaxTypes } from './approximation-options.ts'
import { remez, remezAlgorithm } from './best.ts'
import { chebfun, chebfunAlgorithm } from './chebyshev.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('remezAlgorithm', () => {
  it('exchanges the reference until the error equioscillates, as remez does', () => {
    const f = (x: number): number => Math.exp(x)
    const run = runAlgorithm(remezAlgorithm(f, 3), { maxSteps: 60 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.iterations)
    const result = remez(f, 3)
    expect(result.error).toBe(run.state.error)
    expect(result.iterations).toBe(run.state.iterations)
    expect(result.converged).toBe(true)
    // The equioscillation theorem: the reference has m + n + 2 points where the error is +-E.
    expect(run.state.reference).toHaveLength(5)
    for (const x of toFlat(result.reference))
      expect(Math.abs(f(x) - result.evaluate(x))).toBeCloseTo(Math.abs(result.error), 8)
  })

  it('finds a rational approximation of type (2, 1)', () => {
    const run = runAlgorithm(remezAlgorithm(Math.exp, 2, { n: 1 }), { maxSteps: 60 })
    expect(run.state.reference).toHaveLength(5)
    expect(run.state.denominator).toHaveLength(2)
    expect(Math.abs(run.state.error)).toBeLessThan(1e-2)
  })

  it('serialises, resumes, and logs the error, the largest error and the alternations', () => {
    const alg = remezAlgorithm((x) => Math.exp(x), 3)
    const whole = runAlgorithm(alg, { maxSteps: 60 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 60,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state.iterations).toBe(whole.state.iterations)
    expect(rest.state.error).toBeCloseTo(whole.state.error, 12)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 60, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('remez/error')).toHaveLength(whole.steps)
    expect(metric('remez/alternation').every((a) => a >= 5)).toBe(true)
    // The largest error approaches |E| from above as the reference is exchanged.
    const gap = metric('remez/maxError').map((m, i) => m - Math.abs(metric('remez/error')[i] ?? 0))
    expect(gap.at(-1) ?? 1).toBeLessThan(1e-8)
  })

  it('stops at the iteration cap without convergence', () => {
    const run = runAlgorithm(remezAlgorithm(Math.exp, 3, { maxIterations: 1 }), { maxSteps: 60 })
    expect([run.status, remez(Math.exp, 3, { maxIterations: 1 }).converged]).toEqual(['completed', false])
  })
})

describe('chebfunAlgorithm', () => {
  it('doubles the sample until the coefficients decay and agrees with chebfun', () => {
    const run = runAlgorithm(
      chebfunAlgorithm((x) => sin(x), [0, 20]),
      { maxSteps: 20 },
    )
    expect(run.status).toBe('converged')
    expect(run.steps).toBeGreaterThan(1)
    expect(run.state.points).toBe(2 ** (run.steps + 3) + 1)
    const f = chebfun((x) => sin(x), [0, 20])
    const kept = toFlat(f.coefficients)
    expect(kept.length).toBeLessThanOrEqual(run.state.coefficients.length)
    expect(kept).toEqual(run.state.coefficients.slice(0, kept.length))
  })

  it('fails when the sample would pass the cap, and rejects non-finite values', () => {
    expect(
      runAlgorithm(
        chebfunAlgorithm((x) => exp(x), [-1, 1], { maxPoints: 5 }),
        { maxSteps: 5 },
      ).status,
    ).toBe('failed')
    const kink = (x: Value): Value => abs(x)
    expect(runAlgorithm(chebfunAlgorithm(kink, [-1, 1], { maxPoints: 33 }), { maxSteps: 5 }).status).toBe('failed')
    expect(() => chebfun(kink, [-1, 1], { maxPoints: 33 })).toThrow(/no convergence/)
    const blowUp = (x: Value): Value => sin(exp(exp(exp(x))))
    expect(() => runAlgorithm(chebfunAlgorithm(blowUp, [-3, 3]), { maxSteps: 5 })).toThrow(/not finite/)
  })

  it('serialises and resumes', () => {
    const alg = chebfunAlgorithm((x) => sin(x), [0, 20])
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state.points).toBe(whole.state.points)
    expect(rest.state.coefficients).toEqual(whole.state.coefficients)
    expect(Object.keys(approximationContract)).toHaveLength(7)
  })
})

describe('approximation option tables', () => {
  it('build the options of each algorithm', () => {
    expect(minimaxTypes.names).toEqual(['polynomial', 'rational'])
    expect(minimaxTypes.make({ kind: 'polynomial' }).n).toBe(0)
    expect(chebyshevResolutions.make({ kind: 'strict' }).tol).toBeLessThan(
      chebyshevResolutions.make({ kind: 'standard' }).tol ?? 0,
    )
  })
})
