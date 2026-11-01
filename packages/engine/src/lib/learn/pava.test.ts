import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { isotonicCalibration, poolAdjacentViolators } from './calibration.ts'
import { pavaAlgorithm, pavaContract, pavaFitted, pavaMeans } from './pava.ts'
import { isotonicDirections } from './pava-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('pavaAlgorithm', () => {
  it('pools one violating pair per step and reaches the least-squares fit', () => {
    const run = runAlgorithm(pavaAlgorithm([1, 3, 2, 4]), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(5)
    expect(run.state.merges).toBe(1)
    expect(pavaMeans(run.state)).toEqual([1, 2.5, 4])
    expect(pavaFitted(run.state, 4)).toEqual([1, 2.5, 2.5, 4])
    expect(poolAdjacentViolators([1, 3, 2, 4])).toEqual([1, 2.5, 2.5, 4])
  })

  it('honours weights and the decreasing direction', () => {
    const weighted = poolAdjacentViolators([3, 1, 2, 5], [1, 1, 3, 2])
    expect(weighted[0]).toBeCloseTo(2, 12)
    expect(weighted[2]).toBeCloseTo(2, 12)
    expect(weighted[3]).toBeCloseTo(5, 12)
    const down = runAlgorithm(pavaAlgorithm([4, 2, 3, 1], { direction: { kind: 'decreasing' } }), { maxSteps: 20 })
    const fit = pavaFitted(down.state, 4, isotonicDirections.make({ kind: 'decreasing' }))
    expect(fit).toEqual([4, 2.5, 2.5, 1])
  })

  it('rejects mismatched or negative weights', () => {
    expect(() => pavaAlgorithm([1, 2], { weights: [1] })).toThrow(/weights/)
    expect(() => pavaAlgorithm([1, 2], { weights: [1, -1] })).toThrow(/>= 0/)
  })

  it('resumes from a JSON checkpoint to the same end', () => {
    const y = [5, 1, 4, 2, 8, 3, 9, 7, 7, 10]
    const alg = pavaAlgorithm(y)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const half = runAlgorithm(alg, { maxSteps: 6 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 100, start: { state: stored, step: 6, serialised: true } })
    expect(rest.state.starts).toEqual(whole.state.starts)
    expect(rest.state.merges).toBe(whole.state.merges)
    expect(rest.steps).toBe(whole.steps)
    const expected = pavaMeans(whole.state)
    for (const [j, m] of pavaMeans(rest.state).entries()) expect(m).toBeCloseTo(expected[j] ?? 0, 12)
  })

  it('logs the stack, the pooled means, the merge flag, the violations and the error', () => {
    const rec = recorder()
    runAlgorithm(pavaAlgorithm([1, 3, 2, 4]), {
      maxSteps: 20,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('pava/blocks')).toEqual([1, 2, 3, 2, 3])
    expect(metric('pava/merge')).toEqual([0, 0, 0, 1, 0])
    expect(metric('pava/violations')).toEqual([0, 0, 1, 0, 0])
    expect(metric('pava/error').at(-1)).toBeCloseTo(0.5, 12)
    expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'pava/means')).toHaveLength(5)
    expect(Object.keys(pavaContract)).toHaveLength(5)
  })

  it('is monotone, preserves the weighted mean, and is idempotent on random data', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: -50, max: 50 }), fc.integer({ min: 1, max: 5 })), {
          minLength: 1,
          maxLength: 30,
        }),
        (rows) => {
          const y = rows.map((r) => r[0])
          const w = rows.map((r) => r[1])
          const fit = poolAdjacentViolators(y, w)
          const monotone = fit.every((v, i) => i === 0 || v >= (fit[i - 1] ?? 0) - 1e-12)
          const total = (a: readonly number[]): number => a.reduce((s, v, i) => s + v * (w[i] ?? 0), 0)
          const mean = Math.abs(total(fit) - total(y)) < 1e-8
          const again = poolAdjacentViolators(fit, w)
          return monotone && mean && again.every((v, i) => Math.abs(v - (fit[i] ?? 0)) < 1e-9)
        },
      ),
    )
  })

  it('leaves isotonic calibration unchanged', () => {
    const c = isotonicCalibration([0.1, 0.4, 0.35, 0.8], [0, 1, 0, 1])
    expect(c.params).toEqual({ x: [0.1, 0.35, 0.4, 0.8], y: [0, 0, 1, 1] })
  })
})

describe('isotonicDirections', () => {
  it('names the two directions and their signs', () => {
    expect(isotonicDirections.names).toEqual(['increasing', 'decreasing'])
    expect(isotonicDirections.make(undefined)).toBe(1)
    expect(isotonicDirections.properties({ kind: 'decreasing' }).sign).toBe(-1)
  })
})
