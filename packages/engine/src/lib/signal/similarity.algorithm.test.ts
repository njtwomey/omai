import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { dtw, dtwAlgorithm, dtwAlignment, dtwContract } from './similarity.ts'
import { dtwStepPatterns } from './similarity-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const x = [0, 1, 3, 2, 5, 4, 4, 1]
const y = [0, 0, 2, 3, 3, 5, 3, 1, 1]

describe('dtwAlgorithm', () => {
  it('fills one row per step and traces back the same alignment as dtw', () => {
    for (const stepPattern of ['symmetric2', 'symmetric1', 'asymmetric'] as const) {
      const run = runAlgorithm(dtwAlgorithm(x, y, { stepPattern }), { maxSteps: 50 })
      expect([run.status, run.steps, run.state.row]).toEqual(['converged', x.length, x.length])
      const a = dtwAlignment(run.state)
      expect(dtw(x, y, { stepPattern })).toEqual(a)
      // The path starts and ends at the corners and never moves backwards.
      expect([a.index1[0], a.index2[0]]).toEqual([0, 0])
      expect([a.index1.at(-1), a.index2.at(-1)]).toEqual([x.length - 1, y.length - 1])
      a.index1.forEach((v, k) => {
        if (k > 0) expect(v).toBeGreaterThanOrEqual(a.index1[k - 1] ?? 0)
      })
    }
    expect(dtw([0, 1, 2], [0, 1, 1, 2]).distance).toBe(0)
  })

  it('keeps the accumulated cost of each row at least that of the cheapest cell above it', () => {
    const rec = recorder()
    runAlgorithm(dtwAlgorithm(x, y), {
      maxSteps: 50,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output', 'state'], detail: 'full' },
    })
    const cost = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'dtw/cost' ? [e.value] : []))
    expect(cost).toHaveLength(x.length)
    cost.forEach((c, i) => {
      if (i > 0) expect(c).toBeGreaterThanOrEqual((cost[i - 1] ?? 0) - 1e-12)
    })
    expect(Object.keys(dtwContract)).toHaveLength(3)
  })

  it('honours the window, fails when no alignment fits, and rejects mismatched dimensions', () => {
    const banded = dtw(x, y, { window: 2 })
    expect(banded.distance).toBeGreaterThanOrEqual(dtw(x, y).distance)
    const run = runAlgorithm(dtwAlgorithm([0, 1, 2, 3, 4], [0, 1], { window: 1 }), { maxSteps: 20 })
    expect(run.status).toBe('failed')
    expect(() => dtw([0, 1, 2, 3, 4], [0, 1], { window: 1 })).toThrow(/no alignment/)
    expect(() => dtwAlgorithm([[0, 1]], [0, 1])).toThrow(/dimension/)
  })

  it('serialises, with the infinite costs outside the band, and resumes', () => {
    const alg = dtwAlgorithm(x, y, { window: 3 })
    const whole = runAlgorithm(alg, { maxSteps: 50 })
    const part = runAlgorithm(alg, { maxSteps: 4 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 50,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 4, serialised: true },
    })
    expect(Array.from(rest.state.g)).toEqual(Array.from(whole.state.g))
    expect(Array.from(rest.state.dir)).toEqual(Array.from(whole.state.dir))
  })
})

describe('dtwStepPatterns', () => {
  it('names the patterns and builds options', () => {
    expect(dtwStepPatterns.names).toEqual(['symmetric2', 'symmetric1', 'asymmetric'])
    expect(dtwStepPatterns.make({ kind: 'asymmetric', window: 3 })).toEqual({ stepPattern: 'asymmetric', window: 3 })
    expect(dtwStepPatterns.properties({ kind: 'symmetric1' }).normalisedBy).toBe('none')
  })
})
