import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import { dppAlgorithm, dppContract, sampleDpp, sampleKDpp } from './dpp.ts'

const L = [
  [2, 0.5, 0, 0.1],
  [0.5, 1.5, 0.2, 0],
  [0, 0.2, 1, 0.3],
  [0.1, 0, 0.3, 2.5],
]

describe('sampleDpp and sampleKDpp are runs of dppAlgorithm', () => {
  // Taken from the previous implementation.
  it('give the samples they gave before, exactly', () => {
    expect([0, 1, 2, 3, 4, 5].map((s) => sampleDpp(key(s), L))).toEqual([
      [0, 3],
      [0, 1, 3],
      [0, 2, 3],
      [0, 2, 3],
      [0, 1, 2],
      [2, 3],
    ])
    expect([0, 1, 2, 3, 4, 5].map((s) => sampleKDpp(key(s), L, 2))).toEqual([
      [0, 3],
      [1, 3],
      [1, 3],
      [1, 3],
      [0, 1],
      [2, 3],
    ])
    expect([0, 1, 2].map((s) => sampleKDpp(key(s), L, 3))).toEqual([
      [0, 1, 3],
      [0, 1, 3],
      [0, 2, 3],
    ])
  })
})

describe('dppAlgorithm', () => {
  it('has a state that is JSON, and a run resumes to the same sample', () => {
    const alg = dppAlgorithm(key(3), L, 3)
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    expect(half.chosen).toHaveLength(1)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 9, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.status).toBe('converged')
  })

  it('draws one item per kept eigenvector, shrinking the basis by one each step', () => {
    const alg = dppAlgorithm(key(3), L, 3)
    let s = alg.init(undefined as never)
    expect(s.kept).toBe(3)
    const ctx = { t: 0, wants: () => false, log: () => undefined, emit: () => undefined } as never
    for (let i = 1; i <= 3; i++) {
      s = alg.step(s, ctx)
      expect(s.chosen).toHaveLength(i)
      expect(s.basis).toHaveLength(3 - i)
    }
    expect(new Set(s.chosen).size).toBe(3)
  })

  it('logs the kept count, the item, its probability and the basis each step', () => {
    const rec = recorder()
    runAlgorithm(dppAlgorithm(key(3), L, 2), {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['state'], detail: 'full' },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('dpp/item')).toHaveLength(2)
    expect(metric('dpp/kept')).toEqual([2, 2])
    expect(metric('dpp/probability').every((p) => p > 0 && p <= 1)).toBe(true)
    expect(Object.keys(dppContract)).toEqual(['dpp/kept', 'dpp/item', 'dpp/probability', 'dpp/basis'])
  })

  it('a k-DPP sample has exactly k items; bad k and a rank-deficient kernel throw', () => {
    for (let s = 0; s < 8; s++) expect(sampleKDpp(key(s), L, 2)).toHaveLength(2)
    expect(() => dppAlgorithm(key(0), L, 0)).toThrow()
    expect(() =>
      runAlgorithm(
        dppAlgorithm(
          key(0),
          [
            [1, 1],
            [1, 1],
          ],
          2,
        ),
        { maxSteps: 5 },
      ),
    ).toThrow()
  })
})
