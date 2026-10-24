import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { dominates } from '../optim/index.ts'
import { dict, real } from '../space/index.ts'
import {
  bestTrials,
  createStudy,
  functionObjective,
  lossOf,
  type StudyDirection,
  type Trial,
  tpeSampler,
  tune,
} from './index.ts'

describe('Pareto fronts', () => {
  it('are non-dominated and dominate every other complete trial', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.integer({ min: 0, max: 6 }), { minLength: 2, maxLength: 2 }), {
          minLength: 1,
          maxLength: 30,
        }),
        fc.constantFrom<StudyDirection[]>(['minimize', 'minimize'], ['minimize', 'maximize'], ['maximize', 'maximize']),
        (rows, directions) => {
          const trials: Trial[] = rows.map((values, number) => ({
            number,
            params: 0,
            state: 'complete',
            values,
            intermediate: [],
            source: 'x',
          }))
          const study = createStudy(real(0, 1), { directions, trials })
          const loss = (t: Trial): number[] => directions.map((d, j) => lossOf(d, t.values[j] ?? 0))
          const front = bestTrials(study)
          expect(front.length).toBeGreaterThan(0)
          for (const a of front) for (const b of front) expect(dominates(loss(a), loss(b))).toBe(false)
          const inFront = new Set(front.map((t) => t.number))
          for (const t of trials)
            if (!inFront.has(t.number)) expect(front.some((f) => dominates(loss(f), loss(t)))).toBe(true)
        },
      ),
      { numRuns: 100 },
    )
  })
})

describe('resuming a study', () => {
  it('continues exactly as an uninterrupted run', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), fc.integer({ min: 1, max: 14 }), (seed, cut) => {
        const space = dict({ x: real(-3, 3), y: real(-3, 3) })
        const objective = functionObjective((p) => {
          const { x, y } = p as { x: number; y: number }
          return (x - 1) ** 2 + (y + 0.5) ** 2
        })
        const sampler = tpeSampler({ nStartupTrials: 5 })
        const whole = tune(objective, space, { sampler, nTrials: 15, seed }).study
        const first = tune(objective, space, { sampler, nTrials: cut, seed }).study
        const resumed = tune(objective, space, { sampler, nTrials: 15 - cut, seed, study: first }).study
        expect(resumed.trials).toEqual(whole.trials)
      }),
      { numRuns: 10 },
    )
  })
})
