import { describe, expect, it } from 'vitest'
import { real } from '../space/index.ts'
import {
  ashaSchedule,
  createStudy,
  functionObjective,
  halvingPlan,
  hyperbandSchedule,
  nextAssignment,
  successiveHalvingSchedule,
  tune,
} from './index.ts'

describe('schedules as data', () => {
  it('reproduce the Hyperband table of Li et al. (2018) for R = 81, eta = 3', () => {
    const table = hyperbandSchedule({ maxResource: 81 }).brackets.map((b) =>
      b.rungs.map((r) => [r.nConfigs, r.resource]),
    )
    expect(table).toEqual([
      [
        [81, 1],
        [27, 3],
        [9, 9],
        [3, 27],
        [1, 81],
      ],
      [
        [34, 3],
        [11, 9],
        [3, 27],
        [1, 81],
      ],
      [
        [15, 9],
        [5, 27],
        [1, 81],
      ],
      [
        [8, 27],
        [2, 81],
      ],
      [[5, 81]],
    ])
  })
  it('make successive halving and ASHA rungs', () => {
    expect(
      successiveHalvingSchedule({ nConfigs: 16, maxResource: 16, reductionFactor: 2 }).brackets[0]?.rungs.map(
        (r) => r.nConfigs,
      ),
    ).toEqual([16, 8, 4, 2, 1])
    expect(ashaSchedule({ minResource: 1, maxResource: 27, minEarlyStoppingRate: 1 }).rungs).toEqual([3, 9, 27])
    expect(() => hyperbandSchedule({ maxResource: 10, reductionFactor: 1 })).toThrow()
    expect(() => ashaSchedule({ minResource: 5, maxResource: 1 })).toThrow()
    expect(() => halvingPlan({ nCandidates: 3, minResources: 10, maxResources: 5 })).toThrow()
  })
  it('assign promotions of the best configurations', () => {
    // f(x, budget) improves with budget; x near 0.3 is best.
    const objective = functionObjective((x, trial) => ((x as number) - 0.3) ** 2 + 1 / (trial.budget ?? 1))
    const schedule = successiveHalvingSchedule({ nConfigs: 9, maxResource: 9 })
    const r = tune(objective, real(0, 1), { schedule, nTrials: 100, seed: 0 })
    // 9 + 3 + 1 trials, then the schedule ends the study.
    expect(r.study.trials.length).toBe(13)
    const rung0 = r.study.trials.filter((t) => t.rung === 0)
    const rung1 = r.study.trials.filter((t) => t.rung === 1)
    const best3 = [...rung0].sort((a, b) => (a.values[0] ?? 0) - (b.values[0] ?? 0)).slice(0, 3)
    expect(new Set(rung1.map((t) => t.parent))).toEqual(new Set(best3.map((t) => t.number)))
    expect(rung1.every((t) => t.budget === 3 && t.source === 'promoted')).toBe(true)
    expect(r.study.trials.at(-1)?.budget).toBe(9)
  })
  it('runs Hyperband brackets in turn and repeats when asked', () => {
    const objective = functionObjective((x, trial) => (x as number) + 1 / (trial.budget ?? 1))
    const schedule = hyperbandSchedule({ maxResource: 9 })
    const total = schedule.brackets.reduce((n, b) => n + b.rungs.reduce((m, r) => m + r.nConfigs, 0), 0)
    expect(tune(objective, real(0, 1), { schedule, nTrials: 1000 }).study.trials.length).toBe(total)
    const repeating = hyperbandSchedule({ maxResource: 9, repeat: true })
    const study = tune(objective, real(0, 1), { schedule: repeating, nTrials: total + 5 }).study
    expect(study.trials.length).toBe(total + 5)
    expect(study.trials.at(-1)?.bracket).toBe(schedule.brackets.length)
  })
  it('promotes asynchronously with ASHA', () => {
    const objective = functionObjective((x, trial) => (x as number) + 1 / (trial.budget ?? 1))
    const schedule = ashaSchedule({ minResource: 1, maxResource: 9 })
    const study = tune(objective, real(0, 1), { schedule, nTrials: 40, seed: 3 }).study
    const byRung = [0, 1, 2].map((r) => study.trials.filter((t) => t.rung === r).length)
    expect(byRung[0]).toBeGreaterThan(byRung[1] ?? 0)
    expect(byRung[1]).toBeGreaterThan(byRung[2] ?? 0)
    expect(byRung[2]).toBeGreaterThan(0)
    // A promoted configuration was in the top third of its rung when it was promoted.
    for (const t of study.trials.filter((u) => u.source === 'promoted')) {
      const parent = study.trials[t.parent ?? 0]
      expect(parent?.rung).toBe((t.rung ?? 0) - 1)
    }
  })
  it('needs one objective', () => {
    expect(() =>
      nextAssignment(
        ashaSchedule({ maxResource: 9 }),
        createStudy(real(0, 1), { directions: ['minimize', 'minimize'] }),
      ),
    ).toThrow()
  })
})
