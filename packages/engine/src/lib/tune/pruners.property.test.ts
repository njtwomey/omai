import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { real } from '../space/index.ts'
import {
  functionObjective,
  hyperbandPruner,
  medianPruner,
  type Pruner,
  percentilePruner,
  successiveHalvingPruner,
  tune,
} from './index.ts'

const PRUNERS: readonly (() => Pruner)[] = [
  () => medianPruner({ nStartupTrials: 2 }),
  () => percentilePruner(25, { nStartupTrials: 1 }),
  () => successiveHalvingPruner({ minResource: 1, reductionFactor: 2 }),
  () => successiveHalvingPruner(),
  () => hyperbandPruner({ maxResource: 9 }),
  () => hyperbandPruner(),
]

describe('pruners never prune the best trial', () => {
  it('a trial better than every other at every step runs to the end', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.double({ min: 0, max: 10, noNaN: true }), { minLength: 10, maxLength: 10 }), {
          minLength: 4,
          maxLength: 12,
        }),
        fc.integer({ min: 0, max: PRUNERS.length - 1 }),
        fc.integer({ min: 1, max: 11 }),
        (curves, p, at) => {
          // Insert a curve below every other curve at every step.
          const steps = 10
          const best = Array.from({ length: steps }, (_, s) => Math.min(...curves.map((c) => c[s] ?? 0)) - 1 - s * 1e-3)
          const index = Math.min(at, curves.length)
          const rows = [...curves.slice(0, index), best, ...curves.slice(index)]
          const objective = functionObjective((_x, trial) => {
            const row = rows[trial.number] as number[]
            for (const [s, v] of row.entries()) if (trial.report(s, v)) return v
            return row.at(-1) as number
          })
          const study = tune(objective, real(0, 1), {
            nTrials: rows.length,
            pruner: (PRUNERS[p] as () => Pruner)(),
            constantTrials: Infinity,
          }).study
          expect(study.trials[index]?.state).toBe('complete')
        },
      ),
      { numRuns: 60 },
    )
  })
})
