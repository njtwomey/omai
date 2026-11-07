import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { functionObjective, gridSampler, haltonSampler, type StudySampler, sobolSampler, tune } from './index.ts'

const fixture = loadFixture('tune/samplers')

/** Values equal up to floating point (the QMC points go through the same transform in a different order). */
function close(a: SpaceValue, b: SpaceValue): boolean {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b))
  if (typeof a === 'object' && typeof b === 'object' && a !== null && b !== null) {
    const ra = a as Readonly<Record<string, SpaceValue>>
    const rb = b as Readonly<Record<string, SpaceValue>>
    return Object.keys(rb).every((k) => close(ra[k] as SpaceValue, rb[k] as SpaceValue))
  }
  return a === b
}

const QMC_SPACE = dict({
  x: real(-2, 3),
  y: real(1e-3, 10, { log: true }),
  k: int(0, 8),
  c: choice(['a', 'b', 'c', 'd']),
})

describe('sampler orders against scikit-learn and optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      if (c.fn === 'grid') {
        const grids = c.inputs[0] as Record<string, (string | number | boolean)[]>[]
        const got: SpaceValue[] = []
        for (const g of grids) {
          const names = Object.keys(g).sort()
          const space = dict(Object.fromEntries(names.map((n) => [n, choice(g[n] ?? [])])))
          const study = tune(
            functionObjective(() => 0),
            space,
            { sampler: gridSampler(), nTrials: 1000, constantTrials: Infinity },
          ).study
          got.push(...study.trials.map((t) => t.params))
        }
        expect(got).toEqual(c.output)
        return
      }
      const sampler: StudySampler = c.fn === 'sobol' ? sobolSampler() : haltonSampler()
      const n = Number(c.params['n'])
      const study = tune(
        functionObjective(() => 0),
        QMC_SPACE,
        { sampler, nTrials: n, constantTrials: Infinity },
      ).study
      const want = c.output as SpaceValue[]
      for (const [i, t] of study.trials.entries())
        expect(close(t.params, want[i] as SpaceValue), `trial ${i}`).toBe(true)
    })
  }
})
