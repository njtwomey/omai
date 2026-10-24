import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { halvingPlan } from './index.ts'

const fixture = loadFixture('tune/schedules')

describe('successive-halving plans against scikit-learn', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const out = c.output as { nCandidates: number[]; minResources: number; maxResources: number }
      const p = c.params
      const classes = Number(p['classes'] ?? 0)
      const cv = Number(p['cv'])
      const smallest = p['resource'] === undefined ? cv * 2 * (classes > 0 ? classes : 1) : 1
      const requested = p['minResources']
      const plan = halvingPlan({
        nCandidates: out.nCandidates[0] as number,
        minResources: typeof requested === 'number' ? requested : smallest,
        maxResources: p['maxResources'] === undefined ? Number(p['nSamples']) : Number(p['maxResources']),
        factor: Number(p['factor']),
        exhaust: requested === 'exhaust',
        aggressiveElimination: p['aggressiveElimination'] === true,
      })
      expect(plan).toEqual(c.output)
    })
  }
})
