// The spike and slab setups against their closed form (a two-component mixture whose weights are the branches' evidence),
// across observed values, inclusion probabilities and widths: EP's probability of the slab, mean and sd of the coefficient.
import { describe, expect, it } from 'vitest'
import { spikeSlabPosterior, twoComponentGate } from '../validation/index.ts'
import { epAtom } from './ep-atom-list.ts'
import { verifyAtom } from './ep-atoms.ts'

describe('spike and slab as a gate', () => {
  const cases = [] as { y: number; pi: number; slab: number; spike: number }[]
  for (const y of [0.05, 0.7, 3])
    for (const pi of [0.1, 0.5, 0.9])
      for (const [slab, spike] of [
        [2, 0.1],
        [4, 0.05],
      ] as const)
        cases.push({ y, pi, slab, spike })

  it.each(cases)('EP reports P(slab), mean and sd of the coefficient as the closed form does: %j', (k) => {
    for (const name of ['spike-and-slab', 'spike-and-slab-marginal']) {
      const v = verifyAtom(epAtom(name), k)
      expect(v.run.status).toBe('ran')
      const x = v.check?.rows.find((r) => r.name === (name === 'spike-and-slab' ? 'x' : 'w'))
      expect(x?.meanZ).toBeLessThan(1e-4)
      expect(Math.abs((x?.sdRatio ?? 0) - 1)).toBeLessThan(1e-4)
      if (name === 'spike-and-slab') {
        const z = v.check?.rows.find((r) => r.name === 'z')
        expect(Math.abs((z?.ep.mean ?? 0) - spikeSlabPosterior(k.pi, k.slab, k.spike, k.y, 0.5).p1)).toBeLessThan(1e-6)
      }
    }
  })

  it('has a narrow spike branch around zero and a wide slab branch', () => {
    const g = spikeSlabPosterior(0.5, 2, 0.1, 0.7, 0.5)
    expect(g.updated[0].sd).toBeLessThan(0.1)
    expect(g.updated[1].sd).toBeGreaterThan(0.4)
    expect(Math.abs(g.updated[0].mean)).toBeLessThan(0.05)
    // A value near zero is the spike's, a large one the slab's.
    expect(spikeSlabPosterior(0.5, 2, 0.1, 0.05, 0.5).p1).toBeLessThan(0.3)
    expect(spikeSlabPosterior(0.5, 2, 0.1, 3, 0.5).p1).toBeGreaterThan(0.999)
  })

  it('a two-component gate between equal components does not move the switch', () => {
    expect(
      twoComponentGate(
        0.3,
        [
          { mean: 0, sd: 1 },
          { mean: 0, sd: 1 },
        ],
        0.4,
        1,
      ).p1,
    ).toBeCloseTo(0.3, 12)
  })
})
