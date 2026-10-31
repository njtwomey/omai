import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { columnSelectors, discoverSubgroups } from './index.ts'

describe('subgroup discovery against pysubgroup', () => {
  for (const c of loadFixture('patterns/subgroups').cases) {
    it(c.name, () => {
      const p = c.params
      const selectors = columnSelectors(p['columns'] as Record<string, string[]>)
      const target =
        c.fn === 'mean-shift'
          ? ({ kind: 'numeric', values: p['value'] as number[] } as const)
          : ({ kind: 'binary', positive: p['target'] as number[] } as const)
      const expected = p['qualities'] as number[]
      const options = {
        k: p['k'] as number,
        depth: p['depth'] as number,
        // pysubgroup's StandardQF(a) is (n/N)^a (p/n - p0); its numeric quality is n^a (mean - overall mean) with a = 1.
        measure: 'klosgen' as const,
        a: c.fn === 'mean-shift' ? 1 : (p['a'] as number),
      }
      const exhaustive = discoverSubgroups(selectors, target, { ...options, search: 'exhaustive' })
      expect(exhaustive.length).toBe(expected.length)
      exhaustive.forEach((s, i) => {
        expect(s.quality).toBeCloseTo(expected[i] ?? 0, 10)
      })
      // A wide beam reaches the same optimum on these small tables.
      const beam = discoverSubgroups(selectors, target, { ...options, width: 200 })
      expect(beam[0]?.quality).toBeCloseTo(expected[0] ?? 0, 10)
    })
  }
})
