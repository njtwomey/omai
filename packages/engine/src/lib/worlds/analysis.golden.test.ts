import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { decodeNumber } from '../base/index.ts'
import { bridges, chokePoints, deadEnds, distanceField, fromAscii, regions } from './index.ts'

describe('the cell graph against networkx', () => {
  for (const c of loadFixture('worlds/analysis').cases) {
    it(c.name, () => {
      const p = c.params
      const w = fromAscii(p['rows'] as string[], p['topology'] === 'hex' ? { topology: 'hex' } : {})
      const options = p['connectivity'] === 8 ? { connectivity: 8 as const } : {}
      const r = regions(w, options)
      const groups = Array.from({ length: r.count }, (_, k) =>
        Array.from(r.labels.entries())
          .filter(([, l]) => l === k)
          .map(([cell]) => cell),
      ).sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0))
      expect(groups).toEqual(p['regions'])
      expect(chokePoints(w, options)).toEqual(p['chokePoints'])
      expect(bridges(w, options)).toEqual(p['bridges'])
      expect(deadEnds(w, options)).toEqual(p['deadEnds'])
      const source = p['source'] as number
      expect(Array.from(distanceField(w, source, options))).toEqual(
        (p['hops'] as (number | string)[]).map(decodeNumber),
      )
      const cost = distanceField(w, source, { ...options, weighted: true })
      for (const [i, want] of (p['cost'] as (number | string)[]).map(decodeNumber).entries()) {
        if (Number.isFinite(want)) expect(cost[i]).toBeCloseTo(want, 12)
        else expect(cost[i]).toBe(want)
      }
    })
  }
})
