import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { distanceToWalls, fromAscii, lineOfSight, wallDistanceField, worldGeometry } from './index.ts'

describe('distances and sight against scipy.ndimage and shapely', () => {
  for (const c of loadFixture('worlds/continuous').cases) {
    it(c.name, () => {
      const p = c.params
      const w = fromAscii(p['rows'] as string[])
      if (c.fn === 'transforms') {
        for (const metric of ['euclidean', 'taxicab', 'chessboard'] as const) {
          const got = wallDistanceField(w, { metric })
          const want = p[metric] as number[]
          for (const [i, x] of want.entries()) expect(got[i], `${metric} ${i}`).toBeCloseTo(x, 12)
        }
        return
      }
      const g = worldGeometry(w)
      const distances = p['distances'] as number[]
      for (const [i, point] of (p['points'] as [number, number][]).entries()) {
        expect(distanceToWalls(g, point)).toBeCloseTo(distances[i] ?? 0, 12)
      }
      const centres = p['centres'] as [number, number][]
      for (const [a = 0, b = 0, clear] of p['sight'] as [number, number, boolean][]) {
        const pa = centres[a] ?? [0, 0]
        const pb = centres[b] ?? [0, 0]
        expect(lineOfSight(g, pa, pb), `${pa} to ${pb}`).toBe(clear)
      }
    })
  }
})
