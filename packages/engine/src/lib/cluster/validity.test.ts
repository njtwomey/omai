import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { gapStatistic, kmeansElbow, stabilityIndex } from './validity.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } })

describe('choosing the number of clusters', () => {
  it('the elbow curve falls with k', () => {
    const e = kmeansElbow(data.x, [1, 2, 3, 4], key(0))
    for (let i = 1; i < 4; i++) expect(e.inertia[i] ?? 0).toBeLessThan(e.inertia[i - 1] ?? 0)
  })

  it('the gap statistic finds three blobs', () => {
    const g = gapStatistic(data.x, [1, 2, 3, 4, 5], key(0), { references: 8 })
    expect(g.best).toBe(3)
    expect(g.gap.length).toBe(5)
    expect(() => gapStatistic(data.x, [0], key(0))).toThrow()
  })

  it('the stability index is highest at the true k', () => {
    const s = stabilityIndex(data.x, [2, 3, 5], key(0), { pairs: 6 })
    expect(s.best).toBe(3)
    expect(s.stability[1]).toBeGreaterThan(0.95)
  })
})
