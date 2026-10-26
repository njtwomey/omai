import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { HuberRegressor, quantileRegressor, ransacRegressor, theilSenRegressor } from './robust.ts'

const x = Array.from({ length: 40 }, (_, i) => [i / 10])
const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) + (i % 7 === 0 ? 15 : ((i * 13) % 5) / 25))

describe('robust regression', () => {
  it('resists outliers and flags them', () => {
    const h = new HuberRegressor().fit(x, y)
    expect(h.model.params.coef[0]).toBeCloseTo(2, 0)
    const scores = recordsOf(h.exposition, 'anomalyScores')[0]
    expect((scores?.scores[0] ?? 0) > (scores?.threshold ?? 0)).toBe(true)
    const r = ransacRegressor().fit({ x, y }, { key: key(3) })
    expect(r.params.coef[0]).toBeCloseTo(2, 1)
    expect(recordsOf(r.exposition, 'anomalyScores')[0]?.evidence?.values[0]).toBe(0)
    expect(theilSenRegressor().fit({ x, y }).params.coef[0]).toBeCloseTo(2, 1)
    const q = quantileRegressor({ alpha: 0 }).fit({ x, y })
    expect(q.params.coef[0]).toBeCloseTo(2, 1)
  })
})
