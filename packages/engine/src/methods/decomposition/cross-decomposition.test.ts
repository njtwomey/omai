import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { CCA, PLSRegression, plsSvd } from './cross-decomposition.ts'

const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(i / 2), (i % 5) / 5, i / 30])
const y = x.map((r) => 2 * (r[0] ?? 0) - (r[3] ?? 0))

describe('cross-decomposition', () => {
  it('PLS regression predicts a linear target; CCA and PLS-SVD give scores', () => {
    const m = new PLSRegression({ nComponents: 3 }).fit(x, y)
    expect(m.score(x, y)).toBeGreaterThan(0.99)
    expect(recordsOf(m.model.exposition, 'components')[0]?.scores?.length).toBe(30)
    const y2 = x.map((r) => [r[0] ?? 0, (r[1] ?? 0) + (r[2] ?? 0)])
    expect(new CCA({ nComponents: 2 }).fit(x, y2).transform(x).shape).toEqual([30, 2])
    expect(plsSvd({ nComponents: 1 }).fit({ x, y: y2 }).transform(x).shape).toEqual([30, 1])
  })
})
