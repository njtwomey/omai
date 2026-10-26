import { describe, expect, it } from 'vitest'
import { diabetes } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { LassoLars, lars } from './lars.ts'

describe('least angle regression', () => {
  it('walks the path one knot per step', () => {
    const data = diabetes()
    const f = lars({ nNonzeroCoefs: 4 }).fit(data)
    expect(f.params.coef.filter((v) => v !== 0)).toHaveLength(4)
    const path = recordsOf(f.exposition, 'regularisationPath')[0]
    expect(path?.active?.[4]).toHaveLength(4)
    expect(f.score?.(data)).toBeGreaterThan(0.3)
    const l = new LassoLars({ alpha: 0.5 }).fit(data.x, data.y)
    expect(recordsOf(l.exposition, 'regularisationPath')[0]?.chosen).toBe(0.5)
  })
})
