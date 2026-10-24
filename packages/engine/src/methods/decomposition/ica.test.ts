import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { pearson } from '../../lib/stats/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { FastICA } from './ica.ts'

describe('FastICA', () => {
  it('unmixes a sine and a square wave', () => {
    const t = Array.from({ length: 200 }, (_, i) => (i / 200) * 8)
    const s = t.map((v) => [Math.sin(2 * v), Math.sign(Math.sin(3 * v))])
    const x = s.map(([a = 0, b = 0]) => [a + 0.5 * b, 0.4 * a + b])
    const m = new FastICA({ nComponents: 2 }).fit(x, undefined, { key: key(0) })
    const z = toArray(m.transform(x)) as number[][]
    for (let j = 0; j < 2; j++) {
      const best = Math.max(
        ...[0, 1].map((c) =>
          Math.abs(
            pearson(
              z.map((r) => r[c] ?? 0),
              s.map((r) => r[j] ?? 0),
            ).statistic,
          ),
        ),
      )
      expect(best).toBeGreaterThan(0.95)
    }
    expect(recordsOf(m.model.exposition, 'components')[0]?.components.length).toBe(2)
  })
})
