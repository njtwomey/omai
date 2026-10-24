import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { type AssociationMeasure, collocationCounts, scoreCollocations } from './index.ts'

describe('text/collocations against nltk', () => {
  for (const c of loadFixture('text/collocations').cases) {
    it(c.name, () => {
      const words = c.inputs[0] as string[]
      const window = c.params['window'] as number
      const order = c.fn === 'bigram' ? 2 : 3
      const filtered = !(order === 2 && window === 2)
      const scored = scoreCollocations(
        collocationCounts(words, { order, windowSize: window }),
        c.params['measure'] as AssociationMeasure,
        {
          ...(filtered ? { minFreq: 2 } : {}),
          ...(order === 2 && window === 4 ? { wordFilter: (w: string) => w === '.' } : {}),
        },
      )
      const expected = c.output as [string[], number][]
      const ours = new Map(scored.map((s) => [s.ngram.join(' '), s.score]))
      expect([...ours.keys()].sort()).toEqual(expected.map(([ng]) => ng.join(' ')).sort())
      for (const [ng, s] of expected) {
        const v = ours.get(ng.join(' ')) ?? Number.NaN
        expect(Math.abs(v - s)).toBeLessThanOrEqual(c.atol + c.rtol * Math.abs(s))
      }
      for (let i = 1; i < scored.length; i++)
        expect(scored[i]?.score ?? 0).toBeLessThanOrEqual(scored[i - 1]?.score ?? 0)
    })
  }
})
