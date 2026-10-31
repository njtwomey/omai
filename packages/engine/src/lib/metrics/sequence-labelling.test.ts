import { describe, expect, it } from 'vitest'
import { bioViolations, entityF1, entityScores, entitySpans } from './sequence-labelling.ts'

describe('entity spans', () => {
  it('reads BIO, a stray inside tag, IOB1 and a type change', () => {
    expect(entitySpans(['B-PER', 'I-PER', 'O', 'B-LOC'])).toEqual([
      { type: 'PER', start: 0, end: 2 },
      { type: 'LOC', start: 3, end: 4 },
    ])
    expect(entitySpans(['O', 'I-PER', 'I-PER'])).toEqual([{ type: 'PER', start: 1, end: 3 }])
    expect(entitySpans(['B-PER', 'B-PER'])).toHaveLength(2)
    expect(entitySpans(['I-PER', 'I-LOC'])).toEqual([
      { type: 'PER', start: 0, end: 1 },
      { type: 'LOC', start: 1, end: 2 },
    ])
    expect(entitySpans(['B-PER', 'I-LOC'])).toHaveLength(2)
  })

  it('reads BIOES and BILOU', () => {
    expect(entitySpans(['B-PER', 'I-PER', 'E-PER', 'S-LOC', 'O'])).toEqual([
      { type: 'PER', start: 0, end: 3 },
      { type: 'LOC', start: 3, end: 4 },
    ])
    expect(entitySpans(['B-PER', 'L-PER', 'U-LOC'], 'bilou')).toEqual([
      { type: 'PER', start: 0, end: 2 },
      { type: 'LOC', start: 2, end: 3 },
    ])
  })

  it('finds the inside tags that continue nothing', () => {
    expect(bioViolations(['O', 'I-PER', 'B-PER', 'I-LOC', 'I-LOC', 'B-LOC', 'I-LOC'])).toEqual([1, 3])
    expect(bioViolations(['B-PER', 'I-PER'])).toEqual([])
  })
})

describe('entity scores', () => {
  // The worked example of seqeval's README.
  const yTrue = [
    ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'],
    ['B-PER', 'I-PER', 'O'],
  ]
  const yPred = [
    ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'],
    ['B-PER', 'I-PER', 'O'],
  ]

  it("reproduces seqeval's classification report", () => {
    const s = entityScores(yTrue, yPred)
    expect([s.precision, s.recall, s.f1, s.support]).toEqual([0.5, 0.5, 0.5, 2])
    expect(s.perType['MISC']).toEqual({ precision: 0, recall: 0, f1: 0, support: 1 })
    expect(s.perType['PER']).toEqual({ precision: 1, recall: 1, f1: 1, support: 1 })
    expect([s.macro.precision, s.macro.recall, s.macro.f1]).toEqual([0.5, 0.5, 0.5])
    expect(s.weighted.f1).toBe(0.5)
  })

  it('counts a partly right span as wrong in both precision and recall', () => {
    const s = entityScores([['B-PER', 'I-PER', 'O']], [['B-PER', 'O', 'O']])
    expect([s.precision, s.recall, s.f1]).toEqual([0, 0, 0])
  })

  it('is a metric with the precision and recall in its breakdown', () => {
    const r = entityF1(yTrue, yPred)
    expect(r.estimate.value).toBe(0.5)
    expect(r.breakdown).toEqual({ precision: 0.5, recall: 0.5, support: 2 })
    expect(() => entityF1([['O']], [['O'], ['O']])).toThrow()
    expect(() => entityF1([['O']], [['O', 'O']])).toThrow()
  })

  it('weights types by support in the weighted average', () => {
    const t = [['B-A', 'B-A', 'B-A', 'B-B']]
    const p = [['B-A', 'B-A', 'B-A', 'O']]
    const s = entityScores(t, p)
    expect(s.macro.recall).toBe(0.5)
    expect(s.weighted.recall).toBeCloseTo(0.75, 12)
  })
})
