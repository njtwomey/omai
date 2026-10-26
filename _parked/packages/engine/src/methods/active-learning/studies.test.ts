import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { STRATEGY_NAMES } from './strategies.ts'
import { activeLearningStudy } from './studies.ts'

const { data } = generate(blobs, { key: key(7), knobs: { n: 60, centers: 3, clusterStd: 0.8 } })

describe('active-learning studies', () => {
  it('uncertainty sampling buys labels and learns, beside a random baseline', () => {
    const s = activeLearningStudy(data, { strategy: 'margin', budget: 12, key: key(0) })
    expect(s.initial.length).toBe(6)
    expect(s.queried.length).toBe(12)
    expect(new Set([...s.initial, ...s.queried]).size).toBe(18)
    expect(s.labelled).toEqual([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    expect(s.accuracy.at(-1)).toBeGreaterThan((s.accuracy[0] ?? 1) + 0.1)
    const [curve] = recordsOf(s.exposition, 'learningCurve')
    expect(curve?.validation.length).toBe(13)
    expect(curve?.labels?.validation).toBe('random')
    const [steps] = recordsOf(s.exposition, 'clusteringSteps')
    expect(steps?.assignments[0]?.filter((a) => a >= 0).length).toBe(6)
    expect(steps?.assignments.at(-1)?.filter((a) => a >= 0).length).toBe(18)
    expect(recordsOf(s.exposition, 'decisionSurface').length).toBe(1)
  })

  it('every strategy runs in batches', () => {
    for (const strategy of STRATEGY_NAMES) {
      const s = activeLearningStudy(data, {
        strategy,
        budget: 4,
        batchSize: 2,
        committeeSize: 3,
        candidates: 6,
        baseline: false,
        key: key(1),
      })
      expect(s.queried.length).toBe(4)
      expect(s.labelled).toEqual([6, 8, 10])
    }
  })
})
