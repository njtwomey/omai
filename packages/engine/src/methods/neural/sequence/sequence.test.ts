import { describe, expect, it } from 'vitest'
import { generate } from '../../../lib/data/index.ts'
import { parityTask } from '../../../lib/datasets/index.ts'
import { recordsOf } from '../../../lib/exposition/index.ts'
import { key } from '../../../lib/random/index.ts'
import { recurrentTagger } from './rnn.ts'

describe('recurrentTagger', () => {
  it('learns running parity and records its hidden states at steps', () => {
    const { data } = generate(parityTask, { key: key(0), knobs: { n: 64, minLength: 5, maxLength: 5, running: true } })
    const r = recurrentTagger(data, { epochs: 60, snapshots: 12 }, { key: key(0) })
    expect(r.final).toBe(false)
    expect(r.trainAccuracy).toBeGreaterThan(0.8)
    const [loss, accuracy, positions] = recordsOf(r.exposition, 'learningCurve')
    expect(loss?.sizes.length).toBe(60)
    expect((loss?.train[0] ?? 0) > (loss?.train.at(-1) ?? 1)).toBe(true)
    expect(accuracy?.labels?.validation).toBe('validation token accuracy')
    expect(positions?.sizes).toEqual([1, 2, 3, 4, 5])
    const [hidden, tokens] = recordsOf(r.exposition, 'decisionSurface')
    // Sixteen units by five positions; input, target and prediction by five positions.
    expect([hidden?.values?.length, hidden?.values?.[0]?.length]).toEqual([16, 5])
    expect([tokens?.values?.length, tokens?.values?.[0]?.length]).toEqual([3, 5])
    expect(hidden?.steps?.length).toBeLessThanOrEqual(12)
    expect(r.predict([[2, 3, 3, 2, 3]])[0]?.length).toBe(5)
  })

  it('reads the last position for one-token targets, with a GRU', () => {
    const { data } = generate(parityTask, { key: key(1), knobs: { n: 40, minLength: 3, maxLength: 3 } })
    const r = recurrentTagger(data, { cell: 'gru', hidden: 4, epochs: 3, snapshots: 3 }, { key: key(0) })
    expect(r.final).toBe(true)
    expect(r.predict([[2, 3, 3]])).toHaveLength(1)
    expect(r.describe().nodes.length).toBeGreaterThan(1)
  })
})
