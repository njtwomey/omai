import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { hideLabels, UNLABELLED } from './labels.ts'
import { selfTraining } from './self-training.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.7 } })
const truth = Array.from(toFlat(data.y))
const y = hideLabels(data.y, { perClass: 4, key: key(4) })

describe('self-training', () => {
  it('pseudo-labels the pool round by round', () => {
    const f = selfTraining({ threshold: 0.7 }).fit({ ...data, y }, { key: key(0) })
    const pred = Array.from(toFlat(f.predict(data.x)))
    expect(pred.filter((p, i) => p === truth[i]).length / truth.length).toBeGreaterThan(0.9)
    const [steps] = recordsOf(f.exposition, 'clusteringSteps')
    const labelled = (a: readonly number[]) => a.filter((v) => v !== UNLABELLED).length
    expect(labelled(steps?.assignments[0] ?? [])).toBe(12)
    expect(labelled(steps?.assignments.at(-1) ?? [])).toBeGreaterThan(12)
    const [curve] = recordsOf(f.exposition, 'learningCurve')
    expect(curve?.sizes.length).toBe(steps?.assignments.length)
    expect(f.params.labeledIter.filter((v) => v === 0).length).toBe(12)
  })

  it('kBest adds k rows per round', () => {
    const f = selfTraining({ criterion: 'kBest', kBest: 5, maxIter: 3 }).fit({ ...data, y }, { key: key(0) })
    expect(f.params.transduction.filter((v) => v !== UNLABELLED).length).toBe(12 + 15)
    expect(f.params.terminationCondition).toBe('max_iter')
  })
})
