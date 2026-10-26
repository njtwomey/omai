import { describe, expect, it } from 'vitest'
import { diabetes, digits } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { randomForestRegressor } from '../ensemble/index.ts'
import { lenet } from '../neural/vision/index.ts'
import { evaluateExplanations, sanityChecks } from './evaluation.ts'

describe('evaluation methods', () => {
  it('scores explainers on quality metrics with their agreement and curves', () => {
    const data = takeData(
      diabetes(),
      Array.from({ length: 80 }, (_, i) => i),
    )
    const forest = randomForestRegressor({ nEstimators: 4, maxDepth: 3 }).fit(data, { key: key(1) })
    const r = evaluateExplanations(forest, data, {
      explainers: ['tree', 'sampling'],
      rows: 3,
      background: 8,
      sensitivitySamples: 1,
      key: key(2),
    })
    const [q] = recordsOf(r.exposition, 'explanationQuality')
    expect(q?.explainers).toEqual(['tree', 'sampling'])
    expect(q?.metrics.length).toBe(6)
    expect(q?.agreement?.matrix[0]?.[0]).toBe(1)
    expect(recordsOf(r.exposition, 'curves')[0]?.series.length).toBe(4)
  })

  it('sanity checks randomise a LeNet layer by layer and its labels', () => {
    const images = takeData(
      digits(),
      Array.from({ length: 100 }, (_, i) => i),
    )
    const net = lenet({ epochs: 1, channels: [3, 4], hidden: 8, snapshots: 2 }).fit(images, {
      key: key(3),
      explain: false,
    })
    const r = sanityChecks(net, images, 2, { hyper: { epochs: 1 }, key: key(4) })
    expect(r.cascade.length).toBe(4)
    expect(r.labelSimilarity).toBeDefined()
    expect(recordsOf(r.exposition, 'valueGrids')[0]?.grids.length).toBe(6)
  })
})
