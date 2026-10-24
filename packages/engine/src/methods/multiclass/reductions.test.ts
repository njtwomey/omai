import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet } from '../../testing/supervised.ts'
import { decisionTreeClassifier } from '../trees/index.ts'
import { OneVsRestClassifier, oneVsOneClassifier, outputCodeClassifier } from './reductions.ts'

describe('multiclass reductions', () => {
  it('one-versus-rest, one-versus-one and output codes', () => {
    const flowers = iris()
    const ovr = new OneVsRestClassifier().fit(flowers.x, flowers.y)
    expect(ovr.score(flowers.x, flowers.y)).toBeGreaterThan(0.9)
    expect(recordsOf(ovr.exposition, 'ensembleStages')[0]?.trainScores).toHaveLength(3)
    const ovo = oneVsOneClassifier().fit(flowers)
    expect(ovo.score?.(flowers)).toBeGreaterThan(0.9)
    expect(ovo.predictProba(flowers.x).shape).toEqual([150, 3])
    const blobs = classificationSet('blobs', { n: 90 }, 1)
    const ecoc = outputCodeClassifier({ codeSize: 2 }).fit(blobs, { key: key(1) })
    expect(ecoc.score?.(blobs)).toBeGreaterThan(0.8)
    expect(recordsOf(ecoc.exposition, 'decisionSurface')).toHaveLength(1)
    expect(
      oneVsOneClassifier({ estimator: decisionTreeClassifier({ maxDepth: 2 }) })
        .fit(blobs)
        .score?.(blobs),
    ).toBeGreaterThan(0.8)
  })
})
