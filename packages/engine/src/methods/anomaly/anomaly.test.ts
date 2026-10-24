import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key, normal } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { combineScores, iqrThreshold, madThreshold, normaliseScores } from './combine.ts'
import { IsolationForest, isolationForest } from './isolation.ts'
import { knnDetector, LocalOutlierFactor } from './neighbours.ts'
import { matrixProfileDiscords } from './sequences.ts'
import { EllipticEnvelope, hbos, pcaReconstructionDetector, zScoreDetector } from './statistical.ts'

const inliers = toArray(normal(key(0), [80, 2])) as number[][]
const outliers = [
  [6, 6],
  [-6, 5],
  [7, -6],
  [-7, -7],
]
const x = [...inliers, ...outliers]
const planted = (scores: readonly number[]): number => {
  const cut = [...scores].sort((a, b) => b - a)[3] ?? 0
  return [80, 81, 82, 83].filter((i) => (scores[i] ?? 0) >= cut).length
}

describe('anomaly detectors', () => {
  it('each detector ranks the planted outliers highest and explains its scores', () => {
    const detectors = [
      isolationForest({ nEstimators: 50, contamination: 0.05 }),
      isolationForest({ nEstimators: 50, extended: true }),
      knnDetector({ nNeighbors: 5 }),
      hbos({ bins: 8 }),
      zScoreDetector({ robust: true }),
      pcaReconstructionDetector({ nComponents: 1 }),
    ]
    for (const d of detectors) {
      const f = d.fit({ x }, { key: key(1) })
      const [rec] = recordsOf(f.exposition, 'anomalyScores')
      expect(rec?.scores.length).toBe(84)
      expect(rec?.evidence?.values.length).toBe(84)
      expect(planted(toFlat(f.scoreSamples(x)))).toBeGreaterThanOrEqual(
        d.name === 'pca-reconstruction-detector' ? 2 : 4,
      )
      expect(recordsOf(f.exposition, 'decisionSurface').length).toBe(1)
    }
  })

  it('LOF, the elliptic envelope and the forest classes flag the outliers with -1', () => {
    for (const m of [
      new LocalOutlierFactor({ nNeighbors: 10 }),
      new EllipticEnvelope({ contamination: 0.05 }),
      new IsolationForest({ nEstimators: 50, contamination: 0.05 }),
    ]) {
      const labels = toFlat(m.fit(x, undefined, { key: key(2) }).predict(outliers))
      expect(labels).toEqual([-1, -1, -1, -1])
    }
    const lof = new LocalOutlierFactor({ nNeighbors: 10 }).fit(x).model
    expect(recordsOf(lof.exposition, 'neighbours')[0]?.indices.length).toBe(84)
  })

  it('combines, normalises and thresholds scores', () => {
    const a = [1, 2, 3, 10]
    const b = [0.1, 0.3, 0.2, 5]
    expect(normaliseScores(a, 'minmax')).toEqual([0, 1 / 9, 2 / 9, 1])
    expect(normaliseScores([3, 1, 1], 'rank')).toEqual([1, 0.25, 0.25])
    const c = combineScores([a, b], { combination: 'max', contamination: 0.25 })
    expect(c.scores.indexOf(Math.max(...c.scores))).toBe(3)
    expect(iqrThreshold(a)).toBeLessThan(10)
    expect(madThreshold(a)).toBeLessThan(10)
  })

  it('finds a planted discord in a periodic series', () => {
    const t = Array.from({ length: 120 }, (_, i) => Math.sin((2 * Math.PI * i) / 12) + (i >= 60 && i < 66 ? 2 : 0))
    const rec = matrixProfileDiscords(t, 12)
    const start = rec.scores.indexOf(Math.max(...rec.scores))
    expect(start).toBeGreaterThan(48)
    expect(start).toBeLessThan(66)
  })
})
