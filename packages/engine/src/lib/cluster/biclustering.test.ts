import { describe, expect, it } from 'vitest'
import { makeBiclusters, makeCheckerboard } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm } from '../trace/index.ts'
import { spectralBiclusteringAlgorithm, spectralCoclusteringAlgorithm } from './bicluster-algorithms.ts'
import { consensusScore } from './bicluster-score.ts'
import { spectralBiclustering, spectralCoclustering } from './biclustering.ts'

describe('spectral co-clustering', () => {
  it('recovers planted diagonal biclusters: consensus score 1', () => {
    for (const k of [2, 3, 4]) {
      const truth = makeBiclusters(key(k), [60, 45], k, { noise: 1 })
      const f = spectralCoclustering.with({ nClusters: k }).fit({ x: truth.x }, { key: key(0) })
      expect(consensusScore(f.params, truth)).toBe(1)
      expect(f.params.rows.length).toBe(k)
    }
  })

  it('the tape plays Lloyd steps over the rows and columns together, and the labels path follows them', () => {
    const truth = makeBiclusters(key(5), [40, 30], 3, { noise: 1 })
    const s = spectralCoclustering.with({ nClusters: 3 }).session({ x: truth.x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    const labels = s.probe(s.t, 'bicluster/labels') as { rows: number[]; columns: number[] }
    expect(labels.rows.length).toBe(40)
    expect(labels.columns.length).toBe(30)
    const order = s.probe(s.t, 'bicluster/order') as { rows: number[]; columns: number[] }
    expect([...order.rows].sort((a, b) => a - b)).toEqual(Array.from({ length: 40 }, (_, i) => i))
    const inertia = metricSeries(s.run(), 'solver/inertia').values
    expect(inertia[inertia.length - 1] ?? 0).toBeLessThanOrEqual(inertia[0] ?? 0)
  })

  it('submatrices, shapes and indices of the fitted biclusters', () => {
    const truth = makeBiclusters(key(5), [40, 30], 3, { noise: 0 })
    const f = spectralCoclustering.with({ nClusters: 3 }).fit({ x: truth.x }, { key: key(1) })
    const total = [0, 1, 2].reduce((n, i) => n + f.getShape(i)[0] * f.getShape(i)[1], 0)
    expect(total).toBeLessThanOrEqual(40 * 30)
    const sub = f.getSubmatrix(0, truth.x)
    expect(sub.shape).toEqual(f.getShape(0))
    // A noiseless planted block is constant.
    expect(new Set(toFlat(sub)).size).toBe(1)
    expect(f.getIndices(0).rows.length).toBe(f.getShape(0)[0])
    expect(() => f.getIndices(9)).toThrow(/does not exist/)
  })

  it('the randomised SVD gives the same partition on well separated data; errors are named', () => {
    const truth = makeBiclusters(key(3), [50, 40], 3, { noise: 1 })
    const f = spectralCoclustering.with({ nClusters: 3, svdMethod: 'randomized' }).fit({ x: truth.x }, { key: key(2) })
    expect(consensusScore(f.params, truth)).toBe(1)
    const x = toArray(truth.x) as number[][]
    expect(() => spectralCoclusteringAlgorithm(x, 1, {}, key(0))).toThrow(/nClusters/)
    expect(() => spectralCoclusteringAlgorithm(x, 99, {}, key(0))).toThrow(/nClusters/)
    expect(() =>
      runAlgorithm(
        spectralCoclusteringAlgorithm(
          [
            [0, 0],
            [1, 2],
          ],
          2,
          {},
          key(0),
        ),
        { maxSteps: 1 },
      ),
    ).toThrow(/sums to zero/)
  })
})

describe('spectral biclustering', () => {
  it('recovers a planted checkerboard with every normalisation: consensus score 1', () => {
    const truth = makeCheckerboard(key(1), [60, 45], [3, 2], { noise: 1 })
    for (const method of ['bistochastic', 'scale', 'log'] as const) {
      const f = spectralBiclustering
        .with({ nClusters: [3, 2], method, nComponents: 4, nBest: 2 })
        .fit({ x: truth.x }, { key: key(0) })
      expect(consensusScore(f.params, truth)).toBeGreaterThan(0.99)
      expect(f.params.rows.length).toBe(6)
      expect(f.params.nRowClusters).toBe(3)
    }
  })

  it('clusters the rows first and then the columns, observing both stages', () => {
    const truth = makeCheckerboard(key(1), [40, 30], [2, 2], { noise: 1 })
    const f = spectralBiclustering
      .with({ nClusters: [2, 2], method: 'log', nComponents: 3, nBest: 2 })
      .fit({ x: truth.x }, { key: key(0), logs: 'all' })
    const stage = metricSeries(f.run, 'solver/stage').values
    expect(stage[0]).toBe(0)
    expect(stage[stage.length - 1]).toBe(1)
    expect(stage.every((v, i) => i === 0 || v >= (stage[i - 1] ?? 0))).toBe(true)
  })

  it('takes one count for rows and columns, and rejects impossible ones', () => {
    const x = toArray(makeCheckerboard(key(1), [20, 16], [2, 2], { noise: 0.5 }).x) as number[][]
    const f = spectralBiclustering.with({ nClusters: 2, nComponents: 3, nBest: 2 }).fit({ x })
    expect(f.params.nColumnClusters).toBe(2)
    expect(() => spectralBiclusteringAlgorithm(x, [0, 2], {}, key(0))).toThrow(/row cluster count/)
    expect(() => spectralBiclusteringAlgorithm(x, [2, 99], {}, key(0))).toThrow(/column cluster count/)
    expect(() => spectralBiclusteringAlgorithm(x, 2, { nComponents: 2, nBest: 3 }, key(0))).toThrow(/nBest/)
    expect(() => spectralBiclusteringAlgorithm(x, 2, { nComponents: 40 }, key(0))).toThrow(/singular vectors/)
  })
})
