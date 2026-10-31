import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  batchRanges,
  incrementalPca,
  incrementalPcaAlgorithm,
  pca,
  pcaSvdAlgorithm,
  powerPcaAlgorithm,
  sparsePca,
  sparsePcaAlgorithm,
} from './pca.ts'

const { data } = generate(blobs, { key: key(2), knobs: { n: 60, centers: 3, clusterStd: 1, nFeatures: 4 } })
const column = (
  t: ReturnType<typeof pca.fit>['transform'] extends (x: never) => infer R ? R : never,
  c: number,
): number[] => {
  const f = toFlat(t)
  const k = t.shape[1] ?? 1
  return f.filter((_, i) => i % k === c)
}

describe('PCA', () => {
  it('whitened scores have unit variance; the inverse transform of all components reconstructs', () => {
    const f = pca.with({ nComponents: 4, whiten: true }).fit(data)
    const z = f.transform(data.x)
    for (let c = 0; c < 4; c++) {
      const col = column(z, c)
      const mean = col.reduce((a, v) => a + v, 0) / col.length
      expect(col.reduce((a, v) => a + (v - mean) ** 2, 0) / (col.length - 1)).toBeCloseTo(1, 9)
    }
    const back = toFlat(f.inverseTransform(z))
    const x = toFlat(data.x)
    for (const [i, v] of back.entries()) expect(v).toBeCloseTo(x[i] ?? 0, 9)
  })

  it('power iteration finds the SVD components one per step, in order of variance', () => {
    const exact = pca.with({ nComponents: 3 }).fit(data)
    const power = pca.with({ nComponents: 3, solver: 'power' }).fit(data, { key: key(1), logs: 'all' })
    for (let c = 0; c < 3; c++) {
      expect(power.params.explainedVariance[c]).toBeCloseTo(exact.params.explainedVariance[c] ?? 0, 6)
      for (let j = 0; j < 4; j++)
        expect(power.params.components[c]?.[j]).toBeCloseTo(exact.params.components[c]?.[j] ?? 0, 5)
    }
    expect(metricSeries(power.run, 'solver/found').values).toEqual([0, 1, 2, 3])
    const s = pca.with({ nComponents: 3, solver: 'power' }).session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 1 })
    expect(s.fittedAt(1).params.components.length).toBe(1)
    expect(s.fittedAt(1).transform(data.x).shape).toEqual([60, 1])
  })

  it('the randomised solver agrees with the exact one on the leading components; states round-trip', () => {
    const exact = pca.with({ nComponents: 2 }).fit(data).params
    const rnd = pca.with({ nComponents: 2, solver: 'randomized' }).fit(data, { key: key(3) }).params
    expect(rnd.explainedVariance[0]).toBeCloseTo(exact.explainedVariance[0] ?? 0, 6)
    const state = runAlgorithm(pcaSvdAlgorithm(data.x, 2), { maxSteps: 1 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
    const p = runAlgorithm(powerPcaAlgorithm(data.x, 2), { maxSteps: 2, key: key(0) }).state
    expect(stateFromJson(stateToJson(p))).toEqual(p)
    expect(() => pcaSvdAlgorithm(data.x, 9)).toThrow()
  })

  it('incremental PCA matches PCA on the leading component and folds in one batch per step', () => {
    const exact = pca.with({ nComponents: 2 }).fit(data).params
    const inc = incrementalPca.with({ nComponents: 2, batchSize: 20 }).fit(data).params
    expect(inc.explainedVariance[0]).toBeCloseTo(exact.explainedVariance[0] ?? 0, 2)
    expect(batchRanges(10, 4)).toEqual([
      [0, 4],
      [4, 8],
      [8, 10],
    ])
    const state = runAlgorithm(
      incrementalPcaAlgorithm(data.x, 2, [
        [0, 30],
        [30, 60],
      ]),
      { maxSteps: 2 },
    ).state
    expect(state.seen).toBe(60)
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('sparse PCA with no penalty is the principal axis; with one it zeroes loadings', () => {
    const dense = pca.with({ nComponents: 1 }).fit(data).params.components[0] ?? []
    const none = sparsePca.with({ nComponents: 1, alpha: 0 }).fit(data, { key: key(2) }).params.components[0] ?? []
    for (const [j, v] of dense.entries()) expect(Math.abs(none[j] ?? 0)).toBeCloseTo(Math.abs(v), 4)
    const sparse = sparsePca.with({ nComponents: 2, alpha: 0.7 }).fit(data, { key: key(2) })
    expect(sparse.params.components.flat().filter((v) => v === 0).length).toBeGreaterThan(0)
    expect(sparse.transform(data.x).shape).toEqual([60, 2])
    const state = runAlgorithm(sparsePcaAlgorithm(data.x, 1, { alpha: 0.5 }), { maxSteps: 1, key: key(0) }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})
