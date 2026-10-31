import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { pca } from './pca.ts'
import {
  factorAnalysis,
  factorAnalysisAlgorithm,
  latentCovariance,
  probabilisticPca,
  rotateLoadings,
} from './probabilistic.ts'

const { data } = generate(blobs, { key: key(2), knobs: { n: 80, centers: 3, clusterStd: 1, nFeatures: 5 } })

describe('factor analysis', () => {
  it('the log likelihood rises with every iteration and the model covariance is positive definite', () => {
    const f = factorAnalysis.with({ nComponents: 2 }).fit(data, { logs: 'all' })
    const ll = metricSeries(f.run, 'solver/logLikelihood').values
    for (let t = 1; t < ll.length; t++) expect(ll[t] ?? 0).toBeGreaterThan((ll[t - 1] ?? 0) - 1e-9)
    expect(f.params.components.length).toBe(2)
    expect(f.params.noiseVariance.every((v) => v > 0)).toBe(true)
    expect(latentCovariance(f.params).length).toBe(5)
    expect(f.transform(data.x).shape).toEqual([80, 2])
    expect(toFlat(f.scoreSamples(data.x)).length).toBe(80)
  })

  it('a session probes the loadings at an early step; states round-trip', () => {
    const s = factorAnalysis.with({ nComponents: 2 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 1 })
    expect(s.fittedAt(1).params.components.length).toBe(2)
    const state = runAlgorithm(factorAnalysisAlgorithm(data.x, 2), { maxSteps: 3 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
    expect(() => factorAnalysisAlgorithm(data.x, 9)).toThrow()
  })

  it('rotations keep the loadings span and the communalities', () => {
    const l = [
      [0.8, 0.3],
      [0.7, 0.4],
      [0.2, 0.9],
      [0.3, 0.8],
    ]
    const r = rotateLoadings(l, 'varimax')
    for (const [i, row] of l.entries())
      expect(row.reduce((a, v) => a + v * v, 0)).toBeCloseTo(
        (r[i] ?? []).reduce((a, v) => a + v * v, 0),
        9,
      )
    const f = factorAnalysis.with({ nComponents: 2, rotation: 'varimax' }).fit(data)
    expect(f.params.components.length).toBe(2)
  })
})

describe('probabilistic PCA', () => {
  it('has PCA axes with loadings sqrt(ev - sigma^2) and isotropic noise', () => {
    const p = probabilisticPca.with({ nComponents: 2 }).fit(data).params
    const q = pca.with({ nComponents: 2 }).fit(data).params
    expect(p.noiseVariance.every((v) => Math.abs(v - q.noiseVariance) < 1e-9)).toBe(true)
    for (const [c, w] of p.components.entries())
      expect(Math.hypot(...w)).toBeCloseTo(Math.sqrt((q.explainedVariance[c] ?? 0) - q.noiseVariance), 9)
  })
})
