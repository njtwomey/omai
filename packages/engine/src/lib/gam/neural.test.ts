import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { additiveModel } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'
import { Log, metricSeries } from '../trace/index.ts'
import { additiveRegressor } from './neural.ts'

describe('additiveRegressor smoke', () => {
  it('fits a planted additive model and logs its shape functions', () => {
    const g = generate(additiveModel, { knobs: { n: 200, nFeatures: 3, nInformative: 3, noise: 0.3 }, key: key(1) })
    const data = { x: g.data.x, y: g.data.y }
    const t0 = Date.now()
    const fit = additiveRegressor.with({ epochs: 300, learningRate: 0.05, interiorKnots: 6 }).fit(data, { key: key(2) })
    console.log('ms', Date.now() - t0, 'r2', fit.score?.(data))
    const loss = metricSeries(fit.run, 'train/loss').values
    console.log(loss[0], loss.at(-1))
    console.log(fit.edfTerms, fit.termLabels, fit.exposition.map((e) => e.kind))
    const s = additiveRegressor.session(data, { key: key(2), logs: { 'additive/penalty': Log({ every: 100 }), 'predictor/0/term/0/curve': Log({ every: 100 }) } })
    s.advance({ steps: 100 })
    console.log(JSON.stringify(s.probe(50, 'predictor/0/term/0/curve')).slice(0, 200))
  })
})
