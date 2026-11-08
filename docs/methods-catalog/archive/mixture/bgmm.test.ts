import { generate } from 'omai-engine/data'
import { blobs } from 'omai-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { metricSeries, tensorSeries } from '../../lib/trace/index.ts'
import { bgmm } from './bgmm.ts'

describe('bgmm', () => {
  const { data } = generate(blobs, { knobs: { n: 90, centers: 3, clusterStd: 0.6 }, key: key(3) })

  it('fits, switches off components, and plays as a session of weights and a bound', () => {
    const fitted = bgmm.fit(data, { key: key(1) })
    expect(fitted.params.weights.length).toBe(8)
    expect(fitted.params.weights.filter((w) => w > 0.05).length).toBe(3)
    expect(toFlat(fitted.predict(data.x)).length).toBe(90)
    for (const row of toArray(fitted.predictProba(data.x)) as number[][])
      expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9)
    const s = bgmm.with({ init: 'random' }).session(data, { key: key(1) })
    s.advance({ steps: 6 })
    expect(s.t).toBe(6)
    expect((s.probe(6, 'model/centres') as number[][]).length).toBe(8)
    expect(s.probe(6, 'model/params')).toHaveProperty('covariances')
    expect((s.probe(6, 'model/assignments') as number[]).length).toBe(90)
    const weights = tensorSeries(s.run(), 'solver/weights')
    expect(weights[0]?.values?.length).toBe(8)
    expect(metricSeries(s.run(), 'solver/lowerBound').values.length).toBeGreaterThan(3)
  })
})
