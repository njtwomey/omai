import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { kmeans } from './kmeans.ts'

describe('kmeans', () => {
  const { data } = generate(blobs, { knobs: { n: 60 }, key: key(0) })

  it('fits, explains and predicts', () => {
    const fitted = kmeans.fit(data, { key: key(1) })
    expect(fitted.params.centres.length).toBe(3)
    const steps = fitted.exposition.find((r) => r.kind === 'clusteringSteps')
    expect(steps?.kind === 'clusteringSteps' && steps.assignments.length > 1).toBe(true)
    expect(toFlat(fitted.predict(data.x)).length).toBe(60)
  })

  it('runs as a session with probes', () => {
    const s = kmeans.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 4 })
    expect(s.t).toBeGreaterThan(0)
    expect((s.probe(1, 'model/centres') as unknown[]).length).toBe(3)
  })
})
