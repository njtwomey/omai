import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { gmm } from './gmm.ts'

describe('gmm', () => {
  const { data } = generate(blobs, { knobs: { n: 90 }, key: key(0) })

  it('fits, explains and predicts', () => {
    const fitted = gmm.fit(data, { key: key(1) })
    expect(fitted.params.means.length).toBe(3)
    const steps = fitted.exposition.find((r) => r.kind === 'mixtureSteps')
    expect(steps?.kind === 'mixtureSteps' && steps.objective.length > 1).toBe(true)
    const labels = toFlat(fitted.predict(data.x))
    expect(labels.length).toBe(90)
    expect(labels.every((l) => l >= 0 && l <= 2)).toBe(true)
    for (const row of toArray(fitted.predictProba(data.x)) as number[][])
      expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9)
  })

  it('runs as a session with a density probe', () => {
    const s = gmm.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 4 })
    expect(s.t).toBeGreaterThan(0)
    const surface = s.probe(1, 'model/density', { resolution: 8 }) as { kind: string }
    expect(surface.kind).toBe('decisionSurface')
  })
})
