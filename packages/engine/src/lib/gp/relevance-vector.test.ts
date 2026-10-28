import { describe, expect, it } from 'vitest'
import { rbfKernel } from '../kernels/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { relevanceVectorMachine } from './relevance-vector.ts'

const x = Array.from({ length: 30 }, (_, i) => [-3 + i / 5])
const y = x.map(([v], i) => Math.sin(v as number) + 0.04 * Math.sin(29 * i))
const data = { x, y }
const est = relevanceVectorMachine.with({ kernel: rbfKernel({ lengthScale: 1 }), tol: 1e-3 })

describe('relevanceVectorMachine', () => {
  it('runs as a session: the precisions diverge and basis functions are pruned, with the probe at an early step', () => {
    const s = est.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 3 })
    const record = s.probe(3, 'model/relevance-vectors') as { kind: string; support: number[]; margins: number[] }
    expect(record.kind).toBe('margin')
    expect(record.margins).toHaveLength(30)
    expect(toFlat(s.fittedAt(3).predict([[0]])).length).toBe(1)
    s.advance()
    expect(s.status).toBe('converged')
    const relevant = metricSeries(s.run(), 'solver/relevant').values
    expect(relevant[0] as number).toBeLessThanOrEqual(30)
    expect(relevant[relevant.length - 1] as number).toBeLessThanOrEqual(relevant[0] as number)
    for (let t = 1; t < relevant.length; t++)
      expect(relevant[t] as number).toBeLessThanOrEqual(relevant[t - 1] as number)
    expect(metricSeries(s.run(), 'solver/logEvidence').values.length).toBeGreaterThan(2)
  })

  it('is sparse, fits the curve and is more uncertain away from the data', () => {
    const fitted = est.fit(data, { explain: false })
    expect(fitted.params.relevant.length).toBeLessThan(30)
    expect(fitted.score(data)).toBeGreaterThan(0.98)
    const { std } = fitted.predictWithStd([[0], [12]])
    expect(toFlat(std)[1] as number).toBeGreaterThanOrEqual(toFlat(std)[0] as number)
    expect(fitted.params.vectors).toHaveLength(fitted.params.relevant.length)
  })

  it('revives from JSON', () => {
    const fitted = est.fit(data, { explain: false })
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predict([[1.1]]))).toEqual(toFlat(fitted.predict([[1.1]])))
  })

  it('rejects a non-positive threshold', () => {
    expect(() => est.with({ threshold: 0 }).fit(data)).toThrow()
  })
})
