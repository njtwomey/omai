import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { oneClassSvm, svdd } from './one-class.ts'

const x = Array.from({ length: 30 }, (_, i) => {
  const r = 0.2 + ((i * 7) % 11) / 11
  return [r * Math.cos(i * 2.399), r * Math.sin(i * 2.399)]
})

describe.each([
  ['oneClassSvm', oneClassSvm.with({ nu: 0.2, gamma: 0.5, tol: 1e-6 })],
  ['svdd', svdd.with({ C: 0.2, gamma: 0.5, tol: 1e-6 })],
] as const)('%s', (_name, est) => {
  it('runs as a session with a probe at an early step, and closes its gap', () => {
    const s = est.session({ x }, { key: key(2), logs: 'all' })
    s.advance({ steps: 2 })
    const record = s.probe(2, 'model/support-vectors') as { kind: string; margins: number[] }
    expect(record.kind).toBe('margin')
    expect(record.margins).toHaveLength(30)
    s.advance()
    expect(s.status).toBe('converged')
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap.every((g) => g > -1e-9)).toBe(true)
    const dual = metricSeries(s.run(), 'solver/dual').values
    for (let t = 1; t < dual.length; t++)
      expect(dual[t] as number).toBeGreaterThanOrEqual((dual[t - 1] as number) - 1e-9)
  })

  it('puts the centre of the data inside and a far point outside, and bounds the outlier fraction', () => {
    const fitted = est.fit({ x })
    expect(
      toFlat(
        fitted.predict([
          [0, 0],
          [30, 30],
        ]),
      ),
    ).toEqual([1, -1])
    const outside = toFlat(fitted.predict(x)).filter((v) => v < 0).length
    expect(outside / 30).toBeLessThanOrEqual(0.2 + 1e-9)
    expect(fitted.decisionFunction(x).shape).toEqual([30])
  })
})

describe('svdd', () => {
  it('has a positive squared radius and encloses everything when C is 1', () => {
    const fitted = svdd.with({ C: 1, gamma: 0.5, tol: 1e-9 }).fit({ x })
    expect(fitted.params.radiusSquared as number).toBeGreaterThan(0)
    expect(toFlat(fitted.decisionFunction(x)).every((v) => v > -1e-6)).toBe(true)
  })

  it('refuses a C that cannot hold the unit mass', () => {
    expect(() => svdd.with({ C: 0.01 }).fit({ x })).toThrow(/at least/)
    expect(() => oneClassSvm.with({ nu: 0 }).fit({ x })).toThrow()
  })
})
