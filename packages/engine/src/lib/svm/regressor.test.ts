import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { nuSvr, svr } from './regressor.ts'

const x = Array.from({ length: 24 }, (_, i) => [-3 + (6 * i) / 23])
const data = { x, y: x.map(([v]) => Math.sin(v as number) + 0.1 * Math.cos(7 * (v as number))) }

describe('svr', () => {
  it('runs as a session: an early model is available and the tube probe has a residual per row', () => {
    const est = svr.with({ gamma: 1, C: 5, epsilon: 0.05, tol: 1e-6 })
    const s = est.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 2 })
    const record = s.probe(2, 'model/support-vectors') as { kind: string; margins: number[] }
    expect(record.kind).toBe('margin')
    expect(record.margins).toHaveLength(24)
    expect(toFlat(s.fittedAt(2).predict(x))).toHaveLength(24)
    s.advance()
    expect(s.status).toBe('converged')
    const dual = metricSeries(s.run(), 'solver/dual').values
    for (let t = 1; t < dual.length; t++)
      expect(dual[t] as number).toBeGreaterThanOrEqual((dual[t - 1] as number) - 1e-9)
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap.every((g) => g > -1e-9)).toBe(true)
  })

  it('fits within the tube: points strictly inside carry no weight, and the fit tracks the curve', () => {
    const fitted = svr.with({ gamma: 1, C: 50, epsilon: 0.1, tol: 1e-6 }).fit(data)
    const residual = toFlat(fitted.predict(x)).map((v, i) => Math.abs((data.y[i] as number) - v))
    const inside = residual.filter((r) => r < 0.1 - 1e-4).length
    expect(fitted.params.supportIndex.length).toBeLessThanOrEqual(24 - inside)
    expect(fitted.score(data)).toBeGreaterThan(0.9)
  })

  it('has a nu form that finds the tube: a bigger nu is a narrower tube and more support vectors', () => {
    const narrow = nuSvr.with({ nu: 0.9, gamma: 1, C: 5, tol: 1e-6 }).fit(data)
    const wide = nuSvr.with({ nu: 0.2, gamma: 1, C: 5, tol: 1e-6 }).fit(data)
    expect(narrow.params.tube).toBeLessThanOrEqual(wide.params.tube + 1e-9)
    expect(narrow.params.supportIndex.length).toBeGreaterThanOrEqual(wide.params.supportIndex.length)
    expect(wide.params.supportIndex.length / 24).toBeGreaterThanOrEqual(0.2 - 1e-9)
  })

  it('revives from JSON, and rejects weights on the nu form', () => {
    const est = svr.with({ gamma: 1 })
    const fitted = est.fit(data)
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predict(x))).toEqual(toFlat(fitted.predict(x)))
    expect(() => nuSvr.fit({ ...data, weights: new Array<number>(24).fill(1) })).toThrow(/nu form/)
  })
})
