import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { full } from '../tensor/index.ts'
import { conformalQuantile, splitConformalClassifier, splitConformalRegressor, tuneRaps } from './conformal.ts'
import { rowsOf } from './validation.ts'

const x = Array.from({ length: 60 }, (_, i) => [Math.sin(i) * 2, Math.cos(1.3 * i) * 2])
const y = x.map(([a = 0, b = 0]) => (a + b > 0.5 ? 2 : a - b > 0 ? 1 : 0))

describe('conformal', () => {
  it('tunes RAPS on a held-out share and calibrates on the rest', () => {
    const f = gaussianNB().fit({ x: x.slice(0, 30), y: y.slice(0, 30) })
    const cal = { x: x.slice(30), y: y.slice(30) }
    const t = tuneRaps(f, cal, { alpha: 0.2, key: key(1) })
    expect(t.kReg).toBeGreaterThanOrEqual(1)
    expect(t.kReg).toBeLessThanOrEqual(3)
    expect([0.001, 0.01, 0.1, 0.2, 0.5]).toContain(t.lambda)
    expect(t.calibrationRows.length).toBe(24)
    const c = splitConformalClassifier(
      f,
      { x: t.calibrationRows.map((i) => cal.x[i] ?? []), y: t.calibrationRows.map((i) => cal.y[i] ?? 0) },
      { alpha: 0.2, score: 'raps', ...t },
    )
    expect(c.predictLabels(x.slice(0, 3)).length).toBe(3)
  })

  it('refuses impossible requests clearly', () => {
    expect(() => conformalQuantile([], 0.1)).toThrow()
    expect(() => conformalQuantile([1], 1.5)).toThrow()
    const f = leastSquares().fit({ x, y: y.map((v) => v - 5) })
    expect(() => splitConformalRegressor(f, { x, y: y.map((v) => v - 5) }, { score: 'gamma' })).toThrow(/positive/)
    expect(() => splitConformalRegressor(f, { x, y }, { score: 'normalized' })).toThrow(/scale/)
    const n = splitConformalRegressor(f, { x, y }, { score: 'normalized', scale: (z) => full([rowsOf(z)], 2) })
    expect(n.quantile).toBeGreaterThan(0)
  })
})
