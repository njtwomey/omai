import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { gaussianNB } from '../../testing/learn.ts'
import { toFlat } from '../tensor/index.ts'
import {
  betaCalibration,
  calibratedClassifierCV,
  calibrateFitted,
  isotonicCalibration,
  poolAdjacentViolators,
} from './calibration.ts'
import { revive, serialiseFitted } from './estimator.ts'

const values = fc.array(
  fc.integer({ min: -100, max: 100 }).map((v) => v / 10),
  { minLength: 1, maxLength: 30 },
)

describe('pool adjacent violators', () => {
  it('returns a non-decreasing fit that keeps the weighted mean and is a projection', () => {
    fc.assert(
      fc.property(values, (y) => {
        const w = y.map((_, i) => 1 + (i % 3))
        const f = poolAdjacentViolators(y, w)
        for (let i = 1; i < f.length; i++) expect(f[i] ?? 0).toBeGreaterThanOrEqual((f[i - 1] ?? 0) - 1e-12)
        const mean = (v: readonly number[]) =>
          v.reduce((s, t, i) => s + t * (w[i] ?? 0), 0) / w.reduce((a, b) => a + b, 0)
        expect(mean(f)).toBeCloseTo(mean(y), 9)
        const again = poolAdjacentViolators(f, w)
        again.forEach((v, i) => {
          expect(v).toBeCloseTo(f[i] ?? 0, 12)
        })
      }),
    )
  })

  it('isotonic calibration is monotone in the score and stays in [0, 1]', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: -50, max: 50 }), fc.boolean()), { minLength: 2, maxLength: 40 }),
        (pairs) => {
          const c = isotonicCalibration(
            pairs.map(([s]) => s / 10),
            pairs.map(([, b]) => (b ? 1 : 0)),
          )
          const grid = Array.from({ length: 21 }, (_, i) => -6 + (12 * i) / 20)
          const p = toFlat(c.predict(grid))
          for (let i = 1; i < p.length; i++) expect(p[i] ?? 0).toBeGreaterThanOrEqual((p[i - 1] ?? 0) - 1e-12)
          expect(p.every((v) => v >= 0 && v <= 1)).toBe(true)
        },
      ),
    )
  })
})

describe('calibrated classifiers', () => {
  const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(i)])
  const y = x.map(([a = 0, b = 0]) => (a + 0.3 * b > 0 ? 1 : 0))

  it('revive from JSON and give probabilities that sum to one', () => {
    const est = calibratedClassifierCV({ estimator: gaussianNB(), method: 'isotonic', cv: 3 })
    const f = est.fit({ x, y })
    const again = revive(est, JSON.parse(JSON.stringify(serialiseFitted(f))))
    expect(toFlat(again.predictProba(x))).toEqual(toFlat(f.predictProba(x)))
    const p = toFlat(f.predictProba(x))
    for (let i = 0; i < 30; i++) expect((p[2 * i] ?? 0) + (p[2 * i + 1] ?? 0)).toBeCloseTo(1, 12)
    expect(toFlat(est.with({ method: 'beta' }).fit({ x, y }).predict(x)).length).toBe(30)
  })

  it('calibrate a frozen classifier on held-out rows', () => {
    const fitted = gaussianNB().fit({ x: x.slice(0, 15), y: y.slice(0, 15) })
    const proba = calibrateFitted(fitted, { x: x.slice(15), y: y.slice(15) }, 'sigmoid')
    expect(proba(x).shape).toEqual([30, 2])
  })

  it('beta calibration keeps its coefficients non-negative', () => {
    const p = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4]
    const c = betaCalibration(p, [0, 0, 0, 1, 1, 1, 0, 1])
    expect(c.params.a >= 0 && c.params.b >= 0).toBe(true)
  })
})
