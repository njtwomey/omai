import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import {
  adaptiveConformalInference,
  conformalQuantile,
  crossConformalRegressor,
  intervalCoverage,
  splitConformalClassifier,
  splitConformalRegressor,
} from './conformal.ts'
import type { Fitted } from './estimator.ts'

function regressionData(seed: number, n: number) {
  const z = toFlat(normal(key(seed), [n, 2]))
  const e = toFlat(normal(foldIn(key(seed), 1), [n]))
  const x = Array.from({ length: n }, (_, i) => [z[2 * i] ?? 0, z[2 * i + 1] ?? 0])
  const y = x.map(([a = 0, b = 0], i) => a - 2 * b + (1 + Math.abs(a)) * (e[i] ?? 0))
  return { x, y }
}

describe('conformal coverage on exchangeable data', () => {
  it('split conformal intervals cover at least 1 - alpha on average', () => {
    for (const alpha of [0.1, 0.25]) {
      let covered = 0
      const reps = 40
      for (let r = 0; r < reps; r++) {
        const d = regressionData(100 + r, 160)
        const fitted = leastSquares().fit({ x: d.x.slice(0, 40), y: d.y.slice(0, 40) })
        const c = splitConformalRegressor(fitted, { x: d.x.slice(40, 80), y: d.y.slice(40, 80) }, { alpha })
        covered += intervalCoverage(c.predictInterval(d.x.slice(80)), d.y.slice(80)).coverage
      }
      expect(covered / reps).toBeGreaterThanOrEqual(1 - alpha - 0.02)
      expect(covered / reps).toBeLessThanOrEqual(1 - alpha + 1 / 41 + 0.03)
    }
  })

  it('CV+ intervals cover at least about 1 - alpha', () => {
    let covered = 0
    const reps = 15
    for (let r = 0; r < reps; r++) {
      const d = regressionData(300 + r, 140)
      const c = crossConformalRegressor(
        leastSquares(),
        { x: d.x.slice(0, 40), y: d.y.slice(0, 40) },
        { alpha: 0.2, cv: 5 },
      )
      covered += intervalCoverage(c.predictInterval(d.x.slice(40)), d.y.slice(40)).coverage
    }
    expect(covered / reps).toBeGreaterThanOrEqual(0.77)
  })

  it('classification sets cover the true label for every score', () => {
    // The true model: p(class 1 | u) = u; three classes by splitting the mass.
    const probaOf = (u: number): number[] => [0.6 * (1 - u), 0.4 * (1 - u) + 0.3 * u, 0.7 * u]
    const truth: Fitted = {
      estimator: 'truth',
      kind: 'classifier',
      params: {},
      classes: [0, 1, 2],
      predictProba: (x) => {
        const v = toFlat(
          fromStorage(
            Float64Array.from((x as number[][]).map((r) => r[0] ?? 0)),
            [(x as number[][]).length],
            'float64',
          ),
        )
        return fromStorage(Float64Array.from(v.flatMap(probaOf)), [v.length, 3], 'float64')
      },
    }
    const sample = (seed: number, n: number) => {
      const u = toFlat(uniform(key(seed), [n]))
      const v = toFlat(uniform(foldIn(key(seed), 1), [n]))
      const y = u.map((t, i) => {
        const p = probaOf(t)
        const c = v[i] ?? 0
        return c < (p[0] ?? 0) ? 0 : c < (p[0] ?? 0) + (p[1] ?? 0) ? 1 : 2
      })
      return { x: u.map((t) => [t]), y }
    }
    for (const score of ['lac', 'aps', 'raps'] as const) {
      let covered = 0
      let total = 0
      for (let r = 0; r < 20; r++) {
        const cal = sample(500 + r, 60)
        const test = sample(900 + r, 100)
        const c = splitConformalClassifier(truth, cal, { alpha: 0.2, score, lambda: 0.05, kReg: 1 })
        const sets = c.predictLabels(test.x)
        sets.forEach((s, i) => {
          if (s.includes(test.y[i] ?? -1)) covered++
          total++
        })
      }
      expect(covered / total).toBeGreaterThanOrEqual(0.78)
    }
  })

  it('the conformal quantile is at least the empirical (1 - alpha) quantile', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 1000 }), { minLength: 1, maxLength: 60 }),
        fc.integer({ min: 1, max: 9 }),
        (s, a) => {
          const alpha = a / 10
          const q = conformalQuantile(s, alpha)
          const inside = s.filter((v) => v <= q).length
          expect(inside / s.length).toBeGreaterThanOrEqual(1 - alpha - 1e-12)
        },
      ),
    )
  })
})

describe('adaptive conformal inference', () => {
  it('keeps the long-run error within the Gibbs-Candes bound under any sequence', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -1000, max: 1000 }), { minLength: 50, maxLength: 300 }),
        fc.integer({ min: 1, max: 50 }),
        (ys, g) => {
          const gamma = g / 1000
          const alpha = 0.1
          const r = adaptiveConformalInference(
            ys.map(() => 0),
            ys.map((v, t) => (v / 100) * (1 + (t % 7))),
            { alpha, gamma },
          )
          const err = r.errors.reduce((a, b) => a + b, 0) / r.errors.length
          expect(Math.abs(err - alpha)).toBeLessThanOrEqual(
            (Math.max(alpha, 1 - alpha) + gamma) / (gamma * ys.length) + 1e-12,
          )
        },
      ),
      { numRuns: 40 },
    )
  })
})
