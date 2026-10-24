import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { exp, fromStorage, type Tensor } from '../../tensor/index.ts'
import type { Estimator } from '../estimator.ts'
import { matrixView } from '../validation.ts'
import {
  maxAbsScaler,
  minMaxScaler,
  powerTransformer,
  quantileTransformer,
  robustScaler,
  standardScaler,
} from './scalers.ts'

type Invertible = { readonly transform: (x: Tensor) => Tensor; readonly inverseTransform: (x: Tensor) => Tensor }

function expectRoundTrip(est: Estimator, x: Tensor, tol: number): void {
  const f = est.fit({ x }) as unknown as Invertible
  const back = rows(f.inverseTransform(f.transform(x)))
  rows(x).forEach((r, i) => {
    r.forEach((v, j) => {
      expect(Math.abs((back[i]?.[j] ?? Number.NaN) - v)).toBeLessThanOrEqual(tol * (1 + Math.abs(v)))
    })
  })
}

// Values on a 0.01 grid: realistic data, without the subnormal spacings that no reference handles.
const matrices = fc.array(fc.integer({ min: -500, max: 500 }), { minLength: 36, maxLength: 36 }).map((v) =>
  fromStorage(
    Float64Array.from(v, (t) => t / 100),
    [12, 3],
    'float64',
  ),
)

describe('scalers invert', () => {
  it('standard, min-max, max-abs and robust scalers invert to rounding', () => {
    fc.assert(
      fc.property(matrices, (x) => {
        for (const est of [standardScaler(), minMaxScaler({ featureRange: [-2, 3] }), maxAbsScaler(), robustScaler()]) {
          expectRoundTrip(est, x, 1e-9)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('power transformers invert', () => {
    fc.assert(
      fc.property(matrices, (x) => {
        expectRoundTrip(powerTransformer(), x, 1e-6)
        expectRoundTrip(powerTransformer({ method: 'box-cox' }), exp(x) as Tensor, 1e-6)
      }),
      { numRuns: 10 },
    )
  })

  it('the quantile transformer inverts on the training data and is monotone', () => {
    fc.assert(
      fc.property(matrices, (x) => {
        expectRoundTrip(quantileTransformer({ nQuantiles: 12 }), x, 1e-9)
        const f = quantileTransformer({ nQuantiles: 6 }).fit({ x })
        const t = matrixView(f.transform(x), 'test')
        const m = matrixView(x, 'test')
        for (let j = 0; j < 3; j++) {
          for (let a = 0; a < m.n; a++) {
            for (let b = 0; b < m.n; b++) {
              if ((m.data[a * 3 + j] ?? 0) < (m.data[b * 3 + j] ?? 0)) {
                expect(t.data[a * 3 + j] ?? 0).toBeLessThanOrEqual(t.data[b * 3 + j] ?? 0)
              }
            }
          }
        }
      }),
      { numRuns: 20 },
    )
  })

  it('standardised training data has zero mean and unit variance', () => {
    fc.assert(
      fc.property(matrices, (x) => {
        const t = matrixView(standardScaler().fit({ x }).transform(x), 'test')
        for (let j = 0; j < 3; j++) {
          let s = 0
          let ss = 0
          for (let i = 0; i < t.n; i++) {
            s += t.data[i * 3 + j] ?? 0
            ss += (t.data[i * 3 + j] ?? 0) ** 2
          }
          expect(Math.abs(s / t.n)).toBeLessThan(1e-9)
          expect(ss / t.n === 0 || Math.abs(ss / t.n - 1) < 1e-9).toBe(true)
        }
      }),
      { numRuns: 30 },
    )
  })
})
