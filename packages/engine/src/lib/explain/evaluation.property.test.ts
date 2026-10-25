import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { fromStorage, type Tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  completenessGap,
  complexity,
  deletionCurve,
  explainerAgreement,
  explanationSimilarity,
  faithfulnessCorrelation,
  localisation,
  localLipschitz,
  monotonicity,
  removeAndRetrain,
  sensitivity,
} from './evaluation.ts'

const w = [3, -1, 0.5, 2]
const linear = (x: Tensor): Tensor => {
  const v = toFlat(x)
  const n = x.shape[0] ?? 0
  return fromStorage(
    Float64Array.from({ length: n }, (_, i) => w.reduce((s, c, j) => s + c * (v[i * 4 + j] ?? 0), 0)),
    [n],
    'float64',
  )
}
const x = Array.from({ length: 6 }, (_, i) => [1 + (i % 3), 0.5 + i / 4, 2 - i / 3, 1])
const exact = x.map((r) => r.map((v, j) => (w[j] ?? 0) * v))
const reversed = exact.map((r) => r.map((v) => -v))

describe('evaluation of explanations', () => {
  it('exact attributions delete faster, insert faster and correlate perfectly; reversed ones do worse', () => {
    const good = deletionCurve(linear, x, exact)
    const bad = deletionCurve(linear, x, reversed)
    expect((good.auc[0] ?? 0) < (bad.auc[0] ?? 0)).toBe(true)
    const ins = deletionCurve(linear, x, exact, { mode: 'insertion' })
    expect(ins.curves[0]?.at(-1)).toBeCloseTo(good.curves[0]?.[0] ?? 0, 12)
    expect(ins.auc[0] ?? 0).toBeGreaterThan(deletionCurve(linear, x, reversed, { mode: 'insertion' }).auc[0] ?? 0)
    for (const c of faithfulnessCorrelation(linear, x, exact, { key: key(1) })) expect(c).toBeCloseTo(1, 10)
    // Contributions of mixed sign break monotonicity; positive ones keep it.
    const positive = x.map((r) => r.map((v, j) => Math.abs((w[j] ?? 0) * v)))
    const absModel = (t: Tensor): Tensor => {
      const v = toFlat(t)
      const n = t.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => w.reduce((s, c, j) => s + Math.abs(c) * (v[i * 4 + j] ?? 0), 0)),
        [n],
        'float64',
      )
    }
    expect(monotonicity(absModel, x, positive).every((v) => v === 1)).toBe(true)
    expect(monotonicity(linear, x, exact).some((v) => v < 1)).toBe(true)
    expect(
      completenessGap(exact, toFlat(linear(fromStorage(Float64Array.from(x.flat()), [6, 4], 'float64'))), [0]).every(
        (g) => g < 1e-12,
      ),
    ).toBe(true)
  })

  it('complexity, similarity and localisation stay in range', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 5, maxLength: 5 }),
        fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 5, maxLength: 5 }),
        (a, b) => {
          const c = complexity(a)
          expect(c.sparseness).toBeGreaterThanOrEqual(-1e-12)
          expect(c.sparseness).toBeLessThanOrEqual(1)
          expect(c.entropy).toBeLessThanOrEqual(Math.log(5) + 1e-12)
          const s = explanationSimilarity(a, b)
          expect(Math.abs(s)).toBeLessThanOrEqual(1 + 1e-12)
          expect(explanationSimilarity(b, a)).toBeCloseTo(s, 12)
          const l = localisation(a, [true, false, true, false, false])
          expect(l.mass).toBeGreaterThanOrEqual(0)
          expect(l.mass).toBeLessThanOrEqual(1 + 1e-12)
        },
      ),
      { numRuns: 60 },
    )
    const agree = explainerAgreement([exact, reversed, exact])
    expect(agree[0]?.[2]).toBeCloseTo(1, 12)
    expect(agree[0]?.[1]).toBeCloseTo(1, 12)
    expect(explainerAgreement([exact, reversed], { metric: 'sign' })[0]?.[1]).toBe(0)
  })

  it('a gradient explanation of a linear model is perfectly robust', () => {
    const grad = (v: unknown): number[][] => (toArray(v as Tensor) as number[][]).map(() => [...w])
    const s = sensitivity(grad, x.slice(0, 2), { key: key(2) })
    expect(s.max.every((v) => v === 0)).toBe(true)
    expect(localLipschitz(grad, x.slice(0, 2), { key: key(3) }).every((v) => v === 0)).toBe(true)
  })

  it('ROAR: removing the most relevant features hurts more than removing the least', () => {
    const rows = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(i * 1.7), Math.sin(i * 0.3), (i % 5) / 5])
    const y = rows.map((r) => w.reduce((s, c, j) => s + c * (r[j] ?? 0), 0))
    const attr = rows.map((r) => r.map((v, j) => Math.abs((w[j] ?? 0) * v)))
    const data = { x: rows, y }
    const most = removeAndRetrain(leastSquares(), data, data, attr, attr, { fractions: [0, 0.5] })
    const least = removeAndRetrain(leastSquares(), data, data, attr, attr, { fractions: [0, 0.5], leastFirst: true })
    expect(most.scores[0]).toBeCloseTo(1, 9)
    expect(most.scores[1] ?? 1).toBeLessThan(least.scores[1] ?? 0)
  })
})
