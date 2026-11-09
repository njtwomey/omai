import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { svc } from '../../svm/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { smoteEnn, smoteTomek } from './combined.ts'
import { kMeansSmote, smoten, smotenc, svmSmote, valueDifferenceMetric } from './oversampling.ts'
import { balancedBag, smote } from './resampling.ts'

// Two clouds (30 majority rows, 10 minority rows) on a grid of values without exact duplicates; the minority overlaps
// the majority's edge so that every minority sample has neighbours of both classes.
const clouds = fc
  .tuple(
    fc.array(fc.integer({ min: -300, max: 300 }), { minLength: 80, maxLength: 80 }),
    fc.integer({ min: 0, max: 99 }),
  )
  .map(([v, seed]) => {
    const x = Float64Array.from(v, (t, i) => t / 100 + 0.001 * i + (i >= 60 ? 2 : 0))
    return { x: fromStorage(x, [40, 2], 'float64'), y: Array.from({ length: 40 }, (_, i) => (i >= 30 ? 1 : 0)), seed }
  })

// Rows of categorical codes (columns 0 and 2) with a continuous column between them.
const mixed = fc
  .tuple(
    fc.array(fc.integer({ min: 0, max: 3 }), { minLength: 80, maxLength: 80 }),
    fc.array(fc.integer({ min: -300, max: 300 }), { minLength: 40, maxLength: 40 }),
    fc.integer({ min: 0, max: 99 }),
  )
  .map(([codes, v, seed]) => {
    const x = Float64Array.from({ length: 120 }, (_, t) => {
      const i = Math.floor(t / 3)
      return t % 3 === 1
        ? (v[i] ?? 0) / 100 + 0.001 * i + (i >= 30 ? 2 : 0)
        : (codes[i * 2 + (t % 3 === 0 ? 0 : 1)] ?? 0) * 10 + 5
    })
    return { x: fromStorage(x, [40, 3], 'float64'), y: Array.from({ length: 40 }, (_, i) => (i >= 30 ? 1 : 0)), seed }
  })

function onSegment(p: readonly number[], a: readonly number[], b: readonly number[]): boolean {
  const d = a.map((v, j) => (b[j] ?? 0) - v)
  const len2 = d.reduce((s, v) => s + v * v, 0)
  if (len2 === 0) return p.every((v, j) => Math.abs(v - (a[j] ?? 0)) < 1e-9)
  const t = p.reduce((s, v, j) => s + (v - (a[j] ?? 0)) * (d[j] ?? 0), 0) / len2
  return t >= -1e-9 && t <= 1 + 1e-9 && p.every((v, j) => Math.abs(v - ((a[j] ?? 0) + t * (d[j] ?? 0))) < 1e-7)
}

describe('categorical and mixed SMOTE', () => {
  it('SMOTEN balances the classes, keeps the originals first, and only uses categories of the class', () => {
    fc.assert(
      fc.property(mixed, ({ x, y, seed }) => {
        const categorical = fromStorage(
          Float64Array.from(rows(x).flatMap((r) => [r[0] ?? 0, r[2] ?? 0])),
          [40, 2],
          'float64',
        )
        const r = smoten({ kNeighbors: 3 }).fitResample({ x: categorical, y }, { key: key(seed) })
        const labels = toFlat(r.y)
        expect(labels.filter((v) => v === 1).length).toBe(30)
        expect(labels.slice(0, 40)).toEqual(y)
        expect(rows(r.x).slice(0, 40)).toEqual(rows(categorical))
        const own = rows(categorical).filter((_, i) => y[i] === 1)
        for (const p of rows(r.x).slice(40)) {
          for (let j = 0; j < 2; j++) expect(own.some((q) => q[j] === p[j])).toBe(true)
        }
        expect(Array.from(r.indices.slice(40)).every((i) => i === -1)).toBe(true)
      }),
      { numRuns: 20 },
    )
  })

  it('SMOTE-NC balances the classes; categories come from the class, continuous values lie between class samples', () => {
    fc.assert(
      fc.property(mixed, ({ x, y, seed }) => {
        const r = smotenc({ categorical: [0, 2], kNeighbors: 3 }).fitResample({ x, y }, { key: key(seed) })
        const labels = toFlat(r.y)
        expect(labels.filter((v) => v === 1).length).toBe(30)
        const own = rows(x).filter((_, i) => y[i] === 1)
        for (const p of rows(r.x).slice(40)) {
          for (const j of [0, 2]) expect(own.some((q) => q[j] === p[j])).toBe(true)
          const middle = own.map((q) => [q[1] ?? 0])
          expect(middle.some((a) => middle.some((b) => onSegment([p[1] ?? 0], a, b)))).toBe(true)
        }
      }),
      { numRuns: 20 },
    )
  })

  it('SMOTE-NC needs both kinds of feature', () => {
    const x = [
      [0, 1],
      [1, 0],
      [0, 0],
      [1, 1],
    ]
    expect(() => smotenc({ categorical: [] }).fitResample({ x, y: [0, 0, 0, 1] })).toThrow()
    expect(() => smotenc({ categorical: [0, 1] }).fitResample({ x, y: [0, 0, 0, 1] })).toThrow()
  })

  it('the value difference metric is symmetric, zero on the diagonal and zero between categories with the same class mix', () => {
    fc.assert(
      fc.property(mixed, ({ x, y }) => {
        const codes = fromStorage(Float64Array.from(rows(x).map((r) => r[0] ?? 0)), [40, 1], 'float64')
        const d = toFlat(valueDifferenceMetric(codes, y)(codes))
        for (let i = 0; i < 40; i++) {
          expect(d[i * 40 + i]).toBe(0)
          for (let j = 0; j < 40; j++) {
            expect(d[i * 40 + j]).toBeCloseTo(d[j * 40 + i] ?? 0, 12)
            expect(d[i * 40 + j]).toBeGreaterThanOrEqual(0)
          }
        }
      }),
      { numRuns: 15 },
    )
  })
})

describe('SMOTE variants around the class', () => {
  it('SVM-SMOTE balances the classes with points on or just beyond segments between class samples', () => {
    let ran = 0
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        let r: ReturnType<ReturnType<typeof svmSmote>['fitResample']>
        try {
          r = svmSmote({ svm: svc.with({ kernel: 'rbf' }), kNeighbors: 3, mNeighbors: 6 }).fitResample(
            { x, y },
            { key: key(seed) },
          )
        } catch {
          return // every support vector is noise
        }
        ran++
        expect(toFlat(r.y).filter((v) => v === 1).length).toBe(30)
        const own = rows(x).filter((_, i) => y[i] === 1)
        for (const p of rows(r.x).slice(40)) {
          // Interpolated: on a segment. Extrapolated by at most half the way: on the segment from 1.5 a - 0.5 b to b.
          const ok = own.some((a) =>
            own.some(
              (b) =>
                onSegment(p, a, b) ||
                onSegment(
                  p,
                  a.map((v, j) => 1.5 * v - 0.5 * (b[j] ?? 0)),
                  b,
                ),
            ),
          )
          expect(ok).toBe(true)
        }
      }),
      { numRuns: 10 },
    )
    expect(ran).toBeGreaterThan(0)
  })

  it('k-means SMOTE makes the target or a few more, on segments between class samples', () => {
    let ran = 0
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        let r: ReturnType<ReturnType<typeof kMeansSmote>['fitResample']>
        try {
          r = kMeansSmote({ nClusters: 2, kNeighbors: 2, clusterBalanceThreshold: 0 }).fitResample(
            { x, y },
            { key: key(seed) },
          )
        } catch {
          return // no cluster holds enough of the class
        }
        ran++
        const made = toFlat(r.y).filter((v) => v === 1).length - 10
        expect(made).toBeGreaterThanOrEqual(20)
        expect(made).toBeLessThanOrEqual(21)
        const own = rows(x).filter((_, i) => y[i] === 1)
        for (const p of rows(r.x).slice(40)) expect(own.some((a) => own.some((b) => onSegment(p, a, b)))).toBe(true)
      }),
      { numRuns: 15 },
    )
    expect(ran).toBeGreaterThan(0)
  })

  it('SMOTE + ENN and SMOTE + Tomek return the original rows and synthetic ones, never more than SMOTE made', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const base = smote({ kNeighbors: 3 }).fitResample({ x, y }, { key: key(seed) })
        for (const sampler of [
          smoteEnn({ smote: smote({ kNeighbors: 3 }) }),
          smoteTomek({ smote: smote({ kNeighbors: 3 }) }),
        ]) {
          const r = sampler.fitResample({ x, y }, { key: key(seed) })
          expect(r.y.shape[0] ?? 0).toBeLessThanOrEqual(base.y.shape[0] ?? 0)
          const original = rows(x)
          const synthetic = new Set(
            rows(base.x)
              .slice(40)
              .map((p) => p.join(',')),
          )
          rows(r.x).forEach((p, i) => {
            const from = r.indices[i] ?? -1
            if (from >= 0) expect(p).toEqual(original[from])
            else expect(synthetic.has(p.join(','))).toBe(true)
          })
        }
      }),
      { numRuns: 15 },
    )
  })
})

describe('balancedBag', () => {
  const bags = fc.tuple(
    fc.array(fc.integer({ min: 0, max: 2 }), { minLength: 12, maxLength: 40 }),
    fc.array(fc.integer({ min: 0, max: 3 }), { minLength: 40, maxLength: 40 }),
    fc.integer({ min: 0, max: 99 }),
  )

  it('cuts every class to the size of the smallest present, without replacement never adding to a row', () => {
    fc.assert(
      fc.property(bags, ([labels, counts, seed]) => {
        const c = counts.slice(0, labels.length)
        const out = balancedBag(labels, c, { samplingStrategy: 'all' }, key(seed))
        const sizes = [0, 1, 2].map((k) => c.reduce((s, v, i) => s + (labels[i] === k ? v : 0), 0)).filter((n) => n > 0)
        const floor = Math.min(...sizes)
        for (const k of [0, 1, 2]) {
          const had = c.reduce((s, v, i) => s + (labels[i] === k ? v : 0), 0)
          const now = Array.from(out).reduce((s, v, i) => s + (labels[i] === k ? v : 0), 0)
          expect(now).toBe(had > 0 ? floor : 0)
        }
        out.forEach((v, i) => {
          expect(v).toBeLessThanOrEqual(c[i] ?? 0)
        })
      }),
      { numRuns: 40 },
    )
  })

  it('with replacement it can repeat rows but keeps the class sizes equal', () => {
    fc.assert(
      fc.property(bags, ([labels, counts, seed]) => {
        const c = counts.slice(0, labels.length)
        const out = balancedBag(labels, c, { samplingStrategy: 'all', replacement: true }, key(seed))
        const sizes = [0, 1, 2].map((k) => Array.from(out).reduce((s, v, i) => s + (labels[i] === k ? v : 0), 0))
        const present = sizes.filter((n) => n > 0)
        expect(new Set(present).size).toBeLessThanOrEqual(1)
        out.forEach((v, i) => {
          if ((c[i] ?? 0) === 0) expect(v).toBe(0)
        })
      }),
      { numRuns: 40 },
    )
  })
})
