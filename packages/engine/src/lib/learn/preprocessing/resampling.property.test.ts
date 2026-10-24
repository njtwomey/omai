import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { adasyn, borderlineSmote, randomOverSampler, randomUnderSampler, smote } from './resampling.ts'

// Two Gaussian-ish clouds on a grid of values (no exact duplicates), the second the minority.
const clouds = fc
  .tuple(
    fc.array(fc.integer({ min: -300, max: 300 }), { minLength: 60, maxLength: 60 }),
    fc.integer({ min: 0, max: 99 }),
  )
  .map(([v, seed]) => {
    const x = Float64Array.from(v, (t, i) => t / 100 + (i % 2 === 0 ? 0 : 0.001 * i) + (i >= 40 ? 2 : 0))
    const y = Array.from({ length: 30 }, (_, i) => (i >= 20 ? 1 : 0))
    return { x: fromStorage(x, [30, 2], 'float64'), y, seed }
  })

function onSegment(p: readonly number[], a: readonly number[], b: readonly number[]): boolean {
  const d = a.map((v, j) => (b[j] ?? 0) - v)
  const len2 = d.reduce((s, v) => s + v * v, 0)
  if (len2 === 0) return p.every((v, j) => Math.abs(v - (a[j] ?? 0)) < 1e-9)
  const t = p.reduce((s, v, j) => s + (v - (a[j] ?? 0)) * (d[j] ?? 0), 0) / len2
  return t >= -1e-9 && t <= 1 + 1e-9 && p.every((v, j) => Math.abs(v - ((a[j] ?? 0) + t * (d[j] ?? 0))) < 1e-7)
}

describe('over-sampling', () => {
  it('SMOTE balances the classes with points on segments between minority samples', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const r = smote({ kNeighbors: 3 }).fitResample({ x, y }, { key: key(seed) })
        const labels = toFlat(r.y)
        expect(labels.filter((v) => v === 1).length).toBe(20)
        const original = rows(x)
        const minority = original.filter((_, i) => y[i] === 1)
        rows(r.x)
          .slice(30)
          .forEach((p, s) => {
            expect(labels[30 + s]).toBe(1)
            expect(minority.some((a) => minority.some((b) => onSegment(p, a, b)))).toBe(true)
          })
        expect(rows(smote({ kNeighbors: 3 }).fitResample({ x, y }, { key: key(seed) }).x)).toEqual(rows(r.x))
      }),
      { numRuns: 20 },
    )
  })

  it('ADASYN and borderline SMOTE add minority samples only', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        for (const s of [adasyn({ kNeighbors: 3 }), borderlineSmote({ kNeighbors: 3, mNeighbors: 6 })]) {
          let r: ReturnType<typeof s.fitResample>
          try {
            r = s.fitResample({ x, y }, { key: key(seed) })
          } catch {
            continue // ADASYN refuses when no minority sample has a majority neighbour.
          }
          const labels = toFlat(r.y)
          expect(labels.slice(0, 30)).toEqual(y)
          expect(labels.slice(30).every((v) => v === 1)).toBe(true)
          expect(Array.from(r.indices.slice(30)).every((i) => i === -1)).toBe(true)
        }
      }),
      { numRuns: 20 },
    )
  })

  it('random samplers balance by copying or dropping rows', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const over = randomOverSampler().fitResample({ x, y }, { key: key(seed) })
        expect(toFlat(over.y).filter((v) => v === 1).length).toBe(20)
        const original = rows(x)
        rows(over.x).forEach((r, i) => {
          expect(r).toEqual(original[over.indices[i] ?? -1])
        })
        const under = randomUnderSampler().fitResample({ x, y }, { key: key(seed) })
        expect(toFlat(under.y).filter((v) => v === 0).length).toBe(10)
        expect(new Set(under.indices).size).toBe(20)
      }),
      { numRuns: 20 },
    )
  })
})
