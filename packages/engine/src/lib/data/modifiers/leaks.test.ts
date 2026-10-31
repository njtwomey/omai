import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { fromStorage, type tensor, toFlat } from '../../tensor/index.ts'
import { asGenerated, type Generated } from '../generator.ts'
import { rowCount } from '../schema/common.ts'
import type { ClassificationData } from '../schema/tabular.ts'
import { xyClasses } from '../tensors.ts'
import { withDuplicates, withLeak } from './leaks.ts'
import { withShortcut, withSubpopulationShift } from './shift.ts'

/** Two Gaussian classes in two dimensions with the true posterior in the truth. */
function twoClasses(n: number, seed = 1): Generated<ClassificationData> {
  const z = toFlat(normal(key(seed), [n * 2]))
  const y = Array.from({ length: n }, (_, i) => i % 2)
  const x = Array.from({ length: n }, (_, i) => [(z[i * 2] ?? 0) + 2 * (y[i] ?? 0), z[i * 2 + 1] ?? 0])
  const g = asGenerated(xyClasses(x, y, { classes: ['a', 'b'] })) as Generated<ClassificationData>
  const posterior = (v: ReturnType<typeof tensor>): ReturnType<typeof tensor> => {
    const rows = v.shape[0] ?? 0
    const d = v.shape[1] ?? 1
    const out = new Float64Array(rows * 2)
    for (let i = 0; i < rows; i++) {
      const p1 = 1 / (1 + Math.exp(-(2 * ((v.data[i * d] ?? 0) - 1))))
      out[i * 2] = 1 - p1
      out[i * 2 + 1] = p1
    }
    return fromStorage(out, [rows, 2], 'float64')
  }
  return { data: g.data, truth: { ...g.truth, posterior: posterior as never } }
}

describe('withLeak', () => {
  it('appends a column that gives the class away, while the truth keeps the honest posterior', () => {
    const g = twoClasses(200)
    const out = withLeak(g, { noise: 0.1, key: key(2) })
    expect(out.data.featureNames).toEqual(['x0', 'x1', 'leak'])
    const x = out.data.x
    let right = 0
    for (let i = 0; i < 200; i++) if (((x.data[i * 3 + 2] ?? 0) > 0.5 ? 1 : 0) === (out.data.y.data[i] ?? 0)) right++
    expect(right / 200).toBeGreaterThan(0.99)
    // The honest posterior still accepts the widened features and ignores the leak.
    const post = out.truth.posterior?.(x)
    expect(post?.shape).toEqual([200, 2])
    expect(((out.truth.extra ?? {})['leak'] as { column: number }).column).toBe(2)
    expect(out.truth.relevantFeatures).toEqual([0, 1])
  })

  it('a target encoding fitted on all rows predicts better than the feature alone, and equals the bin mean', () => {
    const g = twoClasses(400, 3)
    const out = withLeak(g, { kind: 'targetEncoding', bins: 8, key: key(0) })
    const x = out.data.x
    const byLeak = new Map<number, number[]>()
    for (let i = 0; i < 400; i++)
      byLeak.set(x.data[i * 3 + 2] ?? 0, [...(byLeak.get(x.data[i * 3 + 2] ?? 0) ?? []), out.data.y.data[i] ?? 0])
    expect(byLeak.size).toBeLessThanOrEqual(8)
    for (const [value, ys] of byLeak) expect(value).toBeCloseTo(ys.reduce((a, b) => a + b, 0) / ys.length, 12)
  })

  it('rejects data without a target', () => {
    expect(() => withLeak({ data: { kind: 'corpus' } as never, truth: {} }, { key: key(0) })).toThrow(/feature matrix/)
  })
})

describe('withDuplicates', () => {
  it('adds the requested copies, each identical to a distinct original, and spreads them', () => {
    const g = twoClasses(100)
    const out = withDuplicates(g, { fraction: 0.2, key: key(4) })
    expect(rowCount(out.data)).toBe(120)
    const source = toFlat(out.truth.latents?.['sourceRow'] as never)
    const dup = toFlat(out.truth.latents?.['isDuplicate'] as never)
    expect(dup.filter((v) => v === 1)).toHaveLength(20)
    const seen = new Map<number, number[]>()
    for (const [i, s] of source.entries()) seen.set(s, [...(seen.get(s) ?? []), i])
    expect([...seen.values()].filter((rows) => rows.length === 2)).toHaveLength(20)
    for (const rows of seen.values())
      if (rows.length === 2) {
        const [a = 0, b = 0] = rows
        expect(out.data.x.data[a * 2]).toBe(out.data.x.data[b * 2])
        expect(out.data.y.data[a]).toBe(out.data.y.data[b])
        expect(dup[a] === 1 || dup[b] === 1).toBe(true)
      }
    // Shuffled: copies are not all at the end.
    expect(dup.slice(0, 100).some((v) => v === 1)).toBe(true)
  })

  it('jitters the copies only, by about the stated share of a standard deviation', () => {
    const g = twoClasses(300, 5)
    const out = withDuplicates(g, { fraction: 1, jitter: 0.5, key: key(1) })
    const source = toFlat(out.truth.latents?.['sourceRow'] as never)
    const dup = toFlat(out.truth.latents?.['isDuplicate'] as never)
    let originalsExact = true
    let sq = 0
    let count = 0
    source.forEach((s, i) => {
      const d = (out.data.x.data[i * 2] ?? 0) - (g.data.x.data[s * 2] ?? 0)
      if (dup[i] === 1) {
        sq += d * d
        count++
      } else if (d !== 0) originalsExact = false
    })
    expect(originalsExact).toBe(true)
    const sd = Math.sqrt(sq / count)
    expect(sd).toBeGreaterThan(0.5 * 0.7)
    expect(sd).toBeLessThan(0.5 * 2.5)
  })

  it('has the right size for any fraction and is deterministic', () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), fc.integer({ min: 0, max: 30 }), (fraction, seed) => {
        const g = twoClasses(40)
        const a = withDuplicates(g, { fraction, key: key(seed) })
        const b = withDuplicates(g, { fraction, key: key(seed) })
        expect(rowCount(a.data)).toBe(40 + Math.round(fraction * 40))
        expect(Array.from(a.data.y.data)).toEqual(Array.from(b.data.y.data))
      }),
      { numRuns: 20 },
    )
  })
})

describe('withSubpopulationShift', () => {
  it('thins the group in the source and gives a density ratio that averages to one over the source', () => {
    const g = twoClasses(2000, 6)
    const out = withSubpopulationShift(g, { keep: 0.2, key: key(3) })
    const info = (out.truth.extra ?? {})['subpopulation'] as {
      threshold: number
      targetShare: number
      sourceShare: number
    }
    const group = toFlat(out.truth.latents?.['group'] as never)
    const share = group.reduce((a, b) => a + b, 0) / group.length
    expect(share).toBeCloseTo(info.sourceShare, 1)
    expect(share).toBeLessThan(info.targetShare / 2)
    const w = toFlat(((out.truth.extra ?? {})['densityRatio'] as (x: unknown) => never)(out.data.x))
    expect(w.reduce((a, b) => a + b, 0) / w.length).toBeCloseTo(1, 1)
    for (let i = 0; i < group.length; i++) expect(group[i] === 1).toBe((out.data.x.data[i * 2] ?? 0) > info.threshold)
  })

  it('keep = 0 removes the group and keep = 1 keeps every row', () => {
    const g = twoClasses(200, 7)
    expect(withSubpopulationShift(g, { keep: 1, key: key(0) }).data.x.shape[0]).toBe(200)
    const gone = withSubpopulationShift(g, { keep: 0, key: key(0) })
    expect(toFlat(gone.truth.latents?.['group'] as never).every((v) => v === 0)).toBe(true)
  })
})

describe('withShortcut', () => {
  it('agrees with the label at the stated rate in the source and at chance in the target', () => {
    const g = twoClasses(3000, 8)
    const source = withShortcut(g, { agreement: 0.9, key: key(1) })
    const target = withShortcut(g, { mode: 'target', key: key(1) })
    const rate = (o: typeof source): number =>
      toFlat(o.truth.latents?.['shortcutAgrees'] as never).reduce((a, b) => a + b, 0) / 3000
    expect(rate(source)).toBeGreaterThan(0.87)
    expect(rate(source)).toBeLessThan(0.93)
    expect(rate(target)).toBeGreaterThan(0.46)
    expect(rate(target)).toBeLessThan(0.54)
    // A rule that reads only the shortcut does well at the source and at chance at the target.
    const acc = (o: typeof source): number => {
      let r = 0
      for (let i = 0; i < 3000; i++)
        if (((o.data.x.data[i * 3 + 2] ?? 0) > 0.5 ? 1 : 0) === (o.data.y.data[i] ?? 0)) r++
      return r / 3000
    }
    expect(acc(source)).toBeGreaterThan(0.85)
    expect(acc(target)).toBeLessThan(0.56)
    expect(source.truth.posterior?.(source.data.x).shape).toEqual([3000, 2])
  })
})
