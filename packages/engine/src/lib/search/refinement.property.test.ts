import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, validateAgainstSchema } from '../trace/index.ts'
import {
  conjunctions,
  coverageSimilarity,
  redundancyFilter,
  refinementBeamSearch,
  refinementBestFirst,
  refinementExhaustive,
  refinementExhaustiveAlgorithm,
  TopK,
} from './index.ts'

/** Random selectors over up to 24 records, with a random 0/1 target. */
const data = fc.integer({ min: 4, max: 24 }).chain((n) =>
  fc.record({
    target: fc.array(fc.integer({ min: 0, max: 1 }), { minLength: n, maxLength: n }),
    selectors: fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: n, maxLength: n }), {
      minLength: 1,
      maxLength: 7,
    }),
    attributes: fc.array(fc.integer({ min: 0, max: 3 }), { minLength: 7, maxLength: 7 }),
  }),
)

function problemOf(d: { target: number[]; selectors: number[][]; attributes: number[] }) {
  const N = d.target.length
  const P = d.target.reduce((a, b) => a + b, 0)
  const counts = (c: Uint8Array): [number, number] => {
    let n = 0
    let p = 0
    c.forEach((v, i) => {
      n += v
      p += v * (d.target[i] ?? 0)
    })
    return [n, p]
  }
  const selectors = d.selectors.map((covers, j) => ({ name: `s${j}`, attribute: `a${d.attributes[j] ?? 0}`, covers }))
  const quality = (c: Uint8Array): number => {
    const [n, p] = counts(c)
    return p / N - (n * P) / (N * N)
  }
  const estimate = (c: Uint8Array): number => (counts(c)[1] / N) * (1 - P / N)
  return {
    pruned: conjunctions(selectors, quality, { optimisticEstimate: estimate }),
    plain: conjunctions(selectors, quality),
  }
}

describe('refinement search properties', () => {
  it('pruned, best-first and unpruned exhaustive searches return the same top-k qualities', () => {
    fc.assert(
      fc.property(data, fc.integer({ min: 1, max: 6 }), fc.integer({ min: 1, max: 3 }), (d, k, maxDepth) => {
        const { pruned, plain } = problemOf(d)
        const truth = refinementExhaustive(plain, { k, maxDepth }).top.map((t) => t.quality)
        for (const r of [refinementExhaustive(pruned, { k, maxDepth }), refinementBestFirst(pruned, { k, maxDepth })]) {
          const got = r.top.map((t) => t.quality)
          expect(got.length).toBe(truth.length)
          got.forEach((q, i) => {
            expect(q).toBeCloseTo(truth[i] ?? 0, 12)
          })
        }
        // Results are sorted, within the depth, and beams find no more than exhaustive search does.
        const beam = refinementBeamSearch(plain, { k, maxDepth, width: 2 })
        for (const r of [beam, refinementExhaustive(plain, { k, maxDepth })]) {
          for (let i = 1; i < r.top.length; i++)
            expect(r.top[i]?.quality).toBeLessThanOrEqual(r.top[i - 1]?.quality ?? 0)
          for (const t of r.top) expect(t.depth).toBeLessThanOrEqual(maxDepth)
        }
        beam.top.forEach((t, i) => {
          expect(t.quality).toBeLessThanOrEqual((truth[i] ?? Number.POSITIVE_INFINITY) + 1e-12)
        })
      }),
      { numRuns: 150 },
    )
  })

  it('keeps no two results as similar as the redundancy threshold', () => {
    fc.assert(
      fc.property(data, fc.double({ min: 0.3, max: 1, noNaN: true }), (d, maxSimilarity) => {
        const { plain } = problemOf(d)
        const r = refinementExhaustive(plain, { k: 5, maxDepth: 2, maxSimilarity })
        for (let i = 0; i < r.top.length; i++) {
          for (let j = i + 1; j < r.top.length; j++) {
            const a = r.top[i]?.coverage ?? []
            const b = r.top[j]?.coverage ?? []
            expect(coverageSimilarity(a, b)).toBeLessThan(maxSimilarity)
          }
        }
        const filtered = redundancyFilter(r.top, { maxSimilarity })
        expect(filtered.length).toBe(r.top.length)
      }),
      { numRuns: 100 },
    )
  })
})

describe('refinement search behaviour', () => {
  it('respects quality floors, budgets, the root option and minimum support, and traces', () => {
    const letters = ['a', 'b', 'c', 'd']
    const p = {
      root: '',
      refine: (s: string) => letters.filter((l) => l > (s.at(-1) ?? '')).map((l) => s + l),
      quality: (s: string) => s.length,
    }
    expect(refinementExhaustive(p, { k: 3, maxDepth: 4, minQuality: 4 }).top.map((t) => t.pattern)).toEqual(['abcd'])
    expect(refinementExhaustive(p, { maxExpansions: 2, maxDepth: 4 }).status).toBe('budget')
    expect(refinementExhaustive({ ...p, quality: () => 1 }, { includeRoot: true, k: 1 }).top[0]?.pattern).toBe('')
    const top = new TopK<string>(2, { maxSimilarity: 0.5 })
    top.add({ pattern: 'x', quality: 2, depth: 1, coverage: [1, 1, 0] })
    expect(top.add({ pattern: 'y', quality: 1, depth: 1, coverage: [1, 1, 0] })).toBe(false)
    expect(() => new TopK(0)).toThrow()
    expect(() => refinementBeamSearch(p, { width: 0 })).toThrow()
    const support = conjunctions(
      [
        { name: 'u', covers: [1, 1, 0, 0] },
        { name: 'v', covers: [0, 1, 1, 0] },
      ],
      (c) => c.reduce((s, v) => s + v, 0),
      { minSupport: 2 },
    )
    expect(refinementExhaustive(support, { maxDepth: 2 }).top.map((t) => support.label?.(t.pattern))).toEqual([
      'u',
      'v',
    ])
    expect(() =>
      conjunctions(
        [
          { name: 'u', covers: [1] },
          { name: 'v', covers: [1, 0] },
        ],
        () => 0,
      ),
    ).toThrow()
    expect(() => coverageSimilarity([1], [1, 0])).toThrow()
    expect(coverageSimilarity([0, 0], [0, 0])).toBe(1)
    const rec = recorder()
    const algorithm = refinementExhaustiveAlgorithm(p)
    refinementExhaustive(p, { tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
  })
})
