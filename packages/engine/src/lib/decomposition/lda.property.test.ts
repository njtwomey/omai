import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray } from '../tensor/index.ts'
import { latentDirichletAllocation } from './lda.ts'

const corpus = fc
  .record({ n: fc.integer({ min: 2, max: 6 }), v: fc.integer({ min: 2, max: 6 }), k: fc.integer({ min: 1, max: 3 }) })
  .chain(({ n, v, k }) =>
    fc.record({
      k: fc.constant(k),
      x: fc.array(fc.array(fc.integer({ min: 0, max: 6 }), { minLength: v, maxLength: v }), {
        minLength: n,
        maxLength: n,
      }),
      start: fc.array(fc.array(fc.double({ min: 0.5, max: 1.5, noNaN: true }), { minLength: v, maxLength: v }), {
        minLength: k,
        maxLength: k,
      }),
    }),
  )

const fit = (x: number[][], start: number[][], k: number) =>
  latentDirichletAllocation
    .with({ nComponents: k, maxIter: 4, componentsInit: start, docInit: 'constant', meanChangeTol: 1e-8 })
    .fit({ x })

describe('latent Dirichlet allocation properties', () => {
  it('topic-word parameters stay above the prior and documents map to the simplex, whatever the counts', () => {
    fc.assert(
      fc.property(corpus, ({ x, start, k }) => {
        const f = fit(x, start, k)
        for (const c of f.params.components.flat()) expect(c).toBeGreaterThanOrEqual(1 / k - 1e-12)
        for (const row of toArray(f.transform(x)) as number[][]) {
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
          for (const p of row) expect(p).toBeGreaterThan(0)
        }
        expect(Number.isFinite(f.params.bound)).toBe(true)
      }),
      { numRuns: 40 },
    )
  })

  it('is invariant to the order of the documents and equivariant to the order of the topics (batch)', () => {
    fc.assert(
      fc.property(corpus, ({ x, start, k }) => {
        const a = fit(x, start, k)
        const reversedDocs = fit([...x].reverse(), start, k)
        const reversedTopics = fit(x, [...start].reverse(), k)
        a.params.components.forEach((row, q) => {
          row.forEach((c, w) => {
            expect(reversedDocs.params.components[q]?.[w] ?? Number.NaN).toBeCloseTo(c, 8)
            expect(reversedTopics.params.components[k - 1 - q]?.[w] ?? Number.NaN).toBeCloseTo(c, 8)
          })
        })
        expect(reversedDocs.params.bound).toBeCloseTo(a.params.bound, 6)
      }),
      { numRuns: 40 },
    )
  })
})
