import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toDense } from '../numerics/sparse/index.ts'
import { toArray } from '../tensor/index.ts'
import { dictInverse, dictTransform, dictVectorize, featureHash } from './feature-dicts.ts'

const record = fc.dictionary(
  fc.constantFrom('a', 'b', 'c', 'd'),
  fc.oneof(fc.double({ min: -10, max: 10, noNaN: true }), fc.constantFrom('x', 'y', 'z')),
  { maxKeys: 4 },
)

describe('feature vectoriser properties', () => {
  it('fit_transform equals fit then transform, and the inverse returns every non-zero entry', () => {
    fc.assert(
      fc.property(fc.array(record, { minLength: 1, maxLength: 6 }), (records) => {
        const { vectorizer, matrix } = dictVectorize(records)
        expect(toArray(toDense(dictTransform(vectorizer, records)))).toEqual(toArray(toDense(matrix)))
        const back = dictInverse(vectorizer, matrix)
        for (const [i, r] of records.entries())
          for (const [name, v] of Object.entries(r)) {
            const key = typeof v === 'string' ? `${name}=${v}` : name
            const value = typeof v === 'string' ? 1 : v
            expect(back[i]?.[key] ?? 0).toBeCloseTo(value, 12)
          }
        expect([...vectorizer.featureNames]).toEqual([...vectorizer.featureNames].sort())
      }),
      { numRuns: 60 },
    )
  })

  it('hashing is linear in the values when signs alternate or not, and every row has at most as many entries as features', () => {
    fc.assert(
      fc.property(record, fc.integer({ min: 2, max: 64 }), (r, width) => {
        const one = toArray(toDense(featureHash([r], { nFeatures: width }))) as number[][]
        const doubled = toArray(
          toDense(
            featureHash(
              [Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'number' ? 2 * v : v]))],
              { nFeatures: width },
            ),
          ),
        ) as number[][]
        const numeric = Object.values(r).every((v) => typeof v === 'number')
        if (numeric) for (const [j, v] of (one[0] ?? []).entries()) expect(doubled[0]?.[j]).toBeCloseTo(2 * v, 10)
        expect((one[0] ?? []).filter((v) => v !== 0).length).toBeLessThanOrEqual(Object.keys(r).length)
      }),
      { numRuns: 60 },
    )
  })
})
