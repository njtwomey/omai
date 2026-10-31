import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { exhaustivePatterns, gsp, prefixSpan } from './index.ts'

const event = fc.uniqueArray(fc.integer({ min: 0, max: 3 }), { minLength: 1, maxLength: 3 })
const sequences = fc.array(fc.array(event, { minLength: 1, maxLength: 4 }), { minLength: 1, maxLength: 8 })
const support = fc.constantFrom(0.2, 0.34, 0.5, 0.75)

describe('sequential pattern mining properties', () => {
  it('PrefixSpan and GSP equal exhaustive enumeration, counts included', () => {
    fc.assert(
      fc.property(sequences, support, (data, minSupport) => {
        const truth = exhaustivePatterns(data, { minSupport, maxLength: 4 })
        expect(prefixSpan(data, { minSupport, maxLength: 4 })).toEqual(truth)
        expect(gsp(data, { minSupport, maxLength: 4 })).toEqual(truth)
      }),
      { numRuns: 120 },
    )
  })

  it('GSP with gap and span constraints equals constrained exhaustive enumeration', () => {
    fc.assert(
      fc.property(
        sequences,
        support,
        fc.record({
          maxGap: fc.option(fc.integer({ min: 1, max: 3 }), { nil: undefined }),
          minGap: fc.option(fc.integer({ min: 1, max: 2 }), { nil: undefined }),
          maxSpan: fc.option(fc.integer({ min: 1, max: 4 }), { nil: undefined }),
        }),
        (data, minSupport, raw) => {
          const constraints = {
            ...(raw.maxGap === undefined ? {} : { maxGap: raw.maxGap }),
            ...(raw.minGap === undefined ? {} : { minGap: raw.minGap }),
            ...(raw.maxSpan === undefined ? {} : { maxSpan: raw.maxSpan }),
          }
          const truth = exhaustivePatterns(data, { minSupport, maxLength: 3 }, constraints)
          expect(gsp(data, { minSupport, maxLength: 3 }, constraints)).toEqual(truth)
        },
      ),
      { numRuns: 120 },
    )
  })
})
