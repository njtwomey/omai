import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray } from '../../tensor/index.ts'
import { addDummyFeature } from './features.ts'

describe('addDummyFeature', () => {
  it('prepends a constant column', () => {
    expect(
      toArray(
        addDummyFeature([
          [1, 2],
          [3, 4],
          [5, 6],
        ]),
      ),
    ).toEqual([
      [1, 1, 2],
      [1, 3, 4],
      [1, 5, 6],
    ])
    expect(toArray(addDummyFeature([[7]], 0))).toEqual([[0, 7]])
  })

  it('keeps NaN and infinities, and refuses a vector', () => {
    expect(toArray(addDummyFeature([[Number.NaN, Number.POSITIVE_INFINITY]]))).toEqual([
      [1, Number.NaN, Number.POSITIVE_INFINITY],
    ])
    expect(() => addDummyFeature([1, 2, 3])).toThrow()
  })

  it('property: the original features follow the constant, unchanged', () => {
    const matrix = fc
      .tuple(fc.integer({ min: 1, max: 8 }), fc.integer({ min: 1, max: 5 }))
      .chain(([n, d]) =>
        fc.array(fc.array(fc.double({ noNaN: true, min: -1e6, max: 1e6 }), { minLength: d, maxLength: d }), {
          minLength: n,
          maxLength: n,
        }),
      )
    fc.assert(
      fc.property(matrix, fc.double({ noNaN: true, min: -10, max: 10 }), (x, value) => {
        const out = toArray(addDummyFeature(x, value)) as number[][]
        expect(out.map((r) => r[0])).toEqual(x.map(() => value))
        expect(out.map((r) => r.slice(1))).toEqual(x)
      }),
    )
  })
})
