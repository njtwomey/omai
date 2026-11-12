import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import type { Tensor } from '../tensor.ts'
import { cast } from './arithmetic.ts'
import { equal } from './compare.ts'
import { oneHot, scatterAdd, segmentSum } from './indexing.ts'
import { broadcastTo, expandDims } from './shape.ts'

// `oneHot` writes its ones directly for concrete labels; the original is the comparison of the labels against the class
// numbers. The two must agree on every label, including ones outside [0, n) (a row of zeros). The index wrapping of the
// gathers and scatters is a plain loop now: negative indices count from the end and out-of-range ones raise.

describe('oneHot and index wrapping', () => {
  it('oneHot of concrete labels equals the comparison of labels with classes', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -3, max: 9 }), { minLength: 0, maxLength: 12 }),
        fc.integer({ min: 1, max: 8 }),
        (labels, n) => {
          const l = cpu.wrap('int32', [labels.length], Int32Array.from(labels))
          const classes = cpu.wrap(
            'int32',
            [n],
            Int32Array.from({ length: n }, (_, i) => i),
          )
          const want = cast(equal(expandDims(l, -1), broadcastTo(classes, [labels.length, n])), 'float64') as Tensor
          const got = oneHot(l, n) as Tensor
          expect(got.shape).toEqual(want.shape)
          expect(Array.from(got.data)).toEqual(Array.from(want.data))
        },
      ),
      { numRuns: 200 },
    )
  })

  it('a matrix of labels gets a trailing class axis', () => {
    const got = oneHot(cpu.wrap('int32', [2, 2], Int32Array.of(0, 2, 1, 5)), 3) as Tensor
    expect(got.shape).toEqual([2, 2, 3])
    expect(Array.from(got.data)).toEqual([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
  })

  it('negative indices wrap and out-of-range ones raise in scatterAdd and segmentSum', () => {
    expect(Array.from((scatterAdd([3], [-1, 0, -3], [1, 2, 4], 0) as Tensor).data)).toEqual([6, 0, 1])
    expect(Array.from((segmentSum([1, 2, 3], [-1, 0, 1], 2) as Tensor).data)).toEqual([2, 4])
    expect(() => scatterAdd([3], [3], [1], 0)).toThrow('scatterAdd: index 3 out of range for length 3')
    expect(() => segmentSum([1], [-3], 2)).toThrow('segmentSum: index -3 out of range for length 2')
  })
})
