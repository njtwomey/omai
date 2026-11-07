import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import * as T from './index.ts'
import { type DType, formatTensor, tensor, tensorFromJson, tensorToJson, toArray, type Value, zeros } from './index.ts'

const inf = Number.POSITIVE_INFINITY
const nan = Number.NaN

describe('formatTensor', () => {
  // [tensor, options, text]: numpy-style nesting, the dtype named unless it is float64, summaries as numpy's `...`
  const rows: [string, Value, Parameters<typeof formatTensor>[1], string][] = [
    ['a scalar', tensor(3), undefined, 'tensor(3)'],
    ['an empty vector', tensor([]), undefined, 'tensor([])'],
    ['an empty matrix keeps its rows', zeros([2, 0]), undefined, 'tensor([[], []])'],
    ['negative zero prints as 0', tensor([-0]), undefined, 'tensor([0])'],
    [
      'four decimals, exponents outside [1e-4, 1e6)',
      tensor([1.23456789, 100000, 1234567, 0.00001, 0.00012345]),
      undefined,
      'tensor([1.2346, 100000, 1.235e+6, 1e-5, 0.0001])',
    ],
    ['precision 2', tensor([1.23456789]), { precision: 2 }, 'tensor([1.23])'],
    ['nan and inf', tensor([nan, inf, -inf, 1.5]), undefined, 'tensor([nan, inf, -inf, 1.5])'],
    ['bool', tensor([1, 0, 1], { dtype: 'bool' }), undefined, 'tensor([true, false, true], dtype=bool)'],
    [
      'int32',
      T.tensor(
        [
          [1, 2],
          [3, -4],
        ],
        { dtype: 'int32' },
      ),
      undefined,
      'tensor([[1, 2], [3, -4]], dtype=int32)',
    ],
    [
      'float32 shows the rounded decimals',
      tensor([0.1, 2], { dtype: 'float32' }),
      undefined,
      'tensor([0.1, 2], dtype=float32)',
    ],
    ['a long vector keeps both ends', T.arange(10), { threshold: 5, edgeItems: 2 }, 'tensor([0, 1, ..., 8, 9])'],
    [
      'an axis not longer than twice the edge is shown whole',
      T.arange(6),
      { threshold: 5, edgeItems: 3 },
      'tensor([0, 1, 2, 3, 4, 5])',
    ],
    [
      'an axis just longer than twice the edge is cut',
      T.arange(7),
      { threshold: 6, edgeItems: 3 },
      'tensor([0, 1, 2, ..., 4, 5, 6])',
    ],
    ['below the threshold nothing is cut', T.arange(5), { threshold: 5 }, 'tensor([0, 1, 2, 3, 4])'],
    [
      'every long axis of a 3-D tensor is cut',
      T.reshape(T.arange(24), [2, 3, 4]),
      { threshold: 10, edgeItems: 1 },
      'tensor([[[0, ..., 3], ..., [8, ..., 11]], [[12, ..., 15], ..., [20, ..., 23]]])',
    ],
  ]
  it('prints nesting, dtypes, special values and summaries as numpy does, and a traced value by shape and dtype', () => {
    for (const [name, x, options, text] of rows) expect(formatTensor(x, options), name).toBe(text)

    let shown = ''
    let refused = ''
    grad((x: Value) => {
      shown = formatTensor(x)
      try {
        tensorToJson(x)
      } catch (e) {
        refused = (e as { code: string }).code
      }
      return T.sum(x)
    })(tensor([1, 2]))
    expect(shown).toBe('traced(shape=[2], dtype=float64)')
    expect(refused).toBe('not-differentiable')
  })
})

describe('tensorToJson and tensorFromJson', () => {
  it('round-trip every dtype, shape and non-finite value (infinities and NaN as strings) and reject data that does not fill the shape', () => {
    const cases: [DType, number[], number[]][] = [
      ['float64', [2, 3], [1.5, -inf, inf, nan, 0, 1e-300]],
      ['float32', [3], [0.1, inf, -3.5]],
      ['int32', [2, 2], [1, -2, 2147483647, -2147483648]],
      ['bool', [4], [1, 0, 0, 1]],
      ['float64', [0], []],
      ['float64', [], [7.25]],
    ]
    for (const [dtype, shape, data] of cases) {
      const t = T.reshape(tensor(data, { dtype }), shape) as T.Tensor
      const json = tensorToJson(t)
      expect(json.dtype).toBe(dtype)
      expect(json.shape).toEqual(shape)
      expect(JSON.parse(JSON.stringify(json))).toEqual(json)
      const back = tensorFromJson(JSON.parse(JSON.stringify(json)))
      expect([back.dtype, back.shape, toArray(back)], `${dtype} ${shape}`).toEqual([dtype, shape, toArray(t)])
    }
    expect(tensorToJson(tensor([1, inf, -inf, nan])).data).toEqual([1, 'inf', '-inf', 'nan'])

    expect(() => tensorFromJson({ dtype: 'float64', shape: [3], data: [1, 2] })).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
  })
})
