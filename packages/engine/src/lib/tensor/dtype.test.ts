import { describe, expect, it } from 'vitest'
import { MdmlError } from '../base/index.ts'
import * as T from './index.ts'
import {
  abs,
  add,
  all,
  any,
  argmax,
  arctan2,
  broadcastShapes,
  ceil,
  clip,
  concat,
  cumprod,
  cumsum,
  div,
  dot,
  equal,
  exp,
  finfo,
  flip,
  floor,
  floorDiv,
  greater,
  iinfo,
  isnan,
  less,
  log,
  logicalAnd,
  logicalNot,
  logsumexp,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  mod,
  mul,
  neg,
  norm,
  normaliseAxis,
  pow,
  prod,
  reciprocal,
  relu,
  reshape,
  resultType,
  round,
  sign,
  sin,
  sort,
  sqrt,
  square,
  std,
  sub,
  sum,
  take,
  tanh,
  tensor,
  toArray,
  tril,
  unique,
  variance,
  weakResultType,
  where,
  type DType,
  type NestedArray,
  type Tensor,
  type Value,
} from './index.ts'

const inf = Number.POSITIVE_INFINITY
const nan = Number.NaN

// One letter per dtype, in promotion order: bool, int32, float32, float64.
const DTYPES: Record<string, DType> = { b: 'bool', i: 'int32', f: 'float32', d: 'float64' }
const STORAGE: Record<DType, string> = {
  bool: 'Uint8Array',
  int32: 'Int32Array',
  float32: 'Float32Array',
  float64: 'Float64Array',
}
const operand = (letter: string, which: 1 | 2): Tensor => {
  const dtype = DTYPES[letter] as DType
  return tensor(letter === 'b' ? (which === 1 ? [1, 0, 1] : [1, 1, 1]) : which === 1 ? [3, -2, 5] : [2, 3, 4], {
    dtype,
  })
}
const flat = (x: NestedArray): number[] => (Array.isArray(x) ? x.flatMap(flat) : [Number(x)])
/** Equal, NaN with NaN, or within a relative tolerance (float32 rounds to 24 bits). */
function sameValues(got: Value, want: NestedArray, rtol: number): boolean {
  const g = flat(toArray(got))
  const w = flat(want)
  return (
    g.length === w.length &&
    g.every((v, k) => {
      const b = w[k] as number
      return (Number.isNaN(v) && Number.isNaN(b)) || v === b || Math.abs(v - b) <= rtol * Math.max(1, Math.abs(b))
    })
  )
}

// Expected result dtypes are numpy's (one-off numpy 2.5 run), with this library's documented departures: bool
// arithmetic is int32 (numpy: bool stays bool for + * max, and `-` and `neg` raise), there is no float16 (a float
// function of bool is float64), and powers, reciprocals and floor/ceil/round of integers are float64 (numpy keeps
// integers for floor/ceil/round since 2.1, and raises on a negative integer power).
const unary: [string, (x: Tensor) => Value, string, number[]][] = [
  ['neg', (x) => neg(x), 'iifd', [-3, 2, -5]],
  ['abs', (x) => abs(x), 'bifd', [3, 2, 5]],
  ['square', (x) => square(x), 'iifd', [9, 4, 25]],
  ['sign', (x) => sign(x), 'bifd', [1, -1, 1]],
  ['reciprocal', (x) => reciprocal(x), 'ddfd', [0.333333333333, -0.5, 0.2]],
  ['exp', (x) => exp(x), 'ddfd', [20.085536923188, 0.135335283237, 148.413159102577]],
  ['sin', (x) => sin(x), 'ddfd', [0.14112000806, -0.909297426826, -0.958924274663]],
  ['sqrt', (x) => sqrt(x), 'ddfd', [1.732050807569, nan, 2.2360679775]],
  ['log', (x) => log(x), 'ddfd', [1.098612288668, nan, 1.609437912434]],
  ['tanh', (x) => tanh(x), 'ddfd', [0.995054753687, -0.964027580076, 0.999909204263]],
  ['floor', (x) => floor(x), 'ddfd', [3, -2, 5]],
  ['ceil', (x) => ceil(x), 'ddfd', [3, -2, 5]],
  ['round', (x) => round(x), 'ddfd', [3, -2, 5]],
  ['relu', (x) => relu(x), 'iifd', [3, 0, 5]],
  ['isnan', (x) => isnan(x), 'bbbb', [0, 0, 0]],
  ['logicalNot', (x) => logicalNot(x), 'bbbb', [0, 0, 0]],
  ['sum', (x) => sum(x), 'iifd', [6]],
  ['prod', (x) => prod(x), 'iifd', [-30]],
  ['max', (x) => max(x), 'bifd', [5]],
  ['min', (x) => min(x), 'bifd', [-2]],
  ['mean', (x) => mean(x), 'ddfd', [2]],
  ['argmax', (x) => argmax(x), 'iiii', [2]],
  ['cumsum', (x) => cumsum(x), 'iifd', [3, 1, 6]],
  ['cumprod', (x) => cumprod(x), 'iifd', [3, -6, -30]],
  ['all', (x) => all(x), 'bbbb', [1]],
  ['any', (x) => any(x), 'bbbb', [1]],
  ['std', (x) => std(x), 'ddfd', [2.943920288776]],
  ['variance', (x) => variance(x), 'ddfd', [8.666666666667]],
  ['norm', (x) => norm(x), 'ddfd', [6.164414002969]],
  ['sort', (x) => sort(x), 'bifd', [-2, 3, 5]],
  ['clip to [0, 2]', (x) => clip(x, 0, 2), 'iifd', [2, 0, 2]],
  ['clip to [0.5, 2]', (x) => clip(x, 0.5, 2), 'ddfd', [2, 0.5, 2]],
  ['+ 1', (x) => add(x, 1), 'iifd', [4, -1, 6]],
  ['+ 1.5', (x) => add(x, 1.5), 'ddfd', [4.5, -0.5, 6.5]],
  ['* 2', (x) => mul(x, 2), 'iifd', [6, -4, 10]],
  ['/ 2', (x) => div(x, 2), 'ddfd', [1.5, -1, 2.5]],
  ['** 2', (x) => pow(x, 2), 'ddfd', [9, 4, 25]],
  ['** -1', (x) => pow(x, -1), 'ddfd', [0.333333333333, -0.5, 0.2]],
  ['% 2', (x) => mod(x, 2), 'iifd', [1, 0, 1]],
  ['// 2', (x) => floorDiv(x, 2), 'iifd', [1, -1, 2]],
  ['maximum with 1', (x) => maximum(x, 1), 'iifd', [3, 1, 5]],
  ['take', (x) => take(x, [2, 0]), 'bifd', [5, 3]],
  ['concat', (x) => concat([x, x]), 'bifd', [3, -2, 5, 3, -2, 5]],
  ['flip', (x) => flip(x, 0), 'bifd', [5, -2, 3]],
  ['matmul x x^T', (x) => matmul(reshape(x, [1, 3]), reshape(x, [3, 1])), 'iifd', [38]],
  ['dot', (x) => dot(x, x), 'iifd', [38]],
  ['tril', (x) => tril(reshape(concat([x, x, x]), [3, 3])), 'bifd', [3, 0, 0, 3, -2, 0, 3, -2, 5]],
  ['where', (x) => where(greater(x, 0), x, x), 'bifd', [3, -2, 5]],
  ['less than 1.5', (x) => less(x, 1.5), 'bbbb', [0, 1, 0]],
  ['unique', (x) => unique(x).values, 'bifd', [-2, 3, 5]],
  ['logsumexp', (x) => logsumexp(x), 'ddfd', [5.127730871635]],
]

describe('the dtype of every operation family, for every input dtype', () => {
  it("one operand: bool, int32, float32 and float64 inputs give numpy's dtype, storage class and values", () => {
    for (const [name, f, letters, want] of unary) {
      for (const [k, letter] of [...'bifd'].entries()) {
        const x = operand(letter, 1)
        const r = f(x) as Tensor
        const label = `${name} on ${x.dtype}`
        expect(r.dtype, label).toBe(DTYPES[letters[k] as string])
        expect(r.data.constructor.name, label).toBe(STORAGE[r.dtype])
        // the values do not depend on the dtype: numpy's float64 answer, or the float64 computation of a bool
        const reference = letter === 'b' ? toArray(f(T.cast(x, 'float64'))) : want
        expect(sameValues(r, reference, letter === 'f' ? 1e-6 : 1e-11), label).toBe(true)
      }
    }
  })

  const binary: [string, (a: Value, b: Value) => Value, string[], number[]][] = [
    ['add', add, ['iifd', 'iidd', 'fdfd', 'dddd'], [5, 1, 9]],
    ['sub', sub, ['iifd', 'iidd', 'fdfd', 'dddd'], [1, -5, 1]],
    ['mul', mul, ['iifd', 'iidd', 'fdfd', 'dddd'], [6, -6, 20]],
    ['div', div, ['ddfd', 'dddd', 'fdfd', 'dddd'], [1.5, -0.666666666667, 1.25]],
    ['pow', pow, ['ddfd', 'dddd', 'fdfd', 'dddd'], [9, -8, 625]],
    ['mod', mod, ['iifd', 'iidd', 'fdfd', 'dddd'], [1, 1, 1]],
    ['floorDiv', floorDiv, ['iifd', 'iidd', 'fdfd', 'dddd'], [1, -1, 1]],
    ['maximum', maximum, ['iifd', 'iidd', 'fdfd', 'dddd'], [3, 3, 5]],
    ['minimum', minimum, ['iifd', 'iidd', 'fdfd', 'dddd'], [2, -2, 4]],
    ['less', less, ['bbbb', 'bbbb', 'bbbb', 'bbbb'], [0, 1, 0]],
    ['equal', equal, ['bbbb', 'bbbb', 'bbbb', 'bbbb'], [0, 0, 0]],
    ['logicalAnd', logicalAnd, ['bbbb', 'bbbb', 'bbbb', 'bbbb'], [1, 1, 1]],
    ['arctan2', arctan2, ['ddfd', 'dddd', 'fdfd', 'dddd'], [0.982793723247, -0.588002603548, 0.896055384571]],
    ['where', (a, b) => where(greater(a, 0), a, b), ['bifd', 'iidd', 'fdfd', 'dddd'], [3, 3, 5]],
    ['concat', (a, b) => concat([a, b]), ['bifd', 'iidd', 'fdfd', 'dddd'], [3, -2, 5, 2, 3, 4]],
    ['matmul (as a dot)', matmul, ['iifd', 'iidd', 'fdfd', 'dddd'], [20]],
  ]
  it('two operands: all sixteen dtype pairs promote as numpy does, and the values are those of float64', () => {
    for (const [name, f, grid, want] of binary) {
      for (const [i, a] of [...'bifd'].entries()) {
        for (const [j, b] of [...'bifd'].entries()) {
          const x = operand(a, 1)
          const y = operand(b, 2)
          const r = f(x, y) as Tensor
          const label = `${name}(${x.dtype}, ${y.dtype})`
          expect(r.dtype, label).toBe(DTYPES[grid[i]?.[j] as string])
          expect(r.data.constructor.name, label).toBe(STORAGE[r.dtype])
          expect(sameValues(r, toArray(f(T.cast(x, 'float64'), T.cast(y, 'float64'))), 1e-6), label).toBe(true)
        }
      }
      expect(sameValues(f(operand('d', 1), operand('d', 2)), want, 1e-11), name).toBe(true)
    }
  })

  it('promotion, weak scalars and machine limits follow numpy', () => {
    const grid = ['bifd', 'iidd', 'fdfd', 'dddd']
    for (const [i, a] of [...'bifd'].entries())
      for (const [j, b] of [...'bifd'].entries())
        expect(resultType(DTYPES[a] as DType, DTYPES[b] as DType), `${a}${b}`).toBe(DTYPES[grid[i]?.[j] as string])
    // [dtype, scalar, result]: a float tensor keeps its dtype; an integer scalar keeps an integer one; 2.5 makes it float64
    const weak: [DType, number, DType][] = [
      ['bool', 1, 'int32'],
      ['bool', 0, 'int32'],
      ['bool', 2.5, 'float64'],
      ['int32', 1, 'int32'],
      ['int32', 2.5, 'float64'],
      ['int32', -2147483648, 'int32'],
      ['int32', 3e9, 'float64'],
      ['float32', 2.5, 'float32'],
      ['float32', 1e300, 'float32'],
      ['float64', 1, 'float64'],
    ]
    for (const [dtype, scalar, want] of weak) expect(weakResultType(dtype, scalar), `${dtype} ${scalar}`).toBe(want)
    expect(finfo('float32')).toEqual({
      eps: 1.1920928955078125e-7,
      max: 3.4028234663852886e38,
      tiny: 1.1754943508222875e-38,
    })
    expect(finfo('float64').tiny).toBe(2.2250738585072014e-308)
    expect([iinfo('int32'), iinfo('bool')]).toEqual([
      { min: -2147483648, max: 2147483647, bits: 32 },
      { min: 0, max: 1, bits: 1 },
    ])
  })

  it('broadcast shapes and axes agree with numpy, empty axes included', () => {
    const shapes: [number[], number[], number[] | null][] = [
      [
        [2, 1],
        [1, 3],
        [2, 3],
      ],
      [[0], [1], [0]],
      [
        [1, 0],
        [3, 1],
        [3, 0],
      ],
      [[], [2, 3], [2, 3]],
      [
        [5, 1, 4],
        [3, 1],
        [5, 3, 4],
      ],
      [[0], [3], null],
      [[2], [3], null],
      [[2, 0], [0, 1], null],
    ]
    for (const [a, b, want] of shapes) {
      if (want === null)
        expect(() => broadcastShapes(a, b), `${a} ${b}`).toThrow(expect.objectContaining({ code: 'shape' }))
      else expect(broadcastShapes(a, b), `${a} ${b}`).toEqual(want)
    }
    expect([normaliseAxis(-1, 3), normaliseAxis(0, 1), normaliseAxis(-3, 3)]).toEqual([2, 0, 0])
    for (const [axis, rank] of [
      [3, 3],
      [-4, 3],
      [0, 0],
      [1.5, 3],
    ] as const)
      expect(() => normaliseAxis(axis, rank), `${axis} ${rank}`).toThrow(MdmlError)
  })
})
