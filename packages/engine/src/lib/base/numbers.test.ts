import { describe, expect, it } from 'vitest'
import {
  allClose,
  allFinite,
  argmaxOf,
  argminOf,
  clamp,
  factorialOf,
  gcdOf,
  isClose,
  kahanSum,
  linspaceNumbers,
  logAddExp,
  logSumExpOf,
  maxOf,
  meanOf,
  minOf,
  pairwiseSum,
  popcount,
  quantileOf,
  range,
  roundHalfEven,
  sigmoidOf,
  softmaxOf,
  toRows,
  varianceOf,
} from './index.ts'

describe('numbers at degenerate arguments', () => {
  it('rejects a zero step and a quantile level outside [0, 1], as numpy and the tensor arange do', () => {
    for (const make of [
      () => range(0, 5, 0),
      () => range(0, 5, Number.NaN),
      () => quantileOf([1, 2], 1.5),
      () => quantileOf([1, 2], -0.1),
      () => quantileOf([1, 2], Number.NaN),
    ])
      expect(make).toThrow(expect.objectContaining({ code: 'value' }))
    expect(range(5, 0, -2)).toEqual([5, 3, 1])
    expect(quantileOf([1, 2, 3, 4], 1)).toBe(4)
    expect(quantileOf([], 0.5)).toBeNaN()
  })

  it('maxOf and minOf take 200,000 values where Math.max(...xs) overflows the stack, and match it on short lists', () => {
    const n = 200_000
    const long = Array.from({ length: n }, (_, i) => ((i * 7919) % n) - 100_000)
    const typed = Float64Array.from(long)
    expect(() => Math.max(...long)).toThrow(RangeError)
    for (const xs of [long, typed, [...long, Number.NaN]]) {
      const hasNaN = xs.length > n
      expect(maxOf(xs)).toBe(hasNaN ? Number.NaN : 99_999)
      expect(minOf(xs)).toBe(hasNaN ? Number.NaN : -100_000)
    }
    for (const xs of [[], [3], [-1, 4, 2], [1, Number.NaN, 2]]) {
      expect(maxOf(xs)).toBe(Math.max(...xs))
      expect(minOf(xs)).toBe(Math.min(...xs))
    }
  })

  it('toRows reshapes a flat row-major array of any array type, reading a missing entry as 0', () => {
    expect(toRows([1, 2, 3, 4, 5, 6], 2, 3)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(toRows(Float64Array.of(1, 2, 3, 4), 2, 2)).toEqual([
      [1, 2],
      [3, 4],
    ])
    expect(toRows([1, 2, 3], 2, 2)).toEqual([
      [1, 2],
      [3, 0],
    ])
    expect(toRows([], 0, 3)).toEqual([])
    expect(toRows([], 2, 0)).toEqual([[], []])
  })

  it('factorialOf is the running product (the loops it replaces) and gcdOf is Euclid on absolute values', () => {
    const loop = (n: number): number => {
      let r = 1
      for (let i = 2; i <= n; i++) r *= i
      return r
    }
    for (const n of [0, 1, 2, 5, 10, 20, 22, 23, 50, 100, 170]) expect(factorialOf(n)).toBe(loop(n))
    expect([factorialOf(171), factorialOf(-1), factorialOf(2.5)]).toEqual([Infinity, Number.NaN, Number.NaN])
    expect([gcdOf(12, 18), gcdOf(12, -18), gcdOf(0, 5), gcdOf(7, 0), gcdOf(0, 0), gcdOf(17, 5)]).toEqual([
      6, 6, 5, 7, 0, 1,
    ])
  })
})

describe('the number helpers agree with numpy and scipy', () => {
  const flat = (x: unknown): unknown[] => (Array.isArray(x) ? x.flatMap(flat) : [x])
  // [name, value under test, numpy/scipy answer] (a one-off numpy 2.5 and scipy run; scalars, booleans and lists)
  const rows: [string, () => unknown, unknown][] = [
    ['isClose: 0.1 + 0.2 and 0.3', () => isClose(0.1 + 0.2, 0.3), true],
    ['isClose: with atol 0 nothing but 0 is close to 0', () => isClose(1e-300, 0), false],
    [
      'isClose: NaN is never close, equal infinities are',
      () => [isClose(NaN, NaN), isClose(Infinity, Infinity), isClose(Infinity, -Infinity), isClose(1, Infinity)],
      [false, true, false, false],
    ],
    [
      'isClose: the tolerance scales with the second argument',
      () => [isClose(1, 1.1, 0.1), isClose(1.1, 1, 0.1)],
      [true, false],
    ],
    ['isClose: atol alone', () => [isClose(1e-9, 0, 0, 1e-8), isClose(1e-7, 0, 0, 1e-8)], [true, false]],
    [
      'allClose: every pair, NaN breaks it',
      () => [
        allClose([1, 2], [1, 2 + 1e-12]),
        allClose([1, NaN], [1, NaN]),
        allClose([1, 2], [1, 3]),
        allClose([], []),
      ],
      [true, false, false, true],
    ],
    [
      'clamp: inside, below, above and NaN',
      () => [clamp(0.5, 0, 1), clamp(-3, 0, 1), clamp(7, 0, 1), clamp(NaN, 0, 1), clamp(Infinity, 0, 1)],
      [0.5, 0, 1, NaN, 1],
    ],
    ['kahanSum: a term cancelled by a larger one', () => kahanSum([1e16, 1, -1e16]), 1],
    ['kahanSum: ten 0.1s are exactly 1', () => kahanSum(Array.from({ length: 10 }, () => 0.1)), 1],
    ['kahanSum: nothing is 0', () => kahanSum([]), 0],
    [
      'kahanSum: an Infinity stays Infinity',
      () => [kahanSum([1, Infinity]), kahanSum([-Infinity, 2]), kahanSum([1e308, 1e308])],
      [Infinity, -Infinity, Infinity],
    ],
    ['kahanSum: Infinity minus Infinity is NaN', () => kahanSum([Infinity, -Infinity, 1]), NaN],
    ['pairwiseSum: 1000 tenths against the exact sum', () => pairwiseSum(Array.from({ length: 1000 }, () => 0.1)), 100],
    [
      'pairwiseSum: nothing is 0, short lists are summed in order',
      () => [pairwiseSum([]), pairwiseSum([1, 2, 3, 4]), pairwiseSum([1e16, 1, -1e16])],
      [0, 10, 0],
    ],
    ['logSumExpOf: two large equal values', () => logSumExpOf([1000, 1000]), 1000.6931471805599],
    [
      'logSumExpOf: all -Infinity, +Infinity, NaN, nothing',
      () => [logSumExpOf([-Infinity, -Infinity]), logSumExpOf([Infinity, 1]), logSumExpOf([NaN, 1]), logSumExpOf([])],
      [-Infinity, Infinity, NaN, -Infinity],
    ],
    ['logSumExpOf: three zeros are log 3', () => logSumExpOf([0, 0, 0]), 1.0986122886681098],
    [
      'argmaxOf and argminOf: the first of ties, -1 for nothing',
      () => [
        argmaxOf([3, 9, 9, 2]),
        argminOf([3, 1, 1, 2]),
        argmaxOf([]),
        argminOf([]),
        argmaxOf([-Infinity, -Infinity]),
      ],
      [1, 1, -1, -1, 0],
    ],
    [
      'sigmoidOf: saturates without NaN, is expit',
      () => [-1000, -40, -1, 0, 1, 40, 1000, Infinity, -Infinity].map(sigmoidOf),
      [0, 4.248354255291589e-18, 0.2689414213699951, 0.5, 0.7310585786300049, 1, 1, 1, 0],
    ],
    [
      'allFinite',
      () => [
        allFinite([1, 2]),
        allFinite([1, NaN]),
        allFinite([Infinity]),
        allFinite([]),
        allFinite(new Float64Array([1, -Infinity])),
      ],
      [true, false, false, true, false],
    ],
    [
      'range: Python and numpy steps',
      () => [range(5), range(2, 10, 3), range(5, 0, -2), range(3, 3), range(0, 1, 0.25)],
      [[0, 1, 2, 3, 4], [2, 5, 8], [5, 3, 1], [], [0, 0.25, 0.5, 0.75]],
    ],
    [
      'linspaceNumbers: numpy linspace, one point, none, descending',
      () => [
        linspaceNumbers(0, 1, 5),
        linspaceNumbers(2, 5, 1),
        linspaceNumbers(0, 1, 0),
        linspaceNumbers(1, 0, 3),
        linspaceNumbers(0, 1, 2),
      ],
      [[0, 0.25, 0.5, 0.75, 1], [2], [], [1, 0.5, 0], [0, 1]],
    ],
    ['linspaceNumbers: the default is 50 points ending at stop', () => linspaceNumbers(0, 1).length, 50],
    [
      'softmaxOf: scipy softmax at huge, -Infinity and ordinary values',
      () => [softmaxOf([1000, 1000]), softmaxOf([0, -Infinity]), softmaxOf([1, 2, 3])],
      [
        [0.5, 0.5],
        [1, 0],
        [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
      ],
    ],
    [
      'logAddExp: numpy logaddexp at the edges',
      () => [
        logAddExp(-Infinity, -Infinity),
        logAddExp(Infinity, Infinity),
        logAddExp(1000, 1000),
        logAddExp(Infinity, -Infinity),
        logAddExp(NaN, 1),
        logAddExp(-1000, -1001),
        logAddExp(0, 0),
      ],
      [-Infinity, Infinity, 1000.6931471805599, Infinity, NaN, -999.6867383124818, Math.LN2],
    ],
    [
      'meanOf and varianceOf: numpy mean and var, NaN when there is nothing left',
      () => [
        meanOf([]),
        meanOf([1, 2, 3, 4]),
        varianceOf([1, 2, 3, 4]),
        varianceOf([1, 2, 3, 4], 1),
        varianceOf([5], 1),
        varianceOf([]),
      ],
      [NaN, 2.5, 1.25, 1.6666666666666667, NaN, NaN],
    ],
    [
      'quantileOf: numpy linear interpolation',
      () => [
        quantileOf([4, 1, 3, 2], 0.3),
        quantileOf([5], 0.5),
        quantileOf([1, 2, 3, 4], 0),
        quantileOf([1, 2, 3, 4], 0.5),
      ],
      [1.9, 5, 1, 2.5],
    ],
    [
      'popcount: the set bits of a 32-bit mask',
      () => [popcount(0), popcount(1), popcount(0b1011), popcount(2 ** 31 - 1), popcount(2 ** 32 - 1)],
      [0, 1, 3, 31, 32],
    ],
    [
      'roundHalfEven: numpy round, ties to even',
      () => [0.5, 1.5, 2.5, -2.5, -1.5, 0.49999999999999994, 1e15 + 0.5].map(roundHalfEven),
      [0, 2, 2, -2, -2, 0, 1000000000000000.0],
    ],
  ]
  it('gives the reference answer for ties, empty lists, NaN, infinities and cancellation', () => {
    const wrong: string[] = []
    for (const [name, thunk, want] of rows) {
      const got = flat(thunk())
      const ref = flat(want)
      const same = (a: unknown, b: unknown): boolean =>
        typeof a === 'number' && typeof b === 'number'
          ? Object.is(a, b) || (Number.isFinite(b) && Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)))
          : a === b
      if (got.length !== ref.length || !got.every((v, k) => same(v, ref[k])))
        wrong.push(`${name}: got ${got}, expected ${ref}`)
    }
    expect(wrong).toEqual([])
  })
})
