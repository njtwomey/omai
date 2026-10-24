import { describe, expect, it } from 'vitest'
import { compareValues, parseLiteral, plain, statedNumber } from './compare.ts'
import { formatValue } from './format.ts'

const tensor = (shape: number[], data: number[], dtype = 'float64') => ({
  kind: 'tensor',
  dtype,
  shape,
  data: dtype === 'bool' ? Uint8Array.from(data) : Float64Array.from(data),
})

describe('statedNumber', () => {
  it('reads the precision a literal states', () => {
    expect(statedNumber('1.25').halfUnit).toBeCloseTo(0.005, 15)
    expect(statedNumber('1.5e-3').halfUnit).toBeCloseTo(0.00005, 15)
    expect(statedNumber('3').halfUnit).toBe(0)
    expect(statedNumber('2e3').halfUnit).toBe(0)
    expect(statedNumber('-Infinity')).toEqual({ kind: 'number', value: Number.NEGATIVE_INFINITY, halfUnit: 0 })
  })
})

describe('parseLiteral', () => {
  it('reads nested literals and refuses expressions', () => {
    expect(parseLiteral("[1, 'a', { b: true, 'c d': null }, undefined,]")).toBeDefined()
    expect(parseLiteral('Math.log(2)')).toBeUndefined()
    expect(parseLiteral('[1, x]')).toBeUndefined()
    expect(parseLiteral('1e')).toBeUndefined()
  })

  it('reads typed arrays in Node notation as arrays', () => {
    expect(parseLiteral('Float64Array [1, 2]')).toEqual(parseLiteral('[1, 2]'))
    expect(parseLiteral('Int32Array(2) [1, 2]')).toEqual(parseLiteral('[1, 2]'))
  })

  it('unescapes strings in either quote', () => {
    expect(parseLiteral("'it\\'s'")).toEqual({ kind: 'atom', value: "it's" })
    expect(parseLiteral('"a\\nb"')).toEqual({ kind: 'atom', value: 'a\nb' })
  })
})

describe('compareValues', () => {
  it('matches numbers to the stated decimals', () => {
    expect(compareValues(1.2345612, 1.23456, '1.23456').pass).toBe(true)
    expect(compareValues(1.2346, 1.23456, '1.23456').pass).toBe(false)
    expect(compareValues(0.1 + 0.2, 0.3, '0.3').pass).toBe(true)
  })

  it('treats integers and computed values as exact up to rounding error', () => {
    expect(compareValues(2.0000000000001, 2, '2').pass).toBe(true)
    expect(compareValues(2.001, 2, '2').pass).toBe(false)
    expect(compareValues(Math.log(0.5) + 1e-17, Math.log(0.5), 'Math.log(0.5)').pass).toBe(true)
    expect(compareValues(Number.NaN, Number.NaN, 'NaN').pass).toBe(true)
  })

  it('compares tensors and typed arrays as nested arrays', () => {
    expect(compareValues(tensor([2, 2], [1, 2, 3, 4]), undefined, '[[1, 2], [3, 4]]').pass).toBe(true)
    expect(compareValues(Float64Array.of(1, 2), [1, 2], 'Float64Array.of(1, 2)').pass).toBe(true)
    expect(compareValues(tensor([2], [1, 0], 'bool'), undefined, '[true, false]').pass).toBe(true)
    expect(compareValues(tensor([], [5]), undefined, '5').pass).toBe(true)
  })

  it('needs the same defined fields and says where values differ', () => {
    expect(compareValues({ a: 1, b: undefined }, undefined, '{ a: 1 }').pass).toBe(true)
    expect(compareValues({ a: 1, b: 2 }, undefined, '{ a: 1 }')).toEqual({ pass: false, message: 'unexpected field b' })
    expect(compareValues({ a: [1, 2] }, undefined, '{ a: [1, 3] }').message).toBe('expected 3 at a[1], got 2')
    expect(compareValues(new Map([['k', 1]]), new Map([['k', 1]]), 'new Map([["k", 1]])').pass).toBe(true)
  })

  it('compares strings, booleans and null strictly', () => {
    expect(compareValues('a', 'a', "'a'").pass).toBe(true)
    expect(compareValues(1, true, 'true').pass).toBe(false)
    expect(compareValues(undefined, null, 'null').pass).toBe(false)
  })
})

describe('plain and formatValue', () => {
  it('turns tensors into nested arrays', () => {
    expect(plain(tensor([2, 1], [1, 2]))).toEqual([[1], [2]])
  })

  it('formats values as one line', () => {
    expect(formatValue(tensor([2], [0.1 + 0.2, 1]))).toBe('[0.3, 1]')
    expect(formatValue({ a: 'x', 'b c': [1, 2], f: () => 1 })).toBe('{ a: \'x\', "b c": [1, 2] }')
    expect(
      formatValue(
        Array.from({ length: 25 }, (_, i) => i),
        { maxItems: 3 },
      ),
    ).toBe('[0, 1, 2, … 22 more]')
    expect(formatValue(new Set([1]))).toBe('Set(1) [1]')
    expect(formatValue(-0)).toBe('-0')
  })
})
