import { describe, expect, it } from 'vitest'
import { encodeNumber, fromJson, type Json, toJson } from '../base/index.ts'
import { tensor, toArray } from '../tensor/index.ts'
import {
  numberFromJson,
  numbersFromJson,
  numbersToJson,
  numberToJson,
  stateFromJson,
  stateToJson,
} from './serialise.ts'

// What a stored checkpoint goes through: JSON text and back.
const viaText = (x: unknown): unknown => stateFromJson(JSON.parse(JSON.stringify(stateToJson(x))) as Json)

describe('state serialisation', () => {
  const states: readonly [string, unknown][] = [
    ['plain nesting', { n: 2, s: 'x', b: true, z: null, a: [1, [2, { k: 3 }]] }],
    [
      'typed arrays',
      { f: Float64Array.of(1, 2.5), i: Int32Array.of(-3, 4), u: Uint8Array.of(7), c: Uint8ClampedArray.of(9) },
    ],
    [
      'non-finite numbers',
      { a: Number.NaN, b: Infinity, c: -Infinity, d: [Number.NaN, 1], e: Float64Array.of(Infinity, Number.NaN) },
    ],
    ['negative zero', { z: -0, t: Float64Array.of(-0, 1) }],
    ['keys that look like tags', { $number: 1, $typed: 'x', $tensor: [1], $object: {} }],
  ]
  it.each(states)('%s survive a round trip through JSON text', (_name, state) => {
    expect(viaText(state)).toEqual(state)
  })

  it('tensors keep dtype, shape and non-finite values, nested anywhere', () => {
    const t = tensor([
      [1, Infinity],
      [Number.NaN, -2],
    ])
    const back = viaText({ w: [t, { u: t }] }) as { w: [typeof t, { u: typeof t }] }
    for (const b of [back.w[0], back.w[1].u]) {
      expect(b.shape).toEqual([2, 2])
      expect(b.dtype).toBe('float64')
      expect(toArray(b)).toEqual(toArray(t))
    }
  })

  it('has no codec of its own: tensor-free states are exactly base toJson, and numbers use encodeNumber', () => {
    const state = { f: Float64Array.of(1, Number.NaN), x: Infinity, y: undefined, a: [undefined, 1] }
    expect(stateToJson(state)).toEqual(toJson(state))
    expect(fromJson(stateToJson(state))).toEqual({ f: Float64Array.of(1, Number.NaN), x: Infinity, a: [null, 1] })
    expect(stateToJson(undefined)).toBeNull()
    expect([Number.NaN, Infinity, -Infinity].map(numberToJson)).toEqual(
      [Number.NaN, Infinity, -Infinity].map(encodeNumber),
    )
  })

  it('refuses what it cannot restore', () => {
    for (const bad of [
      new Map(),
      new Set(),
      () => 1,
      new (class A {})(),
      new DataView(new ArrayBuffer(2)),
      Symbol('s'),
    ]) {
      expect(() => stateToJson({ bad })).toThrow(expect.objectContaining({ code: 'value' }))
    }
  })

  it('numbers: finite stay numbers, the three non-finite ones are strings, the earlier spellings and null still read', () => {
    expect(numbersFromJson(numbersToJson([1.5, Number.NaN, Infinity, -Infinity]))).toEqual([
      1.5,
      Number.NaN,
      Infinity,
      -Infinity,
    ])
    expect(numbersToJson([1.5, 2])).toEqual([1.5, 2])
    expect(JSON.parse(JSON.stringify(numbersToJson([Infinity, Number.NaN])))).toEqual(
      numbersToJson([Infinity, Number.NaN]),
    )
    expect(['Infinity', '-Infinity', 'NaN', null, undefined, 3].map(numberFromJson)).toEqual([
      Infinity,
      -Infinity,
      Number.NaN,
      Number.NaN,
      Number.NaN,
      3,
    ])
    expect(numbersFromJson('not a list')).toEqual([])
  })
})
