import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { type Encodable, fromJson, type Json, JsonSchema, toJson } from './json.ts'
import { resolveConfidenceLevel } from './options.ts'
import { estimate } from './results.ts'

const leaf = fc.oneof(
  fc.constant(null),
  fc.boolean(),
  fc.double(),
  fc.bigInt(),
  fc.string(),
  fc.float64Array(),
  fc.int32Array(),
  fc.uint8Array(),
)
const value = fc.letrec<{ v: Encodable }>((tie) => ({
  v: fc.oneof(
    { depthSize: 'small' },
    leaf,
    fc.array(tie('v'), { maxLength: 4 }),
    fc.dictionary(fc.oneof(fc.string(), fc.constantFrom('$number', '$typed', '$object', '$x', '__proto__')), tie('v'), {
      maxKeys: 4,
    }),
  ),
})).v

describe('toJson and fromJson', () => {
  it('round-trip any encodable value through JSON text', () => {
    fc.assert(
      fc.property(value, (x) => {
        const text = JSON.stringify(toJson(x))
        const parsed: unknown = JSON.parse(text)
        expect(JsonSchema.safeParse(parsed).success).toBe(true)
        expect(fromJson(parsed as Json)).toEqual(x)
      }),
    )
  })

  it('keeps signed zeros, non-finite values and typed-array classes', () => {
    const back = fromJson(toJson({ z: -0, a: new Float32Array([Infinity, -Infinity, Number.NaN]) })) as {
      z: number
      a: Float32Array
    }
    expect(Object.is(back.z, -0)).toBe(true)
    expect(back.a).toBeInstanceOf(Float32Array)
    expect(Array.from(back.a)).toEqual([Infinity, -Infinity, Number.NaN])
  })

  it('drops undefined fields and rejects what JSON cannot hold', () => {
    expect(toJson({ a: 1, b: undefined })).toEqual({ a: 1 })
    expect(() => toJson(new Map() as unknown as Encodable)).toThrow(/class instance/)
    expect(() => fromJson({ $typed: 'Nope', values: [] })).toThrow(/typed array/)
    expect(() => fromJson({ $what: 1 })).toThrow(/unknown tag/)
  })

  it('checks confidence levels', () => {
    expect(resolveConfidenceLevel()).toBe(0.95)
    expect(() => resolveConfidenceLevel(1)).toThrow(/\(0, 1\)/)
    expect(() => resolveConfidenceLevel(Number.NaN)).toThrow()
    // `estimate` takes the same check (it gave an interval of NaN for a level of 1.5).
    expect(() => estimate(1, 0.5, 1.5)).toThrow(expect.objectContaining({ code: 'value' }))
    expect(estimate(1, 0.5, 0.9).level).toBe(0.9)
  })
})
