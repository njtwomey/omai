import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { FLOAT_FORMATS, type FloatFormatName, machineEpsilon, roundToFormat, toBfloat16, toFloat16 } from './index.ts'

/** bfloat16 by its bit definition: the top 16 bits of the float32, rounded to nearest even on the lower 16. */
function bfloat16ByBits(x: number): number {
  const f = new Float32Array(1)
  const u = new Uint32Array(f.buffer)
  f[0] = x
  const bits = u[0] ?? 0
  const exponentAllOnes = (bits & 0x7f800000) === 0x7f800000
  if (exponentAllOnes) return f[0] ?? Number.NaN
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0
  u[0] = (rounded & 0xffff0000) >>> 0
  return f[0] ?? Number.NaN
}

describe('float16 and bfloat16 rounding', () => {
  it('knows the format constants', () => {
    expect(machineEpsilon('float16')).toBe(2 ** -10)
    expect(machineEpsilon('bfloat16')).toBe(2 ** -7)
    expect(FLOAT_FORMATS.float16.max).toBe(65504)
    expect(FLOAT_FORMATS.bfloat16.max).toBe(3.3895313892515355e38)
  })

  it('rounds ties to even and overflows past the largest finite value', () => {
    expect(toFloat16(1 + 2 ** -11)).toBe(1)
    expect(toFloat16(1 + 3 * 2 ** -11)).toBe(1 + 2 ** -9)
    expect(toFloat16(65519)).toBe(65504)
    expect(toFloat16(65520)).toBe(Number.POSITIVE_INFINITY)
    expect(toFloat16(-1e6)).toBe(Number.NEGATIVE_INFINITY)
    expect(toFloat16(2 ** -24)).toBe(2 ** -24)
    expect(toFloat16(2 ** -25)).toBe(0)
    expect(toFloat16(1.5 * 2 ** -25)).toBe(2 ** -24)
    expect(Number.isNaN(toFloat16(Number.NaN))).toBe(true)
    expect(Object.is(toFloat16(-0), -0)).toBe(true)
  })

  it('agrees with the bit-level definition of bfloat16', () => {
    for (const x of [0.1, -3.15, 1e-30, 1e30, 1 + 2 ** -8, 1 + 3 * 2 ** -8, 123456.789, 5e-39])
      expect(toBfloat16(x)).toBe(bfloat16ByBits(x))
    expect(toBfloat16(1e39)).toBe(Number.POSITIVE_INFINITY)
  })

  it('is idempotent, monotone and within half an epsilon in the normal range', () => {
    const formats: FloatFormatName[] = ['float16', 'bfloat16']
    for (const name of formats) {
      const f = FLOAT_FORMATS[name]
      fc.assert(
        fc.property(
          fc.double({ min: -1e4, max: 1e4, noNaN: true }),
          fc.double({ min: -1e4, max: 1e4, noNaN: true }),
          (a, b) => {
            const ra = roundToFormat(a, name)
            expect(roundToFormat(ra, name)).toBe(ra)
            if (a <= b) expect(ra).toBeLessThanOrEqual(roundToFormat(b, name))
            if (Math.abs(a) >= 2 ** f.minExponent)
              expect(Math.abs(ra - a)).toBeLessThanOrEqual((Math.abs(a) * machineEpsilon(name)) / 2 + 1e-300)
          },
        ),
        { numRuns: 300 },
      )
    }
  })
})
