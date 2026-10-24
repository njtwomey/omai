import { describe, expect, it } from 'vitest'
import * as I from './index.ts'

describe('stream coders', () => {
  it('compresses a skewed source close to its entropy', () => {
    const p = [0.9, 0.05, 0.05]
    const f = I.frequencyTable(p, 1000)
    const symbols: number[] = []
    for (let i = 0; i < 2000; i++) symbols.push(i % 20 === 0 ? 1 : i % 20 === 10 ? 2 : 0)
    const spec = { kind: 'static', frequencies: f } as const
    const bits = I.arithmeticEncode(symbols, spec)
    const h = -p.reduce((s, v) => s + v * Math.log2(v), 0)
    expect(bits.length / symbols.length).toBeLessThan(h + 0.01)
    expect(I.arithmeticDecode(bits, symbols.length, spec)).toEqual(symbols)
    const bytes = I.rangeEncode(symbols, spec)
    expect((8 * bytes.length) / symbols.length).toBeLessThan(h + 0.03)
    expect(I.rangeDecode(bytes, symbols.length, spec)).toEqual(symbols)
  })

  it('an adaptive model learns the source without being told', () => {
    const symbols = Array.from({ length: 3000 }, (_, i) => (i % 7 === 0 ? 1 : 0))
    const spec = { kind: 'adaptive', alphabetSize: 4 } as const
    const bits = I.arithmeticEncode(symbols, spec)
    const h = -(1 / 7) * Math.log2(1 / 7) - (6 / 7) * Math.log2(6 / 7)
    expect(bits.length / symbols.length).toBeLessThan(h + 0.02)
    expect(I.arithmeticDecode(bits, symbols.length, spec)).toEqual(symbols)
    expect(I.rangeDecode(I.rangeEncode(symbols, spec), symbols.length, spec)).toEqual(symbols)
  })

  it('encodes the empty sequence and rejects impossible symbols', () => {
    const spec = { kind: 'static', frequencies: [1, 0, 1] } as const
    expect(I.arithmeticDecode(I.arithmeticEncode([], spec), 0, spec)).toEqual([])
    expect(() => I.arithmeticEncode([1], spec)).toThrow()
    expect(() => I.rangeEncode([3], spec)).toThrow()
    expect(() => I.arithmeticEncode([0], { kind: 'static', frequencies: [] })).toThrow()
    expect(() => I.arithmeticEncode([0], { kind: 'static', frequencies: [0.5, 1] })).toThrow()
    expect(() => I.arithmeticEncode([0], { kind: 'static', frequencies: [70000, 1] })).toThrow()
    expect(() => I.frequencyTable([1, 1, 1], 2)).toThrow()
    expect(() => I.frequencyTable([0, 0])).toThrow()
  })
})
