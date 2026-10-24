import { describe, expect, it } from 'vitest'
import * as I from './index.ts'

const bytes = (s: string): number[] => Array.from(new TextEncoder().encode(s))

describe('dictionary codes', () => {
  it('LZW reproduces the textbook example and handles the cScSc case', () => {
    expect(I.lzwEncode('TOBEORNOTTOBEORTOBEORNOT')).toEqual([
      84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, 265, 259, 261, 263,
    ])
    // 'aaaa': the third code refers to the entry being defined.
    expect(I.lzwEncode('aaaa')).toEqual([97, 256, 97])
    expect(I.lzwDecode([97, 256, 97])).toEqual(bytes('aaaa'))
    expect(I.lzwEncode('')).toEqual([])
    expect(I.lzwEncode([0, 1, 0, 1], { alphabetSize: 2 })).toEqual([0, 1, 2])
    expect(() => I.lzwEncode([5], { alphabetSize: 2 })).toThrow()
    expect(() => I.lzwEncode([0], { alphabetSize: 4, maxCodes: 2 })).toThrow()
    expect(() => I.lzwDecode([300])).toThrow()
  })

  it('LZ77 compresses repetition and finds overlapping matches', () => {
    const tokens = I.lz77Encode('aaaaaaaaab')
    expect(tokens).toEqual([
      { offset: 0, length: 0, next: 97 },
      { offset: 1, length: 8, next: 98 },
    ])
    expect(I.lz77Decode(tokens)).toEqual(bytes('aaaaaaaaab'))
    expect(I.lz77Encode('')).toEqual([])
    expect(() => I.lz77Encode('ab', { window: 0 })).toThrow()
    expect(() => I.lz77Decode([{ offset: 2, length: 1 }])).toThrow()
    expect(() => I.lz77Encode([-1])).toThrow()
  })

  it('LZ78 grows its dictionary by one phrase per pair', () => {
    const pairs = I.lz78Encode('abababab')
    expect(pairs.map((p) => p.index)).toEqual([0, 0, 1, 3, 2])
    expect(I.lz78Decode(pairs)).toEqual(bytes('abababab'))
    expect(() => I.lz78Decode([{ index: 4, next: 1 }])).toThrow()
  })

  it('BWT and move-to-front turn context into runs of small numbers', () => {
    const t = I.bwt('abracadabra')
    expect(new TextDecoder().decode(Uint8Array.from(t.last))).toBe('rdarcaaaabb')
    expect(I.inverseBwt(t)).toEqual(bytes('abracadabra'))
    expect(I.bwt('')).toEqual({ last: [], index: 0 })
    expect(I.inverseBwt({ last: [], index: 0 })).toEqual([])
    expect(() => I.inverseBwt({ last: [1], index: 2 })).toThrow()
    const mtf = I.moveToFront(t.last)
    expect(mtf.filter((v) => v === 0).length).toBe(4)
    expect(() => I.moveToFront([3], 3)).toThrow()
    expect(() => I.inverseMoveToFront([3], 3)).toThrow()
  })
})
