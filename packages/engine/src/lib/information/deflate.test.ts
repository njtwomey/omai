import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import * as I from './index.ts'

const STRATEGIES: readonly I.DeflateStrategy[] = ['default', 'filtered', 'huffmanOnly', 'rle', 'fixed']

describe('DEFLATE, zlib and gzip', () => {
  it('round-trips any bytes at every level and strategy', () => {
    fc.assert(
      fc.property(
        fc.oneof(
          fc.uint8Array({ maxLength: 3000 }),
          fc.array(fc.integer({ min: 97, max: 99 }), { maxLength: 3000 }).map((a) => Uint8Array.from(a)),
        ),
        fc.integer({ min: 0, max: 9 }),
        fc.constantFrom(...STRATEGIES),
        (data, level, strategy) => {
          expect(I.inflate(I.deflate(data, { level, strategy }))).toEqual(data)
          expect(I.zlibDecompress(I.zlibCompress(data, { level, strategy }))).toEqual(data)
          expect(I.gzipDecompress(I.gzipCompress(data, { level, strategy }))).toEqual(data)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('stores large inputs in 65535-byte blocks at level 0', () => {
    const data = new Uint8Array(70000).fill(7)
    const packed = I.deflate(data, { level: 0 })
    expect(packed.length).toBe(70000 + 2 * 5)
    expect(Array.from(packed.subarray(0, 5))).toEqual([0, 0xff, 0xff, 0, 0])
    expect(I.inflate(packed)).toEqual(data)
  })

  it('compresses long, repetitive and incompressible data sensibly', () => {
    const runs = new Uint8Array(100000).fill(42)
    expect(I.deflate(runs).length).toBeLessThan(200)
    expect(I.inflate(I.deflate(runs, { level: 1 }))).toEqual(runs)
    const noise = Uint8Array.from({ length: 5000 }, (_, i) => (i * 2654435761) >>> 24)
    expect(I.deflate(noise, { level: 9 }).length).toBeLessThanOrEqual(noise.length + 5 + 5)
  })

  it('parses every optional gzip header field and concatenated members', () => {
    const body = I.deflate('extra')
    const crc = I.crc32('extra')
    const le = (v: number): number[] => [v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff]
    const member = Uint8Array.from([
      0x1f,
      0x8b,
      8,
      4 | 8 | 16 | 2,
      0,
      0,
      0,
      0,
      0,
      3,
      3,
      0,
      1,
      2,
      3,
      ...Array.from(new TextEncoder().encode('name')),
      0,
      ...Array.from(new TextEncoder().encode('comment')),
      0,
      0xab,
      0xcd,
      ...body,
      ...le(crc),
      ...le(5),
    ])
    expect(new TextDecoder().decode(I.gzipDecompress(member))).toBe('extra')
    const two = Uint8Array.from([...member, ...I.gzipCompress('!')])
    expect(new TextDecoder().decode(I.gzipDecompress(two))).toBe('extra!')
  })

  it('rejects corrupt streams', () => {
    const good = I.zlibCompress('hello world, hello world')
    expect(() => I.zlibDecompress(good.slice(0, 4))).toThrow()
    expect(() => I.zlibDecompress([0x78, 0x9d, ...good.slice(2)])).toThrow()
    expect(() => I.zlibDecompress([0x78, 0xbb, ...good.slice(2)])).toThrow()
    const flipped = Uint8Array.from(good)
    flipped[flipped.length - 1] = (flipped[flipped.length - 1] ?? 0) ^ 1
    expect(() => I.zlibDecompress(flipped)).toThrow()
    expect(() => I.zlibDecompress(good.slice(0, good.length - 2))).toThrow()
    expect(() => I.inflate([0x07])).toThrow()
    expect(() => I.inflate([0x01, 5, 0, 0, 0])).toThrow()
    expect(() => I.inflate([0x01, 5, 0, 0xfa, 0xff, 1])).toThrow()
    expect(() => I.inflate([])).toThrow()
    const gz = I.gzipCompress('data')
    expect(() => I.gzipDecompress(gz.slice(0, 8))).toThrow()
    expect(() => I.gzipDecompress([0x1f, 0x8c, ...gz.slice(2)])).toThrow()
    const badCrc = Uint8Array.from(gz)
    badCrc[gz.length - 8] = (badCrc[gz.length - 8] ?? 0) ^ 1
    expect(() => I.gzipDecompress(badCrc)).toThrow()
    const badLength = Uint8Array.from(gz)
    badLength[gz.length - 4] = (badLength[gz.length - 4] ?? 0) ^ 1
    expect(() => I.gzipDecompress(badLength)).toThrow()
    expect(() => I.gzipDecompress(gz.slice(0, gz.length - 3))).toThrow()
    expect(() => I.deflate('x', { level: 10 })).toThrow()
    expect(() => I.asBytes([256])).toThrow()
  })

  it('checksums continue from a running value', () => {
    expect(I.crc32('world', I.crc32('hello '))).toBe(I.crc32('hello world'))
    expect(I.adler32('world', I.adler32('hello '))).toBe(I.adler32('hello world'))
    expect(I.adler32(new Uint8Array(20000).fill(255))).toBe(I.adler32(new Uint8Array(20000).fill(255)))
  })
})
