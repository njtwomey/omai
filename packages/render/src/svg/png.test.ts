import { inflateRawSync, inflateSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { deflateRaw } from './_deflate.ts'
import { base64, encodePng, encodePngAsync, pngDataUrl } from './png.ts'

/** The pixels back from a PNG's image data: inflate, then undo each row's filter. */
function decode(png: Uint8Array): { width: number; height: number; rgba: number[] } {
  const view = new DataView(png.buffer, png.byteOffset)
  const width = view.getUint32(16)
  const height = view.getUint32(20)
  const raw = inflateSync(png.subarray(41, 41 + view.getUint32(33)))
  const stride = width * 4
  const out: number[] = []
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)] ?? 0
    for (let i = 0; i < stride; i++) {
      const x = raw[y * (stride + 1) + 1 + i] ?? 0
      const a = i >= 4 ? (out[y * stride + i - 4] ?? 0) : 0
      const b = y > 0 ? (out[(y - 1) * stride + i] ?? 0) : 0
      const c = y > 0 && i >= 4 ? (out[(y - 1) * stride + i - 4] ?? 0) : 0
      const p = a + b - c
      const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)]
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f] ?? 0
      out.push((x + pred) & 0xff)
    }
  }
  return { width, height, rgba: out }
}

const noisy = (n: number, seed = 1): Uint8Array => {
  let s = seed
  return Uint8Array.from({ length: n }, () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s >> 16
  })
}

describe('deflate', () => {
  it('round-trips through zlib’s inflate: text, runs, noise and the empty input', () => {
    const cases = [
      new Uint8Array(0),
      new TextEncoder().encode('abcabcabcabcabc the quick brown fox jumps over the lazy dog '.repeat(40)),
      new Uint8Array(100_000).fill(9),
      noisy(70_000),
      Uint8Array.from({ length: 50_000 }, (_, i) => (i * 7) % 13),
    ]
    for (const c of cases) expect(new Uint8Array(inflateRawSync(deflateRaw(c)))).toEqual(c)
  })

  it('compresses repetitive data', () => {
    expect(deflateRaw(new Uint8Array(100_000).fill(3)).length).toBeLessThan(2000)
  })
})

describe('png', () => {
  it('writes a valid PNG whose filtered, compressed data decodes back to the pixels', () => {
    const rgba = new Uint8ClampedArray(Array.from({ length: 3 * 2 * 4 }, (_, i) => (i * 37) % 256))
    const png = encodePng({ width: 3, height: 2, rgba })
    const back = decode(png)
    expect([back.width, back.height]).toEqual([3, 2])
    expect(back.rgba).toEqual([...rgba])
    expect(() => encodePng({ width: 2, height: 1, rgba })).toThrow()
    expect(() => encodePng({ width: 0, height: 1, rgba: new Uint8ClampedArray(0) })).toThrow()
  })

  it('compresses flat and smooth images far below their raw size, and round-trips noise', () => {
    const w = 200
    const h = 100
    const flat = new Uint8ClampedArray(w * h * 4).fill(7)
    expect(encodePng({ width: w, height: h, rgba: flat }).length).toBeLessThan(w * h * 4 * 0.02)
    const ramp = new Uint8ClampedArray(w * h * 4).map((_, i) => ((i >> 2) % w) + ((i & 3) === 3 ? 255 : 0))
    expect(encodePng({ width: w, height: h, rgba: ramp }).length).toBeLessThan(w * h * 4 * 0.05)
    expect(decode(encodePng({ width: w, height: h, rgba: ramp })).rgba).toEqual([...ramp])
    const noise = new Uint8ClampedArray(noisy(40 * 30 * 4, 5))
    expect(decode(encodePng({ width: 40, height: 30, rgba: noise })).rgba).toEqual([...noise])
  })

  it('compresses with the platform’s deflate when asked asynchronously', async () => {
    const ramp = new Uint8ClampedArray(64 * 64 * 4).map((_, i) => (i >> 2) % 64)
    const png = await encodePngAsync({ width: 64, height: 64, rgba: ramp })
    expect(decode(png).rgba).toEqual([...ramp])
  })

  it('encodes base64 like Buffer', () => {
    const bytes = new Uint8Array(Array.from({ length: 10000 }, (_, i) => i % 251))
    expect(base64(bytes)).toBe(Buffer.from(bytes).toString('base64'))
    expect(pngDataUrl({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) })).toMatch(/^data:image\/png;base64,/)
  })
})
