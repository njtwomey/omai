import { inflateSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { base64, encodePng, pngDataUrl } from './png.ts'

describe('png', () => {
  it('writes a valid PNG whose image data inflates back to the pixels', () => {
    const rgba = new Uint8ClampedArray(Array.from({ length: 3 * 2 * 4 }, (_, i) => (i * 37) % 256))
    const png = encodePng({ width: 3, height: 2, rgba })
    const view = new DataView(png.buffer)
    expect(view.getUint32(16)).toBe(3)
    expect(view.getUint32(20)).toBe(2)
    const idatLength = view.getUint32(33)
    const raw = inflateSync(png.subarray(41, 41 + idatLength))
    expect(raw.length).toBe(2 * (3 * 4 + 1))
    expect([...raw.subarray(1, 13)]).toEqual([...rgba.subarray(0, 12)])
    expect(() => encodePng({ width: 2, height: 1, rgba })).toThrow()
    expect(() => encodePng({ width: 0, height: 1, rgba: new Uint8ClampedArray(0) })).toThrow()
  })
  it('splits large images into several stored blocks', () => {
    const big = new Uint8ClampedArray(200 * 100 * 4).fill(7)
    const png = encodePng({ width: 200, height: 100, rgba: big })
    const view = new DataView(png.buffer)
    expect(inflateSync(png.subarray(41, 41 + view.getUint32(33))).length).toBe(100 * 801)
  })
  it('encodes base64 like Buffer', () => {
    const bytes = new Uint8Array(Array.from({ length: 10000 }, (_, i) => i % 251))
    expect(base64(bytes)).toBe(Buffer.from(bytes).toString('base64'))
    expect(pngDataUrl({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) })).toMatch(/^data:image\/png;base64,/)
  })
})
