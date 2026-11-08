/**
 * A dependency-free PNG encoder: RGBA pixels written as a PNG file with each scanline filtered and compressed by a
 * small synchronous deflate, or by the platform's `CompressionStream` for smaller files, plus base64 and
 * `data:image/png` URLs for embedding.
 */

import { OmaiError } from 'omai-engine/base'
import type { Pixels } from 'omai-engine/scene'
import { deflateRaw } from './_deflate.ts'

const CRC_TABLE: readonly number[] = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

function crc32(bytes: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff
  for (let i = start; i < end; i++) c = (CRC_TABLE[(c ^ (bytes[i] ?? 0)) & 0xff] ?? 0) ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function adler32(bytes: Uint8Array): number {
  let a = 1
  let b = 0
  for (let i = 0; i < bytes.length; i += 5552) {
    const end = Math.min(bytes.length, i + 5552)
    for (let j = i; j < end; j++) {
      a += bytes[j] ?? 0
      b += a
    }
    a %= 65521
    b %= 65521
  }
  return ((b << 16) | a) >>> 0
}

/** The filtered scanlines of an image (PNG filter types 0 to 4, chosen per row by the least absolute sum). */
function filtered(pixels: Pixels): Uint8Array {
  const { width, height, rgba } = pixels
  const stride = width * 4
  const out = new Uint8Array((stride + 1) * height)
  const candidates = Array.from({ length: 5 }, () => new Uint8Array(stride))
  for (let y = 0; y < height; y++) {
    const row = rgba.subarray(y * stride, (y + 1) * stride)
    const up = y > 0 ? rgba.subarray((y - 1) * stride, y * stride) : undefined
    let best = 0
    let bestScore = Number.POSITIVE_INFINITY
    for (let f = 0; f < 5; f++) {
      const c = candidates[f] as Uint8Array
      let score = 0
      for (let i = 0; i < stride; i++) {
        const x = row[i] as number
        const a = i >= 4 ? (row[i - 4] as number) : 0
        const b = up === undefined ? 0 : (up[i] as number)
        const d = up === undefined || i < 4 ? 0 : (up[i - 4] as number)
        let v: number
        if (f === 0) v = x
        else if (f === 1) v = x - a
        else if (f === 2) v = x - b
        else if (f === 3) v = x - ((a + b) >> 1)
        else {
          const p = a + b - d
          const pa = Math.abs(p - a)
          const pb = Math.abs(p - b)
          const pc = Math.abs(p - d)
          v = x - (pa <= pb && pa <= pc ? a : pb <= pc ? b : d)
        }
        const byte = v & 0xff
        c[i] = byte
        score += byte < 128 ? byte : 256 - byte
      }
      if (score < bestScore) {
        bestScore = score
        best = f
      }
    }
    out[y * (stride + 1)] = best
    out.set(candidates[best] as Uint8Array, y * (stride + 1) + 1)
  }
  return out
}

function checkPixels(pixels: Pixels, name: string): void {
  const { width, height, rgba } = pixels
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0))
    throw new OmaiError('shape', `${name}: width and height must be positive integers`)
  if (rgba.length !== width * height * 4) throw new OmaiError('shape', `${name}: rgba must hold width*height*4`)
}

/** A PNG file around zlib-compressed image data. */
function pngFile(width: number, height: number, zlib: Uint8Array): Uint8Array {
  const ihdr = new Uint8Array(13)
  const view = new DataView(ihdr.buffer)
  view.setUint32(0, width)
  view.setUint32(4, height)
  ihdr.set([8, 6, 0, 0, 0], 8)
  const chunks: [string, Uint8Array][] = [
    ['IHDR', ihdr],
    ['IDAT', zlib],
    ['IEND', new Uint8Array(0)],
  ]
  const size = 8 + chunks.reduce((s, [, d]) => s + 12 + d.length, 0)
  const out = new Uint8Array(size)
  out.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0)
  const dv = new DataView(out.buffer)
  let p = 8
  for (const [type, data] of chunks) {
    dv.setUint32(p, data.length)
    for (let i = 0; i < 4; i++) out[p + 4 + i] = type.charCodeAt(i)
    out.set(data, p + 8)
    dv.setUint32(p + 8 + data.length, crc32(out, p + 4, p + 8 + data.length))
    p += 12 + data.length
  }
  return out
}

/**
 * Encode pixels as a PNG file (8-bit RGBA, no interlacing): each scanline filtered with the PNG filter that suits it
 * best, then compressed by a small synchronous deflate (LZ77 with the fixed Huffman codes), so it runs anywhere,
 * synchronously. `encodePngAsync` compresses harder where the platform has `CompressionStream`.
 *
 * @param pixels - The image.
 * @returns The PNG file's bytes.
 * @throws OmaiError with code `shape` when the pixel buffer does not match the size.
 * @index function RGBA pixels to PNG file bytes, with no dependency: `encodePng(pixels)`.
 * @example
 * ```ts
 * import { encodePng } from 'omai-render/svg'
 * const png = encodePng({ width: 1, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 255]) })
 * [...png.slice(1, 4)].map((c) => String.fromCharCode(c)).join('') // => 'PNG'
 * ```
 */
export function encodePng(pixels: Pixels): Uint8Array {
  checkPixels(pixels, 'encodePng')
  const raw = filtered(pixels)
  const body = deflateRaw(raw)
  const zlib = new Uint8Array(body.length + 6)
  zlib[0] = 0x78
  zlib[1] = 0x01
  zlib.set(body, 2)
  const adler = adler32(raw)
  new DataView(zlib.buffer).setUint32(body.length + 2, adler)
  return pngFile(pixels.width, pixels.height, zlib)
}

/**
 * Encode pixels as a PNG file, compressed with the platform's deflate (`CompressionStream`, in browsers and Node 18
 * and later) for smaller files than {@link encodePng}; where there is none, the same as `encodePng`.
 *
 * @param pixels - The image.
 * @returns A promise of the PNG file's bytes.
 * @throws OmaiError with code `shape` when the pixel buffer does not match the size.
 * @example
 * ```ts
 * import { encodePngAsync } from 'omai-render/svg'
 * const png = await encodePngAsync({ width: 2, height: 2, rgba: new Uint8ClampedArray(16).fill(255) })
 * png[1] // => 80
 * ```
 */
export async function encodePngAsync(pixels: Pixels): Promise<Uint8Array> {
  checkPixels(pixels, 'encodePngAsync')
  if (typeof CompressionStream === 'undefined') return encodePng(pixels)
  const raw = filtered(pixels)
  const stream = new Blob([raw as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream('deflate'))
  const zlib = new Uint8Array(await new Response(stream).arrayBuffer())
  return pngFile(pixels.width, pixels.height, zlib)
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

/**
 * Base64 text of bytes (standard alphabet, padded), without relying on `btoa` or `Buffer`.
 *
 * @param bytes - The bytes.
 * @returns The base64 text.
 * @example
 * ```ts
 * import { base64 } from 'omai-render/svg'
 * base64(new TextEncoder().encode('hi!')) // => 'aGkh'
 * base64(new Uint8Array([0])) // => 'AA=='
 * ```
 */
export function base64(bytes: Uint8Array): string {
  const parts: string[] = []
  let chunk = ''
  for (let i = 0; i < bytes.length; i += 3) {
    const a = bytes[i] ?? 0
    const b = bytes[i + 1] ?? 0
    const c = bytes[i + 2] ?? 0
    const n = (a << 16) | (b << 8) | c
    chunk +=
      (B64[(n >>> 18) & 63] ?? '') +
      (B64[(n >>> 12) & 63] ?? '') +
      (i + 1 < bytes.length ? (B64[(n >>> 6) & 63] ?? '') : '=') +
      (i + 2 < bytes.length ? (B64[n & 63] ?? '') : '=')
    if (chunk.length >= 8192) {
      parts.push(chunk)
      chunk = ''
    }
  }
  parts.push(chunk)
  return parts.join('')
}

/**
 * Pixels as a `data:image/png;base64,...` URL, for an `<img>`, an SVG `<image>` or an ECharts image element.
 *
 * @param pixels - The image.
 * @returns The data URL.
 * @example
 * ```ts
 * import { pngDataUrl } from 'omai-render/svg'
 * pngDataUrl({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) }).startsWith('data:image/png;base64,') // => true
 * ```
 */
export function pngDataUrl(pixels: Pixels): string {
  return `data:image/png;base64,${base64(encodePng(pixels))}`
}
