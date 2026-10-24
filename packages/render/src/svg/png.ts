import { WhiteboxError } from 'whitebox-engine/base'
import type { Pixels } from 'whitebox-engine/scene'

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

/**
 * Encode pixels as a PNG file (8-bit RGBA, no interlacing). The image data is stored in uncompressed deflate blocks, so
 * encoding is fast and needs no compression library; files are about the size of the raw pixels.
 *
 * TODO: compress the image data (deflate with run-length matches would shrink blocky and posterised images a lot).
 *
 * @param pixels - The image.
 * @returns The PNG file's bytes.
 * @throws WhiteboxError with code `shape` when the pixel buffer does not match the size.
 * @example
 * ```ts
 * import { encodePng } from 'whitebox-render/svg'
 * const png = encodePng({ width: 1, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 255]) })
 * [...png.slice(1, 4)].map((c) => String.fromCharCode(c)).join('') // => 'PNG'
 * ```
 */
export function encodePng(pixels: Pixels): Uint8Array {
  const { width, height, rgba } = pixels
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0))
    throw new WhiteboxError('shape', 'encodePng: width and height must be positive integers')
  if (rgba.length !== width * height * 4) throw new WhiteboxError('shape', 'encodePng: rgba must hold width*height*4')
  const stride = width * 4 + 1
  const raw = new Uint8Array(stride * height)
  for (let y = 0; y < height; y++) {
    raw[y * stride] = 0
    raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * stride + 1)
  }
  const blocks = Math.max(1, Math.ceil(raw.length / 65535))
  const zlib = new Uint8Array(2 + raw.length + 5 * blocks + 4)
  zlib[0] = 0x78
  zlib[1] = 0x01
  let at = 2
  for (let b = 0; b < blocks; b++) {
    const start = b * 65535
    const len = Math.min(65535, raw.length - start)
    zlib[at++] = b === blocks - 1 ? 1 : 0
    zlib[at++] = len & 0xff
    zlib[at++] = len >>> 8
    zlib[at++] = ~len & 0xff
    zlib[at++] = (~len >>> 8) & 0xff
    zlib.set(raw.subarray(start, start + len), at)
    at += len
  }
  const adler = adler32(raw)
  zlib[at++] = adler >>> 24
  zlib[at++] = (adler >>> 16) & 0xff
  zlib[at++] = (adler >>> 8) & 0xff
  zlib[at++] = adler & 0xff

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

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

/**
 * Base64 text of bytes (standard alphabet, padded), without relying on `btoa` or `Buffer`.
 *
 * @param bytes - The bytes.
 * @returns The base64 text.
 * @example
 * ```ts
 * import { base64 } from 'whitebox-render/svg'
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
 * import { pngDataUrl } from 'whitebox-render/svg'
 * pngDataUrl({ width: 1, height: 1, rgba: new Uint8ClampedArray(4) }).startsWith('data:image/png;base64,') // => true
 * ```
 */
export function pngDataUrl(pixels: Pixels): string {
  return `data:image/png;base64,${base64(encodePng(pixels))}`
}
