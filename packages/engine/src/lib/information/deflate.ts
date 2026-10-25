/**
 * DEFLATE compression (LZ77 matching followed by Huffman coding), the algorithm inside zip, gzip and PNG: raw `deflate`
 * and `inflate` streams, the zlib and gzip container formats (byte-identical to Python's zlib and gzip at the same
 * level), and the CRC-32 and Adler-32 checksums they use.
 */

import { WhiteboxError } from '../base/index.ts'
import { bitSink, type DeflateStrategy, deflateRaw } from './_deflate.ts'

export type { DeflateStrategy } from './_deflate.ts'

/** Bytes to compress: a byte array, an array of byte values, or a string (encoded as UTF-8). */
export type ByteInput = Uint8Array | ArrayLike<number> | string

/**
 * Convert byte input to a Uint8Array: strings are encoded as UTF-8, arrays must hold integers in [0, 255].
 *
 * @param input - Bytes, byte values or a string.
 * @returns The bytes (the input itself when it is already a Uint8Array).
 * @throws WhiteboxError with code `value` for values that are not bytes.
 * @concept deflate
 * @example
 * ```ts
 * import { asBytes } from 'whitebox-engine/information'
 * Array.from(asBytes('hé')) // => [104, 195, 169]
 * ```
 */
export function asBytes(input: ByteInput): Uint8Array {
  if (typeof input === 'string') return new TextEncoder().encode(input)
  if (input instanceof Uint8Array) return input
  const out = new Uint8Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const b = input[i] ?? 0
    if (!Number.isInteger(b) || b < 0 || b > 255) throw new WhiteboxError('value', `bytes: ${b} is not a byte`)
    out[i] = b
  }
  return out
}

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

/**
 * The CRC-32 of the gzip and PNG formats (reflected polynomial 0xEDB88320), as zlib.crc32.
 *
 * @param input - The bytes.
 * @param crc - A running CRC to continue (default 0).
 * @returns The CRC as an unsigned 32-bit integer.
 * @concept checksums
 * @example
 * ```ts
 * import { crc32 } from 'whitebox-engine/information'
 * crc32('123456789') // => 3421780262
 * ```
 */
export function crc32(input: ByteInput, crc = 0): number {
  const bytes = asBytes(input)
  let c = ~crc >>> 0
  for (const b of bytes) c = (CRC_TABLE[(c ^ b) & 0xff] ?? 0) ^ (c >>> 8)
  return ~c >>> 0
}

/**
 * The Adler-32 checksum of the zlib format (two sums modulo 65521), as zlib.adler32.
 *
 * @param input - The bytes.
 * @param adler - A running checksum to continue (default 1).
 * @returns The checksum as an unsigned 32-bit integer.
 * @concept checksums
 * @example
 * ```ts
 * import { adler32 } from 'whitebox-engine/information'
 * adler32('Wikipedia') // => 300286872
 * ```
 */
export function adler32(input: ByteInput, adler = 1): number {
  const bytes = asBytes(input)
  let a = adler & 0xffff
  let b = (adler >>> 16) & 0xffff
  for (let i = 0; i < bytes.length; ) {
    const end = Math.min(bytes.length, i + 5552)
    for (; i < end; i++) {
      a += bytes[i] ?? 0
      b += a
    }
    a %= 65521
    b %= 65521
  }
  return ((b << 16) | a) >>> 0
}

/** Options of the DEFLATE compressors. */
export interface DeflateOptions {
  /** 0 (stored) to 9 (best), default 6, as zlib. */
  readonly level?: number
  /** zlib's strategy (default `'default'`). */
  readonly strategy?: DeflateStrategy
}

function checkLevel(level: number): void {
  if (!Number.isInteger(level) || level < 0 || level > 9)
    throw new WhiteboxError('value', `deflate: level must be an integer from 0 to 9, got ${level}`)
}

/**
 * Compress to a raw DEFLATE stream (RFC 1951): LZ77 with a 32 KiB window and hash chains, then Huffman coding of
 * literals, lengths and distances in fixed, dynamic or stored blocks. A port of zlib's compressor (lazy matching from
 * level 4, its block-type choice and tree construction), so the output is byte-identical to zlib 1.2 with
 * windowBits -15 and memLevel 8 (at level 0, to zlib given a large enough output buffer).
 *
 * @param input - The bytes (or a string, as UTF-8).
 * @param options - Level and strategy.
 * @returns The compressed bytes.
 * @concept deflate
 * @example
 * ```ts
 * import { deflate, inflate } from 'whitebox-engine/information'
 * const packed = deflate('abcabcabcabcabcabc')
 * packed.length // => 7
 * new TextDecoder().decode(inflate(packed)) // => 'abcabcabcabcabcabc'
 * ```
 */
export function deflate(input: ByteInput, options: DeflateOptions = {}): Uint8Array {
  const level = options.level ?? 6
  checkLevel(level)
  const sink = bitSink()
  deflateRaw(asBytes(input), level, options.strategy ?? 'default', sink)
  return sink.bytes()
}

/** Huffman decoding table: counts of codes per length and symbols in canonical order (Adler's puff). */
interface Huffman {
  readonly count: Int32Array
  readonly symbol: Int32Array
}

function construct(lengths: ArrayLike<number>, n: number): { readonly table: Huffman; readonly left: number } {
  const count = new Int32Array(16)
  const symbol = new Int32Array(n)
  for (let s = 0; s < n; s++) count[lengths[s] ?? 0] = (count[lengths[s] ?? 0] ?? 0) + 1
  if ((count[0] ?? 0) === n) return { table: { count, symbol }, left: 0 }
  let left = 1
  for (let len = 1; len < 16; len++) {
    left <<= 1
    left -= count[len] ?? 0
    if (left < 0) return { table: { count, symbol }, left }
  }
  const offs = new Int32Array(16)
  for (let len = 1; len < 15; len++) offs[len + 1] = (offs[len] ?? 0) + (count[len] ?? 0)
  for (let s = 0; s < n; s++) {
    const l = lengths[s] ?? 0
    if (l !== 0) {
      symbol[offs[l] ?? 0] = s
      offs[l] = (offs[l] ?? 0) + 1
    }
  }
  return { table: { count, symbol }, left }
}

const LENGTH_BASE = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
]
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const DIST_BASE = [
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
  8193, 12289, 16385, 24577,
]
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
const CODE_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]

const FIXED = ((): { readonly lencode: Huffman; readonly distcode: Huffman } => {
  const lengths = new Int32Array(288)
  for (let s = 0; s < 288; s++) lengths[s] = s < 144 ? 8 : s < 256 ? 9 : s < 280 ? 7 : 8
  const dist = new Int32Array(30).fill(5)
  return { lencode: construct(lengths, 288).table, distcode: construct(dist, 30).table }
})()

/** Inflate a raw DEFLATE stream starting at `start`; returns the output and the position after the stream. */
function inflateAt(data: Uint8Array, start: number): { readonly output: Uint8Array; readonly end: number } {
  let pos = start
  let bitbuf = 0
  let bitcnt = 0
  let out = new Uint8Array(Math.max(1024, 4 * (data.length - start)))
  let size = 0
  const fail = (why: string): never => {
    throw new WhiteboxError('value', `inflate: ${why}`)
  }
  const push = (b: number): void => {
    if (size === out.length) {
      const grown = new Uint8Array(out.length * 2)
      grown.set(out)
      out = grown
    }
    out[size++] = b
  }
  const bits = (need: number): number => {
    let val = bitbuf
    while (bitcnt < need) {
      if (pos >= data.length) fail('unexpected end of data')
      val |= (data[pos++] ?? 0) << bitcnt
      bitcnt += 8
    }
    bitbuf = val >>> need
    bitcnt -= need
    return val & ((1 << need) - 1)
  }
  const decode = (h: Huffman): number => {
    let code = 0
    let first = 0
    let index = 0
    for (let len = 1; len < 16; len++) {
      code |= bits(1)
      const count = h.count[len] ?? 0
      if (code - count < first) return h.symbol[index + (code - first)] ?? 0
      index += count
      first += count
      first <<= 1
      code <<= 1
    }
    return fail('invalid Huffman code')
  }
  const codes = (lencode: Huffman, distcode: Huffman): void => {
    for (;;) {
      let symbol = decode(lencode)
      if (symbol < 256) push(symbol)
      else if (symbol === 256) return
      else {
        symbol -= 257
        if (symbol >= 29) fail('invalid length code')
        const len = (LENGTH_BASE[symbol] ?? 0) + bits(LENGTH_EXTRA[symbol] ?? 0)
        const ds = decode(distcode)
        if (ds >= 30) fail('invalid distance code')
        const dist = (DIST_BASE[ds] ?? 0) + bits(DIST_EXTRA[ds] ?? 0)
        if (dist > size) fail('distance too far back')
        for (let i = 0; i < len; i++) push(out[size - dist] ?? 0)
      }
    }
  }
  let last = 0
  do {
    last = bits(1)
    const type = bits(2)
    if (type === 0) {
      bitbuf = 0
      bitcnt = 0
      if (pos + 4 > data.length) fail('unexpected end of data')
      const len = (data[pos] ?? 0) | ((data[pos + 1] ?? 0) << 8)
      const nlen = (data[pos + 2] ?? 0) | ((data[pos + 3] ?? 0) << 8)
      pos += 4
      if (len !== (~nlen & 0xffff)) fail('stored block length check failed')
      if (pos + len > data.length) fail('unexpected end of data')
      for (let i = 0; i < len; i++) push(data[pos + i] ?? 0)
      pos += len
    } else if (type === 1) codes(FIXED.lencode, FIXED.distcode)
    else if (type === 2) {
      const nlen = bits(5) + 257
      const ndist = bits(5) + 1
      const ncode = bits(4) + 4
      if (nlen > 286 || ndist > 30) fail('bad counts')
      const lengths = new Int32Array(320)
      for (let i = 0; i < ncode; i++) lengths[CODE_ORDER[i] ?? 0] = bits(3)
      const lencode = construct(lengths, 19)
      if (lencode.left !== 0) fail('incomplete code-length code')
      for (let index = 0; index < nlen + ndist; ) {
        const symbol = decode(lencode.table)
        if (symbol < 16) lengths[index++] = symbol
        else {
          let len = 0
          let repeat: number
          if (symbol === 16) {
            if (index === 0) fail('repeat with no first length')
            len = lengths[index - 1] ?? 0
            repeat = 3 + bits(2)
          } else if (symbol === 17) repeat = 3 + bits(3)
          else repeat = 11 + bits(7)
          if (index + repeat > nlen + ndist) fail('too many lengths')
          for (; repeat > 0; repeat--) lengths[index++] = len
        }
      }
      if ((lengths[256] ?? 0) === 0) fail('no end-of-block code')
      const lit = construct(lengths, nlen)
      // As puff: an incomplete code is allowed only when it is a single code of length one.
      if (lit.left < 0 || (lit.left > 0 && nlen !== (lit.table.count[0] ?? 0) + (lit.table.count[1] ?? 0)))
        fail('incomplete literal/length code')
      const dist = construct(lengths.subarray(nlen), ndist)
      if (dist.left < 0 || (dist.left > 0 && ndist !== (dist.table.count[0] ?? 0) + (dist.table.count[1] ?? 0)))
        fail('incomplete distance code')
      codes(lit.table, dist.table)
    } else fail('invalid block type')
  } while (last === 0)
  return { output: out.slice(0, size), end: pos }
}

/**
 * Decompress a raw DEFLATE stream (RFC 1951): stored, fixed-Huffman and dynamic-Huffman blocks, with the checks of
 * Adler's reference decoder `puff` (over-subscribed or incomplete codes, distances too far back, truncation).
 *
 * @param input - The compressed bytes.
 * @returns The decompressed bytes.
 * @throws WhiteboxError with code `value` for invalid or truncated data.
 * @concept deflate
 * @example
 * ```ts
 * import { inflate } from 'whitebox-engine/information'
 * // A fixed-Huffman block holding 'a'.
 * Array.from(inflate([0x4b, 0x04, 0x00])) // => [97]
 * ```
 */
export function inflate(input: ByteInput): Uint8Array {
  return inflateAt(asBytes(input), 0).output
}

/**
 * Compress to the zlib format (RFC 1950): a two-byte header (window size and level), the DEFLATE stream and the Adler-32
 * of the data, byte-identical to Python's zlib.compress at the same level.
 *
 * @param input - The bytes (or a string, as UTF-8).
 * @param options - Level and strategy.
 * @returns The zlib stream.
 * @concept deflate
 * @example
 * ```ts
 * import { zlibCompress } from 'whitebox-engine/information'
 * Array.from(zlibCompress('')) // => [120, 156, 3, 0, 0, 0, 0, 1]
 * ```
 */
export function zlibCompress(input: ByteInput, options: DeflateOptions = {}): Uint8Array {
  const bytes = asBytes(input)
  const level = options.level ?? 6
  checkLevel(level)
  const strategy = options.strategy ?? 'default'
  const flags =
    strategy === 'huffmanOnly' || strategy === 'rle' || strategy === 'fixed' || level < 2
      ? 0
      : level < 6
        ? 1
        : level === 6
          ? 2
          : 3
  let header = (8 + ((15 - 8) << 4)) << 8
  header |= flags << 6
  header += 31 - (header % 31)
  const body = deflate(bytes, { level, strategy })
  const check = adler32(bytes)
  const out = new Uint8Array(body.length + 6)
  out[0] = header >>> 8
  out[1] = header & 0xff
  out.set(body, 2)
  out[body.length + 2] = check >>> 24
  out[body.length + 3] = (check >>> 16) & 0xff
  out[body.length + 4] = (check >>> 8) & 0xff
  out[body.length + 5] = check & 0xff
  return out
}

/**
 * Decompress the zlib format, checking the header and the Adler-32 trailer.
 *
 * @param input - The zlib stream.
 * @returns The decompressed bytes.
 * @throws WhiteboxError with code `value` for a bad header, a preset dictionary, corrupt data or a checksum mismatch.
 * @concept deflate
 * @example
 * ```ts
 * import { zlibCompress, zlibDecompress } from 'whitebox-engine/information'
 * new TextDecoder().decode(zlibDecompress(zlibCompress('hello, hello'))) // => 'hello, hello'
 * ```
 */
export function zlibDecompress(input: ByteInput): Uint8Array {
  const data = asBytes(input)
  if (data.length < 6) throw new WhiteboxError('value', 'zlibDecompress: stream too short')
  const cmf = data[0] ?? 0
  const flg = data[1] ?? 0
  if ((cmf & 0x0f) !== 8 || cmf >> 4 > 7 || ((cmf << 8) | flg) % 31 !== 0)
    throw new WhiteboxError('value', 'zlibDecompress: invalid header')
  if (flg & 0x20) throw new WhiteboxError('value', 'zlibDecompress: preset dictionaries are not supported')
  const { output, end } = inflateAt(data, 2)
  if (end + 4 > data.length) throw new WhiteboxError('value', 'zlibDecompress: missing Adler-32 trailer')
  const want =
    (((data[end] ?? 0) << 24) | ((data[end + 1] ?? 0) << 16) | ((data[end + 2] ?? 0) << 8) | (data[end + 3] ?? 0)) >>> 0
  if (adler32(output) !== want) throw new WhiteboxError('value', 'zlibDecompress: Adler-32 mismatch')
  return output
}

/** Options of {@link gzipCompress}. */
export interface GzipOptions extends DeflateOptions {
  /** Modification time in seconds since 1970 (default 0: unknown). */
  readonly mtime?: number
  /** Operating system code (default 255: unknown, as Python's gzip module writes). */
  readonly os?: number
  /** An original file name to store (FNAME, Latin-1). */
  readonly name?: string
}

/**
 * Compress to the gzip format (RFC 1952): a header (with the modification time, extra flags 2 for level 9 and 4 for
 * level 1 as Python's gzip module sets them, and the OS), the DEFLATE stream, the CRC-32 and the length modulo 2^32.
 * With the same mtime this is byte-identical to Python's gzip.compress.
 *
 * @param input - The bytes (or a string, as UTF-8).
 * @param options - Level, strategy, mtime, OS and file name.
 * @returns The gzip member.
 * @concept deflate
 * @example
 * ```ts
 * import { gzipCompress } from 'whitebox-engine/information'
 * Array.from(gzipCompress('', { level: 9 }).subarray(0, 10)) // => [31, 139, 8, 0, 0, 0, 0, 0, 2, 255]
 * ```
 */
export function gzipCompress(input: ByteInput, options: GzipOptions = {}): Uint8Array {
  const bytes = asBytes(input)
  const level = options.level ?? 9
  checkLevel(level)
  const mtime = options.mtime ?? 0
  const name = options.name === undefined ? undefined : Uint8Array.from(options.name, (c) => c.charCodeAt(0) & 0xff)
  const header = [
    0x1f,
    0x8b,
    8,
    name === undefined ? 0 : 8,
    mtime & 0xff,
    (mtime >>> 8) & 0xff,
    (mtime >>> 16) & 0xff,
    (mtime >>> 24) & 0xff,
    level === 9 ? 2 : level === 1 ? 4 : 0,
    options.os ?? 255,
  ]
  const body = deflate(bytes, { level, ...(options.strategy === undefined ? {} : { strategy: options.strategy }) })
  const crc = crc32(bytes)
  const size = bytes.length >>> 0
  const tail = [crc, size].flatMap((v) => [v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff])
  const nameBytes = name === undefined ? [] : [...name, 0]
  return Uint8Array.from([...header, ...nameBytes, ...body, ...tail])
}

/**
 * Decompress gzip data: one or more members, each with its header fields (extra field, name, comment, header CRC)
 * skipped and its CRC-32 and length checked.
 *
 * @param input - The gzip bytes.
 * @returns The concatenated decompressed members.
 * @throws WhiteboxError with code `value` for a bad header, corrupt data or a trailer mismatch.
 * @concept deflate
 * @example
 * ```ts
 * import { gzipCompress, gzipDecompress } from 'whitebox-engine/information'
 * new TextDecoder().decode(gzipDecompress(gzipCompress('gzip me', { name: 'a.txt' }))) // => 'gzip me'
 * ```
 */
export function gzipDecompress(input: ByteInput): Uint8Array {
  const data = asBytes(input)
  const parts: Uint8Array[] = []
  let pos = 0
  const fail = (why: string): never => {
    throw new WhiteboxError('value', `gzipDecompress: ${why}`)
  }
  do {
    if (pos + 10 > data.length) fail('truncated header')
    if (data[pos] !== 0x1f || data[pos + 1] !== 0x8b || data[pos + 2] !== 8) fail('not a gzip member')
    const flags = data[pos + 3] ?? 0
    pos += 10
    if (flags & 4) {
      const xlen = (data[pos] ?? 0) | ((data[pos + 1] ?? 0) << 8)
      pos += 2 + xlen
    }
    for (const bit of [8, 16]) {
      if (flags & bit) {
        while (pos < data.length && data[pos] !== 0) pos++
        pos++
      }
    }
    if (flags & 2) pos += 2
    if (pos > data.length) fail('truncated header')
    const { output, end } = inflateAt(data, pos)
    if (end + 8 > data.length) fail('missing trailer')
    const word = (i: number): number =>
      ((data[i] ?? 0) | ((data[i + 1] ?? 0) << 8) | ((data[i + 2] ?? 0) << 16) | ((data[i + 3] ?? 0) << 24)) >>> 0
    if (crc32(output) !== word(end)) fail('CRC-32 mismatch')
    if (output.length >>> 0 !== word(end + 4)) fail('length mismatch')
    parts.push(output)
    pos = end + 8
  } while (pos < data.length)
  const total = parts.reduce((s, p) => s + p.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}
