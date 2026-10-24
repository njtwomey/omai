// A small synchronous deflate (RFC 1951) for PNG and other encoders: LZ77 over a 32 KiB window with hash chains, and
// the fixed Huffman codes, in one final block. Not as tight as zlib's dynamic codes, but fast, dependency-free and
// several times smaller than stored blocks on images with flat areas or repeats.

const LENGTH_BASE = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
]
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const DIST_BASE = [
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
  8193, 12289, 16385, 24577,
]
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]

/** The code index (0-based, of codes 257..285) and extra bits of a match length 3..258. */
const LENGTH_CODE: Uint8Array = (() => {
  const out = new Uint8Array(259)
  for (let c = 0; c < LENGTH_BASE.length; c++) {
    const base = LENGTH_BASE[c] as number
    const next = c + 1 < LENGTH_BASE.length ? (LENGTH_BASE[c + 1] as number) : 259
    for (let l = base; l < next && l <= 258; l++) out[l] = c
  }
  out[258] = 28
  return out
})()

function distCode(d: number): number {
  let lo = 0
  let hi = DIST_BASE.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if ((DIST_BASE[mid] as number) <= d) lo = mid
    else hi = mid - 1
  }
  return lo
}

class BitWriter {
  private buf: Uint8Array
  private pos = 0
  private bits = 0
  private count = 0
  constructor(size: number) {
    this.buf = new Uint8Array(Math.max(64, size))
  }
  /** Write `n` bits of `value`, least significant first. */
  write(value: number, n: number): void {
    this.bits |= value << this.count
    this.count += n
    while (this.count >= 8) {
      if (this.pos >= this.buf.length) {
        const grown = new Uint8Array(this.buf.length * 2)
        grown.set(this.buf)
        this.buf = grown
      }
      this.buf[this.pos++] = this.bits & 0xff
      this.bits >>>= 8
      this.count -= 8
    }
  }
  /** Write a Huffman code of `n` bits, most significant first. */
  code(value: number, n: number): void {
    let r = 0
    for (let i = 0; i < n; i++) r |= ((value >> i) & 1) << (n - 1 - i)
    this.write(r, n)
  }
  finish(): Uint8Array {
    if (this.count > 0) this.write(0, 8 - this.count)
    return this.buf.subarray(0, this.pos)
  }
}

function literal(w: BitWriter, v: number): void {
  if (v < 144) w.code(0x30 + v, 8)
  else if (v < 256) w.code(0x190 + v - 144, 9)
  else if (v < 280) w.code(v - 256, 7)
  else w.code(0xc0 + v - 280, 8)
}

const WINDOW = 32768
const HASH = 1 << 15
const CHAIN = 48

/**
 * Raw deflate data of bytes (one final block with the fixed Huffman codes).
 *
 * @param data - The bytes.
 * @returns The compressed stream (no zlib header).
 */
export function deflateRaw(data: Uint8Array): Uint8Array {
  const n = data.length
  const w = new BitWriter(n / 2 + 64)
  w.write(1, 1) // final block
  w.write(1, 2) // fixed Huffman codes
  const head = new Int32Array(HASH).fill(-1)
  const prev = new Int32Array(WINDOW)
  const hash = (i: number): number =>
    (((data[i] as number) << 10) ^ ((data[i + 1] as number) << 5) ^ (data[i + 2] as number)) & (HASH - 1)
  const insert = (i: number): void => {
    if (i + 2 >= n) return
    const h = hash(i)
    prev[i & (WINDOW - 1)] = head[h] as number
    head[h] = i
  }
  let i = 0
  while (i < n) {
    let bestLen = 0
    let bestDist = 0
    if (i + 2 < n) {
      let j = head[hash(i)] as number
      let chain = CHAIN
      const max = Math.min(258, n - i)
      while (j >= 0 && i - j <= WINDOW && chain-- > 0) {
        if (data[j + bestLen] === data[i + bestLen]) {
          let l = 0
          while (l < max && data[j + l] === data[i + l]) l++
          if (l > bestLen) {
            bestLen = l
            bestDist = i - j
            if (l === max) break
          }
        }
        const p = prev[j & (WINDOW - 1)] as number
        if (p >= j) break
        j = p
      }
    }
    if (bestLen >= 3) {
      const lc = LENGTH_CODE[bestLen] as number
      literal(w, 257 + lc)
      const le = LENGTH_EXTRA[lc] as number
      if (le > 0) w.write(bestLen - (LENGTH_BASE[lc] as number), le)
      const dc = distCode(bestDist)
      w.code(dc, 5)
      const de = DIST_EXTRA[dc] as number
      if (de > 0) w.write(bestDist - (DIST_BASE[dc] as number), de)
      for (let k = 0; k < bestLen; k++) insert(i + k)
      i += bestLen
    } else {
      literal(w, data[i] as number)
      insert(i)
      i++
    }
  }
  literal(w, 256)
  return w.finish()
}
