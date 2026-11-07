/**
 * A port of zlib's compressor (deflate.c and trees.c, version 1.2.12, by Jean-loup Gailly and Mark Adler) for one-shot
 * compression with windowBits 15 and memLevel 8, so that the output matches zlib byte for byte. Private: the public
 * functions are in deflate.ts, and names follow zlib's so the two can be read side by side.
 */

/** zlib's compression strategies. */
export type DeflateStrategy = 'default' | 'filtered' | 'huffmanOnly' | 'rle' | 'fixed'

const MAX_BITS = 15
const MAX_BL_BITS = 7
const LENGTH_CODES = 29
const LITERALS = 256
const L_CODES = LITERALS + 1 + LENGTH_CODES
const D_CODES = 30
const BL_CODES = 19
const HEAP_SIZE = 2 * L_CODES + 1
const END_BLOCK = 256
const REP_3_6 = 16
const REPZ_3_10 = 17
const REPZ_11_138 = 18
const MIN_MATCH = 3
const MAX_MATCH = 258
const MIN_LOOKAHEAD = MAX_MATCH + MIN_MATCH + 1
const W_BITS = 15
const W_SIZE = 1 << W_BITS
const W_MASK = W_SIZE - 1
const MAX_DIST = W_SIZE - MIN_LOOKAHEAD
const WINDOW_SIZE = 2 * W_SIZE
const HASH_BITS = 8 + 7
const HASH_SIZE = 1 << HASH_BITS
const HASH_MASK = HASH_SIZE - 1
const HASH_SHIFT = Math.floor((HASH_BITS + MIN_MATCH - 1) / MIN_MATCH)
const LIT_BUFSIZE = 1 << (8 + 6)
const SYM_END = (LIT_BUFSIZE - 1) * 3
const TOO_FAR = 4096
const WIN_INIT = MAX_MATCH
const MAX_STORED = 65535

const EXTRA_LBITS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const EXTRA_DBITS = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
const EXTRA_BLBITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7]
const BL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]

/** good, lazy, nice, chain and which compressor, by level (zlib's configuration_table). */
const CONFIG: readonly (readonly [number, number, number, number, 'stored' | 'fast' | 'slow'])[] = [
  [0, 0, 0, 0, 'stored'],
  [4, 4, 8, 4, 'fast'],
  [4, 5, 16, 8, 'fast'],
  [4, 6, 32, 32, 'fast'],
  [4, 4, 16, 16, 'slow'],
  [8, 16, 32, 32, 'slow'],
  [8, 16, 128, 128, 'slow'],
  [8, 32, 128, 256, 'slow'],
  [32, 128, 258, 1024, 'slow'],
  [32, 258, 258, 4096, 'slow'],
]

function reverseBits(code: number, len: number): number {
  let res = 0
  let c = code
  for (let i = 0; i < len; i++) {
    res = (res << 1) | (c & 1)
    c >>>= 1
  }
  return res
}

/** Codes from code lengths (zlib's gen_codes): canonical, then bit-reversed for LSB-first output. */
function genCodes(len: ArrayLike<number>, maxCode: number, blCount: ArrayLike<number>, code: Int32Array): void {
  const next = new Int32Array(MAX_BITS + 1)
  let c = 0
  for (let bits = 1; bits <= MAX_BITS; bits++) {
    c = (c + (blCount[bits - 1] ?? 0)) << 1
    next[bits] = c
  }
  for (let n = 0; n <= maxCode; n++) {
    const l = len[n] ?? 0
    if (l === 0) continue
    code[n] = reverseBits(next[l] ?? 0, l)
    next[l] = (next[l] ?? 0) + 1
  }
}

/** The fixed tables of trees.c (tr_static_init). */
const STATIC = ((): {
  baseLength: Int32Array
  lengthCode: Uint8Array
  baseDist: Int32Array
  distCode: Uint8Array
  ltreeLen: Int32Array
  ltreeCode: Int32Array
  dtreeLen: Int32Array
  dtreeCode: Int32Array
} => {
  const baseLength = new Int32Array(LENGTH_CODES)
  const lengthCode = new Uint8Array(256)
  let length = 0
  let code = 0
  for (code = 0; code < LENGTH_CODES - 1; code++) {
    baseLength[code] = length
    for (let n = 0; n < 1 << (EXTRA_LBITS[code] ?? 0); n++) lengthCode[length++] = code
  }
  lengthCode[length - 1] = code
  const baseDist = new Int32Array(D_CODES)
  const distCode = new Uint8Array(512)
  let dist = 0
  for (code = 0; code < 16; code++) {
    baseDist[code] = dist
    for (let n = 0; n < 1 << (EXTRA_DBITS[code] ?? 0); n++) distCode[dist++] = code
  }
  dist >>= 7
  for (; code < D_CODES; code++) {
    baseDist[code] = dist << 7
    for (let n = 0; n < 1 << ((EXTRA_DBITS[code] ?? 0) - 7); n++) distCode[256 + dist++] = code
  }
  const ltreeLen = new Int32Array(L_CODES + 2)
  const blCount = new Int32Array(MAX_BITS + 1)
  for (let n = 0; n < L_CODES + 2; n++) {
    const l = n <= 143 ? 8 : n <= 255 ? 9 : n <= 279 ? 7 : 8
    ltreeLen[n] = l
    blCount[l] = (blCount[l] ?? 0) + 1
  }
  const ltreeCode = new Int32Array(L_CODES + 2)
  genCodes(ltreeLen, L_CODES + 1, blCount, ltreeCode)
  const dtreeLen = new Int32Array(D_CODES).fill(5)
  const dtreeCode = Int32Array.from({ length: D_CODES }, (_, n) => reverseBits(n, 5))
  return { baseLength, lengthCode, baseDist, distCode, ltreeLen, ltreeCode, dtreeLen, dtreeCode }
})()

const dCode = (dist: number): number =>
  dist < 256 ? (STATIC.distCode[dist] ?? 0) : (STATIC.distCode[256 + (dist >>> 7)] ?? 0)

/** An LSB-first bit writer (zlib's bi_buf and pending buffer). */
export interface BitSink {
  bits(value: number, length: number): void
  windup(): void
  byte(b: number): void
  bytes(): Uint8Array
}

/** An LSB-first bit writer. */
export function bitSink(): BitSink {
  let out = new Uint8Array(1024)
  let size = 0
  let buf = 0
  let valid = 0
  const put = (b: number): void => {
    if (size === out.length) {
      const grown = new Uint8Array(out.length * 2)
      grown.set(out)
      out = grown
    }
    out[size++] = b & 0xff
  }
  return {
    bits: (value, length) => {
      buf |= (value & ((1 << length) - 1)) << valid
      valid += length
      while (valid >= 8) {
        put(buf & 0xff)
        buf >>>= 8
        valid -= 8
      }
    },
    windup: () => {
      if (valid > 0) put(buf & 0xff)
      buf = 0
      valid = 0
    },
    byte: (b) => {
      put(b)
    },
    bytes: () => out.slice(0, size),
  }
}

/** A dynamic tree: frequencies, codes, parents and lengths (zlib's ct_data unions kept apart). */
interface DeflateTree {
  readonly freq: Int32Array
  readonly code: Int32Array
  readonly dad: Int32Array
  readonly len: Int32Array
  maxCode: number
}

interface StaticDesc {
  readonly staticLen?: Int32Array
  readonly extra: readonly number[]
  readonly extraBase: number
  readonly elems: number
  readonly maxLength: number
}

const L_DESC: StaticDesc = {
  staticLen: STATIC.ltreeLen,
  extra: EXTRA_LBITS,
  extraBase: LITERALS + 1,
  elems: L_CODES,
  maxLength: MAX_BITS,
}
const D_DESC: StaticDesc = {
  staticLen: STATIC.dtreeLen,
  extra: EXTRA_DBITS,
  extraBase: 0,
  elems: D_CODES,
  maxLength: MAX_BITS,
}
const BL_DESC: StaticDesc = { extra: EXTRA_BLBITS, extraBase: 0, elems: BL_CODES, maxLength: MAX_BL_BITS }

function tree(size: number): DeflateTree {
  return {
    freq: new Int32Array(size),
    code: new Int32Array(size),
    dad: new Int32Array(size),
    len: new Int32Array(size),
    maxCode: 0,
  }
}

/**
 * Compress `input` exactly as zlib's deflate() with one call and Z_FINISH (windowBits 15, memLevel 8) would, writing the
 * raw DEFLATE stream to `sink`.
 */
export function deflateRaw(input: Uint8Array, level: number, strategy: DeflateStrategy, sink: BitSink): void {
  const [goodMatch, maxLazy, niceMatch0, maxChain, kind] = CONFIG[level] ?? CONFIG[6] ?? [8, 16, 128, 128, 'slow']
  const window = new Uint8Array(WINDOW_SIZE)
  const prev = new Uint16Array(W_SIZE)
  const head = new Uint16Array(HASH_SIZE)
  let next = 0
  const availIn = (): number => input.length - next
  let strstart = 0
  let blockStart = 0
  let lookahead = 0
  let insert = 0
  let insH = 0
  let matchLength = MIN_MATCH - 1
  let prevLength = MIN_MATCH - 1
  let matchAvailable = false
  let matchStart = 0
  let prevMatch = 0
  let highWater = 0

  const ltree = tree(HEAP_SIZE)
  const dtree = tree(2 * D_CODES + 1)
  const bltree = tree(2 * BL_CODES + 1)
  const heap = new Int32Array(2 * L_CODES + 1)
  const depth = new Uint8Array(2 * L_CODES + 1)
  const blCount = new Int32Array(MAX_BITS + 1)
  let heapLen = 0
  let heapMax = 0
  let optLen = 0
  let staticLen = 0
  const sym = new Uint8Array(LIT_BUFSIZE * 3)
  let symNext = 0

  const updateHash = (h: number, c: number): number => ((h << HASH_SHIFT) ^ c) & HASH_MASK
  /** INSERT_STRING: returns the previous head of the chain. */
  const insertString = (str: number): number => {
    insH = updateHash(insH, window[str + MIN_MATCH - 1] ?? 0)
    const matchHead = head[insH] ?? 0
    prev[str & W_MASK] = matchHead
    head[insH] = str
    return matchHead
  }

  const slideHash = (): void => {
    for (let n = 0; n < HASH_SIZE; n++) {
      const m = head[n] ?? 0
      head[n] = m >= W_SIZE ? m - W_SIZE : 0
    }
    for (let n = 0; n < W_SIZE; n++) {
      const m = prev[n] ?? 0
      prev[n] = m >= W_SIZE ? m - W_SIZE : 0
    }
  }

  const fillWindow = (): void => {
    do {
      let more = WINDOW_SIZE - lookahead - strstart
      if (strstart >= W_SIZE + MAX_DIST) {
        window.copyWithin(0, W_SIZE, W_SIZE + (W_SIZE - more))
        matchStart -= W_SIZE
        strstart -= W_SIZE
        blockStart -= W_SIZE
        if (insert > strstart) insert = strstart
        slideHash()
        more += W_SIZE
      }
      if (availIn() === 0) break
      const n = Math.min(availIn(), more)
      window.set(input.subarray(next, next + n), strstart + lookahead)
      next += n
      lookahead += n
      if (lookahead + insert >= MIN_MATCH) {
        let str = strstart - insert
        insH = window[str] ?? 0
        insH = updateHash(insH, window[str + 1] ?? 0)
        while (insert > 0) {
          insH = updateHash(insH, window[str + MIN_MATCH - 1] ?? 0)
          prev[str & W_MASK] = head[insH] ?? 0
          head[insH] = str
          str++
          insert--
          if (lookahead + insert < MIN_MATCH) break
        }
      }
    } while (lookahead < MIN_LOOKAHEAD && availIn() !== 0)
    if (highWater < WINDOW_SIZE) {
      const curr = strstart + lookahead
      if (highWater < curr) {
        const init = Math.min(WINDOW_SIZE - curr, WIN_INIT)
        window.fill(0, curr, curr + init)
        highWater = curr + init
      } else if (highWater < curr + WIN_INIT) {
        const init = Math.min(curr + WIN_INIT - highWater, WINDOW_SIZE - highWater)
        window.fill(0, highWater, highWater + init)
        highWater += init
      }
    }
  }

  const longestMatch = (curMatch0: number): number => {
    let curMatch = curMatch0
    let chainLength = maxChain
    const scan = strstart
    let bestLen = prevLength
    let nice = niceMatch0
    const limit = strstart > MAX_DIST ? strstart - MAX_DIST : 0
    const strend = strstart + MAX_MATCH
    let scanEnd1 = window[scan + bestLen - 1] ?? 0
    let scanEnd = window[scan + bestLen] ?? 0
    if (prevLength >= goodMatch) chainLength >>>= 2
    if (nice > lookahead) nice = lookahead
    for (;;) {
      const match = curMatch
      if (
        window[match + bestLen] === scanEnd &&
        window[match + bestLen - 1] === scanEnd1 &&
        window[match] === window[scan] &&
        window[match + 1] === window[scan + 1]
      ) {
        // Bytes 0 and 1 match and the hash makes byte 2 match; compare from byte 3 (zlib does not look at byte 2).
        let s = scan + 2
        let m = match + 2
        do {
          s++
          m++
        } while (window[s] === window[m] && s < strend)
        const len = Math.min(MAX_MATCH, s - scan)
        if (len > bestLen) {
          matchStart = curMatch
          bestLen = len
          if (len >= nice) break
          scanEnd1 = window[scan + bestLen - 1] ?? 0
          scanEnd = window[scan + bestLen] ?? 0
        }
      }
      curMatch = prev[curMatch & W_MASK] ?? 0
      if (curMatch <= limit || --chainLength === 0) break
    }
    return Math.min(bestLen, lookahead)
  }

  const initBlock = (): void => {
    ltree.freq.fill(0, 0, L_CODES)
    dtree.freq.fill(0, 0, D_CODES)
    bltree.freq.fill(0, 0, BL_CODES)
    ltree.freq[END_BLOCK] = 1
    optLen = 0
    staticLen = 0
    symNext = 0
  }
  initBlock()

  const smaller = (t: DeflateTree, n: number, m: number): boolean =>
    (t.freq[n] ?? 0) < (t.freq[m] ?? 0) || ((t.freq[n] ?? 0) === (t.freq[m] ?? 0) && (depth[n] ?? 0) <= (depth[m] ?? 0))

  const pqdownheap = (t: DeflateTree, k0: number): void => {
    let k = k0
    const v = heap[k] ?? 0
    let j = k << 1
    while (j <= heapLen) {
      if (j < heapLen && smaller(t, heap[j + 1] ?? 0, heap[j] ?? 0)) j++
      if (smaller(t, v, heap[j] ?? 0)) break
      heap[k] = heap[j] ?? 0
      k = j
      j <<= 1
    }
    heap[k] = v
  }

  const genBitlen = (t: DeflateTree, desc: StaticDesc): void => {
    const maxCode = t.maxCode
    blCount.fill(0)
    t.len[heap[heapMax] ?? 0] = 0
    let overflow = 0
    let h = heapMax + 1
    for (; h < HEAP_SIZE; h++) {
      const n = heap[h] ?? 0
      let bits = (t.len[t.dad[n] ?? 0] ?? 0) + 1
      if (bits > desc.maxLength) {
        bits = desc.maxLength
        overflow++
      }
      t.len[n] = bits
      if (n > maxCode) continue
      blCount[bits] = (blCount[bits] ?? 0) + 1
      const xbits = n >= desc.extraBase ? (desc.extra[n - desc.extraBase] ?? 0) : 0
      const f = t.freq[n] ?? 0
      optLen += f * (bits + xbits)
      if (desc.staticLen !== undefined) staticLen += f * ((desc.staticLen[n] ?? 0) + xbits)
    }
    if (overflow === 0) return
    do {
      let bits = desc.maxLength - 1
      while ((blCount[bits] ?? 0) === 0) bits--
      blCount[bits] = (blCount[bits] ?? 0) - 1
      blCount[bits + 1] = (blCount[bits + 1] ?? 0) + 2
      blCount[desc.maxLength] = (blCount[desc.maxLength] ?? 0) - 1
      overflow -= 2
    } while (overflow > 0)
    for (let bits = desc.maxLength; bits !== 0; bits--) {
      let n = blCount[bits] ?? 0
      while (n !== 0) {
        const m = heap[--h] ?? 0
        if (m > maxCode) continue
        if ((t.len[m] ?? 0) !== bits) {
          optLen += (bits - (t.len[m] ?? 0)) * (t.freq[m] ?? 0)
          t.len[m] = bits
        }
        n--
      }
    }
  }

  const buildTree = (t: DeflateTree, desc: StaticDesc): void => {
    const elems = desc.elems
    let maxCode = -1
    heapLen = 0
    heapMax = HEAP_SIZE
    for (let n = 0; n < elems; n++) {
      if ((t.freq[n] ?? 0) !== 0) {
        heap[++heapLen] = maxCode = n
        depth[n] = 0
      } else t.len[n] = 0
    }
    while (heapLen < 2) {
      const node = maxCode < 2 ? ++maxCode : 0
      heap[++heapLen] = node
      t.freq[node] = 1
      depth[node] = 0
      optLen--
      if (desc.staticLen !== undefined) staticLen -= desc.staticLen[node] ?? 0
    }
    t.maxCode = maxCode
    for (let n = heapLen >> 1; n >= 1; n--) pqdownheap(t, n)
    let node = elems
    do {
      const n = heap[1] ?? 0
      heap[1] = heap[heapLen--] ?? 0
      pqdownheap(t, 1)
      const m = heap[1] ?? 0
      heap[--heapMax] = n
      heap[--heapMax] = m
      t.freq[node] = (t.freq[n] ?? 0) + (t.freq[m] ?? 0)
      depth[node] = Math.max(depth[n] ?? 0, depth[m] ?? 0) + 1
      t.dad[n] = node
      t.dad[m] = node
      heap[1] = node++
      pqdownheap(t, 1)
    } while (heapLen >= 2)
    heap[--heapMax] = heap[1] ?? 0
    genBitlen(t, desc)
    genCodes(t.len, maxCode, blCount, t.code)
  }

  /** The length at n, with zlib's guard of 0xffff just past max_code. */
  const lenAt = (t: DeflateTree, n: number, maxCode: number): number => (n === maxCode + 1 ? 0xffff : (t.len[n] ?? 0))

  const scanTree = (t: DeflateTree, maxCode: number): void => {
    let prevlen = -1
    let nextlen = lenAt(t, 0, maxCode)
    let count = 0
    let maxCount = 7
    let minCount = 4
    if (nextlen === 0) {
      maxCount = 138
      minCount = 3
    }
    for (let n = 0; n <= maxCode; n++) {
      const curlen = nextlen
      nextlen = lenAt(t, n + 1, maxCode)
      if (++count < maxCount && curlen === nextlen) continue
      if (count < minCount) bltree.freq[curlen] = (bltree.freq[curlen] ?? 0) + count
      else if (curlen !== 0) {
        if (curlen !== prevlen) bltree.freq[curlen] = (bltree.freq[curlen] ?? 0) + 1
        bltree.freq[REP_3_6] = (bltree.freq[REP_3_6] ?? 0) + 1
      } else if (count <= 10) bltree.freq[REPZ_3_10] = (bltree.freq[REPZ_3_10] ?? 0) + 1
      else bltree.freq[REPZ_11_138] = (bltree.freq[REPZ_11_138] ?? 0) + 1
      count = 0
      prevlen = curlen
      if (nextlen === 0) {
        maxCount = 138
        minCount = 3
      } else if (curlen === nextlen) {
        maxCount = 6
        minCount = 3
      } else {
        maxCount = 7
        minCount = 4
      }
    }
  }

  const sendCode = (c: number, t: { readonly code: ArrayLike<number>; readonly len: ArrayLike<number> }): void => {
    sink.bits(t.code[c] ?? 0, t.len[c] ?? 0)
  }

  const sendTree = (t: DeflateTree, maxCode: number): void => {
    let prevlen = -1
    let nextlen = lenAt(t, 0, maxCode)
    let count = 0
    let maxCount = 7
    let minCount = 4
    if (nextlen === 0) {
      maxCount = 138
      minCount = 3
    }
    for (let n = 0; n <= maxCode; n++) {
      const curlen = nextlen
      nextlen = lenAt(t, n + 1, maxCode)
      if (++count < maxCount && curlen === nextlen) continue
      if (count < minCount) {
        do sendCode(curlen, bltree)
        while (--count !== 0)
      } else if (curlen !== 0) {
        if (curlen !== prevlen) {
          sendCode(curlen, bltree)
          count--
        }
        sendCode(REP_3_6, bltree)
        sink.bits(count - 3, 2)
      } else if (count <= 10) {
        sendCode(REPZ_3_10, bltree)
        sink.bits(count - 3, 3)
      } else {
        sendCode(REPZ_11_138, bltree)
        sink.bits(count - 11, 7)
      }
      count = 0
      prevlen = curlen
      if (nextlen === 0) {
        maxCount = 138
        minCount = 3
      } else if (curlen === nextlen) {
        maxCount = 6
        minCount = 3
      } else {
        maxCount = 7
        minCount = 4
      }
    }
  }

  const buildBlTree = (): number => {
    scanTree(ltree, ltree.maxCode)
    scanTree(dtree, dtree.maxCode)
    buildTree(bltree, BL_DESC)
    let maxBlindex = BL_CODES - 1
    for (; maxBlindex >= 3; maxBlindex--) if ((bltree.len[BL_ORDER[maxBlindex] ?? 0] ?? 0) !== 0) break
    optLen += 3 * (maxBlindex + 1) + 5 + 5 + 4
    return maxBlindex
  }

  const compressBlock = (
    lt: { readonly code: ArrayLike<number>; readonly len: ArrayLike<number> },
    dt: { readonly code: ArrayLike<number>; readonly len: ArrayLike<number> },
  ): void => {
    for (let sx = 0; sx < symNext; ) {
      let dist = (sym[sx] ?? 0) | ((sym[sx + 1] ?? 0) << 8)
      let lc = sym[sx + 2] ?? 0
      sx += 3
      if (dist === 0) sendCode(lc, lt)
      else {
        let code = STATIC.lengthCode[lc] ?? 0
        sendCode(code + LITERALS + 1, lt)
        let extra = EXTRA_LBITS[code] ?? 0
        if (extra !== 0) {
          lc -= STATIC.baseLength[code] ?? 0
          sink.bits(lc, extra)
        }
        dist--
        code = dCode(dist)
        sendCode(code, dt)
        extra = EXTRA_DBITS[code] ?? 0
        if (extra !== 0) {
          dist -= STATIC.baseDist[code] ?? 0
          sink.bits(dist, extra)
        }
      }
    }
    sendCode(END_BLOCK, lt)
  }

  const storedBlock = (start: number, storedLen: number, last: boolean): void => {
    sink.bits(last ? 1 : 0, 3)
    sink.windup()
    sink.byte(storedLen & 0xff)
    sink.byte((storedLen >>> 8) & 0xff)
    sink.byte(~storedLen & 0xff)
    sink.byte((~storedLen >>> 8) & 0xff)
    for (let i = 0; i < storedLen; i++) sink.byte(window[start + i] ?? 0)
  }

  const flushBlock = (last: boolean): void => {
    const storedLen = strstart - blockStart
    buildTree(ltree, L_DESC)
    buildTree(dtree, D_DESC)
    const maxBlindex = buildBlTree()
    let optLenb = (optLen + 3 + 7) >>> 3
    const staticLenb = (staticLen + 3 + 7) >>> 3
    if (staticLenb <= optLenb || strategy === 'fixed') optLenb = staticLenb
    if (storedLen + 4 <= optLenb && blockStart >= 0) storedBlock(blockStart, storedLen, last)
    else if (staticLenb === optLenb) {
      sink.bits((1 << 1) + (last ? 1 : 0), 3)
      compressBlock({ code: STATIC.ltreeCode, len: STATIC.ltreeLen }, { code: STATIC.dtreeCode, len: STATIC.dtreeLen })
    } else {
      sink.bits((2 << 1) + (last ? 1 : 0), 3)
      const lcodes = ltree.maxCode + 1
      const dcodes = dtree.maxCode + 1
      sink.bits(lcodes - 257, 5)
      sink.bits(dcodes - 1, 5)
      sink.bits(maxBlindex + 1 - 4, 4)
      for (let rank = 0; rank <= maxBlindex; rank++) sink.bits(bltree.len[BL_ORDER[rank] ?? 0] ?? 0, 3)
      sendTree(ltree, lcodes - 1)
      sendTree(dtree, dcodes - 1)
      compressBlock(ltree, dtree)
    }
    initBlock()
    if (last) sink.windup()
    blockStart = strstart
  }

  const tallyLit = (c: number): boolean => {
    sym[symNext++] = 0
    sym[symNext++] = 0
    sym[symNext++] = c
    ltree.freq[c] = (ltree.freq[c] ?? 0) + 1
    return symNext === SYM_END
  }
  const tallyDist = (distance: number, length: number): boolean => {
    sym[symNext++] = distance & 0xff
    sym[symNext++] = (distance >>> 8) & 0xff
    sym[symNext++] = length
    const lcode = (STATIC.lengthCode[length] ?? 0) + LITERALS + 1
    ltree.freq[lcode] = (ltree.freq[lcode] ?? 0) + 1
    const d = dCode(distance - 1)
    dtree.freq[d] = (dtree.freq[d] ?? 0) + 1
    return symNext === SYM_END
  }

  if (level === 0) {
    // deflate_stored with all the input and room for all the output: blocks of up to 65535 bytes.
    let offset = 0
    do {
      const len = Math.min(MAX_STORED, input.length - offset)
      const last = offset + len === input.length
      sink.bits(last ? 1 : 0, 3)
      sink.windup()
      sink.byte(len & 0xff)
      sink.byte((len >>> 8) & 0xff)
      sink.byte(~len & 0xff)
      sink.byte((~len >>> 8) & 0xff)
      for (let i = 0; i < len; i++) sink.byte(input[offset + i] ?? 0)
      offset += len
    } while (offset < input.length)
    return
  }

  if (strategy === 'huffmanOnly') {
    for (;;) {
      if (lookahead === 0) {
        fillWindow()
        if (lookahead === 0) break
      }
      matchLength = 0
      const flush = tallyLit(window[strstart] ?? 0)
      lookahead--
      strstart++
      if (flush) flushBlock(false)
    }
    flushBlock(true)
    return
  }

  if (strategy === 'rle') {
    for (;;) {
      if (lookahead <= MAX_MATCH) {
        fillWindow()
        if (lookahead === 0) break
      }
      matchLength = 0
      if (lookahead >= MIN_MATCH && strstart > 0) {
        const p = window[strstart - 1] ?? 0
        if (p === window[strstart] && p === window[strstart + 1] && p === window[strstart + 2]) {
          const strend = strstart + MAX_MATCH
          let s = strstart + 2
          do s++
          while (p === window[s] && s < strend)
          matchLength = Math.min(MAX_MATCH, s - strstart)
          if (matchLength > lookahead) matchLength = lookahead
        }
      }
      let flush: boolean
      if (matchLength >= MIN_MATCH) {
        flush = tallyDist(1, matchLength - MIN_MATCH)
        lookahead -= matchLength
        strstart += matchLength
        matchLength = 0
      } else {
        flush = tallyLit(window[strstart] ?? 0)
        lookahead--
        strstart++
      }
      if (flush) flushBlock(false)
    }
    flushBlock(true)
    return
  }

  if (kind === 'fast') {
    for (;;) {
      if (lookahead < MIN_LOOKAHEAD) {
        fillWindow()
        if (lookahead === 0) break
      }
      let hashHead = 0
      if (lookahead >= MIN_MATCH) hashHead = insertString(strstart)
      if (hashHead !== 0 && strstart - hashHead <= MAX_DIST) matchLength = longestMatch(hashHead)
      let flush: boolean
      if (matchLength >= MIN_MATCH) {
        flush = tallyDist(strstart - matchStart, matchLength - MIN_MATCH)
        lookahead -= matchLength
        if (matchLength <= maxLazy && lookahead >= MIN_MATCH) {
          matchLength--
          do {
            strstart++
            insertString(strstart)
          } while (--matchLength !== 0)
          strstart++
        } else {
          strstart += matchLength
          matchLength = 0
          insH = window[strstart] ?? 0
          insH = updateHash(insH, window[strstart + 1] ?? 0)
        }
      } else {
        flush = tallyLit(window[strstart] ?? 0)
        lookahead--
        strstart++
      }
      if (flush) flushBlock(false)
    }
    flushBlock(true)
    return
  }

  // deflate_slow: lazy matching.
  for (;;) {
    if (lookahead < MIN_LOOKAHEAD) {
      fillWindow()
      if (lookahead === 0) break
    }
    let hashHead = 0
    if (lookahead >= MIN_MATCH) hashHead = insertString(strstart)
    prevLength = matchLength
    prevMatch = matchStart
    matchLength = MIN_MATCH - 1
    if (hashHead !== 0 && prevLength < maxLazy && strstart - hashHead <= MAX_DIST) {
      matchLength = longestMatch(hashHead)
      if (
        matchLength <= 5 &&
        (strategy === 'filtered' || (matchLength === MIN_MATCH && strstart - matchStart > TOO_FAR))
      )
        matchLength = MIN_MATCH - 1
    }
    if (prevLength >= MIN_MATCH && matchLength <= prevLength) {
      const maxInsert = strstart + lookahead - MIN_MATCH
      const flush = tallyDist(strstart - 1 - prevMatch, prevLength - MIN_MATCH)
      lookahead -= prevLength - 1
      prevLength -= 2
      do {
        if (++strstart <= maxInsert) insertString(strstart)
      } while (--prevLength !== 0)
      matchAvailable = false
      matchLength = MIN_MATCH - 1
      strstart++
      if (flush) flushBlock(false)
    } else if (matchAvailable) {
      const flush = tallyLit(window[strstart - 1] ?? 0)
      if (flush) flushBlock(false)
      strstart++
      lookahead--
    } else {
      matchAvailable = true
      strstart++
      lookahead--
    }
  }
  if (matchAvailable) {
    tallyLit(window[strstart - 1] ?? 0)
    matchAvailable = false
  }
  flushBlock(true)
}
