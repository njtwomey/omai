/**
 * Dictionary and transform coding of symbol sequences: the LZ77 sliding-window parse, the LZ78 growing-dictionary parse
 * and Lempel-Ziv-Welch (LZW) coding, each with its decoder, and the Burrows-Wheeler transform and move-to-front coding
 * (with their inverses), which rearrange data so that it compresses better, as in bzip2.
 */

import { OmaiError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** A sequence of symbols: a string (its UTF-8 bytes) or integers. */
export type SymbolInput = string | ArrayLike<number>

function symbolsOf(input: SymbolInput, name: string): number[] {
  const xs = typeof input === 'string' ? Array.from(new TextEncoder().encode(input)) : Array.from(input)
  if (xs.some((x) => !Number.isSafeInteger(x) || x < 0))
    throw new OmaiError('value', `${name}: symbols must be non-negative integers`)
  return xs
}

/** An LZ77 token: copy `length` symbols from `offset` back, then emit `next` (absent when the input ends). */
export interface Lz77Token {
  readonly offset: number
  readonly length: number
  readonly next?: number
}

/** The paths the Lempel-Ziv parses log, as a type. */
export type LzPath = 'lz/position' | 'lz/match' | 'lz/offset' | 'lz/phrase' | 'lz/ratio'

/**
 * What the Lempel-Ziv parses log per step: the position parsed up to, the length of the match (LZ77: the longest match
 * in the window; complexity: the longest candidate in the phrase), the offset of the match, the number of phrases (tokens)
 * so far, and the ratio of phrases to symbols consumed.
 */
export const lzContract: Contract<unknown, LzPath> = defineContract({
  'lz/position': scalar({ description: 'the symbols consumed so far' }),
  'lz/match': scalar({ description: 'the length of the match found at this step' }),
  'lz/offset': scalar({ description: 'how far back the match starts (0 for none)' }),
  'lz/phrase': scalar({ description: 'phrases (tokens) so far' }),
  'lz/ratio': scalar({
    description: 'phrases per symbol consumed: low means a compressible sequence',
    direction: 'lower',
  }),
})

/** Options of {@link lz77Algorithm} and {@link lz77Encode}. */
export interface Lz77Options {
  /** How far back a match may start (default 4096). */
  readonly window?: number
  /** The longest match (default 255). */
  readonly maxLength?: number
}

/**
 * The state of {@link lz77Algorithm}: the position reached and the tokens emitted. Steps push tokens in place; keep a
 * copy of a state you want to look at later.
 */
export interface Lz77State {
  /** The next symbol to parse. */
  position: number
  readonly tokens: { offset: number; length: number; next: number }[]
}

/**
 * The LZ77 parse (Ziv and Lempel 1977) as a stepped algorithm, one token per step: at the current position find the
 * longest match in the preceding window (overlapping the position if it likes, nearest first among equals), emit
 * (offset, length, next symbol) and move past them. It logs the position, the match length and offset, the tokens so far
 * and their ratio to the symbols consumed, and is done (`converged`) when the input is consumed. The state serialises.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @param options - Window size (default 4096) and longest match (default 255).
 * @returns The algorithm.
 * @throws OmaiError with code `value` unless window and maxLength are at least 1.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz77Algorithm } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(lz77Algorithm('abababc'), { maxSteps: 20 })
 * [run.steps, run.state.tokens.map((t) => t.length)] // => [3, [0, 0, 4]]
 * ```
 */
export function lz77Algorithm(input: SymbolInput, options: Lz77Options = {}): Algorithm<Lz77State, LzPath> {
  const xs = symbolsOf(input, 'lz77Encode')
  const window = options.window ?? 4096
  const maxLength = options.maxLength ?? 255
  if (!(window >= 1 && maxLength >= 1)) throw new OmaiError('value', 'lz77Encode: window and maxLength must be >= 1')
  return {
    name: 'information/lz77',
    config: { window, maxLength, n: xs.length },
    init: () => ({ position: 0, tokens: [] }),
    step: (s, ctx) => {
      const i = s.position
      let bestLength = 0
      let bestOffset = 0
      for (let offset = 1; offset <= Math.min(window, i); offset++) {
        let l = 0
        while (l < maxLength && i + l < xs.length - 1 && xs[i + l - offset] === xs[i + l]) l++
        if (l > bestLength) {
          bestLength = l
          bestOffset = offset
        }
      }
      // Matches stop one short of the end, so every token carries its next symbol.
      s.tokens.push({ offset: bestOffset, length: bestLength, next: xs[i + bestLength] as number })
      s.position = i + bestLength + 1
      ctx.log('lz/position', s.position)
      ctx.log('lz/match', bestLength)
      ctx.log('lz/offset', bestOffset)
      ctx.log('lz/phrase', s.tokens.length)
      ctx.log('lz/ratio', s.tokens.length / s.position)
      return s
    },
    done: (s) => (s.position >= xs.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as Lz77State,
  }
}

/**
 * The LZ77 parse (Ziv and Lempel 1977): at each position, the longest match in the preceding window (overlapping the
 * current position if it likes, nearest first among equals) becomes a token (offset, length, next symbol).
 * {@link lz77Algorithm} run to the end.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @param options - Window size (default 4096) and longest match (default 255).
 * @returns The tokens.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz77Encode } from 'omai-engine/information'
 * lz77Encode('abababc').map((t) => [t.offset, t.length, t.next]) // => [[0, 0, 97], [0, 0, 98], [2, 4, 99]]
 * ```
 */
export function lz77Encode(input: SymbolInput, options: Lz77Options = {}): Lz77Token[] {
  return runAlgorithm(lz77Algorithm(input, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state.tokens
}

/**
 * Decode LZ77 tokens.
 *
 * @param tokens - Tokens from {@link lz77Encode}.
 * @returns The symbols.
 * @throws OmaiError with code `value` for an offset that reaches before the start.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz77Decode } from 'omai-engine/information'
 * lz77Decode([{ offset: 0, length: 0, next: 97 }, { offset: 1, length: 3, next: 98 }]) // => [97, 97, 97, 97, 98]
 * ```
 */
export function lz77Decode(tokens: readonly Lz77Token[]): number[] {
  const out: number[] = []
  for (const t of tokens) {
    if (t.length > 0 && (t.offset < 1 || t.offset > out.length))
      throw new OmaiError('value', 'lz77Decode: offset out of range')
    for (let j = 0; j < t.length; j++) out.push(out[out.length - t.offset] as number)
    if (t.next !== undefined) out.push(t.next)
  }
  return out
}

/** An LZ78 pair: the dictionary phrase `index` (0 for the empty phrase) followed by `next` (absent at the end). */
export interface Lz78Pair {
  readonly index: number
  readonly next?: number
}

/**
 * The LZ78 parse (Ziv and Lempel 1978): each pair extends the longest phrase seen so far by one symbol and adds the
 * result to the dictionary, so the dictionary grows with the input instead of sliding.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @returns The pairs.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz78Encode } from 'omai-engine/information'
 * lz78Encode([0, 1, 0, 0, 1, 0]).map((p) => [p.index, p.next]) // => [[0, 0], [0, 1], [1, 0], [2, 0]]
 * ```
 */
export function lz78Encode(input: SymbolInput): Lz78Pair[] {
  const xs = symbolsOf(input, 'lz78Encode')
  const dictionary = new Map<string, number>()
  const pairs: Lz78Pair[] = []
  let phrase = 0
  for (const x of xs) {
    const key = `${phrase},${x}`
    const found = dictionary.get(key)
    if (found !== undefined) phrase = found
    else {
      pairs.push({ index: phrase, next: x })
      dictionary.set(key, dictionary.size + 1)
      phrase = 0
    }
  }
  if (phrase !== 0) pairs.push({ index: phrase })
  return pairs
}

/**
 * Decode LZ78 pairs.
 *
 * @param pairs - Pairs from {@link lz78Encode}.
 * @returns The symbols.
 * @throws OmaiError with code `value` for an index not yet in the dictionary.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz78Decode } from 'omai-engine/information'
 * lz78Decode([{ index: 0, next: 5 }, { index: 1, next: 6 }, { index: 2 }]) // => [5, 5, 6, 5, 6]
 * ```
 */
export function lz78Decode(pairs: readonly Lz78Pair[]): number[] {
  const phrases: number[][] = [[]]
  const out: number[] = []
  for (const p of pairs) {
    const base = phrases[p.index]
    if (base === undefined) throw new OmaiError('value', `lz78Decode: unknown phrase ${p.index}`)
    const phrase = p.next === undefined ? base : [...base, p.next]
    out.push(...phrase)
    phrases.push(phrase)
  }
  return out
}

/** Options of the LZW coder. */
export interface LzwOptions {
  /** Symbols 0..alphabetSize-1 start in the dictionary (default 256, bytes). */
  readonly alphabetSize?: number
  /** The dictionary stops growing at this many entries (default 4096, 12-bit codes). */
  readonly maxCodes?: number
}

/**
 * Lempel-Ziv-Welch coding (Welch 1984): the dictionary starts with every single symbol; each code is the longest
 * phrase in the dictionary, and that phrase plus the next symbol becomes the next entry. No symbols are sent, only codes.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @param options - Alphabet size and dictionary limit.
 * @returns The codes.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lzwEncode } from 'omai-engine/information'
 * lzwEncode('TOBEORNOTTOBEORTOBEORNOT').slice(8) // => [84, 256, 258, 260, 265, 259, 261, 263]
 * ```
 */
export function lzwEncode(input: SymbolInput, options: LzwOptions = {}): number[] {
  const xs = symbolsOf(input, 'lzwEncode')
  const alphabet = options.alphabetSize ?? 256
  const maxCodes = options.maxCodes ?? 4096
  if (maxCodes < alphabet) throw new OmaiError('value', 'lzwEncode: maxCodes is smaller than the alphabet')
  const dictionary = new Map<string, number>()
  let size = alphabet
  const codes: number[] = []
  let w = -1
  for (const x of xs) {
    if (x >= alphabet) throw new OmaiError('value', `lzwEncode: symbol ${x} is outside the alphabet`)
    if (w < 0) {
      w = x
      continue
    }
    const key = `${w},${x}`
    const found = dictionary.get(key)
    if (found !== undefined) w = found
    else {
      codes.push(w)
      if (size < maxCodes) dictionary.set(key, size++)
      w = x
    }
  }
  if (w >= 0) codes.push(w)
  return codes
}

/**
 * Decode LZW codes, rebuilding the dictionary one step behind the encoder (including the case where a code refers to
 * the entry being defined).
 *
 * @param codes - Codes from {@link lzwEncode}.
 * @param options - The same alphabet size and dictionary limit as the encoder.
 * @returns The symbols.
 * @throws OmaiError with code `value` for an impossible code.
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lzwDecode, lzwEncode } from 'omai-engine/information'
 * new TextDecoder().decode(Uint8Array.from(lzwDecode(lzwEncode('abababab')))) // => 'abababab'
 * ```
 */
export function lzwDecode(codes: readonly number[], options: LzwOptions = {}): number[] {
  const alphabet = options.alphabetSize ?? 256
  const maxCodes = options.maxCodes ?? 4096
  const entries: number[][] = Array.from({ length: alphabet }, (_, i) => [i])
  const out: number[] = []
  let previous: number[] | undefined
  for (const c of codes) {
    let entry = entries[c]
    if (entry === undefined) {
      if (c !== entries.length || previous === undefined) throw new OmaiError('value', `lzwDecode: bad code ${c}`)
      entry = [...previous, previous[0] as number]
    }
    out.push(...entry)
    if (previous !== undefined && entries.length < maxCodes) entries.push([...previous, entry[0] as number])
    previous = entry
  }
  return out
}

/** A Burrows-Wheeler transform: the last column of the sorted rotations and the row of the original. */
export interface BwtResult {
  readonly last: readonly number[]
  readonly index: number
}

/**
 * The Burrows-Wheeler transform (1994): sort all rotations of the input and keep the last column and the position of
 * the input among them. Symbols that precede similar contexts end up together, so the output has long runs that
 * move-to-front and run-length coding then exploit (as in bzip2). Rotations are sorted by prefix doubling.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @returns The last column and the index.
 * @concept burrows-wheeler
 * @example
 * ```ts
 * import { bwt } from 'omai-engine/information'
 * const t = bwt('banana')
 * new TextDecoder().decode(Uint8Array.from(t.last)) // => 'nnbaaa'
 * t.index // => 3
 * ```
 */
export function bwt(input: SymbolInput): BwtResult {
  const xs = symbolsOf(input, 'bwt')
  const n = xs.length
  if (n === 0) return { last: [], index: 0 }
  let rank = Int32Array.from(xs)
  let order = Array.from({ length: n }, (_, i) => i)
  for (let k = 1; ; k <<= 1) {
    const r = rank
    const key = (i: number): [number, number] => [r[i] ?? 0, r[(i + k) % n] ?? 0]
    order = order.sort((a, b) => {
      const [a0, a1] = key(a)
      const [b0, b1] = key(b)
      return a0 - b0 || a1 - b1 || a - b
    })
    const next = new Int32Array(n)
    for (let j = 1; j < n; j++) {
      const [p0, p1] = key(order[j - 1] ?? 0)
      const [c0, c1] = key(order[j] ?? 0)
      next[order[j] ?? 0] = (next[order[j - 1] ?? 0] ?? 0) + (p0 !== c0 || p1 !== c1 ? 1 : 0)
    }
    rank = next
    if ((next[order[n - 1] ?? 0] ?? 0) === n - 1 || k >= n) break
  }
  return { last: order.map((i) => xs[(i + n - 1) % n] as number), index: order.indexOf(0) }
}

/**
 * Invert the Burrows-Wheeler transform by the last-to-first mapping: the k-th occurrence of a symbol in the last
 * column is its k-th occurrence in the sorted first column.
 *
 * @param transform - The last column and index from {@link bwt}.
 * @returns The original symbols.
 * @throws OmaiError with code `value` for an index out of range.
 * @concept burrows-wheeler
 * @example
 * ```ts
 * import { inverseBwt } from 'omai-engine/information'
 * new TextDecoder().decode(Uint8Array.from(inverseBwt({ last: [110, 110, 98, 97, 97, 97], index: 3 }))) // => 'banana'
 * ```
 */
export function inverseBwt(transform: BwtResult): number[] {
  const last = transform.last
  const n = last.length
  if (n === 0) return []
  if (!(transform.index >= 0 && transform.index < n)) throw new OmaiError('value', 'inverseBwt: index out of range')
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (last[a] ?? 0) - (last[b] ?? 0) || a - b)
  // order[j] is the row of the last column holding the j-th symbol of the sorted first column.
  const out = new Array<number>(n)
  let row = order[transform.index] ?? 0
  for (let i = 0; i < n; i++) {
    out[i] = last[row] as number
    row = order[row] ?? 0
  }
  return out
}

/**
 * Move-to-front coding: each symbol is replaced by its position in a list that starts as 0..alphabetSize-1, and is then
 * moved to the front, so recently used symbols get small numbers.
 *
 * @param input - The symbols (a string is taken as its UTF-8 bytes).
 * @param alphabetSize - The alphabet size (default 256).
 * @returns The positions.
 * @concept burrows-wheeler
 * @example
 * ```ts
 * import { moveToFront } from 'omai-engine/information'
 * moveToFront([1, 1, 1, 0, 0, 2], 3) // => [1, 0, 0, 1, 0, 2]
 * ```
 */
export function moveToFront(input: SymbolInput, alphabetSize = 256): number[] {
  const xs = symbolsOf(input, 'moveToFront')
  const list = Array.from({ length: alphabetSize }, (_, i) => i)
  return xs.map((x) => {
    const j = list.indexOf(x)
    if (j < 0) throw new OmaiError('value', `moveToFront: symbol ${x} is outside the alphabet`)
    list.splice(j, 1)
    list.unshift(x)
    return j
  })
}

/**
 * Invert move-to-front coding.
 *
 * @param positions - Positions from {@link moveToFront}.
 * @param alphabetSize - The same alphabet size (default 256).
 * @returns The symbols.
 * @throws OmaiError with code `value` for a position outside the list.
 * @concept burrows-wheeler
 * @example
 * ```ts
 * import { inverseMoveToFront } from 'omai-engine/information'
 * inverseMoveToFront([1, 0, 0, 1, 0, 2], 3) // => [1, 1, 1, 0, 0, 2]
 * ```
 */
export function inverseMoveToFront(positions: readonly number[], alphabetSize = 256): number[] {
  const list = Array.from({ length: alphabetSize }, (_, i) => i)
  return positions.map((j) => {
    const x = list[j]
    if (x === undefined) throw new OmaiError('value', `inverseMoveToFront: position ${j} is outside the list`)
    list.splice(j, 1)
    list.unshift(x)
    return x
  })
}
