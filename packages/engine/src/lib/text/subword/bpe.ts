/**
 * Byte-pair encoding (BPE), the subword tokenisation of GPT-style models: training repeatedly merges the most frequent
 * adjacent pair of symbols into a new vocabulary entry. GPT-2's byte-level alphabet and conversions, BPE training as a
 * traced algorithm, models loaded from a vocabulary and merges, and encoding of a word as HF tokenizers does (with
 * dropout and byte fallback).
 */

import { heap, type Json, WhiteboxError } from '../../base/index.ts'
import { type Key, randomBits, split } from '../../random/index.ts'
import { type Algorithm, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import { compareCodePoints } from '../_classes.ts'
import { utf8Bytes } from '../hashing.ts'
import { namesOf } from './_shared.ts'

// ---------------------------------------------------------------------------------------------------------------
// The GPT-2 byte alphabet.

let byteChars: string[] | undefined
let charBytes: Map<string, number> | undefined

/**
 * GPT-2's byte-to-character table (`bytes_to_unicode`): printable bytes map to themselves and the others to code points
 * from 256 up, so every byte string is a printable string.
 *
 * @returns The character of each byte 0..255.
 * @concept byte-level-bpe
 * @example
 * ```ts
 * import { bytesToUnicode } from 'whitebox-engine/text/subword'
 * const t = bytesToUnicode()
 * [t[32], t[65], t[10]] // => ['Ġ', 'A', 'Ċ']
 * ```
 */
export function bytesToUnicode(): string[] {
  if (byteChars === undefined) {
    const bs: number[] = []
    for (let b = 33; b <= 126; b++) bs.push(b)
    for (let b = 161; b <= 172; b++) bs.push(b)
    for (let b = 174; b <= 255; b++) bs.push(b)
    const cs = [...bs]
    let n = 0
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b)
        cs.push(256 + n)
        n++
      }
    }
    const table = new Array<string>(256).fill('')
    for (const [i, b] of bs.entries()) table[b] = String.fromCodePoint(cs[i] ?? 0)
    byteChars = table
    charBytes = new Map(table.map((c, b) => [c, b]))
  }
  return [...byteChars]
}

/**
 * The 256 characters of the byte alphabet, in code point order (HF's ByteLevel.alphabet, the initial alphabet of
 * byte-level BPE training).
 *
 * @returns The characters.
 * @concept byte-level-bpe
 * @example
 * ```ts
 * import { byteLevelAlphabet } from 'whitebox-engine/text/subword'
 * byteLevelAlphabet().length // => 256
 * ```
 */
export function byteLevelAlphabet(): string[] {
  return bytesToUnicode().sort(compareCodePoints)
}

/**
 * A string as byte-level characters: its UTF-8 bytes through {@link bytesToUnicode}.
 *
 * @param text - The text.
 * @returns The byte-level string.
 * @concept byte-level-bpe
 * @example
 * ```ts
 * import { toByteLevel } from 'whitebox-engine/text/subword'
 * toByteLevel(' héllo') // => 'ĠhÃ©llo'
 * ```
 */
export function toByteLevel(text: string): string {
  const table = bytesToUnicode()
  let out = ''
  for (const b of utf8Bytes(text)) out += table[b] ?? ''
  return out
}

let decoder: TextDecoder | undefined

/**
 * Decode a byte-level string back to text (invalid UTF-8 becomes U+FFFD; characters outside the byte alphabet are
 * kept as they are).
 *
 * @param text - The byte-level string.
 * @returns The text.
 * @concept byte-level-bpe
 * @example
 * ```ts
 * import { fromByteLevel } from 'whitebox-engine/text/subword'
 * fromByteLevel('ĠhÃ©llo') // => ' héllo'
 * ```
 */
export function fromByteLevel(text: string): string {
  bytesToUnicode()
  const map = charBytes as Map<string, number>
  decoder ??= new TextDecoder('utf-8', { fatal: false })
  const bytes: number[] = []
  let out = ''
  const flush = (): void => {
    if (bytes.length > 0) out += (decoder as TextDecoder).decode(Uint8Array.from(bytes))
    bytes.length = 0
  }
  for (const ch of text) {
    const b = map.get(ch)
    if (b === undefined) {
      flush()
      out += ch
    } else bytes.push(b)
  }
  flush()
  return out
}

// ---------------------------------------------------------------------------------------------------------------
// Training (HF tokenizers' BpeTrainer).

/** Options of BPE training. */
export interface BpeTrainingOptions {
  /** The vocabulary size to reach, specials and alphabet included (default 30000). */
  readonly vocabSize?: number
  /** Stop when the best pair occurs fewer times (default 0). */
  readonly minFrequency?: number
  /** Special tokens, given the first ids. */
  readonly specialTokens?: readonly string[]
  /** Keep at most this many characters of the alphabet (the most frequent). */
  readonly limitAlphabet?: number
  /** Characters always in the alphabet. */
  readonly initialAlphabet?: readonly string[]
  /** The prefix marking symbols inside a word (WordPiece's `##`). */
  readonly continuingSubwordPrefix?: string
  /** The suffix marking the last symbol of a word (`</w>` in the original BPE). */
  readonly endOfWordSuffix?: string
  /** Never form tokens longer than this many symbols. */
  readonly maxTokenLength?: number
  /** Train on the UTF-8 bytes of words through the GPT-2 byte alphabet, with all 256 bytes in the alphabet. */
  readonly byteLevel?: boolean
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** A trained or loaded BPE model (HF tokenizers' `models.BPE`). */
export interface BpeModel {
  readonly vocab: ReadonlyMap<string, number>
  /** The merges in rank order. */
  readonly merges: readonly (readonly [string, string])[]
  readonly unkToken?: string
  readonly continuingSubwordPrefix?: string
  readonly endOfWordSuffix?: string
  /** Merge consecutive unknown tokens into one (default false). */
  readonly fuseUnk?: boolean
  /** Spell unknown characters as `<0xXX>` byte tokens when the vocabulary has them. */
  readonly byteFallback?: boolean
  /** Return a whole word that is in the vocabulary without merging. */
  readonly ignoreMerges?: boolean
  /** Merge dropout probability (BPE-dropout, Provilkov et al. 2020). */
  readonly dropout?: number
}

/** The state of BPE training: words as symbol ids, pair statistics, the vocabulary and the merges so far. */
export interface BpeTrainingState {
  readonly words: Int32Array[]
  readonly lengths: Int32Array[]
  readonly counts: readonly number[]
  readonly pairCounts: Map<number, number>
  readonly where: Map<number, Set<number>>
  readonly queue: { push: (m: QueuedPair) => void; pop: () => QueuedPair | undefined; readonly size: number }
  readonly tokens: string[]
  readonly index: Map<string, number>
  readonly merges: [string, string][]
  /** The last merge: its pair, new token and count (undefined before the first). */
  readonly last?: { readonly pair: readonly [string, string]; readonly token: string; readonly count: number }
  readonly stopped: boolean
}

/** A candidate merge in the priority queue. */
export interface QueuedPair {
  readonly a: number
  readonly b: number
  readonly count: number
}

/** Pairs of ids as one number (ids are below 2^26). */
const pairKey = (a: number, b: number): number => a * 67108864 + b

/**
 * BPE training (Sennrich et al. 2016) as an {@link Algorithm}, following HF tokenizers' BpeTrainer: the alphabet
 * (sorted by code point, after the special tokens) seeds the vocabulary; each step merges the most frequent adjacent
 * pair (ties: the pair with the smallest ids), until the vocabulary size, the frequency floor or the pairs run out.
 * Each step records the merged pair, the new token and its count.
 *
 * @param wordCounts - Pre-tokenised words with their counts.
 * @param options - Vocabulary size, frequency floor, specials, alphabet limits, prefix and suffix, byte level.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for invalid sizes.
 * @concept byte-pair-encoding
 * @example
 * ```ts
 * import { bpeTrainingAlgorithm } from 'whitebox-engine/text/subword'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const alg = bpeTrainingAlgorithm(new Map([['low', 5], ['lower', 2], ['newest', 6], ['widest', 3]]), { vocabSize: 16 })
 * runAlgorithm(alg, { maxSteps: 100 }).state.merges // => [['e', 's'], ['es', 't'], ['l', 'o'], ['lo', 'w'], ['e', 'w'], ['n', 'ew']]
 * ```
 */
export function bpeTrainingAlgorithm(
  wordCounts: ReadonlyMap<string, number> | Iterable<readonly [string, number]>,
  options: BpeTrainingOptions = {},
): Algorithm<BpeTrainingState> {
  const vocabSize = options.vocabSize ?? 30000
  if (!Number.isInteger(vocabSize) || vocabSize < 1)
    throw new WhiteboxError('value', 'trainBpe: vocabSize must be a positive integer')
  const byteLevel = options.byteLevel === true
  const entries = [...wordCounts].map(([w, c]): [string, number] => [byteLevel ? toByteLevel(w) : w, c])
  const prefix = options.continuingSubwordPrefix
  const suffix = options.endOfWordSuffix
  const maxLength = options.maxTokenLength ?? Number.POSITIVE_INFINITY
  const minFrequency = options.minFrequency ?? 0
  return {
    name: 'bpeTraining',
    config: { vocabSize, minFrequency, byteLevel, prefix: prefix ?? null, suffix: suffix ?? null } as Json,
    init: () => {
      const tokens: string[] = []
      const index = new Map<string, number>()
      const add = (t: string): void => {
        if (!index.has(t)) {
          index.set(t, tokens.length)
          tokens.push(t)
        }
      }
      for (const s of options.specialTokens ?? []) add(s)
      // The alphabet: weighted character counts, forced initial characters, the limit, then code point order.
      const alphabet = new Map<string, number>()
      for (const [w, c] of entries) for (const ch of w) alphabet.set(ch, (alphabet.get(ch) ?? 0) + c)
      const initial = [...(options.initialAlphabet ?? []), ...(byteLevel ? byteLevelAlphabet() : [])]
      for (const ch of initial) alphabet.set(ch, Number.POSITIVE_INFINITY)
      let kept = [...alphabet]
      if (options.limitAlphabet !== undefined && kept.length > options.limitAlphabet) {
        kept.sort((x, y) => x[1] - y[1])
        kept = kept.slice(kept.length - options.limitAlphabet)
      }
      kept.sort((x, y) => compareCodePoints(x[0], y[0]))
      for (const [ch] of kept) add(ch)
      const allowed = new Set(kept.map(([ch]) => ch))
      // Words as symbols, with the prefix inside words and the suffix on the last symbol.
      const words: Int32Array[] = []
      const lengths: Int32Array[] = []
      const counts: number[] = []
      for (const [w, c] of entries) {
        const chars = Array.from(w)
        const ids: number[] = []
        for (const [k, ch] of chars.entries()) {
          if (!allowed.has(ch)) continue
          let s = ch
          if (k > 0 && prefix !== undefined) s = prefix + s
          if (k === chars.length - 1 && suffix !== undefined) s += suffix
          add(s)
          ids.push(index.get(s) ?? 0)
        }
        words.push(Int32Array.from(ids))
        lengths.push(new Int32Array(ids.length).fill(1))
        counts.push(c)
      }
      const pairCounts = new Map<number, number>()
      const where = new Map<number, Set<number>>()
      for (const [i, w] of words.entries()) {
        for (let k = 0; k + 1 < w.length; k++) {
          const key = pairKey(w[k] ?? 0, w[k + 1] ?? 0)
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + (counts[i] ?? 0))
          const set = where.get(key) ?? new Set<number>()
          set.add(i)
          where.set(key, set)
        }
      }
      const queue = heap<QueuedPair>((x, y) => y.count - x.count || x.a - y.a || x.b - y.b)
      for (const [key, c] of pairCounts)
        if (c > 0) queue.push({ a: Math.floor(key / 67108864), b: key % 67108864, count: c })
      return { words, lengths, counts, pairCounts, where, queue, tokens, index, merges: [], stopped: false }
    },
    step: (state) => {
      if (state.tokens.length >= vocabSize) return { ...state, stopped: true }
      for (;;) {
        const top = state.queue.pop()
        if (top === undefined) return { ...state, stopped: true }
        const key = pairKey(top.a, top.b)
        const current = state.pairCounts.get(key) ?? 0
        if (top.count !== current) {
          if (current > 0) state.queue.push({ ...top, count: current })
          continue
        }
        if (top.count < 1 || minFrequency > top.count) return { ...state, stopped: true }
        const left = state.tokens[top.a] ?? ''
        let right = state.tokens[top.b] ?? ''
        if (prefix !== undefined && right.startsWith(prefix)) right = right.slice(prefix.length)
        const token = left + right
        let id = state.index.get(token)
        if (id === undefined) {
          id = state.tokens.length
          state.tokens.push(token)
          state.index.set(token, id)
        }
        state.merges.push([state.tokens[top.a] ?? '', state.tokens[top.b] ?? ''])
        const touched = new Map<number, Set<number>>()
        for (const wi of [...(state.where.get(key) ?? [])].sort((x, y) => x - y)) {
          mergeWord(state, wi, top.a, top.b, id, maxLength, touched)
        }
        for (const [k, set] of touched) {
          const c = state.pairCounts.get(k) ?? 0
          const existing = state.where.get(k) ?? new Set<number>()
          for (const w of set) existing.add(w)
          state.where.set(k, existing)
          if (c > 0) state.queue.push({ a: Math.floor(k / 67108864), b: k % 67108864, count: c })
        }
        return {
          ...state,
          last: { pair: [state.tokens[top.a] ?? '', state.tokens[top.b] ?? ''], token, count: top.count },
        }
      }
    },
    done: (state) => (state.stopped ? 'converged' : undefined),
    observe: (state) => ({
      metrics: { vocabSize: state.tokens.length, count: state.last?.count ?? 0 },
      ...(state.last === undefined
        ? {}
        : {
            messages: [
              {
                kind: 'merge',
                payload: { pair: [...state.last.pair], token: state.last.token, count: state.last.count },
              },
            ],
          }),
    }),
  }
}

/** Merge every occurrence of (a, b) in one word, updating pair counts (HF's Word::merge). */
function mergeWord(
  state: BpeTrainingState,
  wi: number,
  a: number,
  b: number,
  id: number,
  maxLength: number,
  touched: Map<number, Set<number>>,
): void {
  const w = state.words[wi] as Int32Array
  const len = state.lengths[wi] as Int32Array
  const c = state.counts[wi] ?? 0
  const change = (x: number, y: number, delta: number): void => {
    const k = pairKey(x, y)
    state.pairCounts.set(k, (state.pairCounts.get(k) ?? 0) + delta * c)
    if (delta > 0) {
      const set = touched.get(k) ?? new Set<number>()
      set.add(wi)
      touched.set(k, set)
    }
  }
  const ids: number[] = Array.from(w)
  const ls: number[] = Array.from(len)
  let i = 0
  while (i < ids.length) {
    if (ids[i] === a && i + 1 < ids.length && ids[i + 1] === b) {
      const newLen = (ls[i] ?? 0) + (ls[i + 1] ?? 0)
      if (i > 0) {
        change(ids[i - 1] ?? 0, a, -1)
        if ((ls[i - 1] ?? 0) + newLen < maxLength) change(ids[i - 1] ?? 0, id, 1)
      }
      ids.splice(i, 2, id)
      ls.splice(i, 2, newLen)
      if (i < ids.length - 1) {
        change(b, ids[i + 1] ?? 0, -1)
        if ((ls[i + 1] ?? 0) + newLen < maxLength) change(id, ids[i + 1] ?? 0, 1)
      }
    }
    i++
  }
  state.words[wi] = Int32Array.from(ids)
  state.lengths[wi] = Int32Array.from(ls)
}

/**
 * Train a BPE model on word counts ({@link bpeTrainingAlgorithm}, traced when a tracer is given). With `byteLevel`
 * the vocabulary is over GPT-2 byte characters and needs no unknown token.
 *
 * @param wordCounts - Pre-tokenised words with their counts.
 * @param options - Training options and an unknown token for the model.
 * @returns The model.
 * @concept byte-pair-encoding
 * @example
 * ```ts
 * import { trainBpe, bpeTokenize } from 'whitebox-engine/text/subword'
 * const model = trainBpe(new Map([['low', 5], ['lower', 2], ['newest', 6], ['widest', 3]]), { vocabSize: 16 })
 * bpeTokenize(model, 'newer').map((t) => t.token) // => ['new', 'e', 'r']
 * ```
 */
export function trainBpe(
  wordCounts: ReadonlyMap<string, number> | Iterable<readonly [string, number]>,
  options: BpeTrainingOptions & { readonly unkToken?: string } = {},
): BpeModel {
  const entries = [...wordCounts]
  const alg = bpeTrainingAlgorithm(entries, options)
  // Each merge shortens some word, so merges are bounded by the total number of symbols.
  const symbols = entries.reduce(
    (n, [w]) => n + (options.byteLevel === true ? utf8Bytes(w).length : Array.from(w).length),
    0,
  )
  const result = runAlgorithm(alg, {
    maxSteps: (options.vocabSize ?? 30000) + symbols + 1,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return bpeModel(new Map(result.state.tokens.map((t, i) => [t, i])), result.state.merges, {
    ...(options.unkToken === undefined ? {} : { unkToken: options.unkToken }),
    ...(options.continuingSubwordPrefix === undefined
      ? {}
      : { continuingSubwordPrefix: options.continuingSubwordPrefix }),
    ...(options.endOfWordSuffix === undefined ? {} : { endOfWordSuffix: options.endOfWordSuffix }),
  })
}

// ---------------------------------------------------------------------------------------------------------------
// Encoding.

/** A BPE model with its merge ranks precomputed. */
interface Ranked extends BpeModel {
  readonly ranks: ReadonlyMap<number, readonly [number, number]>
}

const rankCache = new WeakMap<BpeModel, Map<number, readonly [number, number]>>()

function ranksOf(model: BpeModel): ReadonlyMap<number, readonly [number, number]> {
  let r = rankCache.get(model)
  if (r === undefined) {
    r = new Map()
    const prefix = model.continuingSubwordPrefix
    for (const [rank, [x, y]] of model.merges.entries()) {
      const a = model.vocab.get(x)
      const b = model.vocab.get(y)
      const merged = x + (prefix !== undefined && y.startsWith(prefix) ? y.slice(prefix.length) : y)
      const id = model.vocab.get(merged)
      if (a === undefined || b === undefined || id === undefined)
        throw new WhiteboxError('value', `bpeModel: merge ${x} ${y} uses tokens outside the vocabulary`)
      r.set(pairKey(a, b), [rank, id])
    }
    rankCache.set(model, r)
  }
  return r
}

/**
 * A BPE model from a vocabulary and merges (as stored in an HF `tokenizer.json`).
 *
 * @param vocab - Token to id.
 * @param merges - Merges in rank order.
 * @param options - Unknown token, prefix and suffix, unknown fusing, byte fallback, dropout, ignoreMerges.
 * @returns The model.
 * @throws WhiteboxError with code `value` for a merge whose tokens are not in the vocabulary.
 * @concept byte-pair-encoding
 * @example
 * ```ts
 * import { bpeModel, bpeTokenize } from 'whitebox-engine/text/subword'
 * const m = bpeModel(new Map([['a', 0], ['b', 1], ['ab', 2]]), [['a', 'b']])
 * bpeTokenize(m, 'abab').map((t) => t.id) // => [2, 2]
 * ```
 */
export function bpeModel(
  vocab: ReadonlyMap<string, number>,
  merges: readonly (readonly [string, string])[],
  options: Omit<BpeModel, 'vocab' | 'merges'> = {},
): BpeModel {
  const model: BpeModel = { vocab: new Map(vocab), merges: merges.map(([a, b]) => [a, b] as const), ...options }
  ranksOf(model)
  return model
}

/** A token of an encoded word: its text, id and code-unit offsets within the word. */
export interface SubwordToken {
  readonly token: string
  readonly id: number
  readonly start: number
  readonly end: number
}

/** A uniform number in [0, 1) from each 32-bit draw. */
function uniforms(key: Key, n: number): Float64Array {
  const bits = randomBits(key, Math.max(n, 1))
  return Float64Array.from(bits, (b) => b / 4294967296)
}

/**
 * Encode one pre-tokenised word with a BPE model (HF's BPE.tokenize): split into characters (prefix inside, suffix
 * on the last), map unknown characters to the unknown token (or byte tokens with byte fallback), then apply merges
 * lowest rank first, leftmost first. With dropout and a key, each candidate merge is skipped with that probability.
 *
 * @param model - The model.
 * @param word - The word (already pre-tokenised and, for byte-level models, byte-mapped).
 * @param options - Dropout probability (default the model's) and the key for it.
 * @returns The tokens with ids and offsets in the word.
 * @throws WhiteboxError with code `value` for an unknown character without an unknown token.
 * @concept byte-pair-encoding
 * @example
 * ```ts
 * import { bpeModel, bpeTokenize } from 'whitebox-engine/text/subword'
 * const m = bpeModel(new Map([['[UNK]', 0], ['a', 1], ['b', 2], ['ab', 3]]), [['a', 'b']], { unkToken: '[UNK]' })
 * bpeTokenize(m, 'abz').map((t) => [t.token, t.start, t.end]) // => [['ab', 0, 2], ['[UNK]', 2, 3]]
 * ```
 */
export function bpeTokenize(
  model: BpeModel,
  word: string,
  options: { readonly dropout?: number; readonly key?: Key } = {},
): SubwordToken[] {
  if (word.length === 0) return []
  const direct = model.vocab.get(word)
  if (model.ignoreMerges === true && direct !== undefined)
    return [{ token: word, id: direct, start: 0, end: word.length }]
  const ranks = (model as Partial<Ranked>).ranks ?? ranksOf(model)
  const prefix = model.continuingSubwordPrefix
  const suffix = model.endOfWordSuffix
  const unk = model.unkToken === undefined ? undefined : model.vocab.get(model.unkToken)
  // Symbols: id, start and end in the word.
  let symbols: { id: number; start: number; end: number }[] = []
  const chars = Array.from(word)
  let at = 0
  for (const [k, ch] of chars.entries()) {
    let s = ch
    if (k > 0 && prefix !== undefined) s = prefix + s
    if (k === chars.length - 1 && suffix !== undefined) s += suffix
    const id = model.vocab.get(s)
    const start = at
    at += ch.length
    if (id !== undefined) {
      symbols.push({ id, start, end: at })
      continue
    }
    if (model.byteFallback === true) {
      const bytes = Array.from(utf8Bytes(ch), (b) =>
        model.vocab.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`),
      )
      if (bytes.every((b) => b !== undefined)) {
        for (const b of bytes) symbols.push({ id: b ?? 0, start, end: at })
        continue
      }
    }
    if (unk === undefined)
      throw new WhiteboxError('value', `bpeTokenize: ${ch} is not in the vocabulary and there is no unknown token`)
    const last = symbols[symbols.length - 1]
    if (model.fuseUnk === true && last !== undefined && last.id === unk) last.end = at
    else symbols.push({ id: unk, start, end: at })
  }
  const dropout = options.dropout ?? model.dropout ?? 0
  let draws: Float64Array | undefined
  let drawAt = 0
  let key = options.key
  const skip = (): boolean => {
    if (dropout <= 0) return false
    if (key === undefined) throw new WhiteboxError('value', 'bpeTokenize: dropout needs a key')
    if (draws === undefined || drawAt >= draws.length) {
      const [k1, k2] = split(key) as [Key, Key]
      key = k2
      draws = uniforms(k1, 64)
      drawAt = 0
    }
    return (draws[drawAt++] ?? 1) < dropout
  }
  // Repeatedly merge the best-ranked adjacent pair (leftmost on ties); with dropout, skipped candidates are tried
  // again after the next merge, as HF does.
  for (;;) {
    const candidates: { pos: number; rank: number; id: number }[] = []
    for (let i = 0; i + 1 < symbols.length; i++) {
      const r = ranks.get(pairKey(symbols[i]?.id ?? 0, symbols[i + 1]?.id ?? 0))
      if (r !== undefined) candidates.push({ pos: i, rank: r[0], id: r[1] })
    }
    candidates.sort((x, y) => x.rank - y.rank || x.pos - y.pos)
    const chosen = candidates.find(() => !skip())
    if (chosen === undefined) break
    const l = symbols[chosen.pos] as { id: number; start: number; end: number }
    const r = symbols[chosen.pos + 1] as { id: number; start: number; end: number }
    symbols = [
      ...symbols.slice(0, chosen.pos),
      { id: chosen.id, start: l.start, end: r.end },
      ...symbols.slice(chosen.pos + 2),
    ]
  }
  const names = namesOf(model)
  return symbols.map((s) => ({ token: names.get(s.id) ?? '', id: s.id, start: s.start, end: s.end }))
}
