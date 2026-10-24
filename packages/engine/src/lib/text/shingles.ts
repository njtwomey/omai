import { WhiteboxError } from '../base/index.ts'
import { fnv1a, utf8Bytes } from './hashing.ts'

/** Padding of n-grams (nltk's pad_sequence): n - 1 symbols on the left and/or right. */
export interface Padding<T> {
  readonly padLeft?: boolean
  readonly padRight?: boolean
  /** The left pad symbol (default `<s>`). */
  readonly leftPad?: T | string
  /** The right pad symbol (default `</s>`). */
  readonly rightPad?: T | string
}

function checkOrder(n: number, name: string): void {
  if (!Number.isInteger(n) || n < 1) throw new WhiteboxError('value', `${name}: n must be a positive integer`)
}

function pad<T>(items: readonly T[], n: number, p: Padding<T>): (T | string)[] {
  const left = p.padLeft === true ? Array.from({ length: n - 1 }, () => p.leftPad ?? '<s>') : []
  const right = p.padRight === true ? Array.from({ length: n - 1 }, () => p.rightPad ?? '</s>') : []
  return [...left, ...items, ...right]
}

/**
 * The n-grams of a sequence (nltk's `ngrams`), optionally padded with n - 1 boundary symbols on either side.
 *
 * @param items - The tokens (or characters).
 * @param n - The order.
 * @param padding - Boundary padding.
 * @returns The n-grams, in order.
 * @throws WhiteboxError with code `value` for an order below 1.
 * @concept n-grams
 * @example
 * ```ts
 * import { ngrams } from 'whitebox-engine/text'
 * ngrams(['a', 'b', 'c'], 2, { padLeft: true }) // => [['<s>', 'a'], ['a', 'b'], ['b', 'c']]
 * ```
 */
export function ngrams<T>(items: readonly T[], n: number, padding: Padding<T> = {}): (T | string)[][] {
  checkOrder(n, 'ngrams')
  const seq = pad(items, n, padding)
  const out: (T | string)[][] = []
  for (let i = 0; i + n <= seq.length; i++) out.push(seq.slice(i, i + n))
  return out
}

/**
 * All n-grams with min <= n <= max (nltk's `everygrams`), ordered by start position and then by length.
 *
 * @param items - The tokens.
 * @param minLength - The smallest order (default 1).
 * @param maxLength - The largest order (default the sequence length).
 * @returns The n-grams.
 * @concept n-grams
 * @example
 * ```ts
 * import { everygrams } from 'whitebox-engine/text'
 * everygrams(['a', 'b', 'c'], 1, 2) // => [['a'], ['a', 'b'], ['b'], ['b', 'c'], ['c']]
 * ```
 */
export function everygrams<T>(items: readonly T[], minLength = 1, maxLength: number = items.length): T[][] {
  const out: T[][] = []
  for (let i = 0; i < items.length; i++)
    for (let n = minLength; n <= maxLength && i + n <= items.length; n++) out.push(items.slice(i, i + n))
  return out
}

/**
 * Skip-grams (nltk's `skipgrams`, Guthrie et al. 2006): the n-grams that may skip up to k tokens in total, each
 * starting at its first token, in order of start and then of the combinations of the following n + k - 1 tokens.
 *
 * @param items - The tokens.
 * @param n - The order (at least 2).
 * @param k - The largest total skip.
 * @returns The skip-grams.
 * @throws WhiteboxError with code `value` for an order below 2 or a negative skip.
 * @concept n-grams
 * @example
 * ```ts
 * import { skipgrams } from 'whitebox-engine/text'
 * skipgrams(['a', 'b', 'c', 'd'], 2, 1) // => [['a', 'b'], ['a', 'c'], ['b', 'c'], ['b', 'd'], ['c', 'd']]
 * ```
 */
export function skipgrams<T>(items: readonly T[], n: number, k: number): T[][] {
  if (!Number.isInteger(n) || n < 2) throw new WhiteboxError('value', 'skipgrams: n must be an integer >= 2')
  if (!Number.isInteger(k) || k < 0) throw new WhiteboxError('value', 'skipgrams: k must be a non-negative integer')
  const out: T[][] = []
  const w = n + k
  for (let i = 0; i < items.length; i++) {
    // The window of n + k - 1 tokens after the head (shorter at the end: nltk pads with a sentinel and drops
    // combinations ending in it, which is the same as only combining real tokens whose last pick is real).
    const tail = items.slice(i + 1, i + w)
    const pick: number[] = []
    const choose = (start: number): void => {
      if (pick.length === n - 1) {
        out.push([items[i] as T, ...pick.map((p) => tail[p] as T)])
        return
      }
      for (let p = start; p < tail.length; p++) {
        pick.push(p)
        choose(p + 1)
        pick.pop()
      }
    }
    choose(0)
  }
  return out
}

/**
 * Counts of the n-grams of a token list, keyed by the tokens joined with a separator (by default one that cannot occur
 * in text): the n-gram statistics behind BLEU, chrF, ROUGE and distinct-n.
 *
 * @param tokens - The tokens (or characters).
 * @param n - The order.
 * @param separator - The join separator (default U+0000).
 * @returns The n-gram counts, in first-seen order.
 * @concept n-grams
 * @example
 * ```ts
 * import { ngramCounts } from 'whitebox-engine/text'
 * [...ngramCounts(['a', 'b', 'a', 'b'], 2, ' ')] // => [['a b', 2], ['b a', 1]]
 * ```
 */
export function ngramCounts(tokens: readonly string[], n: number, separator = '\u0000'): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i + n <= tokens.length; i++) {
    const key = tokens.slice(i, i + n).join(separator)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return counts
}

/** Options of the shingle extractor. */
export interface ShingleOptions {
  /** Shingle size: characters (code points) for `char`, tokens for `word` (default 3 for characters, 2 for words). */
  readonly size?: number
  /** Shingle characters of the text or tokens of a token list (default `char` for a string, `word` for a list). */
  readonly unit?: 'char' | 'word'
  /** Pad with n - 1 boundary symbols on both sides (character shingles pad with spaces, word shingles with `<s>`). */
  readonly pad?: boolean
  /** Join word shingles with this separator (default a space). */
  readonly separator?: string
}

/**
 * The shingle extractor shared by matching, fingerprinting and fastText: the contiguous k-grams of the characters of a
 * text or of a token list, as strings, in order (with repeats). Text shorter than k gives the whole text.
 *
 * @param input - A text (character shingles) or a token list (word shingles).
 * @param options - Size, unit, padding and separator.
 * @returns The shingles in order.
 * @throws WhiteboxError with code `value` for a size below 1.
 * @concept shingling
 * @example
 * ```ts
 * import { shingles } from 'whitebox-engine/text'
 * [shingles('abcd'), shingles(['a', 'rose', 'is', 'a'], { size: 2 })] // => [['abc', 'bcd'], ['a rose', 'rose is', 'is a']]
 * ```
 */
export function shingles(input: string | readonly string[], options: ShingleOptions = {}): string[] {
  const unit = options.unit ?? (typeof input === 'string' ? 'char' : 'word')
  const items =
    typeof input === 'string' ? (unit === 'char' ? Array.from(input) : input.split(/\s+/u).filter(Boolean)) : [...input]
  const k = options.size ?? (unit === 'char' ? 3 : 2)
  checkOrder(k, 'shingles')
  const sep = unit === 'char' ? '' : (options.separator ?? ' ')
  const seq =
    options.pad === true
      ? pad(
          items,
          k,
          unit === 'char'
            ? { padLeft: true, padRight: true, leftPad: ' ', rightPad: ' ' }
            : { padLeft: true, padRight: true },
        )
      : items
  if (seq.length === 0) return []
  if (seq.length < k) return [seq.join(sep)]
  const out: string[] = []
  for (let i = 0; i + k <= seq.length; i++) out.push(seq.slice(i, i + k).join(sep))
  return out
}

/**
 * The set of character k-shingles of a text (Broder's shingling of documents).
 *
 * @param text - The text.
 * @param k - The shingle size (default 3).
 * @returns The distinct shingles, in first-seen order.
 * @concept shingling
 * @example
 * ```ts
 * import { characterShingles } from 'whitebox-engine/text'
 * [...characterShingles('abab', 2)] // => ['ab', 'ba']
 * ```
 */
export function characterShingles(text: string, k = 3): Set<string> {
  return new Set(shingles(text, { size: k, unit: 'char' }))
}

/**
 * w-shingling (Broder 1997): the set of contiguous w-token shingles of a token list.
 *
 * @param tokens - The tokens.
 * @param w - The shingle width (default 4, Broder's choice).
 * @returns The distinct shingles, as space-joined strings.
 * @concept shingling
 * @example
 * ```ts
 * import { wShingling } from 'whitebox-engine/text'
 * [...wShingling(['a', 'rose', 'is', 'a', 'rose', 'is', 'a', 'rose'], 4)] // => ['a rose is a', 'rose is a rose', 'is a rose is']
 * ```
 */
export function wShingling(tokens: readonly string[], w = 4): Set<string> {
  return new Set(shingles(tokens, { size: w, unit: 'word' }))
}

/**
 * Weighted shingles: each distinct shingle with a weight, by default its count (a multiset for weighted Jaccard and
 * weighted MinHash), or a weight function of the shingle and its count (for example tf-idf).
 *
 * @param input - A text or token list (as {@link shingles}).
 * @param options - Shingle options and a weight function.
 * @returns The weights, in first-seen order.
 * @concept shingling
 * @example
 * ```ts
 * import { weightedShingles } from 'whitebox-engine/text'
 * [...weightedShingles('aaaa', { size: 2 })] // => [['aa', 3]]
 * ```
 */
export function weightedShingles(
  input: string | readonly string[],
  options: ShingleOptions & { readonly weight?: (shingle: string, count: number) => number } = {},
): Map<string, number> {
  const counts = new Map<string, number>()
  for (const s of shingles(input, options)) counts.set(s, (counts.get(s) ?? 0) + 1)
  const weight = options.weight
  if (weight === undefined) return counts
  return new Map([...counts].map(([s, c]) => [s, weight(s, c)]))
}

/**
 * fastText's character n-grams of a word (Bojanowski et al. 2017): the n-grams, minn <= n <= maxn, of `<word>`
 * counted in characters (over the UTF-8 bytes, as fastText), excluding the lone boundary markers; the whole `<word>`
 * is included when it is short enough, as fastText does.
 *
 * @param word - The word.
 * @param minn - The smallest n-gram (default 3).
 * @param maxn - The largest n-gram (default 6).
 * @returns The n-grams, in fastText's order (by start, then length).
 * @concept subword-embeddings
 * @example
 * ```ts
 * import { fastTextNgrams } from 'whitebox-engine/text'
 * fastTextNgrams('where', 3, 4) // => ['<wh', '<whe', 'whe', 'wher', 'her', 'here', 'ere', 'ere>', 're>']
 * ```
 */
export function fastTextNgrams(word: string, minn = 3, maxn = 6): string[] {
  const chars = Array.from(`<${word}>`)
  const out: string[] = []
  for (let i = 0; i < chars.length; i++) {
    let gram = ''
    for (let n = 1, j = i; j < chars.length && n <= maxn; n++) {
      gram += chars[j++]
      if (n >= minn && !(n === 1 && (i === 0 || j === chars.length))) out.push(gram)
    }
  }
  return out
}

/**
 * The bucket of each fastText character n-gram: FNV-1a with sign-extended bytes (fastText's `hash`), modulo the
 * number of buckets (gensim's `ft_ngram_hashes`); fastText's input matrix stores n-gram i at row nwords + bucket.
 *
 * @param word - The word.
 * @param minn - The smallest n-gram (default 3).
 * @param maxn - The largest n-gram (default 6).
 * @param buckets - The number of hash buckets (default 2,000,000, fastText's default).
 * @returns The bucket of each n-gram, in {@link fastTextNgrams} order.
 * @throws WhiteboxError with code `value` for a non-positive bucket count.
 * @concept subword-embeddings
 * @example
 * ```ts
 * import { fastTextBuckets } from 'whitebox-engine/text'
 * fastTextBuckets('a', 3, 3, 1000) // => [600]
 * ```
 */
export function fastTextBuckets(word: string, minn = 3, maxn = 6, buckets = 2_000_000): number[] {
  if (!Number.isInteger(buckets) || buckets < 1)
    throw new WhiteboxError('value', 'fastTextBuckets: buckets must be a positive integer')
  return fastTextNgrams(word, minn, maxn).map((g) => fnv1a(utf8Bytes(g), { signedBytes: true }) % buckets)
}
