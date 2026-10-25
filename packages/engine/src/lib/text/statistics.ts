/**
 * Corpus and vocabulary statistics: fitting Zipf's law to word frequencies and Heaps' law to vocabulary growth, lexical
 * diversity measures (type-token ratio and its corrected forms), tokeniser statistics such as fertility and
 * compression, and topic coherence as gensim's CoherenceModel computes it.
 */

import { MdmlError } from '../base/index.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { toFlat } from '../tensor/index.ts'
import { utf8Bytes } from './hashing.ts'

/** Counts of each type, from tokens or given directly. */
function frequencies(input: readonly string[] | ReadonlyMap<string, number>): number[] {
  if (input instanceof Map) return [...input.values()]
  const m = new Map<string, number>()
  for (const t of input as readonly string[]) m.set(t, (m.get(t) ?? 0) + 1)
  return [...m.values()]
}

/** Least squares fit y = a + b x. */
function lineFit(x: readonly number[], y: readonly number[]): [number, number] {
  const coef = toFlat(
    lstsq(
      x.map((v) => [1, v]),
      y.map((v) => [v]),
    ),
  )
  return [coef[0] ?? 0, coef[1] ?? 0]
}

/**
 * Fit Zipf's law f(r) ~ C / r^s to the rank-frequency distribution of a text: by least squares on log f against log r
 * (`loglog`, numpy.polyfit's line), or by maximum likelihood for the finite Zipf distribution over the observed ranks
 * (`mle`, scipy.stats.zipfian with n the number of types: the root of the score equation).
 *
 * @param input - Tokens, or counts by type.
 * @param options - The method (default `loglog`).
 * @returns The exponent s and, for `loglog`, the constant C.
 * @throws MdmlError with code `value` for fewer than two types.
 * @concept zipf-law
 * @example
 * ```ts
 * import { zipfFit } from 'mdml-engine/text'
 * zipfFit(new Map([['a', 60], ['b', 30], ['c', 20], ['d', 15]])).exponent // => 1
 * ```
 */
export function zipfFit(
  input: readonly string[] | ReadonlyMap<string, number>,
  options: { readonly method?: 'loglog' | 'mle' } = {},
): { readonly exponent: number; readonly constant: number } {
  const f = frequencies(input).sort((a, b) => b - a)
  if (f.length < 2) throw new MdmlError('value', 'zipfFit: need at least two types')
  const ranks = f.map((_, i) => i + 1)
  if (options.method !== 'mle') {
    const [a, b] = lineFit(
      ranks.map(Math.log),
      f.map((v) => Math.log(v)),
    )
    return { exponent: -b, constant: Math.exp(a) }
  }
  const n = f.length
  let total = 0
  let weighted = 0
  for (const [i, c] of f.entries()) {
    total += c
    weighted += c * Math.log(i + 1)
  }
  // d/ds log L = -sum f log r + N * sum r^-s log r / sum r^-s.
  const score = (s: number): number => {
    let num = 0
    let den = 0
    for (let r = 1; r <= n; r++) {
      const w = r ** -s
      num += w * Math.log(r)
      den += w
    }
    return -weighted + (total * num) / den
  }
  let hi = 1
  while (score(hi) > 0 && hi < 1e3) hi *= 2
  const root = brentq(score, 1e-9, hi, { xtol: 1e-14 })
  let norm = 0
  for (let r = 1; r <= n; r++) norm += r ** -root.root
  return { exponent: root.root, constant: total / norm }
}

/**
 * Fit Heaps' law V(N) = K N^beta to the vocabulary growth of a token sequence: least squares on log V against log N
 * over every prefix length.
 *
 * @param tokens - The tokens.
 * @returns K and beta.
 * @throws MdmlError with code `value` for fewer than two tokens.
 * @concept heaps-law
 * @example
 * ```ts
 * import { heapsFit } from 'mdml-engine/text'
 * heapsFit(['a', 'b', 'c', 'd']).beta // => 1
 * ```
 */
export function heapsFit(tokens: readonly string[]): { readonly k: number; readonly beta: number } {
  if (tokens.length < 2) throw new MdmlError('value', 'heapsFit: need at least two tokens')
  const seen = new Set<string>()
  const xs: number[] = []
  const ys: number[] = []
  for (const [i, t] of tokens.entries()) {
    seen.add(t)
    xs.push(Math.log(i + 1))
    ys.push(Math.log(seen.size))
  }
  const [a, b] = lineFit(xs, ys)
  return { k: Math.exp(a), beta: b }
}

/**
 * Lexical diversity: the type-token ratio V/N, Guiraud's root TTR V/sqrt(N), Herdan's log TTR log V / log N, the
 * moving-average TTR over windows (Covington and McFall 2010; the plain TTR when the text is shorter than the window),
 * and the proportion of hapax legomena among the types.
 *
 * @param tokens - The tokens.
 * @param options - The MATTR window (default 50).
 * @returns The measures (NaN for an empty text).
 * @concept lexical-diversity
 * @example
 * ```ts
 * import { lexicalDiversity } from 'mdml-engine/text'
 * lexicalDiversity(['a', 'b', 'a', 'c'], { window: 2 }) // => { types: 3, tokens: 4, ttr: 0.75, rootTtr: 1.5, logTtr: 0.7924812503605781, mattr: 1, hapaxRatio: 0.6666666666666666 }
 * ```
 */
export function lexicalDiversity(
  tokens: readonly string[],
  options: { readonly window?: number } = {},
): {
  readonly types: number
  readonly tokens: number
  readonly ttr: number
  readonly rootTtr: number
  readonly logTtr: number
  readonly mattr: number
  readonly hapaxRatio: number
} {
  const n = tokens.length
  const counts = new Map<string, number>()
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1)
  const v = counts.size
  let hapax = 0
  for (const c of counts.values()) if (c === 1) hapax++
  const w = options.window ?? 50
  let mattr: number
  if (n === 0) mattr = Number.NaN
  else if (n <= w) mattr = v / n
  else {
    const win = new Map<string, number>()
    for (let i = 0; i < w; i++) win.set(tokens[i] ?? '', (win.get(tokens[i] ?? '') ?? 0) + 1)
    let sum = win.size / w
    for (let i = w; i < n; i++) {
      const out = tokens[i - w] ?? ''
      const c = (win.get(out) ?? 0) - 1
      if (c === 0) win.delete(out)
      else win.set(out, c)
      win.set(tokens[i] ?? '', (win.get(tokens[i] ?? '') ?? 0) + 1)
      sum += win.size / w
    }
    mattr = sum / (n - w + 1)
  }
  return {
    types: v,
    tokens: n,
    ttr: n === 0 ? Number.NaN : v / n,
    rootTtr: n === 0 ? Number.NaN : v / Math.sqrt(n),
    logTtr: n < 2 ? Number.NaN : Math.log(v) / Math.log(n),
    mattr,
    hapaxRatio: v === 0 ? Number.NaN : hapax / v,
  }
}

/**
 * Tokeniser statistics: fertility (subword tokens per word), the proportion of words split into more than one token,
 * and the compression ratio (UTF-8 bytes, and characters, per token).
 *
 * @param words - The words of a text.
 * @param tokenize - The subword tokeniser of one word.
 * @returns The statistics.
 * @concept tokenizer-evaluation
 * @example
 * ```ts
 * import { tokenizerStatistics } from 'mdml-engine/text'
 * tokenizerStatistics(['unhappy', 'cat'], (w) => (w === 'unhappy' ? ['un', 'happy'] : [w])) // => { fertility: 1.5, continuedWords: 0.5, bytesPerToken: 3.3333333333333335, charactersPerToken: 3.3333333333333335 }
 * ```
 */
export function tokenizerStatistics(
  words: readonly string[],
  tokenize: (word: string) => readonly string[],
): {
  readonly fertility: number
  readonly continuedWords: number
  readonly bytesPerToken: number
  readonly charactersPerToken: number
} {
  let tokens = 0
  let split = 0
  let bytes = 0
  let chars = 0
  for (const w of words) {
    const t = tokenize(w).length
    tokens += t
    if (t > 1) split++
    bytes += utf8Bytes(w).length
    chars += Array.from(w).length
  }
  return {
    fertility: words.length === 0 ? Number.NaN : tokens / words.length,
    continuedWords: words.length === 0 ? Number.NaN : split / words.length,
    bytesPerToken: tokens === 0 ? Number.NaN : bytes / tokens,
    charactersPerToken: tokens === 0 ? Number.NaN : chars / tokens,
  }
}

const EPSILON = 1e-12

/** Boolean occurrence counts of words and word pairs over (virtual) documents, given the words present in each. */
function occurrences(present: readonly ReadonlySet<string>[]): {
  readonly single: Map<string, number>
  readonly pair: Map<string, number>
  readonly docs: number
} {
  const single = new Map<string, number>()
  const pair = new Map<string, number>()
  for (const set of present) {
    for (const w of set) single.set(w, (single.get(w) ?? 0) + 1)
    for (const a of set)
      for (const b of set) if (a !== b) pair.set(`${a}\u0000${b}`, (pair.get(`${a}\u0000${b}`) ?? 0) + 1)
  }
  return { single, pair, docs: present.length }
}

/**
 * The words present in each sliding window of a text as gensim's WordOccurrenceAccumulator tracks them: the first
 * window's words, then at each step the word leaving the window is unmarked (even when it occurs again inside it)
 * and the entering word marked; a text shorter than the window is one window.
 */
function slidingPresence(text: readonly string[], w: number, relevant: ReadonlySet<string>): Set<string>[] {
  if (text.length <= w) return [new Set(text.filter((t) => relevant.has(t)))]
  const out: Set<string>[] = []
  const cur = new Set(text.slice(0, w).filter((t) => relevant.has(t)))
  out.push(new Set(cur))
  for (let i = 1; i + w <= text.length; i++) {
    cur.delete(text[i - 1] ?? '')
    const entering = text[i + w - 1] ?? ''
    if (relevant.has(entering)) cur.add(entering)
    out.push(new Set(cur))
  }
  return out
}

/**
 * Topic coherence as gensim's CoherenceModel computes it: `u_mass` (Mimno et al. 2011; each word against every earlier
 * word, log((D(w, w*) / D + 1e-12) / (D(w*) / D)) over boolean documents) or `c_npmi` (Bouma's normalised PMI of every
 * ordered pair over boolean sliding windows of 10 tokens, counted as gensim's accumulator counts them), averaged per
 * topic and over topics.
 *
 * @param topics - The top words of each topic.
 * @param texts - The tokenised corpus.
 * @param options - The measure (default `u_mass`) and window size for `c_npmi` (default 10).
 * @returns The coherence of each topic and their mean.
 * @concept topic-coherence
 * @example
 * ```ts
 * import { topicCoherence } from 'mdml-engine/text'
 * const texts = [['cat', 'dog'], ['cat', 'dog'], ['car', 'road']]
 * topicCoherence([['cat', 'dog']], texts).perTopic.map((v) => Math.round(v * 1e6) / 1e6) // => [0]
 * ```
 */
export function topicCoherence(
  topics: readonly (readonly string[])[],
  texts: readonly (readonly string[])[],
  options: { readonly measure?: 'u_mass' | 'c_npmi'; readonly windowSize?: number } = {},
): { readonly perTopic: number[]; readonly mean: number } {
  const relevant = new Set(topics.flat())
  const measure = options.measure ?? 'u_mass'
  const w = options.windowSize ?? 10
  const present =
    measure === 'c_npmi'
      ? texts.flatMap((t) => slidingPresence(t, w, relevant))
      : texts.map((t) => new Set(t.filter((x) => relevant.has(x))))
  const { single, pair, docs: n } = occurrences(present)
  const perTopic = topics.map((topic) => {
    const values: number[] = []
    if (measure === 'u_mass') {
      for (let i = 1; i < topic.length; i++)
        for (let j = 0; j < i; j++) {
          const w = topic[i] ?? ''
          const s = topic[j] ?? ''
          const co = pair.get(`${w}\u0000${s}`) ?? 0
          values.push(Math.log((co / n + EPSILON) / ((single.get(s) ?? 0) / n)))
        }
    } else {
      for (const [i, w] of topic.entries())
        for (const [j, s] of topic.entries()) {
          if (i === j) continue
          const co = pair.get(`${w}\u0000${s}`) ?? 0
          const pmi = Math.log((co / n + EPSILON) / (((single.get(w) ?? 0) / n) * ((single.get(s) ?? 0) / n)))
          values.push(pmi / -Math.log(co / n + EPSILON))
        }
    }
    return values.reduce((a, b) => a + b, 0) / values.length
  })
  return { perTopic, mean: perTopic.reduce((a, b) => a + b, 0) / perTopic.length }
}
