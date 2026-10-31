/**
 * N-gram language models with the smoothers of Chen and Goodman (1998): maximum likelihood, add-k, Witten-Bell,
 * absolute discounting, interpolated Kneser-Ney (continuation counts below the top order) and its modified form with
 * three discounts, and Katz backoff over Good-Turing discounts. A model is built from the counts of padded n-grams,
 * which accumulate one sentence per step as an {@link Algorithm}; it answers probabilities, whole next-word
 * distributions, sentence log-probabilities, perplexity and samples. Words rarer than a cutoff become `<unk>`.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { categorical, foldIn, type Key } from '../random/index.ts'
import type { Algorithm } from '../trace/index.ts'

/** How the probabilities of unseen events are made. */
export type NgramSmoothing =
  | 'mle'
  | 'add-k'
  | 'witten-bell'
  | 'absolute-discounting'
  | 'kneser-ney'
  | 'modified-kneser-ney'
  | 'katz'

/** Options of {@link ngramModel}, {@link countNgrams} and {@link ngramTrainingAlgorithm}. */
export interface NgramOptions {
  /** The order n (default 3). */
  readonly order?: number
  /** The smoother (default `kneser-ney`). */
  readonly smoothing?: NgramSmoothing
  /** The pseudo-count of `add-k` (default 1). */
  readonly k?: number
  /** The discount of `absolute-discounting` and `kneser-ney`; default estimated per order as n1 / (n1 + 2 n2). */
  readonly discount?: number
  /** The three discounts of `modified-kneser-ney`; default estimated per order (Chen and Goodman's formulas). */
  readonly discounts?: readonly [number, number, number]
  /** The count above which Katz backoff leaves counts undiscounted (default 5). */
  readonly katzCutoff?: number
  /** Words seen fewer times become `<unk>` (default 1: none). */
  readonly minCount?: number
  /** The lowest order: `uniform` interpolates it with 1 / |V| (default), `mle` leaves the relative frequencies (NLTK's). */
  readonly unigram?: 'uniform' | 'mle'
  /** Sentence-start padding (default `<s>`). */
  readonly bos?: string
  /** Sentence-end token (default `</s>`). */
  readonly eos?: string
  /** The unknown-word token (default `<unk>`). */
  readonly unk?: string
}

const SEP = '\u0000'

/** The counts of padded n-grams by order: `counts[m - 1]` maps a gram's words joined by NUL to its count. */
export interface NgramCounts {
  readonly order: number
  readonly bos: string
  readonly eos: string
  readonly counts: readonly Map<string, number>[]
  /** The sentences counted. */
  readonly sentences: number
}

/**
 * Count the padded n-grams of orders 1 to n over sentences: each sentence gets n - 1 start tokens and an end token, and
 * every real word and the end token is the last word of one gram per order.
 *
 * @param sentences - Token lists.
 * @param options - The order and the padding tokens.
 * @returns The counts by order.
 * @throws MdmlError with code `value` for an order below 1.
 * @concept language-models
 * @example
 * ```ts
 * import { countNgrams } from 'mdml-engine/text'
 * const c = countNgrams([['a', 'b'], ['a', 'c']], { order: 2 })
 * [c.counts[0]?.get('a'), c.counts[1]?.get(['<s>', 'a'].join('\u0000'))] // => [2, 2]
 * ```
 */
export function countNgrams(sentences: readonly (readonly string[])[], options: NgramOptions = {}): NgramCounts {
  const acc = emptyCounts(options)
  for (const s of sentences) addSentence(acc, s)
  return acc
}

function emptyCounts(options: NgramOptions): { -readonly [K in keyof NgramCounts]: NgramCounts[K] } {
  const order = options.order ?? 3
  if (!Number.isInteger(order) || order < 1)
    throw new MdmlError('value', 'countNgrams: order must be a positive integer')
  return {
    order,
    bos: options.bos ?? '<s>',
    eos: options.eos ?? '</s>',
    counts: Array.from({ length: order }, () => new Map<string, number>()),
    sentences: 0,
  }
}

function addSentence(acc: { -readonly [K in keyof NgramCounts]: NgramCounts[K] }, sentence: readonly string[]): void {
  const padded = [...new Array<string>(acc.order - 1).fill(acc.bos), ...sentence, acc.eos]
  for (let i = acc.order - 1; i < padded.length; i++) {
    for (let m = 1; m <= acc.order; m++) {
      const key = padded.slice(i - m + 1, i + 1).join(SEP)
      const table = acc.counts[m - 1] as Map<string, number>
      table.set(key, (table.get(key) ?? 0) + 1)
    }
  }
  acc.sentences++
}

/** The counts of counts of a table of numbers: how many grams have each value. */
function countsOfCounts(values: Iterable<number>): Map<number, number> {
  const out = new Map<number, number>()
  for (const v of values) out.set(v, (out.get(v) ?? 0) + 1)
  return out
}

/**
 * The Good-Turing adjustment of observed counts (Good 1953): a word seen r times is treated as seen
 * r* = (r + 1) N(r + 1) / N(r) times, where N(r) is the number of words seen r times, and the probability of all
 * unseen words together is N(1) / N. Where N(r + 1) is zero the count is left as it is.
 *
 * @param counts - The observed counts of the seen events.
 * @returns The adjusted count for each distinct observed count, and the total probability of the unseen.
 * @concept language-models
 * @example
 * ```ts
 * import { goodTuring } from 'mdml-engine/text'
 * const g = goodTuring([1, 1, 1, 2, 2, 3])
 * [g.adjusted.get(1), g.adjusted.get(2), g.unseenMass] // => [1.3333333333333333, 1.5, 0.3]
 * ```
 */
export function goodTuring(counts: readonly number[]): { adjusted: Map<number, number>; unseenMass: number } {
  const nr = countsOfCounts(counts)
  const total = counts.reduce((a, b) => a + b, 0)
  const adjusted = new Map<number, number>()
  for (const [r, n] of nr) {
    const next = nr.get(r + 1)
    adjusted.set(r, next === undefined ? r : ((r + 1) * next) / n)
  }
  return { adjusted, unseenMass: total === 0 ? 0 : (nr.get(1) ?? 0) / total }
}

/** Chen and Goodman's discounts D1, D2, D3+ from the counts of counts n1..n4 (absolute discounting uses D1 = D2 = D3 = n1 / (n1 + 2 n2)). */
function estimateDiscounts(values: Iterable<number>, modified: boolean): [number, number, number] {
  const nr = countsOfCounts(values)
  const n1 = nr.get(1) ?? 0
  const n2 = nr.get(2) ?? 0
  const n3 = nr.get(3) ?? 0
  const n4 = nr.get(4) ?? 0
  if (n1 === 0 || n2 === 0) return [0.5, 0.5, 0.5]
  const y = n1 / (n1 + 2 * n2)
  if (!modified) return [y, y, y]
  const d1 = 1 - (2 * y * n2) / n1
  const d2 = n3 > 0 ? 2 - (3 * y * n3) / n2 : d1
  const d3 = n3 > 0 && n4 > 0 ? 3 - (4 * y * n4) / n3 : d2
  return [d1, d2, d3]
}

/** A context's statistics at one order. */
interface Context {
  /** The sum of the adjusted counts of the words that follow it. */
  total: number
  /** The adjusted counts of those words. */
  next: Map<string, number>
  n1: number
  n2: number
  n3: number
}

/** A built n-gram language model. */
export interface NgramModel {
  readonly order: number
  readonly smoothing: NgramSmoothing
  /** The words the model predicts (the vocabulary without the start padding), sorted. */
  readonly vocabulary: readonly string[]
  /** The sentence-start, sentence-end and unknown tokens. */
  readonly bos: string
  readonly eos: string
  readonly unk: string
  /** `P(word | context)`: the last order - 1 words of the context are used; unseen words are read as `<unk>`. */
  prob(word: string, context?: readonly string[]): number
  /** `P(. | context)` over `vocabulary`, in its order. */
  distribution(context?: readonly string[]): Float64Array
  /** The natural log of the probability of a sentence, with its end token. */
  sentenceLogProb(sentence: readonly string[]): number
  /** The perplexity over sentences: exp of the mean negative log probability per predicted token (the end token included). */
  perplexity(sentences: readonly (readonly string[])[]): number
  /** A sentence drawn word by word until the end token or `maxLength` words. */
  sample(key: Key, maxLength?: number): string[]
  /** The probability mass the model leaves for the words never seen after a context at the highest order. */
  unseenMass(context?: readonly string[]): number
}

/**
 * Build a language model from n-gram counts: apply the cutoff for rare words, form the continuation counts when
 * Kneser-Ney asks for them, estimate the discounts, and return the model.
 *
 * @param c - The counts ({@link countNgrams}).
 * @param options - The smoother and its knobs, the cutoff and the unknown token (the order is the counts').
 * @returns The model.
 * @throws MdmlError with code `value` for an unknown smoother or a bad discount.
 * @concept language-models
 * @example
 * ```ts
 * import { countNgrams, ngramFromCounts } from 'mdml-engine/text'
 * const m = ngramFromCounts(countNgrams([['a', 'b'], ['a', 'c']], { order: 1 }), { smoothing: 'mle' })
 * m.prob('a') // => 0.3333333333333333
 * ```
 */
export function ngramFromCounts(c: NgramCounts, options: NgramOptions = {}): NgramModel {
  const smoothing = options.smoothing ?? 'kneser-ney'
  const unk = options.unk ?? '<unk>'
  const minCount = options.minCount ?? 1
  const order = c.order
  const bos = c.bos
  const eos = c.eos
  const k = options.k ?? 1
  if (options.discount !== undefined && !(options.discount >= 0 && options.discount < 1))
    throw new MdmlError('value', 'ngramModel: the discount must be in [0, 1)')
  // Rare words become <unk>: re-key every gram.
  const unigram = c.counts[0] as Map<string, number>
  const known = new Set<string>()
  let anyUnk = false
  for (const [w, n] of unigram) {
    if (n >= minCount || w === eos) known.add(w)
    else anyUnk = true
  }
  const raw: Map<string, number>[] = c.counts.map((t) => {
    if (!anyUnk) return t
    const out = new Map<string, number>()
    for (const [g, n] of t) {
      const key = g
        .split(SEP)
        .map((w) => (known.has(w) || w === bos ? w : unk))
        .join(SEP)
      out.set(key, (out.get(key) ?? 0) + n)
    }
    return out
  })
  const words = new Set<string>([eos, unk])
  for (const g of raw[0]?.keys() ?? []) words.add(g)
  words.delete(bos)
  const vocabulary = [...words].sort()
  const V = vocabulary.length
  const kn = smoothing === 'kneser-ney' || smoothing === 'modified-kneser-ney'
  // The count each gram carries at each level: raw counts, or continuation counts below the top order for KN.
  const adjusted: Map<string, number>[] = raw.map((t, i) => {
    if (!kn || i === order - 1) return t
    const out = new Map<string, number>()
    const preds = new Map<string, Set<string>>()
    for (const h of raw[i + 1]?.keys() ?? []) {
      const cut = h.indexOf(SEP)
      const tail = h.slice(cut + 1)
      const set = preds.get(tail) ?? new Set<string>()
      set.add(h.slice(0, cut))
      preds.set(tail, set)
    }
    for (const [g, n] of t) out.set(g, g.startsWith(bos + SEP) ? n : (preds.get(g)?.size ?? 0))
    return out
  })
  // Contexts by level.
  const levels: Map<string, Context>[] = adjusted.map((t, i) => {
    const ctx = new Map<string, Context>()
    for (const [g, n] of t) {
      if (n <= 0) continue
      const cut = g.lastIndexOf(SEP)
      const h = cut < 0 ? '' : g.slice(0, cut)
      const w = cut < 0 ? g : g.slice(cut + 1)
      let e = ctx.get(h)
      if (e === undefined) {
        e = { total: 0, next: new Map(), n1: 0, n2: 0, n3: 0 }
        ctx.set(h, e)
      }
      e.total += n
      e.next.set(w, n)
      if (n === 1) e.n1++
      else if (n === 2) e.n2++
      else e.n3++
    }
    if (i === 0 && !ctx.has('')) ctx.set('', { total: 0, next: new Map(), n1: 0, n2: 0, n3: 0 })
    return ctx
  })
  const discounts: (readonly [number, number, number])[] = adjusted.map((t) => {
    const positive = [...t.values()].filter((v) => v > 0)
    if (smoothing === 'modified-kneser-ney') return options.discounts ?? estimateDiscounts(positive, true)
    const d = options.discount ?? estimateDiscounts(positive, false)[0]
    return [d, d, d]
  })
  const cutoff = options.katzCutoff ?? 5
  // Katz discount ratios d_r by level.
  const katz: Map<number, number>[] = adjusted.map((t) => {
    const out = new Map<number, number>()
    if (smoothing !== 'katz') return out
    const nr = countsOfCounts(t.values())
    const n1 = nr.get(1) ?? 0
    const tail = ((cutoff + 1) * (nr.get(cutoff + 1) ?? 0)) / (n1 === 0 ? 1 : n1)
    for (let r = 1; r <= cutoff; r++) {
      const nrr = nr.get(r) ?? 0
      const star = nrr === 0 ? r : ((r + 1) * (nr.get(r + 1) ?? 0)) / nrr
      const d = tail >= 1 || nrr === 0 ? 1 : (star / r - tail) / (1 - tail)
      out.set(r, d > 0 && d <= 1 ? d : 1)
    }
    return out
  })
  const mleUnigram = options.unigram === 'mle'
  const key = (h: readonly string[]): string => h.join(SEP)
  const mapped = (w: string): string => (words.has(w) ? w : unk)
  const mapContext = (context: readonly string[]): string[] => {
    const ctx = (order > 1 ? context.slice(-(order - 1)) : []).map((w) => (w === bos ? w : mapped(w)))
    while (ctx.length < order - 1) ctx.unshift(bos)
    return ctx
  }

  // P(w | h) at level m (h has m - 1 words), interpolated forms.
  const interpolated = (m: number, h: readonly string[], w: string): number => {
    const level = levels[m - 1] as Map<string, Context>
    const e = level.get(key(h))
    if (m === 1) {
      const top = level.get('') as Context
      const n = top.next.get(w) ?? 0
      if (top.total === 0) return 1 / V
      if (smoothing === 'witten-bell')
        return mleUnigram ? n / top.total : (n + top.next.size / V) / (top.total + top.next.size)
      if (mleUnigram) return n / top.total
      return withBase(top, n, 1 / V, discounts[0] as readonly [number, number, number])
    }
    const lower = (): number => interpolated(m - 1, h.slice(1), w)
    if (e === undefined) return lower()
    const n = e.next.get(w) ?? 0
    if (smoothing === 'witten-bell') return (n + e.next.size * lower()) / (e.total + e.next.size)
    return withBase(e, n, lower(), discounts[m - 1] as readonly [number, number, number])
  }
  function withBase(e: Context, n: number, base: number, d: readonly [number, number, number]): number {
    const dn = n === 0 ? 0 : n === 1 ? d[0] : n === 2 ? d[1] : d[2]
    const gamma =
      smoothing === 'modified-kneser-ney'
        ? (d[0] * e.n1 + d[1] * e.n2 + d[2] * e.n3) / e.total
        : (d[0] * e.next.size) / e.total
    return (n - dn) / e.total + gamma * base
  }
  // Katz: the probability of w after h at level m, and the leftover weights.
  const katzProb = (m: number, h: readonly string[], w: string): number => {
    const level = levels[m - 1] as Map<string, Context>
    if (m === 1) {
      const top = level.get('') as Context
      const n = top.next.get(w) ?? 0
      const d = katz[0] as Map<number, number>
      if (n > 0) return ((d.get(n) ?? 1) * n) / top.total
      let seen = 0
      for (const x of top.next.values()) seen += ((d.get(x) ?? 1) * x) / top.total
      return Math.max(0, 1 - seen) / Math.max(1, V - top.next.size)
    }
    const e = level.get(key(h))
    if (e === undefined) return katzProb(m - 1, h.slice(1), w)
    const n = e.next.get(w) ?? 0
    const d = katz[m - 1] as Map<number, number>
    if (n > 0) return ((d.get(n) ?? 1) * n) / e.total
    // alpha(h) = (1 - sum of discounted seen) / (1 - sum of lower-order probabilities of those words)
    let seen = 0
    let lowerSeen = 0
    const lowerH = h.slice(1)
    for (const [x, c] of e.next) {
      seen += ((d.get(c) ?? 1) * c) / e.total
      lowerSeen += katzProb(m - 1, lowerH, x)
    }
    const alpha = lowerSeen >= 1 ? 0 : Math.max(0, 1 - seen) / (1 - lowerSeen)
    return alpha * katzProb(m - 1, lowerH, w)
  }
  const prob = (word: string, context: readonly string[] = []): number => {
    const h = mapContext(context)
    const w = mapped(word)
    if (smoothing === 'katz') return katzProb(order, h, w)
    if (smoothing === 'add-k') {
      const e = (levels[order - 1] as Map<string, Context>).get(key(h))
      return ((e?.next.get(w) ?? 0) + k) / ((e?.total ?? 0) + k * V)
    }
    if (smoothing === 'mle') {
      let m = order
      let hh = h
      // Unsmoothed: the highest order that has seen the context (a model is not defined on unseen contexts).
      let e = (levels[m - 1] as Map<string, Context>).get(key(hh))
      while (e === undefined && m > 1) {
        m--
        hh = hh.slice(1)
        e = (levels[m - 1] as Map<string, Context>).get(key(hh))
      }
      return e === undefined || e.total === 0 ? 0 : (e.next.get(w) ?? 0) / e.total
    }
    return interpolated(order, h, w)
  }
  const distribution = (context: readonly string[] = []): Float64Array =>
    Float64Array.from(vocabulary, (w) => prob(w, context))
  const sentenceLogProb = (sentence: readonly string[]): number => {
    const padded = [...new Array<string>(order - 1).fill(bos), ...sentence, eos]
    let lp = 0
    for (let i = order - 1; i < padded.length; i++)
      lp += Math.log(prob(padded[i] as string, padded.slice(Math.max(0, i - order + 1), i)))
    return lp
  }
  return {
    order,
    smoothing,
    vocabulary,
    bos,
    eos,
    unk,
    prob,
    distribution,
    sentenceLogProb,
    perplexity: (sentences) => {
      let lp = 0
      let n = 0
      for (const s of sentences) {
        lp += sentenceLogProb(s)
        n += s.length + 1
      }
      return Math.exp(-lp / n)
    },
    sample: (key0, maxLength = 50) => {
      const out: string[] = []
      for (let i = 0; i < maxLength; i++) {
        const p = distribution([...new Array<string>(order - 1).fill(bos), ...out])
        const z = p.reduce((a, b) => a + b, 0)
        const w =
          vocabulary[
            categorical(
              foldIn(key0, i),
              Array.from(p, (x) => x / z),
              [1],
            ).data[0] ?? 0
          ] ?? eos
        if (w === eos) break
        out.push(w)
      }
      return out
    },
    unseenMass: (context = []) => {
      const e = (levels[order - 1] as Map<string, Context>).get(key(mapContext(context)))
      let seen = 0
      for (const w of e?.next.keys() ?? []) seen += prob(w, context)
      return Math.max(0, 1 - seen)
    },
  }
}

/**
 * Count and build in one call: an n-gram language model of tokenised sentences.
 *
 * @param sentences - Token lists.
 * @param options - Order, smoother and its knobs, cutoff and tokens.
 * @returns The model.
 * @concept language-models
 * @example
 * ```ts
 * import { ngramModel } from 'mdml-engine/text'
 * const m = ngramModel([['a', 'b'], ['a', 'c'], ['a', 'b']], { order: 2, smoothing: 'add-k', k: 1 })
 * m.prob('b', ['a']) // => 0.375
 * ```
 */
export function ngramModel(sentences: readonly (readonly string[])[], options: NgramOptions = {}): NgramModel {
  return ngramFromCounts(countNgrams(sentences, options), options)
}

/** The state of {@link ngramTrainingAlgorithm}: sentences counted so far and the counts by order as entries. */
export interface NgramTrainingState {
  readonly counts: NgramCounts
  readonly next: number
  readonly tokens: number
}

/**
 * N-gram counting as an {@link Algorithm}: one sentence per step, so the counts, the number of distinct n-grams and
 * (with held-out sentences) the perplexity of the model so far can be watched as the corpus arrives.
 *
 * @param sentences - The training sentences, in the order they arrive.
 * @param options - The model's options.
 * @param heldOut - Sentences to score the model on after every step.
 * @returns The algorithm.
 * @concept language-models
 * @example
 * ```ts
 * import { ngramTrainingAlgorithm } from 'mdml-engine/text'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = ngramTrainingAlgorithm([['a', 'b'], ['a', 'c']], { order: 2 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.counts.sentences // => 2
 * ```
 */
export function ngramTrainingAlgorithm(
  sentences: readonly (readonly string[])[],
  options: NgramOptions = {},
  heldOut: readonly (readonly string[])[] = [],
): Algorithm<NgramTrainingState> {
  return {
    name: 'ngramTraining',
    config: { order: options.order ?? 3, smoothing: options.smoothing ?? 'kneser-ney', sentences: sentences.length },
    init: () => ({ counts: emptyCounts(options), next: 0, tokens: 0 }),
    step: (s) => {
      const sentence = sentences[s.next] ?? []
      addSentence(s.counts as { -readonly [K in keyof NgramCounts]: NgramCounts[K] }, sentence)
      return { counts: s.counts, next: s.next + 1, tokens: s.tokens + sentence.length + 1 }
    },
    done: (s) => (s.next >= sentences.length ? 'converged' : undefined),
    observe: (s) => {
      const metrics: Record<string, number> = {
        documents: s.next,
        ngrams: s.counts.counts.reduce((a, t) => a + t.size, 0),
      }
      if (heldOut.length > 0) metrics['perplexity'] = ngramFromCounts(s.counts, options).perplexity(heldOut)
      return { metrics }
    },
    serialise: (s) =>
      ({
        counts: s.counts.counts.map((t) => [...t]),
        sentences: s.counts.sentences,
        next: s.next,
        tokens: s.tokens,
      }) as unknown as Json,
    deserialise: (j) => {
      const o = j as unknown as { counts: [string, number][][]; sentences: number; next: number; tokens: number }
      const base = emptyCounts(options)
      return {
        counts: { ...base, counts: o.counts.map((t) => new Map(t)), sentences: o.sentences },
        next: o.next,
        tokens: o.tokens,
      }
    },
  }
}
