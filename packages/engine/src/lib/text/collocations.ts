/**
 * Collocation finding: discovering word pairs and triples that occur together more often than chance, as nltk does.
 * Bigram and trigram counts within a window, the association measures (Student's t, chi-square, PMI, likelihood ratio,
 * Poisson-Stirling, Jaccard and others) from their marginals, and scoring and ranking every n-gram.
 */

import { OmaiError } from '../base/index.ts'
import { hypergeometric } from '../dists/index.ts'
import { item } from '../tensor/index.ts'
import { compareCodePoints } from './_classes.ts'

/** The marginals of a bigram: its count, the counts of its two words, and the number of words. */
export interface BigramMarginals {
  readonly nii: number
  readonly nix: number
  readonly nxi: number
  readonly nxx: number
}

/** The marginals of a trigram: its count, its pair counts (12, 1_3, 23), its word counts, and the number of words. */
export interface TrigramMarginals {
  readonly niii: number
  readonly pairs: readonly [number, number, number]
  readonly words: readonly [number, number, number]
  readonly nxxx: number
}

const SMALL = 1e-20

/** The contingency table of a bigram (n_ii, n_oi, n_io, n_oo). */
function bigramTable(m: BigramMarginals): [number, number, number, number] {
  const noi = m.nxi - m.nii
  const nio = m.nix - m.nii
  return [m.nii, noi, nio, m.nxx - m.nii - noi - nio]
}

/** The contingency table of a trigram (nltk's order). */
function trigramTable(m: TrigramMarginals): number[] {
  const [iix, ixi, xii] = m.pairs
  const [ixx, xix, xxi] = m.words
  const iii = m.niii
  const oii = xii - iii
  const ioi = ixi - iii
  const iio = iix - iii
  const ooi = xxi - iii - oii - ioi
  const oio = xix - iii - oii - iio
  const ioo = ixx - iii - ioi - iio
  const ooo = m.nxxx - iii - oii - ioi - iio - ooi - oio - ioo
  return [iii, oii, ioi, ooi, iio, oio, ioo, ooo]
}

/** nltk's expected cell counts of an n-way contingency table under independence. */
function expected(table: readonly number[], n: number): number[] {
  let all = 0
  for (const v of table) all += v
  if (n === 2) return table.map((v, i) => ((v + (table[i ^ 1] ?? 0)) * (v + (table[i ^ 2] ?? 0))) / all)
  return table.map((_, i) => {
    let prod = 1
    for (let b = 0; b < n; b++) {
      const bit = 1 << b
      let s = 0
      for (let x = 0; x < 2 ** n; x++) if ((x & bit) === (i & bit)) s += table[x] ?? 0
      prod *= s
    }
    return prod / all ** (n - 1)
  })
}

/** An association measure of n-grams (nltk's BigramAssocMeasures and TrigramAssocMeasures). */
export type AssociationMeasure =
  | 'rawFreq'
  | 'studentT'
  | 'chiSq'
  | 'pmi'
  | 'likelihoodRatio'
  | 'poissonStirling'
  | 'jaccard'
  | 'miLike'
  | 'dice'
  | 'phiSq'
  | 'fisher'

/** Counts in the form shared by both orders: the n-gram count, the product of word counts, the number of words. */
function general(
  count: number,
  wordProduct: number,
  total: number,
  table: readonly number[],
  n: number,
  measure: AssociationMeasure,
  power: number,
): number {
  switch (measure) {
    case 'rawFreq':
      return count / total
    case 'studentT':
      return (count - wordProduct / total ** (n - 1)) / Math.sqrt(count + SMALL)
    case 'chiSq': {
      const e = expected(table, n)
      return table.reduce((s, o, i) => s + (o - (e[i] ?? 0)) ** 2 / ((e[i] ?? 0) + SMALL), 0)
    }
    case 'miLike':
      return count ** power / wordProduct
    case 'pmi':
      return Math.log2(count * total ** (n - 1)) - Math.log2(wordProduct)
    case 'likelihoodRatio': {
      const e = expected(table, n)
      return 2 * table.reduce((s, o, i) => s + o * Math.log(o / ((e[i] ?? 0) + SMALL) + SMALL), 0)
    }
    case 'poissonStirling': {
      const ex = wordProduct / total ** (n - 1)
      return count * (Math.log2(count / ex) - 1)
    }
    case 'jaccard': {
      let s = 0
      for (let i = 0; i < table.length - 1; i++) s += table[i] ?? 0
      return (table[0] ?? 0) / s
    }
    default:
      throw new OmaiError('value', `association: ${measure} is only defined for bigrams`)
  }
}

/**
 * A bigram association score from its marginals, as nltk's BigramAssocMeasures: raw frequency, Student's t, chi-square
 * (n times phi squared), PMI, the likelihood ratio, Poisson-Stirling, Jaccard, MI-like, Dice, phi squared, and
 * Fisher's exact test (the one-sided p-value, scipy's `alternative='less'`).
 *
 * @param measure - The measure.
 * @param m - The marginals.
 * @param options - The power of MI-like (default 3).
 * @returns The score.
 * @concept collocations
 * @example
 * ```ts
 * import { bigramAssociation } from 'omai-engine/text'
 * bigramAssociation('pmi', { nii: 4, nix: 8, nxi: 4, nxx: 64 }) // => 3
 * ```
 */
export function bigramAssociation(
  measure: AssociationMeasure,
  m: BigramMarginals,
  options: { readonly power?: number } = {},
): number {
  const table = bigramTable(m)
  const [nii, noi, nio, noo] = table
  const phiSq = (): number => (nii * noo - nio * noi) ** 2 / ((nii + nio) * (nii + noi) * (nio + noo) * (noi + noo))
  switch (measure) {
    case 'phiSq':
      return phiSq()
    case 'chiSq':
      return m.nxx * phiSq()
    case 'dice':
      return (2 * m.nii) / (m.nix + m.nxi)
    case 'fisher': {
      // nltk orders the table [[n_ii, n_io], [n_oi, n_oo]] from its (n_ii, n_oi, n_io, n_oo) unpacked as
      // (n_ii, n_io, n_oi, n_oo), so the second cell of the first row is n_oi.
      // scipy reads the table as int64, truncating the fractional counts of wide windows.
      const [a, b, c, d] = [nii, noi, nio, noo].map(Math.trunc) as [number, number, number, number]
      return item(hypergeometric(a + b + c + d, a + b, a + c).cdf(a))
    }
    default:
      return general(m.nii, m.nix * m.nxi, m.nxx, table, 2, measure, options.power ?? 3)
  }
}

/**
 * A trigram association score from its marginals, as nltk's TrigramAssocMeasures (raw frequency, Student's t,
 * chi-square, PMI, the likelihood ratio, Poisson-Stirling, Jaccard, MI-like).
 *
 * @param measure - The measure.
 * @param m - The marginals.
 * @param options - The power of MI-like (default 3).
 * @returns The score.
 * @throws OmaiError with code `value` for a bigram-only measure.
 * @concept collocations
 * @example
 * ```ts
 * import { trigramAssociation } from 'omai-engine/text'
 * trigramAssociation('rawFreq', { niii: 2, pairs: [3, 2, 2], words: [4, 3, 2], nxxx: 20 }) // => 0.1
 * ```
 */
export function trigramAssociation(
  measure: AssociationMeasure,
  m: TrigramMarginals,
  options: { readonly power?: number } = {},
): number {
  const [a, b, c] = m.words
  return general(m.niii, a * b * c, m.nxxx, trigramTable(m), 3, measure, options.power ?? 3)
}

/** Bigram (or trigram) and word counts of a word sequence, for collocation finding. */
export interface CollocationCounts {
  readonly order: 2 | 3
  readonly windowSize: number
  readonly words: ReadonlyMap<string, number>
  /** n-gram counts keyed by the words joined with U+0000. */
  readonly ngrams: ReadonlyMap<string, number>
  readonly pairs: ReadonlyMap<string, number>
  readonly wildcards: ReadonlyMap<string, number>
  readonly total: number
}

const SEP = '\u0000'

/**
 * Count bigrams or trigrams of a word sequence within a window, as nltk's Bigram/TrigramCollocationFinder.from_words:
 * every window of `windowSize` words (padded at the end) contributes the pairs (or triples) of its first word with the
 * later ones.
 *
 * @param words - The words.
 * @param options - Order (default 2) and window size (default the order).
 * @returns The counts.
 * @throws OmaiError with code `value` for a window smaller than the order.
 * @concept collocations
 * @example
 * ```ts
 * import { collocationCounts } from 'omai-engine/text'
 * collocationCounts(['a', 'b', 'a', 'b']).ngrams.get('a\u0000b') // => 2
 * ```
 */
export function collocationCounts(
  words: readonly string[],
  options: { readonly order?: 2 | 3; readonly windowSize?: number } = {},
): CollocationCounts {
  const order = options.order ?? 2
  const ws = options.windowSize ?? order
  if (!Number.isInteger(ws) || ws < order)
    throw new OmaiError('value', `collocationCounts: the window must be at least ${order}`)
  const wfd = new Map<string, number>()
  const ngramFd = new Map<string, number>()
  const pairs = new Map<string, number>()
  const wildcards = new Map<string, number>()
  const inc = (m: Map<string, number>, k: string): void => {
    m.set(k, (m.get(k) ?? 0) + 1)
  }
  for (let i = 0; i < words.length; i++) {
    const w1 = words[i] ?? ''
    const rest: (string | undefined)[] = words.slice(i + 1, i + ws)
    while (rest.length < ws - 1) rest.push(undefined)
    if (order === 2) {
      inc(wfd, w1)
      for (const w2 of rest) if (w2 !== undefined) inc(ngramFd, w1 + SEP + w2)
      continue
    }
    for (let a = 0; a < rest.length; a++) {
      for (let b = a + 1; b < rest.length; b++) {
        inc(wfd, w1)
        const w2 = rest[a]
        if (w2 === undefined) continue
        inc(pairs, w1 + SEP + w2)
        const w3 = rest[b]
        if (w3 === undefined) continue
        inc(wildcards, w1 + SEP + w3)
        inc(ngramFd, w1 + SEP + w2 + SEP + w3)
      }
    }
  }
  let total = 0
  for (const c of wfd.values()) total += c
  return { order, windowSize: ws, words: wfd, ngrams: ngramFd, pairs, wildcards, total }
}

/** Options of {@link scoreCollocations}. */
export interface CollocationOptions {
  /** Drop n-grams seen fewer times (nltk's apply_freq_filter). */
  readonly minFreq?: number
  /** Drop n-grams containing a word for which this holds (apply_word_filter). */
  readonly wordFilter?: (word: string) => boolean
  /** The power of MI-like. */
  readonly power?: number
}

/**
 * Score every n-gram with an association measure and sort by score (descending), then by the n-gram (nltk's
 * `score_ngrams`); bigram counts in a wider window are divided by window size - 1, as nltk does.
 *
 * @param counts - The counts ({@link collocationCounts}).
 * @param measure - The association measure.
 * @param options - Frequency and word filters.
 * @returns The n-grams with their scores.
 * @concept collocations
 * @example
 * ```ts
 * import { collocationCounts, scoreCollocations } from 'omai-engine/text'
 * const words = 'new york is big and new york is old but york is far'.split(' ')
 * scoreCollocations(collocationCounts(words), 'likelihoodRatio', { minFreq: 2 })[0]?.ngram // => ['york', 'is']
 * ```
 */
export function scoreCollocations(
  counts: CollocationCounts,
  measure: AssociationMeasure,
  options: CollocationOptions = {},
): { readonly ngram: string[]; readonly score: number }[] {
  const out: { ngram: string[]; score: number }[] = []
  const power = options.power === undefined ? {} : { power: options.power }
  for (const [key, freq] of counts.ngrams) {
    if (options.minFreq !== undefined && freq < options.minFreq) continue
    const ngram = key.split(SEP)
    if (options.wordFilter !== undefined && ngram.some(options.wordFilter)) continue
    const [w1 = '', w2 = '', w3 = ''] = ngram
    let score: number
    if (counts.order === 2) {
      const nii = freq / (counts.windowSize - 1)
      if (nii === 0) continue
      score = bigramAssociation(
        measure,
        { nii, nix: counts.words.get(w1) ?? 0, nxi: counts.words.get(w2) ?? 0, nxx: counts.total },
        power,
      )
    } else {
      score = trigramAssociation(
        measure,
        {
          niii: freq,
          pairs: [
            counts.pairs.get(w1 + SEP + w2) ?? 0,
            counts.wildcards.get(w1 + SEP + w3) ?? 0,
            counts.pairs.get(w2 + SEP + w3) ?? 0,
          ],
          words: [counts.words.get(w1) ?? 0, counts.words.get(w2) ?? 0, counts.words.get(w3) ?? 0],
          nxxx: counts.total,
        },
        power,
      )
    }
    out.push({ ngram, score })
  }
  const byNgram = (a: readonly string[], b: readonly string[]): number => {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const c = compareCodePoints(a[i] ?? '', b[i] ?? '')
      if (c !== 0) return c
    }
    return a.length - b.length
  }
  return out.sort((a, b) => b.score - a.score || byNgram(a.ngram, b.ngram))
}
