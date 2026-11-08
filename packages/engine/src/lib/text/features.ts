/**
 * Document vectorisers that turn text into feature matrices, as in scikit-learn: the shared analyser (preprocessing,
 * word or character n-grams, stop words), bag-of-words counts, tf-idf weighting with smoothed inverse document
 * frequency, and the hashing trick. Also BM25 ranking of documents for a query, as rank_bm25 computes it.
 */

import { OmaiError } from '../base/index.ts'
import { type CsrMatrix, csr } from '../numerics/sparse/index.ts'
import { toFlat } from '../tensor/index.ts'
import { compareCodePoints, PY_SPACE } from './_classes.ts'
import { murmurhash3 } from './hashing.ts'
import { type AccentStripping, stripAccents } from './normalise.ts'
import { stopWords } from './stopwords.ts'
import { SKLEARN_TOKEN_PATTERN, wordTokenize } from './tokenise.ts'

/** Options of the analyzer shared by the vectorisers (scikit-learn's `_VectorizerMixin`). */
export interface AnalyzerOptions {
  /** `word` (default), `char`, `char_wb` (character n-grams inside word boundaries) or a function. */
  readonly analyzer?: 'word' | 'char' | 'char_wb' | ((doc: string) => readonly string[])
  /** Lowercase before tokenising (default true). */
  readonly lowercase?: boolean
  /** Accent stripping (`unicode` or `ascii`; default none). */
  readonly stripAccents?: Exclude<AccentStripping, 'nonspacing'>
  /** The token pattern of the word analyzer (default two or more word characters). */
  readonly tokenPattern?: RegExp
  /** A tokeniser replacing the token pattern. */
  readonly tokenizer?: (doc: string) => readonly string[]
  /** Stop words removed by the word analyzer: `english` (scikit-learn's list) or a list. */
  readonly stopWords?: 'english' | readonly string[]
  /** The smallest and largest n-gram (default [1, 1]). */
  readonly ngramRange?: readonly [number, number]
}

const WHITE_RUNS = new RegExp(`[${PY_SPACE}][${PY_SPACE}]+`, 'gu')
const PY_SPLIT = new RegExp(`[${PY_SPACE}]+`, 'u')

/**
 * The analyzer of scikit-learn's vectorisers: preprocessing (lowercasing, accent stripping), then word tokens with stop
 * words removed and word n-grams, or character n-grams (whitespace runs collapsed), or character n-grams within
 * space-padded words (`char_wb`).
 *
 * @param doc - The document.
 * @param options - The analyzer options.
 * @returns The features of the document, in scikit-learn's order.
 * @throws OmaiError with code `value` for an invalid n-gram range.
 * @concept bag-of-words
 * @example
 * ```ts
 * import { analyze } from 'omai-engine/text'
 * analyze('The cat sat', { ngramRange: [1, 2], stopWords: 'english' }) // => ['cat', 'sat', 'cat sat']
 * analyze('ab cd', { analyzer: 'char_wb', ngramRange: [2, 3] }) // => [' a', 'ab', 'b ', ' ab', 'ab ', ' c', 'cd', 'd ', ' cd', 'cd ']
 * ```
 */
export function analyze(doc: string, options: AnalyzerOptions = {}): string[] {
  if (typeof options.analyzer === 'function') return [...options.analyzer(doc)]
  const [minN, maxN] = options.ngramRange ?? [1, 1]
  if (!(Number.isInteger(minN) && Number.isInteger(maxN) && minN >= 1 && maxN >= minN))
    throw new OmaiError('value', 'analyze: ngramRange must be integers 1 <= min <= max')
  let text = options.lowercase === false ? doc : doc.toLowerCase()
  if (options.stripAccents !== undefined) text = stripAccents(text, options.stripAccents)
  const analyzer = options.analyzer ?? 'word'
  if (analyzer === 'word') {
    const stop =
      options.stopWords === undefined
        ? undefined
        : new Set(options.stopWords === 'english' ? stopWords('sklearn') : options.stopWords)
    let tokens =
      options.tokenizer === undefined
        ? wordTokenize(text, options.tokenPattern ?? SKLEARN_TOKEN_PATTERN)
        : [...options.tokenizer(text)]
    if (stop !== undefined) tokens = tokens.filter((t) => !stop.has(t))
    if (maxN === 1) return tokens
    const out = minN === 1 ? [...tokens] : []
    for (let n = Math.max(minN, 2); n <= Math.min(maxN, tokens.length); n++)
      for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(' '))
    return out
  }
  text = text.replace(WHITE_RUNS, ' ')
  if (analyzer === 'char') {
    const chars = Array.from(text)
    const out: string[] = []
    for (let n = minN; n <= Math.min(maxN, chars.length); n++)
      for (let i = 0; i + n <= chars.length; i++) out.push(chars.slice(i, i + n).join(''))
    return out
  }
  if (analyzer !== 'char_wb') throw new OmaiError('value', `analyze: unknown analyzer ${String(analyzer)}`)
  const out: string[] = []
  for (const word of text.split(PY_SPLIT).filter((w) => w.length > 0)) {
    const w = Array.from(` ${word} `)
    for (let n = minN; n <= maxN; n++) {
      let offset = 0
      out.push(w.slice(offset, offset + n).join(''))
      while (offset + n < w.length) {
        offset++
        out.push(w.slice(offset, offset + n).join(''))
      }
      if (offset === 0) break
    }
  }
  return out
}

/** Options of {@link countVectorize}. */
export interface CountVectorizerOptions extends AnalyzerOptions {
  /** Ignore features in more documents than this: a proportion if at most 1 (default 1), otherwise a count. */
  readonly maxDf?: number
  /** Ignore features in fewer documents than this: a count if at least 1 (default 1), otherwise a proportion. */
  readonly minDf?: number
  /** Keep only the most frequent features (by total count). */
  readonly maxFeatures?: number
  /** Indicator values instead of counts. */
  readonly binary?: boolean
  /** A fixed vocabulary (feature to column). */
  readonly vocabulary?: ReadonlyMap<string, number> | readonly string[]
}

/** A fitted count vectoriser: its options and vocabulary. */
export interface CountVectorizer {
  readonly options: CountVectorizerOptions
  readonly vocabulary: ReadonlyMap<string, number>
  /** Feature names in column order. */
  readonly featureNames: readonly string[]
  /** Features dropped by the document-frequency and size limits. */
  readonly stopWords: ReadonlySet<string>
}

/** Rows of (column, value) pairs to a CSR matrix with sorted columns. */
function toCsr(rows: readonly ReadonlyMap<number, number>[], width: number): CsrMatrix {
  const indptr = [0]
  const indices: number[] = []
  const values: number[] = []
  for (const r of rows) {
    for (const [c, v] of [...r].sort((a, b) => a[0] - b[0])) {
      indices.push(c)
      values.push(v)
    }
    indptr.push(indices.length)
  }
  return csr({ indptr, indices, values: Float64Array.from(values), shape: [rows.length, width] })
}

function countRows(
  docs: readonly string[],
  vocabulary: ReadonlyMap<string, number>,
  options: AnalyzerOptions & { readonly binary?: boolean },
): Map<number, number>[] {
  return docs.map((d) => {
    const row = new Map<number, number>()
    for (const f of analyze(d, options)) {
      const j = vocabulary.get(f)
      if (j !== undefined) row.set(j, options.binary === true ? 1 : (row.get(j) ?? 0) + 1)
    }
    return row
  })
}

/**
 * Fit a bag-of-words count vectoriser (scikit-learn's CountVectorizer.fit_transform): analyze each document, build an
 * alphabetical vocabulary, drop features by document frequency and keep the most frequent, and count.
 *
 * @param docs - The documents.
 * @param options - Analyzer, document-frequency and size limits, binary counts, fixed vocabulary.
 * @returns The fitted vectoriser and the document-term count matrix.
 * @throws OmaiError with code `value` when no feature remains.
 * @concept bag-of-words
 * @example
 * ```ts
 * import { countVectorize } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * const { vectorizer, matrix } = countVectorize(['the cat', 'the cat sat on the mat'])
 * [vectorizer.featureNames, toArray(toDense(matrix))] // => [['cat', 'mat', 'on', 'sat', 'the'], [[1, 0, 0, 0, 1], [1, 1, 1, 1, 2]]]
 * ```
 */
export function countVectorize(
  docs: readonly string[],
  options: CountVectorizerOptions = {},
): { readonly vectorizer: CountVectorizer; readonly matrix: CsrMatrix } {
  if (options.vocabulary !== undefined) {
    const vocabulary = Array.isArray(options.vocabulary)
      ? new Map((options.vocabulary as readonly string[]).map((f, i) => [f, i]))
      : new Map(options.vocabulary as ReadonlyMap<string, number>)
    const featureNames = [...vocabulary].sort((a, b) => a[1] - b[1]).map(([f]) => f)
    const vectorizer = { options, vocabulary, featureNames, stopWords: new Set<string>() }
    return { vectorizer, matrix: countTransform(vectorizer, docs) }
  }
  const counts = docs.map((d) => {
    const m = new Map<string, number>()
    for (const f of analyze(d, options)) m.set(f, (m.get(f) ?? 0) + 1)
    return m
  })
  const df = new Map<string, number>()
  const tf = new Map<string, number>()
  for (const m of counts)
    for (const [f, c] of m) {
      df.set(f, (df.get(f) ?? 0) + 1)
      tf.set(f, (tf.get(f) ?? 0) + c)
    }
  let names = [...df.keys()].sort(compareCodePoints)
  const n = docs.length
  const maxDf = options.maxDf ?? 1
  const minDf = options.minDf ?? 1
  const high = maxDf <= 1 ? maxDf * n : maxDf
  const low = minDf >= 1 ? minDf : minDf * n
  const removed = new Set<string>()
  names = names.filter((f) => {
    const d = df.get(f) ?? 0
    const keep = d <= high && d >= low
    if (!keep) removed.add(f)
    return keep
  })
  if (options.maxFeatures !== undefined && names.length > options.maxFeatures) {
    const ranked = names
      .map((f, i) => [f, i] as const)
      .sort((a, b) => (tf.get(b[0]) ?? 0) - (tf.get(a[0]) ?? 0) || a[1] - b[1])
    const keep = new Set(ranked.slice(0, options.maxFeatures).map(([f]) => f))
    for (const f of names) if (!keep.has(f)) removed.add(f)
    names = names.filter((f) => keep.has(f))
  }
  if (names.length === 0 && docs.length > 0)
    throw new OmaiError('value', 'countVectorize: after pruning, no terms remain; lower minDf or raise maxDf')
  const vocabulary = new Map(names.map((f, i) => [f, i]))
  const vectorizer = { options, vocabulary, featureNames: names, stopWords: removed }
  return { vectorizer, matrix: toCsr(countRows(docs, vocabulary, options), names.length) }
}

/**
 * Count the vocabulary's features in new documents (CountVectorizer.transform); unseen features are ignored.
 *
 * @param vectorizer - A fitted count vectoriser.
 * @param docs - The documents.
 * @returns The document-term count matrix.
 * @concept bag-of-words
 * @example
 * ```ts
 * import { countVectorize, countTransform } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(countTransform(countVectorize(['a cat', 'a dog']).vectorizer, ['dog dog bird']))) // => [[0, 2]]
 * ```
 */
export function countTransform(vectorizer: CountVectorizer, docs: readonly string[]): CsrMatrix {
  return toCsr(countRows(docs, vectorizer.vocabulary, vectorizer.options), vectorizer.featureNames.length)
}

/** Options of the tf-idf weighting (scikit-learn's TfidfTransformer). */
export interface TfidfOptions {
  /** Row normalisation: `l2` (default), `l1` or none. */
  readonly norm?: 'l1' | 'l2' | 'none'
  /** Weight by idf (default true). */
  readonly useIdf?: boolean
  /** Add one to document frequencies, as if a document contained every term (default true). */
  readonly smoothIdf?: boolean
  /** Replace tf by 1 + log(tf) (default false). */
  readonly sublinearTf?: boolean
}

/**
 * The idf of each column of a document-term count matrix: log((1 + n) / (1 + df)) + 1 (smoothed, default) or
 * log(n / df) + 1.
 *
 * @param counts - The count matrix.
 * @param options - Smoothing.
 * @returns The idf of each column.
 * @concept tf-idf
 * @example
 * ```ts
 * import { countVectorize, inverseDocumentFrequency } from 'omai-engine/text'
 * inverseDocumentFrequency(countVectorize(['ab cd', 'ab']).matrix, { smoothIdf: false }) // => [1, 1.6931471805599454]
 * ```
 */
export function inverseDocumentFrequency(counts: CsrMatrix, options: { readonly smoothIdf?: boolean } = {}): number[] {
  const [n, m] = counts.shape
  const df = new Array<number>(m).fill(0)
  for (let k = 0; k < counts.indices.length; k++) {
    const j = counts.indices[k] ?? 0
    df[j] = (df[j] ?? 0) + 1
  }
  const smooth = options.smoothIdf !== false ? 1 : 0
  return df.map((d) => Math.log((n + smooth) / (d + smooth)) + 1)
}

/**
 * Tf-idf weighting of a count matrix (TfidfTransformer.fit_transform): tf (or 1 + log tf) times idf, rows normalised.
 *
 * @param counts - The count matrix.
 * @param options - Normalisation, idf use and smoothing, sublinear tf.
 * @param idf - The idf to use (default: computed from `counts`).
 * @returns The weighted matrix.
 * @concept tf-idf
 * @example
 * ```ts
 * import { countVectorize, tfidfWeights } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(tfidfWeights(countVectorize(['ab cd', 'ab']).matrix, { norm: 'none', smoothIdf: false }))) // => [[1, 1.6931471805599454], [1, 0]]
 * ```
 */
export function tfidfWeights(counts: CsrMatrix, options: TfidfOptions = {}, idf?: readonly number[]): CsrMatrix {
  const weights = idf ?? inverseDocumentFrequency(counts, options)
  const values = Array.from(toFlat(counts.values))
  for (let k = 0; k < values.length; k++) {
    let v = values[k] ?? 0
    if (options.sublinearTf === true) v = 1 + Math.log(v)
    if (options.useIdf !== false) v *= weights[counts.indices[k] ?? 0] ?? 1
    values[k] = v
  }
  const norm = options.norm ?? 'l2'
  if (norm !== 'none') {
    for (let i = 0; i < counts.shape[0]; i++) {
      const a = counts.indptr[i] ?? 0
      const b = counts.indptr[i + 1] ?? 0
      let s = 0
      for (let k = a; k < b; k++) s += norm === 'l1' ? Math.abs(values[k] ?? 0) : (values[k] ?? 0) ** 2
      const scale = norm === 'l1' ? s : Math.sqrt(s)
      if (scale > 0) for (let k = a; k < b; k++) values[k] = (values[k] ?? 0) / scale
    }
  }
  return csr({ indptr: counts.indptr, indices: counts.indices, values: Float64Array.from(values), shape: counts.shape })
}

/** A fitted tf-idf vectoriser: a count vectoriser and the idf of its features. */
export interface TfidfVectorizer {
  readonly counts: CountVectorizer
  readonly options: TfidfOptions
  readonly idf: readonly number[]
}

/**
 * Fit a tf-idf vectoriser (scikit-learn's TfidfVectorizer.fit_transform): counts, then tf-idf weights.
 *
 * @param docs - The documents.
 * @param options - Count vectoriser and tf-idf options.
 * @returns The fitted vectoriser and the tf-idf matrix.
 * @concept tf-idf
 * @index function TF-IDF document vectors; count and hashing vectorisers beside it.
 * @example
 * ```ts
 * import { tfidfVectorize } from 'omai-engine/text'
 * tfidfVectorize(['a cat', 'a dog', 'cat dog']).vectorizer.idf // => [1.2876820724517808, 1.2876820724517808]
 * ```
 */
export function tfidfVectorize(
  docs: readonly string[],
  options: CountVectorizerOptions & TfidfOptions = {},
): { readonly vectorizer: TfidfVectorizer; readonly matrix: CsrMatrix } {
  const { vectorizer, matrix } = countVectorize(docs, options)
  const idf = inverseDocumentFrequency(matrix, options)
  return { vectorizer: { counts: vectorizer, options, idf }, matrix: tfidfWeights(matrix, options, idf) }
}

/**
 * Tf-idf weights of new documents with a fitted vectoriser (TfidfVectorizer.transform).
 *
 * @param vectorizer - The fitted vectoriser.
 * @param docs - The documents.
 * @returns The tf-idf matrix.
 * @concept tf-idf
 * @example
 * ```ts
 * import { tfidfVectorize, tfidfTransform } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(tfidfTransform(tfidfVectorize(['a cat', 'a dog']).vectorizer, ['cat']))) // => [[1, 0]]
 * ```
 */
export function tfidfTransform(vectorizer: TfidfVectorizer, docs: readonly string[]): CsrMatrix {
  return tfidfWeights(countTransform(vectorizer.counts, docs), vectorizer.options, vectorizer.idf)
}

/** Options of {@link hashingVectorize}. */
export interface HashingOptions extends AnalyzerOptions {
  /** The number of columns (default 2^20). */
  readonly nFeatures?: number
  /** Flip the sign of features whose hash is negative, so collisions cancel on average (default true). */
  readonly alternateSign?: boolean
  /** Row normalisation (default `l2`). */
  readonly norm?: 'l1' | 'l2' | 'none'
  /** Indicator values (default false). */
  readonly binary?: boolean
}

/**
 * The hashing trick (scikit-learn's HashingVectorizer, Weinberger et al. 2009): each feature goes to column
 * |h| mod nFeatures of its signed 32-bit MurmurHash3 h, with sign -1 when h < 0 (alternateSign), summed and normalised.
 *
 * @param docs - The documents.
 * @param options - Analyzer, width, sign alternation, normalisation and binary values.
 * @returns The document-feature matrix.
 * @throws OmaiError with code `value` for a non-positive width.
 * @concept feature-hashing
 * @example
 * ```ts
 * import { hashingVectorize } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(hashingVectorize(['foo foo bar'], { nFeatures: 4, norm: 'none' }))) // => [[-2, 1, 0, 0]]
 * ```
 */
export function hashingVectorize(docs: readonly string[], options: HashingOptions = {}): CsrMatrix {
  const width = options.nFeatures ?? 2 ** 20
  if (!Number.isInteger(width) || width < 1)
    throw new OmaiError('value', 'hashingVectorize: nFeatures must be a positive integer')
  const rows = docs.map((d) => {
    const row = new Map<number, number>()
    for (const f of analyze(d, options)) {
      const h = murmurhash3(f)
      const j = Math.abs(h) % width
      const v = options.alternateSign === false || h >= 0 ? 1 : -1
      row.set(j, (row.get(j) ?? 0) + v)
    }
    if (options.binary === true) for (const j of row.keys()) row.set(j, 1)
    return row
  })
  return tfidfWeights(toCsr(rows, width), { useIdf: false, norm: options.norm ?? 'l2' })
}

// ---------------------------------------------------------------------------------------------------------------
// BM25 (Robertson et al. 1994) and BM25+ (Lv and Zhai 2011).

/** Options of {@link bm25}. */
export interface Bm25Options {
  /** `okapi` (default) or `plus`, as rank_bm25's BM25Okapi and BM25Plus. */
  readonly variant?: 'okapi' | 'plus'
  /** Term-frequency saturation (default 1.5). */
  readonly k1?: number
  /** Length normalisation (default 0.75). */
  readonly b?: number
  /** Okapi: negative idfs become epsilon times the mean idf (default 0.25). */
  readonly epsilon?: number
  /** BM25+: the lower bound added for each query term (default 1). */
  readonly delta?: number
}

/** A BM25 index of a tokenised corpus. */
export interface Bm25Index {
  readonly options: Required<Bm25Options>
  readonly idf: ReadonlyMap<string, number>
  readonly termFrequencies: readonly ReadonlyMap<string, number>[]
  readonly lengths: readonly number[]
  readonly averageLength: number
}

/**
 * Index a tokenised corpus for BM25 ranking as rank_bm25 does: Okapi idf log((N - n + 0.5) / (n + 0.5)), negative
 * values replaced by epsilon times the mean idf; BM25+ idf log((N + 1) / n).
 *
 * @param corpus - Token lists.
 * @param options - Variant and parameters.
 * @returns The index.
 * @throws OmaiError with code `value` for an empty corpus.
 * @concept bm25
 * @index function BM25 ranking index.
 * @example
 * ```ts
 * import { bm25 } from 'omai-engine/text'
 * bm25([['a', 'b'], ['b'], ['c']]).idf.get('a') // => 0.5108256237659907
 * ```
 */
export function bm25(corpus: readonly (readonly string[])[], options: Bm25Options = {}): Bm25Index {
  if (corpus.length === 0) throw new OmaiError('value', 'bm25: empty corpus')
  const opts: Required<Bm25Options> = {
    variant: options.variant ?? 'okapi',
    k1: options.k1 ?? 1.5,
    b: options.b ?? 0.75,
    epsilon: options.epsilon ?? 0.25,
    delta: options.delta ?? 1,
  }
  const termFrequencies = corpus.map((doc) => {
    const m = new Map<string, number>()
    for (const t of doc) m.set(t, (m.get(t) ?? 0) + 1)
    return m
  })
  const df = new Map<string, number>()
  for (const m of termFrequencies) for (const t of m.keys()) df.set(t, (df.get(t) ?? 0) + 1)
  const n = corpus.length
  const idf = new Map<string, number>()
  if (opts.variant === 'plus') for (const [t, d] of df) idf.set(t, Math.log((n + 1) / d))
  else {
    let sum = 0
    const negative: string[] = []
    for (const [t, d] of df) {
      const v = Math.log(n - d + 0.5) - Math.log(d + 0.5)
      idf.set(t, v)
      sum += v
      if (v < 0) negative.push(t)
    }
    const eps = (opts.epsilon * sum) / df.size
    for (const t of negative) idf.set(t, eps)
  }
  const lengths = corpus.map((d) => d.length)
  return { options: opts, idf, termFrequencies, lengths, averageLength: lengths.reduce((s, l) => s + l, 0) / n }
}

/**
 * The BM25 score of every document of an index for a query (rank_bm25's get_scores): the sum over query tokens of
 * idf times the saturated, length-normalised term frequency (plus delta for BM25+, for every query term in the
 * vocabulary, as rank_bm25 does).
 *
 * @param index - The BM25 index.
 * @param query - The query tokens.
 * @returns The score of each document.
 * @concept bm25
 * @example
 * ```ts
 * import { bm25, bm25Scores } from 'omai-engine/text'
 * bm25Scores(bm25([['a', 'b'], ['b'], ['c']]), ['a']) // => [0.4170005091967271, 0, 0]
 * ```
 */
export function bm25Scores(index: Bm25Index, query: readonly string[]): number[] {
  const { k1, b, delta, variant } = index.options
  return index.termFrequencies.map((tf, i) => {
    const norm = k1 * (1 - b + (b * (index.lengths[i] ?? 0)) / index.averageLength)
    let s = 0
    for (const q of query) {
      const f = tf.get(q) ?? 0
      const w = index.idf.get(q) ?? 0
      s += variant === 'plus' ? w * (delta + (f * (k1 + 1)) / (norm + f)) : (w * (f * (k1 + 1))) / (f + norm)
    }
    return s
  })
}
