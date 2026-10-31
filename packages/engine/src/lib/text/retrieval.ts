/**
 * Retrieval over a positional inverted index: postings lists with term frequencies and positions that grow one
 * document at a time, the scoring models of lexical retrieval as a mode (TF-IDF cosine, BM25, the binary independence
 * model with or without relevance information, and query likelihood under Dirichlet, Jelinek-Mercer or absolute-discount
 * smoothing), each term's contribution to a score, and phrase, proximity and boolean queries answered from the postings.
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { Algorithm } from '../trace/index.ts'

/** One document's entry in a term's postings list. */
export interface Posting {
  readonly doc: number
  /** The positions of the term in the document, increasing. */
  readonly positions: readonly number[]
}

/** A positional inverted index over tokenised documents. */
export interface InvertedIndex {
  /** The number of documents. */
  readonly documents: number
  /** The length of each document in tokens. */
  readonly lengths: readonly number[]
  /** The postings of each term, in document order. */
  readonly postings: ReadonlyMap<string, readonly Posting[]>
  /** The total number of tokens. */
  readonly tokens: number
  /** Add a document and return its id. */
  add(tokens: readonly string[]): number
}

/**
 * An empty inverted index; `add` appends documents (ids 0, 1, ...), extending each term's postings list in place.
 *
 * @returns The index.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { invertedIndex } from 'mdml-engine/text'
 * const ix = invertedIndex()
 * ix.add(['to', 'be', 'or', 'not', 'to', 'be'])
 * ix.add(['to', 'do'])
 * [ix.documents, ix.postings.get('to')?.map((p) => p.positions)] // => [2, [[0, 4], [0]]]
 * ```
 */
export function invertedIndex(): InvertedIndex {
  const postings = new Map<string, Posting[]>()
  const lengths: number[] = []
  const self = {
    get documents(): number {
      return lengths.length
    },
    lengths,
    postings,
    tokens: 0,
    add(tokens: readonly string[]): number {
      const doc = lengths.length
      const positions = new Map<string, number[]>()
      for (const [i, t] of tokens.entries()) {
        const list = positions.get(t)
        if (list === undefined) positions.set(t, [i])
        else list.push(i)
      }
      for (const [t, p] of positions) {
        const list = postings.get(t)
        if (list === undefined) postings.set(t, [{ doc, positions: p }])
        else list.push({ doc, positions: p })
      }
      lengths.push(tokens.length)
      self.tokens += tokens.length
      return doc
    },
  }
  return self
}

/**
 * Index a tokenised corpus.
 *
 * @param corpus - Token lists.
 * @returns The index.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { indexCorpus } from 'mdml-engine/text'
 * indexCorpus([['a', 'b'], ['b', 'c']]).postings.get('b')?.length // => 2
 * ```
 */
export function indexCorpus(corpus: readonly (readonly string[])[]): InvertedIndex {
  const ix = invertedIndex()
  for (const d of corpus) ix.add(d)
  return ix
}

/** The state of {@link indexingAlgorithm}: the index and the documents added. */
export interface IndexingState {
  readonly index: InvertedIndex
  readonly next: number
}

/**
 * Index building as an {@link Algorithm}: one document per step, so the postings lists, the vocabulary and the
 * document frequencies can be watched as the corpus arrives.
 *
 * @param corpus - Token lists, in arrival order.
 * @returns The algorithm.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { indexingAlgorithm } from 'mdml-engine/text'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(indexingAlgorithm([['a'], ['a', 'b']]), { maxSteps: 5 }).state.index.postings.size // => 2
 * ```
 */
export function indexingAlgorithm(corpus: readonly (readonly string[])[]): Algorithm<IndexingState> {
  return {
    name: 'invertedIndex',
    config: { documents: corpus.length },
    init: () => ({ index: invertedIndex(), next: 0 }),
    step: (s) => {
      s.index.add(corpus[s.next] ?? [])
      return { index: s.index, next: s.next + 1 }
    },
    done: (s) => (s.next >= corpus.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { documents: s.index.documents, vocabulary: s.index.postings.size, tokens: s.index.tokens },
    }),
    serialise: (s) => ({ next: s.next, corpus: corpus.slice(0, s.next).map((d) => [...d]) }) as unknown as Json,
    deserialise: (j) => {
      const o = j as unknown as { next: number; corpus: string[][] }
      return { index: indexCorpus(o.corpus), next: o.next }
    },
  }
}

/**
 * The number of documents containing a term.
 *
 * @param index - The inverted index.
 * @param term - The term.
 * @returns The document frequency.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { documentFrequency, indexCorpus } from 'mdml-engine/text'
 * documentFrequency(indexCorpus([['a', 'b'], ['b']]), 'b') // => 2
 * ```
 */
export function documentFrequency(index: InvertedIndex, term: string): number {
  return index.postings.get(term)?.length ?? 0
}

/** A scoring model of lexical retrieval. */
export type RetrievalModel =
  | { readonly kind: 'tfidf' }
  | { readonly kind: 'bm25'; readonly k1?: number; readonly b?: number; readonly idf?: 'robertson' | 'lucene' }
  | { readonly kind: 'bim'; readonly relevant?: readonly number[] }
  | {
      readonly kind: 'query-likelihood'
      readonly smoothing?: 'dirichlet' | 'jelinek-mercer' | 'absolute'
      readonly mu?: number
      readonly lambda?: number
      readonly delta?: number
    }

/** A document's score for a query with the contribution of each query term. */
export interface ScoredDocument {
  readonly doc: number
  readonly score: number
  readonly contributions: Readonly<Record<string, number>>
}

const countOf = (q: readonly string[]): Map<string, number> => {
  const m = new Map<string, number>()
  for (const t of q) m.set(t, (m.get(t) ?? 0) + 1)
  return m
}

/**
 * Score every document of an index for a query under a retrieval model, with the contribution of each query term
 * (the terms' contributions sum to the score). TF-IDF is the cosine of `(1 + log tf) log(N / df)` weights; BM25 is
 * Robertson and Sparck Jones's idf (`log((N - n + 0.5) / (n + 0.5))`, or Lucene's `log(1 + ...)`) times the saturated,
 * length-normalised term frequency, summed over the query's terms with their query frequencies; the binary independence
 * model sums the RSJ weight of the query terms present in a document (with relevance judgements `relevant`, the
 * relevance-weighted form); query likelihood is the log probability of the query under the document's smoothed unigram
 * model (Zhai and Lafferty 2004), for every query term of the collection.
 *
 * @param index - The inverted index.
 * @param query - The query tokens.
 * @param model - The model (default BM25 with k1 = 1.2, b = 0.75).
 * @returns The scores, one per document in order, each with its term contributions.
 * @throws MdmlError with code `value` for a bad parameter.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { indexCorpus, scoreDocuments } from 'mdml-engine/text'
 * const ix = indexCorpus([['a', 'b'], ['b', 'c'], ['c', 'c']])
 * scoreDocuments(ix, ['a'], { kind: 'bim' }).map((s) => Math.round(s.score * 100) / 100) // => [0.51, 0, 0]
 * ```
 */
export function scoreDocuments(
  index: InvertedIndex,
  query: readonly string[],
  model: RetrievalModel = { kind: 'bm25' },
): ScoredDocument[] {
  const N = index.documents
  const contributions: Record<string, number>[] = Array.from({ length: N }, () => ({}))
  const qtf = countOf(query)
  const avgdl = N === 0 ? 0 : index.tokens / N
  if (model.kind === 'tfidf') {
    const idf = (df: number): number => Math.log(N / df)
    const norms = new Float64Array(N)
    for (const list of index.postings.values()) {
      const w = idf(list.length)
      for (const p of list) norms[p.doc] = (norms[p.doc] ?? 0) + ((1 + Math.log(p.positions.length)) * w) ** 2
    }
    let qnorm = 0
    const qw = new Map<string, number>()
    for (const [t, f] of qtf) {
      const df = documentFrequency(index, t)
      if (df === 0) continue
      const w = (1 + Math.log(f)) * idf(df)
      qw.set(t, w)
      qnorm += w * w
    }
    for (const [t, w] of qw) {
      const list = index.postings.get(t) ?? []
      const d = idf(list.length)
      for (const p of list) {
        const dn = Math.sqrt(norms[p.doc] ?? 0)
        const c = dn === 0 || qnorm === 0 ? 0 : (w * (1 + Math.log(p.positions.length)) * d) / (dn * Math.sqrt(qnorm))
        ;(contributions[p.doc] as Record<string, number>)[t] = c
      }
    }
  } else if (model.kind === 'bm25') {
    const k1 = model.k1 ?? 1.2
    const b = model.b ?? 0.75
    if (!(k1 >= 0 && b >= 0 && b <= 1)) throw new MdmlError('value', 'scoreDocuments: need k1 >= 0 and b in [0, 1]')
    for (const [t, f] of qtf) {
      const list = index.postings.get(t)
      if (list === undefined) continue
      const n = list.length
      const idf = model.idf === 'lucene' ? Math.log(1 + (N - n + 0.5) / (n + 0.5)) : Math.log((N - n + 0.5) / (n + 0.5))
      for (const p of list) {
        const tf = p.positions.length
        const norm = k1 * (1 - b + (b * (index.lengths[p.doc] ?? 0)) / avgdl)
        ;(contributions[p.doc] as Record<string, number>)[t] = f * idf * ((tf * (k1 + 1)) / (tf + norm))
      }
    }
  } else if (model.kind === 'bim') {
    const rel = new Set(model.relevant ?? [])
    const R = rel.size
    for (const t of qtf.keys()) {
      const list = index.postings.get(t)
      if (list === undefined) continue
      const n = list.length
      const r = list.filter((p) => rel.has(p.doc)).length
      const w = Math.log(((r + 0.5) * (N - n - R + r + 0.5)) / ((n - r + 0.5) * (R - r + 0.5)))
      for (const p of list) (contributions[p.doc] as Record<string, number>)[t] = w
    }
  } else {
    const smoothing = model.smoothing ?? 'dirichlet'
    const mu = model.mu ?? 2000
    const lambda = model.lambda ?? 0.1
    const delta = model.delta ?? 0.7
    if (smoothing === 'dirichlet' && !(mu > 0)) throw new MdmlError('value', 'scoreDocuments: mu must be positive')
    if (smoothing === 'jelinek-mercer' && !(lambda > 0 && lambda < 1))
      throw new MdmlError('value', 'scoreDocuments: lambda must be in (0, 1)')
    if (smoothing === 'absolute' && !(delta >= 0 && delta < 1))
      throw new MdmlError('value', 'scoreDocuments: delta must be in [0, 1)')
    const uniqueTerms: number[] = new Array<number>(N).fill(0)
    if (smoothing === 'absolute')
      for (const list of index.postings.values())
        for (const p of list) uniqueTerms[p.doc] = (uniqueTerms[p.doc] ?? 0) + 1
    for (const [t, f] of qtf) {
      const list = index.postings.get(t)
      if (list === undefined) continue
      const pc = list.reduce((a, p) => a + p.positions.length, 0) / index.tokens
      const tfs = new Map(list.map((p) => [p.doc, p.positions.length]))
      for (let d = 0; d < N; d++) {
        const len = index.lengths[d] ?? 0
        const tf = tfs.get(d) ?? 0
        let p: number
        if (smoothing === 'dirichlet') p = (tf + mu * pc) / (len + mu)
        else if (smoothing === 'jelinek-mercer') p = (1 - lambda) * (len === 0 ? 0 : tf / len) + lambda * pc
        else p = len === 0 ? pc : (Math.max(tf - delta, 0) + delta * (uniqueTerms[d] ?? 0) * pc) / len
        ;(contributions[d] as Record<string, number>)[t] = f * Math.log(p)
      }
    }
  }
  return contributions.map((c, doc) => ({ doc, score: Object.values(c).reduce((a, b) => a + b, 0), contributions: c }))
}

/**
 * The top k documents for a query: the highest scores first, ties by document id, documents that match no query term
 * left out (for query likelihood every document matches, so all are ranked).
 *
 * @param index - The inverted index.
 * @param query - The query tokens.
 * @param model - The model.
 * @param k - How many (default all).
 * @returns The ranked documents with their term contributions.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { indexCorpus, search } from 'mdml-engine/text'
 * const ix = indexCorpus([['a', 'b'], ['b', 'c'], ['c', 'c'], ['d'], ['e'], ['f']])
 * search(ix, ['c'], { kind: 'bm25' }, 1).map((r) => r.doc) // => [2]
 * ```
 */
export function search(
  index: InvertedIndex,
  query: readonly string[],
  model: RetrievalModel = { kind: 'bm25' },
  k: number = Number.POSITIVE_INFINITY,
): ScoredDocument[] {
  const scored = scoreDocuments(index, query, model)
  const keep =
    model.kind === 'query-likelihood' ? scored : scored.filter((s) => Object.keys(s.contributions).length > 0)
  return keep.sort((a, b) => b.score - a.score || a.doc - b.doc).slice(0, k)
}

/**
 * The documents containing a phrase, from the positional postings: the terms occur in order with at most `slop` other
 * tokens between consecutive terms (slop 0: adjacent, the exact phrase).
 *
 * @param index - The inverted index.
 * @param phrase - The phrase's tokens.
 * @param slop - Tokens allowed between consecutive terms (default 0).
 * @returns The matching document ids in increasing order.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { indexCorpus, phraseSearch } from 'mdml-engine/text'
 * const ix = indexCorpus([['new', 'york', 'city'], ['york', 'new'], ['new', 'big', 'york']])
 * [phraseSearch(ix, ['new', 'york']), phraseSearch(ix, ['new', 'york'], 1)] // => [[0], [0, 2]]
 * ```
 */
export function phraseSearch(index: InvertedIndex, phrase: readonly string[], slop = 0): number[] {
  if (phrase.length === 0) return []
  const lists = phrase.map((t) => index.postings.get(t))
  if (lists.some((l) => l === undefined)) return []
  const byDoc = (list: readonly Posting[]): Map<number, readonly number[]> =>
    new Map(list.map((p) => [p.doc, p.positions]))
  const maps = lists.map((l) => byDoc(l as readonly Posting[]))
  const out: number[] = []
  for (const [doc, first] of maps[0] as Map<number, readonly number[]>) {
    let starts = [...first]
    for (let i = 1; i < phrase.length && starts.length > 0; i++) {
      const next = maps[i]?.get(doc)
      if (next === undefined) {
        starts = []
        break
      }
      const reach = new Set<number>()
      for (const s of starts) for (const q of next) if (q > s && q - s - 1 <= slop) reach.add(q)
      starts = [...reach]
    }
    if (starts.length > 0) out.push(doc)
  }
  return out.sort((a, b) => a - b)
}

/**
 * Boolean retrieval from the postings: documents containing all of the `all` terms and any of the `any` terms (when
 * given), and none of the `none` terms.
 *
 * @param index - The inverted index.
 * @param query - The terms to require, to allow and to exclude.
 * @returns The matching document ids in increasing order.
 * @concept information-retrieval
 * @example
 * ```ts
 * import { booleanSearch, indexCorpus } from 'mdml-engine/text'
 * const ix = indexCorpus([['a', 'b'], ['a', 'c'], ['b', 'c']])
 * booleanSearch(ix, { all: ['a'], none: ['c'] }) // => [0]
 * ```
 */
export function booleanSearch(
  index: InvertedIndex,
  query: { readonly all?: readonly string[]; readonly any?: readonly string[]; readonly none?: readonly string[] },
): number[] {
  const docs = (t: string): Set<number> => new Set((index.postings.get(t) ?? []).map((p) => p.doc))
  let result = new Set<number>(Array.from({ length: index.documents }, (_, i) => i))
  for (const t of query.all ?? []) {
    const d = docs(t)
    result = new Set([...result].filter((x) => d.has(x)))
  }
  if (query.any !== undefined && query.any.length > 0) {
    const any = new Set(query.any.flatMap((t) => [...docs(t)]))
    result = new Set([...result].filter((x) => any.has(x)))
  }
  for (const t of query.none ?? []) for (const x of docs(t)) result.delete(x)
  return [...result].sort((a, b) => a - b)
}
