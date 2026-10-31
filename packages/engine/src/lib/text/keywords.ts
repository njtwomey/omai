/**
 * Keyword extraction and extractive summarisation: RAKE (Rose et al. 2010), which splits text at stop words and
 * punctuation into candidate phrases and scores a word by its degree in the candidate co-occurrence graph over its
 * frequency, and TextRank (Mihalcea and Tarau 2004), which runs PageRank on a graph of words co-occurring in a window,
 * or of sentences linked by their overlap, and reads the keywords or the summary off the scores.
 */

import { type Graph, graph, pagerank } from '../graph/index.ts'
import { stopWords } from './stopwords.ts'

/** A scored keyword phrase. */
export interface KeywordPhrase {
  readonly phrase: string
  readonly score: number
}

/** Options of {@link rake}. */
export interface RakeOptions {
  /** The words that end a candidate phrase (default the scikit-learn English list). */
  readonly stopWords?: Iterable<string>
  /** Candidate phrases with more words than this are dropped (default no limit). */
  readonly maxWords?: number
  /** Words shorter than this many characters end a phrase too (default 1: none). */
  readonly minChars?: number
}

const FRAGMENT = /[.!?,;:\t\\"()'’–—\n[\]{}]|\s-\s/u
const WORD = /[\p{L}\p{N}_+/-]+/gu

/**
 * RAKE: rapid automatic keyword extraction. The text is cut into fragments at punctuation, each fragment is cut at stop
 * words into candidate phrases, the degree of a word is the total length of the candidate phrases it occurs in (counting
 * every occurrence) and its score is degree over frequency, and a phrase scores the sum of its words.
 *
 * @param text - The text.
 * @param options - The stop words, the longest phrase and the shortest word.
 * @returns The distinct candidate phrases, best first (ties by first occurrence).
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { rake } from 'mdml-engine/text'
 * rake('linear constraints over natural numbers; minimal set of solutions', { stopWords: ['over', 'of'] })[0] // => { phrase: 'linear constraints', score: 4 }
 * ```
 */
export function rake(text: string, options: RakeOptions = {}): KeywordPhrase[] {
  const stop = new Set(options.stopWords ?? stopWords())
  const minChars = options.minChars ?? 1
  const candidates: string[][] = []
  for (const fragment of text.toLowerCase().split(FRAGMENT)) {
    let current: string[] = []
    for (const word of fragment.match(WORD) ?? []) {
      if (stop.has(word) || word.length < minChars) {
        if (current.length > 0) candidates.push(current)
        current = []
      } else current.push(word)
    }
    if (current.length > 0) candidates.push(current)
  }
  const frequency = new Map<string, number>()
  const degree = new Map<string, number>()
  for (const c of candidates)
    for (const w of c) {
      frequency.set(w, (frequency.get(w) ?? 0) + 1)
      degree.set(w, (degree.get(w) ?? 0) + c.length)
    }
  const scored = new Map<string, number>()
  for (const c of candidates) {
    if (c.length > (options.maxWords ?? Number.POSITIVE_INFINITY)) continue
    const phrase = c.join(' ')
    if (!scored.has(phrase))
      scored.set(
        phrase,
        c.reduce((s, w) => s + (degree.get(w) ?? 0) / (frequency.get(w) ?? 1), 0),
      )
  }
  return [...scored].map(([phrase, score]) => ({ phrase, score })).sort((a, b) => b.score - a.score)
}

/** A word graph for TextRank. */
export interface WordGraph {
  readonly graph: Graph
  /** The words, node `i` being `words[i]`. */
  readonly words: readonly string[]
}

/** Options of {@link textRankGraph}. */
export interface TextRankGraphOptions {
  /** Words within this many positions (counted among the kept words) are linked (default 2). */
  readonly window?: number
  /** Which words become nodes (default: not a stop word, at least two characters). */
  readonly keep?: (word: string) => boolean
  /** Weight an edge by how often the pair co-occurs (default) or just 1. */
  readonly weighted?: boolean
}

/**
 * The co-occurrence graph of TextRank: the words that pass the filter, in order, each linked to the words after it
 * within the window (an undirected edge, weighted by the number of co-occurrences).
 *
 * @param tokens - The text's tokens, lowercased.
 * @param options - Window, word filter and weighting.
 * @returns The graph and its words.
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { textRankGraph } from 'mdml-engine/text'
 * const g = textRankGraph(['graph', 'ranks', 'words', 'graph'], { keep: () => true, window: 2 })
 * [g.words, g.graph.edges.length] // => [['graph', 'ranks', 'words'], 3]
 * ```
 */
export function textRankGraph(tokens: readonly string[], options: TextRankGraphOptions = {}): WordGraph {
  const stop = new Set(stopWords())
  const keep = options.keep ?? ((w: string): boolean => w.length > 1 && !stop.has(w))
  const kept = tokens.filter(keep)
  const words = [...new Set(kept)]
  const index = new Map(words.map((w, i) => [w, i]))
  const counts = new Map<string, number>()
  const window = options.window ?? 2
  for (let i = 0; i < kept.length; i++)
    for (let j = i + 1; j < Math.min(kept.length, i + window); j++) {
      const a = index.get(kept[i] as string) as number
      const b = index.get(kept[j] as string) as number
      if (a === b) continue
      const key = a < b ? `${a},${b}` : `${b},${a}`
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
  const edges = [...counts].map(([k, c]) => {
    const [a, b] = k.split(',').map(Number) as [number, number]
    return [a, b, options.weighted === false ? 1 : c] as [number, number, number]
  })
  return { graph: graph(edges, { nodes: words.length }), words }
}

/**
 * TextRank keywords: PageRank on the word co-occurrence graph, the top k words by score, and with `merge` the
 * keywords that are adjacent in the text joined into phrases (Mihalcea and Tarau's post-processing).
 *
 * @param tokens - The text's tokens, lowercased.
 * @param options - The graph options, the damping `alpha` (default 0.85), `topK` (default 10) and `merge`.
 * @returns The ranked words, and the merged phrases when asked.
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { textRank } from 'mdml-engine/text'
 * textRank(['graph', 'model', 'graph', 'node', 'graph', 'edge'], { keep: () => true, topK: 1 }).words[0]?.word // => 'graph'
 * ```
 */
export function textRank(
  tokens: readonly string[],
  options: TextRankGraphOptions & { readonly alpha?: number; readonly topK?: number; readonly merge?: boolean } = {},
): { readonly words: { readonly word: string; readonly score: number }[]; readonly phrases: string[] } {
  const g = textRankGraph(tokens, options)
  const scores = pagerank(g.graph, { alpha: options.alpha ?? 0.85 })
  const ranked = g.words.map((word, i) => ({ word, score: scores[i] ?? 0 })).sort((a, b) => b.score - a.score)
  const top = ranked.slice(0, options.topK ?? 10)
  const phrases: string[] = []
  if (options.merge === true) {
    const chosen = new Set(top.map((t) => t.word))
    let run: string[] = []
    const flush = (): void => {
      if (run.length > 1) phrases.push(run.join(' '))
      run = []
    }
    for (const t of tokens) {
      if (chosen.has(t)) run.push(t)
      else flush()
    }
    flush()
  }
  return { words: top, phrases: [...new Set(phrases)] }
}

/**
 * Mihalcea and Tarau's sentence similarity: the number of words two sentences share over the sum of the logarithms of
 * their lengths (0 when either has fewer than two words).
 *
 * @param a - A sentence's tokens.
 * @param b - Another sentence's tokens.
 * @returns The similarity.
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { overlapSimilarity } from 'mdml-engine/text'
 * Math.round(overlapSimilarity(['a', 'b', 'c'], ['b', 'c', 'd', 'e']) * 1000) // => 805
 * ```
 */
export function overlapSimilarity(a: readonly string[], b: readonly string[]): number {
  if (a.length < 2 || b.length < 2) return 0
  const sb = new Set(b)
  const shared = new Set(a.filter((w) => sb.has(w))).size
  return shared / (Math.log(a.length) + Math.log(b.length))
}

/**
 * The sentence graph of TextRank: a node per sentence, an undirected edge weighted by the similarity of each pair
 * with positive similarity.
 *
 * @param sentences - Each sentence's tokens.
 * @param similarity - The similarity (default {@link overlapSimilarity}).
 * @returns The graph.
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { sentenceGraph } from 'mdml-engine/text'
 * sentenceGraph([['a', 'b'], ['a', 'b', 'c'], ['x', 'y']]).edges.length // => 1
 * ```
 */
export function sentenceGraph(
  sentences: readonly (readonly string[])[],
  similarity: (a: readonly string[], b: readonly string[]) => number = overlapSimilarity,
): Graph {
  const edges: [number, number, number][] = []
  for (let i = 0; i < sentences.length; i++)
    for (let j = i + 1; j < sentences.length; j++) {
      const s = similarity(sentences[i] as string[], sentences[j] as string[])
      if (s > 0) edges.push([i, j, s])
    }
  return graph(edges, { nodes: sentences.length })
}

/**
 * Extractive summary by TextRank: the k sentences with the highest PageRank on the sentence graph, in text order.
 *
 * @param sentences - Each sentence's tokens.
 * @param k - How many sentences (default 3).
 * @param options - The damping `alpha` (default 0.85) and the similarity.
 * @returns The chosen sentence indices in text order, and every sentence's score.
 * @concept keyword-extraction
 * @example
 * ```ts
 * import { textRankSentences } from 'mdml-engine/text'
 * const s = [['cat', 'sat', 'mat'], ['cat', 'sat', 'rug'], ['dog', 'ran', 'far'], ['cat', 'mat', 'rug']]
 * textRankSentences(s, 1).chosen.length // => 1
 * ```
 */
export function textRankSentences(
  sentences: readonly (readonly string[])[],
  k = 3,
  options: {
    readonly alpha?: number
    readonly similarity?: (a: readonly string[], b: readonly string[]) => number
  } = {},
): { readonly chosen: number[]; readonly scores: Float64Array } {
  const scores = pagerank(sentenceGraph(sentences, options.similarity), { alpha: options.alpha ?? 0.85 })
  const chosen = Array.from(scores, (s, i) => [s, i] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, i]) => i)
    .sort((a, b) => a - b)
  return { chosen, scores }
}
