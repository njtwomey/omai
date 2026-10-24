import { WhiteboxError } from '../base/index.ts'
import { ddot, dnrm2, svd } from '../numerics/linalg/index.ts'
import { type Key, randomBits } from '../random/index.ts'
import { asTensor, type TensorLike, toArray } from '../tensor/index.ts'
import { compareCodePoints } from './_classes.ts'

/** A word-by-something matrix: one row per word. */
export interface WordMatrix {
  readonly words: readonly string[]
  readonly rows: readonly Float64Array[]
}

/** Options of {@link cooccurrence}. */
export interface CooccurrenceOptions {
  /** Context words on each side (default 2). */
  readonly window?: number
  /** `uniform` counts (default) or `harmonic` (1 / distance, GloVe's weighting). */
  readonly weighting?: 'uniform' | 'harmonic'
  /** Ignore words seen fewer times (default 1). */
  readonly minCount?: number
}

/**
 * The symmetric word-context co-occurrence matrix of tokenised sentences: for every word, the (weighted) count of
 * each word within `window` positions on either side, over a vocabulary in code point order.
 *
 * @param sentences - Token lists.
 * @param options - Window, distance weighting and minimum count.
 * @returns The vocabulary and the co-occurrence rows.
 * @concept distributional-semantics
 * @example
 * ```ts
 * import { cooccurrence } from 'whitebox-engine/text'
 * const m = cooccurrence([['a', 'b', 'c']], { window: 1 })
 * [m.words, m.rows.map((r) => Array.from(r))] // => [['a', 'b', 'c'], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]]
 * ```
 */
export function cooccurrence(sentences: readonly (readonly string[])[], options: CooccurrenceOptions = {}): WordMatrix {
  const window = options.window ?? 2
  if (!Number.isInteger(window) || window < 1)
    throw new WhiteboxError('value', 'cooccurrence: window must be a positive integer')
  const counts = new Map<string, number>()
  for (const s of sentences) for (const w of s) counts.set(w, (counts.get(w) ?? 0) + 1)
  const words = [...counts]
    .filter(([, c]) => c >= (options.minCount ?? 1))
    .map(([w]) => w)
    .sort(compareCodePoints)
  const index = new Map(words.map((w, i) => [w, i]))
  const rows = words.map(() => new Float64Array(words.length))
  for (const s of sentences) {
    for (let i = 0; i < s.length; i++) {
      const a = index.get(s[i] ?? '')
      if (a === undefined) continue
      for (let d = 1; d <= window; d++) {
        for (const j of [i - d, i + d]) {
          const b = j >= 0 && j < s.length ? index.get(s[j] ?? '') : undefined
          if (b === undefined) continue
          const row = rows[a] as Float64Array
          row[b] = (row[b] ?? 0) + (options.weighting === 'harmonic' ? 1 / d : 1)
        }
      }
    }
  }
  return { words, rows }
}

/**
 * Pointwise mutual information of a co-occurrence matrix, log(p(w, c) / (p(w) p_alpha(c))), with the context
 * distribution raised to `alpha` (0.75 in word2vec and Levy et al. 2015; 1 is plain PMI); `positive` keeps the
 * positive part (PPMI) and `shift` subtracts log k (shifted PPMI). Zero counts give -Infinity, or 0 when positive.
 *
 * @param m - The co-occurrence matrix.
 * @param options - Positive part, shift and context smoothing.
 * @returns The PMI matrix.
 * @concept distributional-semantics
 * @example
 * ```ts
 * import { pmi } from 'whitebox-engine/text'
 * Array.from(pmi({ words: ['a', 'b'], rows: [Float64Array.of(2, 2), Float64Array.of(2, 2)] }).rows[0] ?? []) // => [0, 0]
 * ```
 */
export function pmi(
  m: WordMatrix,
  options: { readonly positive?: boolean; readonly shift?: number; readonly alpha?: number } = {},
): WordMatrix {
  const alpha = options.alpha ?? 1
  const n = m.rows.length
  const width = m.rows[0]?.length ?? 0
  let total = 0
  const rowSums = m.rows.map((r) => {
    let s = 0
    for (const v of r) s += v
    total += s
    return s
  })
  const colSums = new Float64Array(width)
  for (const r of m.rows) for (let j = 0; j < width; j++) colSums[j] = (colSums[j] ?? 0) + (r[j] ?? 0)
  let smoothed = 0
  for (const c of colSums) smoothed += c ** alpha
  const shift = Math.log(options.shift ?? 1)
  const rows = m.rows.map((r, i) =>
    Float64Array.from(r, (v, j) => {
      if (n === 0) return 0
      const pwc = v / total
      const pw = (rowSums[i] ?? 0) / total
      const pc = (colSums[j] ?? 0) ** alpha / smoothed
      const value = Math.log(pwc / (pw * pc)) - shift
      if (options.positive === true) return Math.max(value, 0)
      return value
    }),
  )
  return { words: m.words, rows }
}

/** The rows of a matrix (any TensorLike or rows). */
function rowsOf(x: TensorLike | readonly Float64Array[]): Float64Array[] {
  if (Array.isArray(x) && x.every((r) => r instanceof Float64Array)) return [...(x as readonly Float64Array[])]
  return (toArray(asTensor(x as TensorLike)) as number[][]).map((r) => Float64Array.from(r))
}

/**
 * Latent semantic analysis (Deerwester et al. 1990): the rank-k truncated SVD X ~ U_k S_k V_k^T of a document-term
 * matrix (tf-idf, counts), with documents as rows of U_k S_k and terms as rows of V_k S_k (singular vectors signed so
 * each left vector's largest entry is positive).
 *
 * @param x - The document-term matrix (rows are documents).
 * @param k - The number of dimensions.
 * @returns The singular values, document vectors and term vectors.
 * @throws WhiteboxError with code `value` for k outside [1, min(m, n)].
 * @concept latent-semantic-analysis
 * @example
 * ```ts
 * import { lsa } from 'whitebox-engine/text'
 * lsa([[1, 1, 0], [1, 1, 0], [0, 0, 1]], 1).singularValues // => [2]
 * ```
 */
export function lsa(
  x: TensorLike | readonly Float64Array[],
  k: number,
): { readonly singularValues: number[]; readonly documents: Float64Array[]; readonly terms: Float64Array[] } {
  const rows = rowsOf(x)
  const m = rows.length
  const n = rows[0]?.length ?? 0
  if (!Number.isInteger(k) || k < 1 || k > Math.min(m, n))
    throw new WhiteboxError('value', 'lsa: k must be in [1, min(m, n)]')
  const { u, s, vh } = svd(rows.map((r) => Array.from(r)))
  const U = toArray(u) as number[][]
  const S = (toArray(s) as number[]).slice(0, k)
  const V = toArray(vh) as number[][]
  return {
    singularValues: S,
    documents: U.map((r) => Float64Array.from(S, (sv, j) => (r[j] ?? 0) * sv)),
    terms: Array.from({ length: n }, (_, t) => Float64Array.from(S, (sv, j) => (V[j]?.[t] ?? 0) * sv)),
  }
}

/**
 * Dense word vectors from an SVD of a (P)PMI matrix (Levy, Goldberg and Dagan 2015): rows of U_k S_k^p, with the
 * eigenvalue weighting p (0.5 by default; 1 is the plain truncated SVD, 0 drops the singular values).
 *
 * @param m - A word-context matrix (typically PPMI).
 * @param k - The dimension.
 * @param options - The eigenvalue weighting.
 * @returns The word vectors.
 * @throws WhiteboxError with code `value` for k outside [1, min(rows, cols)].
 * @concept distributional-semantics
 * @example
 * ```ts
 * import { svdWordVectors } from 'whitebox-engine/text'
 * svdWordVectors({ words: ['a', 'b'], rows: [Float64Array.of(4, 0), Float64Array.of(0, 1)] }, 1).rows.map((r) => Array.from(r)) // => [[2], [0]]
 * ```
 */
export function svdWordVectors(m: WordMatrix, k: number, options: { readonly power?: number } = {}): WordMatrix {
  const n = Math.min(m.rows.length, m.rows[0]?.length ?? 0)
  if (!Number.isInteger(k) || k < 1 || k > n)
    throw new WhiteboxError('value', 'svdWordVectors: k must be in [1, min(rows, cols)]')
  const { u, s } = svd(m.rows.map((r) => Array.from(r)))
  const U = toArray(u) as number[][]
  const S = (toArray(s) as number[]).slice(0, k).map((v) => v ** (options.power ?? 0.5))
  return { words: m.words, rows: U.map((r) => Float64Array.from(S, (sv, j) => (r[j] ?? 0) * sv)) }
}

/**
 * Random indexing (Kanerva, Kristoferson and Holst 2000): each word gets a sparse ternary index vector (`nonzeros`
 * entries of +1 and -1 at random positions, from the key), and its context vector is the sum of the index vectors of
 * the words around it.
 *
 * @param sentences - Token lists.
 * @param key - The key for the index vectors.
 * @param options - Dimension (default 300), nonzeros (default 8) and window (default 2).
 * @returns The context vectors (vocabulary in code point order).
 * @throws WhiteboxError with code `value` for nonzeros above the dimension.
 * @concept random-indexing
 * @example
 * ```ts
 * import { randomIndexing } from 'whitebox-engine/text'
 * import { key } from 'whitebox-engine/random'
 * randomIndexing([['a', 'b']], key(0), { dimension: 16, nonzeros: 2, window: 1 }).rows[0]?.reduce((s, v) => s + Math.abs(v), 0) // => 2
 * ```
 */
export function randomIndexing(
  sentences: readonly (readonly string[])[],
  key: Key,
  options: { readonly dimension?: number; readonly nonzeros?: number; readonly window?: number } = {},
): WordMatrix {
  const dim = options.dimension ?? 300
  const nz = options.nonzeros ?? 8
  if (!(nz >= 1 && nz <= dim)) throw new WhiteboxError('value', 'randomIndexing: nonzeros must be in [1, dimension]')
  const counts = cooccurrence(sentences, { window: options.window ?? 2 })
  const words = counts.words
  // Index vectors: a partial Fisher-Yates shuffle of the positions, signs from further draws.
  const bits = randomBits(key, words.length * nz * 2)
  const index = words.map((_, w) => {
    const v = new Float64Array(dim)
    const positions = Array.from({ length: dim }, (_, i) => i)
    for (let k = 0; k < nz; k++) {
      const r = (bits[(w * nz + k) * 2] ?? 0) % (dim - k)
      const pick = positions[k + r] ?? 0
      positions[k + r] = positions[k] ?? 0
      positions[k] = pick
      v[pick] = (bits[(w * nz + k) * 2 + 1] ?? 0) % 2 === 0 ? 1 : -1
    }
    return v
  })
  const rows = counts.rows.map((r) => {
    const out = new Float64Array(dim)
    for (let j = 0; j < r.length; j++) {
      const c = r[j] ?? 0
      if (c === 0) continue
      const iv = index[j] as Float64Array
      for (let d = 0; d < dim; d++) out[d] = (out[d] ?? 0) + c * (iv[d] ?? 0)
    }
    return out
  })
  return { words, rows }
}

function unit(v: ArrayLike<number>): Float64Array {
  const n = dnrm2(v)
  return Float64Array.from(v as ArrayLike<number>, (x) => (n > 0 ? x / n : 0))
}

/**
 * The words nearest a word or vector by cosine similarity (gensim's `most_similar` with one positive term), the
 * query word excluded.
 *
 * @param vectors - The word vectors.
 * @param query - A word of the vocabulary or a vector.
 * @param topn - How many (default 10).
 * @returns The words and similarities, most similar first.
 * @throws WhiteboxError with code `value` for an unknown word.
 * @concept distributional-semantics
 * @example
 * ```ts
 * import { nearestWords } from 'whitebox-engine/text'
 * const v = { words: ['cat', 'dog', 'car'], rows: [Float64Array.of(1, 0.1), Float64Array.of(1, 0.2), Float64Array.of(0, 1)] }
 * nearestWords(v, 'cat', 1)[0]?.word // => 'dog'
 * ```
 */
export function nearestWords(
  vectors: WordMatrix,
  query: string | ArrayLike<number>,
  topn = 10,
): { readonly word: string; readonly similarity: number }[] {
  let q: Float64Array
  const exclude = new Set<number>()
  if (typeof query === 'string') {
    const i = vectors.words.indexOf(query)
    if (i < 0) throw new WhiteboxError('value', `nearestWords: ${query} is not in the vocabulary`)
    q = unit(vectors.rows[i] ?? [])
    exclude.add(i)
  } else q = unit(query)
  return rank(vectors, q, exclude, topn)
}

function rank(
  vectors: WordMatrix,
  q: Float64Array,
  exclude: ReadonlySet<number>,
  topn: number,
  score: (row: Float64Array) => number = (r) => ddot(unit(r), q),
): { word: string; similarity: number }[] {
  const scored: { word: string; similarity: number; i: number }[] = []
  for (const [i, r] of vectors.rows.entries())
    if (!exclude.has(i)) scored.push({ word: vectors.words[i] ?? '', similarity: score(r), i })
  scored.sort((a, b) => b.similarity - a.similarity || a.i - b.i)
  return scored.slice(0, topn).map(({ word, similarity }) => ({ word, similarity }))
}

/**
 * Word analogies a : b :: c : ? (Mikolov et al. 2013): 3CosAdd, the word nearest unit(b) - unit(a) + unit(c)
 * (gensim's `most_similar(positive=[b, c], negative=[a])`), or 3CosMul (Levy and Goldberg 2014; gensim's
 * `most_similar_cosmul`), cos(d, b) cos(d, c) / (cos(d, a) + 1e-6) with cosines shifted to [0, 1]. The three query words
 * are excluded.
 *
 * @param vectors - The word vectors.
 * @param a - The first word.
 * @param b - Its counterpart.
 * @param c - The word whose counterpart is sought.
 * @param options - The method (default `3cosadd`) and how many answers (default 1).
 * @returns The answers with their scores.
 * @throws WhiteboxError with code `value` for an unknown word.
 * @concept distributional-semantics
 * @example
 * ```ts
 * import { analogy } from 'whitebox-engine/text'
 * const v = { words: ['man', 'woman', 'king', 'queen', 'apple'], rows: [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0.2, 0]].map((r) => Float64Array.from(r)) }
 * analogy(v, 'man', 'woman', 'king')[0]?.word // => 'queen'
 * ```
 */
export function analogy(
  vectors: WordMatrix,
  a: string,
  b: string,
  c: string,
  options: { readonly method?: '3cosadd' | '3cosmul'; readonly topn?: number } = {},
): { readonly word: string; readonly similarity: number }[] {
  const idx = [a, b, c].map((w) => {
    const i = vectors.words.indexOf(w)
    if (i < 0) throw new WhiteboxError('value', `analogy: ${w} is not in the vocabulary`)
    return i
  })
  const [ua, ub, uc] = idx.map((i) => unit(vectors.rows[i] ?? [])) as [Float64Array, Float64Array, Float64Array]
  const exclude = new Set(idx)
  if (options.method === '3cosmul') {
    const shift = (x: number): number => (1 + x) / 2
    return rank(vectors, ua, exclude, options.topn ?? 1, (r) => {
      const d = unit(r)
      return (shift(ddot(d, ub)) * shift(ddot(d, uc))) / (shift(ddot(d, ua)) + 1e-6)
    })
  }
  const target = unit(Float64Array.from(ua, (x, i) => -x + (ub[i] ?? 0) + (uc[i] ?? 0)))
  return rank(vectors, target, exclude, options.topn ?? 1)
}
