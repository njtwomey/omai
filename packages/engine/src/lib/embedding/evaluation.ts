/**
 * Evaluating word embeddings: the analogy test (a : b :: c : d by 3CosAdd or 3CosMul over the whole vocabulary or a
 * candidate set, Mikolov et al. 2013), the correlation of cosine similarity with judged similarity (Spearman, as in
 * WordSim-353), and the neighbour purity of planted clusters. Cosine neighbours and single analogies are
 * `text/representations.ts`'s `nearestWords` and `analogy`, which these loop over.
 */

import { MdmlError } from '../base/index.ts'
import { ddot, dnrm2 } from '../numerics/linalg/index.ts'
import { spearman } from '../stats/index.ts'
import { analogy, nearestWords, type WordMatrix } from '../text/index.ts'

/** The outcome of an analogy test. */
export interface AnalogyResult {
  /** The share of answered questions whose top answer is the expected word. */
  readonly accuracy: number
  readonly correct: number
  /** The questions answered (those with all four words in the vocabulary). */
  readonly total: number
  /** The questions skipped for a word outside the vocabulary. */
  readonly skipped: number
}

/**
 * The analogy test: for each quadruple `(a, b, c, d)`, whether the word nearest to `b - a + c` (3CosAdd) or by
 * 3CosMul, with a, b and c excluded, is `d`. A candidate set restricts the answers (the target words of a planted
 * corpus, say), which raises the chance level from 1 / V to 1 / |candidates|.
 *
 * @param vectors - The word vectors.
 * @param quadruples - `[a, b, c, d]` word tuples.
 * @param options - The method (default `3cosadd`) and an optional candidate set for the answer.
 * @returns The accuracy and the counts.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { analogyAccuracy } from 'mdml-engine/embedding'
 * const v = { words: ['man', 'woman', 'king', 'queen', 'apple'], rows: [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0.2, 0]].map((r) => Float64Array.from(r)) }
 * analogyAccuracy(v, [['man', 'woman', 'king', 'queen']]).accuracy // => 1
 * ```
 */
export function analogyAccuracy(
  vectors: WordMatrix,
  quadruples: readonly (readonly [string, string, string, string])[],
  options: { readonly method?: '3cosadd' | '3cosmul'; readonly candidates?: readonly string[] } = {},
): AnalogyResult {
  const index = new Set(vectors.words)
  const allowed = options.candidates === undefined ? undefined : new Set(options.candidates)
  let correct = 0
  let total = 0
  let skipped = 0
  for (const [a, b, c, d] of quadruples) {
    if (![a, b, c, d].every((w) => index.has(w))) {
      skipped++
      continue
    }
    let space = vectors
    if (allowed !== undefined) {
      const keep = vectors.words
        .map((w, i) => (allowed.has(w) || w === a || w === b || w === c ? i : -1))
        .filter((i) => i >= 0)
      space = {
        words: keep.map((i) => vectors.words[i] as string),
        rows: keep.map((i) => vectors.rows[i] as Float64Array),
      }
    }
    const answer = analogy(space, a, b, c, { method: options.method ?? '3cosadd', topn: 1 })[0]?.word
    total++
    if (answer === d) correct++
  }
  return { accuracy: total === 0 ? Number.NaN : correct / total, correct, total, skipped }
}

/**
 * The Spearman correlation between the cosine similarity of word pairs and their judged similarity (the WordSim-353
 * protocol), over the pairs whose words are both in the vocabulary.
 *
 * @param vectors - The word vectors.
 * @param pairs - `[word1, word2, similarity]` triples.
 * @returns The correlation and the number of pairs used.
 * @throws MdmlError with code `value` for fewer than three usable pairs.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { similarityCorrelation } from 'mdml-engine/embedding'
 * const v = { words: ['a', 'b', 'c', 'd'], rows: [[1, 0], [1, 0.1], [0, 1], [0.1, 1]].map((r) => Float64Array.from(r)) }
 * similarityCorrelation(v, [['a', 'b', 9], ['c', 'd', 8], ['a', 'c', 1], ['b', 'd', 2]]).rho > 0.9 // => true
 * ```
 */
export function similarityCorrelation(
  vectors: WordMatrix,
  pairs: readonly (readonly [string, string, number])[],
): { readonly rho: number; readonly pairs: number } {
  const at = new Map(vectors.words.map((w, i) => [w, i]))
  const cos: number[] = []
  const judged: number[] = []
  for (const [a, b, s] of pairs) {
    const i = at.get(a)
    const j = at.get(b)
    if (i === undefined || j === undefined) continue
    const u = vectors.rows[i] as Float64Array
    const v = vectors.rows[j] as Float64Array
    cos.push(ddot(u, v) / (dnrm2(u) * dnrm2(v) || 1))
    judged.push(s)
  }
  if (cos.length < 3) throw new MdmlError('value', 'similarityCorrelation: needs at least three usable pairs')
  return { rho: spearman(cos, judged).statistic, pairs: cos.length }
}

/**
 * The share of each word's k cosine neighbours that fall in its own cluster, averaged over the words that have a
 * cluster: 1 when the clusters are perfectly separated in the embedding, about the cluster's share of the vocabulary
 * when the vectors carry no cluster information.
 *
 * @param vectors - The word vectors.
 * @param clusterOf - The cluster of each word that has one.
 * @param k - Neighbours per word (default 5).
 * @returns The mean purity.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { neighbourPurity } from 'mdml-engine/embedding'
 * const v = { words: ['a', 'b', 'c', 'd'], rows: [[1, 0], [1, 0.1], [0, 1], [0.1, 1]].map((r) => Float64Array.from(r)) }
 * neighbourPurity(v, { a: 0, b: 0, c: 1, d: 1 }, 1) // => 1
 * ```
 */
export function neighbourPurity(vectors: WordMatrix, clusterOf: Readonly<Record<string, number>>, k = 5): number {
  const members = vectors.words.filter((w) => w in clusterOf)
  const sub: WordMatrix = {
    words: members,
    rows: members.map((w) => vectors.rows[vectors.words.indexOf(w)] as Float64Array),
  }
  let hit = 0
  let total = 0
  for (const w of members)
    for (const n of nearestWords(sub, w, k)) {
      total++
      if (clusterOf[n.word] === clusterOf[w]) hit++
    }
  return total === 0 ? Number.NaN : hit / total
}
