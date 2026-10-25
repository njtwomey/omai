/**
 * The Unigram language-model tokeniser (Kudo 2018; SentencePiece and HF tokenizers): each word is segmented into the
 * pieces that maximise the product of their probabilities. Viterbi and n-best segmentation, sampling of segmentations
 * for subword regularisation, and training that prunes a large seed vocabulary by EM, as a traced algorithm.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { type Key, randomBits } from '../../random/index.ts'
import { type Algorithm, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import { utf8Bytes } from '../hashing.ts'
import {
  insertNode,
  kBestPaths,
  type Lattice,
  type LatticeNode,
  lattice,
  logSumExpHf,
  populateMarginal,
  viterbiPath,
} from './_lattice.ts'
import type { SubwordToken } from './bpe.ts'

/** A Unigram language model over subword pieces (Kudo 2018; HF tokenizers' `models.Unigram`). */
export interface UnigramModel {
  /** The pieces with their log probabilities, in id order. */
  readonly pieces: readonly (readonly [string, number])[]
  readonly unkId?: number
  /** Spell unknown pieces as `<0xXX>` byte tokens when the vocabulary has them. */
  readonly byteFallback?: boolean
  /** Merge consecutive unknown characters into one token (default true, as HF). */
  readonly fuseUnk?: boolean
}

/** The derived tables of a model: piece ids, a trie over code points and the smallest score. */
interface Tables {
  readonly ids: Map<string, number>
  readonly trie: TrieNode
  readonly minScore: number
}

interface TrieNode {
  readonly children: Map<string, TrieNode>
  id: number
}

const tableCache = new WeakMap<UnigramModel, Tables>()

function tablesOf(model: UnigramModel): Tables {
  let t = tableCache.get(model)
  if (t === undefined) {
    const ids = new Map<string, number>()
    const trie: TrieNode = { children: new Map(), id: -1 }
    let minScore = Number.POSITIVE_INFINITY
    for (const [i, [piece, score]] of model.pieces.entries()) {
      ids.set(piece, i)
      let cur = trie
      for (const ch of piece) {
        let next = cur.children.get(ch)
        if (next === undefined) {
          next = { children: new Map(), id: -1 }
          cur.children.set(ch, next)
        }
        cur = next
      }
      cur.id = i
      if (score < minScore) minScore = score
    }
    t = { ids, trie, minScore }
    tableCache.set(model, t)
  }
  return t
}

/** HF's penalty of an unknown character below the smallest piece score. */
const UNK_PENALTY = 10

/**
 * A Unigram model from pieces and scores (as stored in an HF `tokenizer.json`).
 *
 * @param pieces - The pieces with their scores, in id order.
 * @param options - The unknown piece's id, byte fallback and unknown fusing.
 * @returns The model.
 * @throws MdmlError with code `value` for an unknown id outside the pieces or an empty vocabulary with one.
 * @concept unigram-lm
 * @example
 * ```ts
 * import { unigramModel, unigramTokenize } from 'mdml-engine/text/subword'
 * const m = unigramModel([['<unk>', 0], ['a', -1], ['b', -1], ['ab', -1.5]], { unkId: 0 })
 * unigramTokenize(m, 'abba').map((t) => t.token) // => ['ab', 'b', 'a']
 * ```
 */
export function unigramModel(
  pieces: readonly (readonly [string, number])[],
  options: Omit<UnigramModel, 'pieces'> = {},
): UnigramModel {
  if (options.unkId !== undefined && !(options.unkId >= 0 && options.unkId < pieces.length))
    throw new MdmlError('value', 'unigramModel: unkId is outside the vocabulary')
  const model: UnigramModel = { pieces: pieces.map(([p, s]) => [p, s] as const), ...options }
  tablesOf(model)
  return model
}

/** Fill a lattice with every piece starting at every position, and an unknown node where no single character fits. */
function populateNodes(model: UnigramModel, l: Lattice): void {
  const { trie, minScore } = tablesOf(model)
  const unkScore = minScore - UNK_PENALTY
  for (let pos = 0; pos < l.chars.length; pos++) {
    let cur: TrieNode | undefined = trie
    let single = false
    for (let k = pos; k < l.chars.length; k++) {
      cur = cur.children.get(l.chars[k] ?? '')
      if (cur === undefined) break
      if (cur.id >= 0) {
        insertNode(l, pos, k - pos + 1, model.pieces[cur.id]?.[1] ?? 0, cur.id)
        if (k === pos) single = true
      }
    }
    if (!single && model.unkId !== undefined) insertNode(l, pos, 1, unkScore, model.unkId)
  }
}

function modelLattice(model: UnigramModel, sentence: string): Lattice {
  const l = lattice(sentence, model.pieces.length + 1, model.pieces.length + 2)
  populateNodes(model, l)
  return l
}

/** Tokens of a path: pieces with ids and code-unit offsets, unknowns fused, byte fallback applied. */
function pathTokens(model: UnigramModel, l: Lattice, path: readonly LatticeNode[]): SubwordToken[] {
  const { ids } = tablesOf(model)
  const offsets: number[] = [0]
  for (const ch of l.chars) offsets.push((offsets[offsets.length - 1] ?? 0) + ch.length)
  const pieces: { text: string; start: number; end: number; unk: boolean }[] = []
  for (const nd of path) {
    const start = offsets[nd.pos] ?? 0
    const end = offsets[nd.pos + nd.length] ?? 0
    const text = l.chars.slice(nd.pos, nd.pos + nd.length).join('')
    const unk = nd.id === model.unkId
    const last = pieces[pieces.length - 1]
    if (unk && model.fuseUnk !== false && last?.unk === true) {
      last.text += text
      last.end = end
    } else pieces.push({ text, start, end, unk })
  }
  const out: SubwordToken[] = []
  for (const p of pieces) {
    const id = ids.get(p.text)
    if (id !== undefined) {
      out.push({ token: p.text, id, start: p.start, end: p.end })
      continue
    }
    if (model.byteFallback === true) {
      const bytes = Array.from(utf8Bytes(p.text), (b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`)
      if (bytes.every((b) => ids.has(b))) {
        for (const b of bytes) out.push({ token: b, id: ids.get(b) ?? 0, start: p.start, end: p.end })
        continue
      }
    }
    if (model.unkId === undefined)
      throw new MdmlError('value', `unigramTokenize: no piece covers ${p.text} and there is no unknown piece`)
    out.push({ token: p.text, id: model.unkId, start: p.start, end: p.end })
  }
  return out
}

/**
 * The most probable segmentation of a pre-tokenised word (Viterbi; HF's Unigram.tokenize): pieces with ids and
 * code-unit offsets, consecutive unknown characters fused, byte fallback when enabled.
 *
 * @param model - The model.
 * @param word - The word.
 * @returns The tokens.
 * @throws MdmlError with code `value` when a character has no piece and there is no unknown piece.
 * @concept unigram-lm
 * @example
 * ```ts
 * import { unigramModel, unigramTokenize } from 'mdml-engine/text/subword'
 * const m = unigramModel([['<unk>', 0], ['h', -2], ['e', -2], ['he', -3]], { unkId: 0 })
 * unigramTokenize(m, 'hexx').map((t) => [t.token, t.id]) // => [['he', 3], ['xx', 0]]
 * ```
 */
export function unigramTokenize(model: UnigramModel, word: string): SubwordToken[] {
  if (word.length === 0) return []
  const l = modelLattice(model, word)
  const path = viterbiPath(l)
  if (path.length === 0) throw new MdmlError('value', `unigramTokenize: no segmentation of ${word}`)
  return pathTokens(model, l, path)
}

/**
 * The n best segmentations of a word with their log probabilities (k-best Viterbi, SentencePiece's NBestEncode).
 *
 * @param model - The model.
 * @param word - The word.
 * @param n - How many segmentations.
 * @returns The segmentations, best first.
 * @concept unigram-lm
 * @example
 * ```ts
 * import { unigramModel, unigramNbest } from 'mdml-engine/text/subword'
 * const m = unigramModel([['a', -1], ['b', -1], ['ab', -1.5]])
 * unigramNbest(m, 'ab', 2).map((s) => [s.tokens.map((t) => t.token), s.score]) // => [[['ab'], -1.5], [['a', 'b'], -2]]
 * ```
 */
export function unigramNbest(
  model: UnigramModel,
  word: string,
  n: number,
): { readonly tokens: SubwordToken[]; readonly score: number }[] {
  if (word.length === 0) return [{ tokens: [], score: 0 }]
  const l = modelLattice(model, word)
  return kBestPaths(l, n).map((path) => ({
    tokens: pathTokens(model, l, path),
    score: path.reduce((s, nd) => s + nd.score, 0),
  }))
}

/**
 * Sample a segmentation from p(x) proportional to exp(alpha * score(x)) (subword regularisation, Kudo 2018; as
 * SentencePiece's SampleEncode): forward filtering of the scaled scores, then backward sampling.
 *
 * @param model - The model.
 * @param word - The word.
 * @param key - The key.
 * @param alpha - The smoothing (inverse temperature; default 1).
 * @returns The sampled tokens.
 * @concept subword-regularisation
 * @example
 * ```ts
 * import { unigramModel, unigramSample } from 'mdml-engine/text/subword'
 * import { key } from 'mdml-engine/random'
 * const m = unigramModel([['a', -1], ['b', -1], ['ab', -1.5]])
 * unigramSample(m, 'ab', key(0), 0).map((t) => t.token).join('') // => 'ab'
 * ```
 */
export function unigramSample(model: UnigramModel, word: string, key: Key, alpha = 1): SubwordToken[] {
  if (word.length === 0) return []
  const l = modelLattice(model, word)
  const n = l.chars.length
  const fwd = new Float64Array(l.nodes.length)
  for (let pos = 0; pos <= n; pos++) {
    const ends = l.end[pos] ?? []
    for (const r of l.begin[pos] ?? []) {
      let acc = Number.NEGATIVE_INFINITY
      for (const [k, left] of ends.entries())
        acc = logSumExpHf(acc, (fwd[left.nodeId] ?? 0) + alpha * left.score, k === 0)
      fwd[r.nodeId] = acc
    }
  }
  const draws = randomBits(key, n + 1)
  const path: LatticeNode[] = []
  let cur = l.begin[n]?.[0]
  let d = 0
  while (cur !== undefined && cur.pos > 0) {
    const ends = l.end[cur.pos] ?? []
    const logits = ends.map((left) => (fwd[left.nodeId] ?? 0) + alpha * left.score)
    let hi = Number.NEGATIVE_INFINITY
    for (const v of logits) if (v > hi) hi = v
    const weights = logits.map((v) => Math.exp(v - hi))
    let total = 0
    for (const w of weights) total += w
    let u = ((draws[d++] ?? 0) / 4294967296) * total
    let pick = ends.length - 1
    for (const [k, w] of weights.entries()) {
      if (u < w) {
        pick = k
        break
      }
      u -= w
    }
    const chosen = ends[pick]
    if (chosen === undefined) break
    path.push(chosen)
    cur = chosen
  }
  return pathTokens(model, l, path.reverse())
}

// ---------------------------------------------------------------------------------------------------------------
// Training (HF tokenizers' UnigramTrainer, after SentencePiece).

/** Options of Unigram training. */
export interface UnigramTrainingOptions {
  /** The final vocabulary size, specials included (default 8000). */
  readonly vocabSize?: number
  /** Special tokens, given the first ids with score 0. */
  readonly specialTokens?: readonly string[]
  /** The unknown token (one of the specials, or added first). */
  readonly unkToken?: string
  /** Characters always kept. */
  readonly initialAlphabet?: readonly string[]
  /** The longest seed piece, in characters (default 16). */
  readonly maxPieceLength?: number
  /** EM iterations between prunings (default 2). */
  readonly nSubIterations?: number
  /** The fraction of pieces kept at each pruning (default 0.75). */
  readonly shrinkingFactor?: number
  /** The most seed pieces (default 1,000,000). */
  readonly seedSize?: number
  /** The most pruning rounds, a guard against inputs that cannot shrink (default 100). */
  readonly maxRounds?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of Unigram training. */
export interface UnigramTrainingState {
  readonly pieces: readonly (readonly [string, number])[]
  /** `em` (EM sub-iterations), `prune`, or `done`. */
  readonly phase: 'em' | 'prune' | 'done'
  readonly subIteration: number
  readonly rounds: number
  readonly objective: number
  readonly model?: UnigramModel
}

/** HF's digamma (asymptotic series after shifting to x >= 7). */
function digamma(x0: number): number {
  let x = x0
  let result = 0
  while (x < 7) {
    result -= 1 / x
    x += 1
  }
  x -= 0.5
  const xx = 1 / x
  const xx2 = xx * xx
  const xx4 = xx2 * xx2
  return result + Math.log(x) + (1 / 24) * xx2 - (7 / 960) * xx4 + (31 / 8064) * xx4 * xx2 - (127 / 30720) * xx4 * xx4
}

/** The suffix array of a code point array (prefix doubling). */
function suffixArray(s: readonly number[]): Int32Array {
  const n = s.length
  const sa = Int32Array.from({ length: n }, (_, i) => i)
  let rank = Int32Array.from(s)
  const tmp = new Int32Array(n)
  for (let k = 1; ; k *= 2) {
    const key = (i: number): readonly [number, number] => [rank[i] ?? 0, i + k < n ? (rank[i + k] ?? 0) : -1]
    const sorted = Array.from(sa).sort((a, b) => {
      const [a0, a1] = key(a)
      const [b0, b1] = key(b)
      return a0 - b0 || a1 - b1
    })
    sa.set(sorted)
    tmp[sa[0] ?? 0] = 0
    for (let i = 1; i < n; i++) {
      const [p0, p1] = key(sa[i - 1] ?? 0)
      const [c0, c1] = key(sa[i] ?? 0)
      tmp[sa[i] ?? 0] = (tmp[sa[i - 1] ?? 0] ?? 0) + (p0 !== c0 || p1 !== c1 ? 1 : 0)
    }
    rank = Int32Array.from(tmp)
    if ((rank[sa[n - 1] ?? 0] ?? 0) === n - 1 || k > n) break
  }
  return sa
}

/** The internal nodes of the suffix tree (esaxx's suffixtree): substring start, length and frequency. */
function suffixTreeNodes(t: readonly number[]): { offset: number; length: number; freq: number }[] {
  const n = t.length
  if (n === 0) return []
  const sa = suffixArray(t)
  const psi = new Int32Array(n)
  psi[sa[0] ?? 0] = sa[n - 1] ?? 0
  for (let i = 1; i < n; i++) psi[sa[i] ?? 0] = sa[i - 1] ?? 0
  const plcp = new Int32Array(n)
  let h = 0
  for (let i = 0; i < n; i++) {
    const j = psi[i] ?? 0
    while (i + h < n && j + h < n && t[i + h] === t[j + h]) h++
    plcp[i] = h
    if (h > 0) h--
  }
  const H = new Int32Array(n)
  for (let i = 0; i < n; i++) H[i] = plcp[sa[i] ?? 0] ?? 0
  H[0] = -1
  const out: { offset: number; length: number; freq: number }[] = []
  const stack: [number, number][] = [[-1, -1]]
  for (let i = 0; ; i++) {
    let cur: [number, number] = [i, i === n ? -1 : (H[i] ?? 0)]
    let cand = stack[stack.length - 1] ?? [-1, -1]
    while (cand[1] > cur[1]) {
      if (i - cand[0] > 1) out.push({ offset: sa[cand[0]] ?? 0, length: cand[1], freq: i - cand[0] })
      cur = [cand[0], cur[1]]
      stack.pop()
      cand = stack[stack.length - 1] ?? [-1, -1]
    }
    if (cand[1] < cur[1]) stack.push(cur)
    if (i === n) break
    stack.push([i, n - (sa[i] ?? 0) + 1])
  }
  return out
}

/** Lexicographic comparison of code point arrays. */
function compareArrays(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return (a[i] ?? 0) - (b[i] ?? 0)
  return a.length - b.length
}

/** Seed pieces: every character, then frequent substrings scored by frequency times length (log probabilities). */
function seedPieces(
  sentences: readonly (readonly [string, number])[],
  maxPieceLength: number,
  seedSize: number,
): [string, number][] {
  const flat: number[] = []
  const chars = new Map<number, number>()
  for (const [s, c] of sentences) {
    if (s.length === 0) continue
    for (const ch of s) {
      const cp = ch.codePointAt(0) ?? 0
      flat.push(cp)
      if (cp !== 0) chars.set(cp, (chars.get(cp) ?? 0) + c)
    }
    flat.push(0)
  }
  const subs: { score: number; cps: number[] }[] = []
  for (const nd of suffixTreeNodes(flat)) {
    if (nd.length <= 1 || nd.length > maxPieceLength) continue
    const cps = flat.slice(nd.offset, nd.offset + nd.length)
    if (cps.includes(0)) continue
    subs.push({ score: nd.freq * nd.length, cps })
  }
  const seeds: [string, number][] = [...chars]
    .sort((a, b) => b[1] - a[1] || b[0] - a[0])
    .map(([cp, c]) => [String.fromCodePoint(cp), c])
  subs.sort((a, b) => b.score - a.score || compareArrays(b.cps, a.cps))
  for (const s of subs) {
    if (seeds.length >= seedSize) break
    seeds.push([String.fromCodePoint(...s.cps), s.score])
  }
  let sum = 0
  for (const [, v] of seeds) sum += v
  const logsum = Math.log(sum)
  return seeds.map(([p, v]) => [p, Math.log(v) - logsum])
}

/** One E step and one M step (HF's run_e_step and run_m_step, with the Bayesian digamma update). */
function emStep(
  pieces: readonly (readonly [string, number])[],
  sentences: readonly (readonly [string, number])[],
): { pieces: [string, number][]; objective: number } {
  const model = unigramModel(pieces, { unkId: 0 })
  const expected = new Float64Array(pieces.length)
  let total = 0
  for (const [, c] of sentences) total += c
  let objective = 0
  for (const [s, c] of sentences) {
    const z = populateMarginal(modelLattice(model, s), c, expected)
    objective -= z / total
  }
  const kept: [string, number][] = []
  let sum = 0
  for (const [i, [piece]] of pieces.entries()) {
    if (i === 0) {
      kept.push([piece, Number.NaN])
      continue
    }
    const f = expected[i] ?? 0
    if (f < 0.5) continue
    kept.push([piece, f])
    sum += f
  }
  const logsum = digamma(sum)
  return { pieces: kept.map(([p, f]) => [p, digamma(f) - logsum]), objective }
}

/** One pruning round (HF's prune_sentence_pieces, including its use of the number of pieces in logsum_alt). */
function prune(
  pieces: readonly (readonly [string, number])[],
  sentences: readonly (readonly [string, number])[],
  vocabSize: number,
  shrinking: number,
): [string, number][] {
  const model = unigramModel(pieces, { unkId: 0 })
  const alwaysKeep = pieces.map(() => true)
  const alternatives: number[][] = pieces.map(() => [])
  for (const [id, [piece]] of pieces.entries()) {
    if (id === 0) {
      alwaysKeep[id] = false
      continue
    }
    const nbests = kBestPaths(modelLattice(model, piece), 2)
    if (nbests.length === 1) alwaysKeep[id] = true
    else if ((nbests[0]?.length ?? 0) >= 2) alwaysKeep[id] = false
    else if ((nbests[0]?.length ?? 0) === 1) {
      alwaysKeep[id] = true
      for (const nd of nbests[1] ?? []) alternatives[id]?.push(nd.id)
    }
  }
  let vsum = 0
  const freq = new Float64Array(pieces.length)
  const inverted: number[][] = pieces.map(() => [])
  for (const [i, [s, c]] of sentences.entries()) {
    vsum += c
    for (const nd of viterbiPath(modelLattice(model, s))) {
      freq[nd.id] = (freq[nd.id] ?? 0) + c
      inverted[nd.id]?.push(i)
    }
  }
  let sum = 0
  for (const f of freq) sum += f
  const logsum = Math.log(sum)
  const kept: [string, number][] = [[pieces[0]?.[0] ?? '', pieces[0]?.[1] ?? Number.NaN]]
  const candidates: [number, number][] = []
  for (const [id, [piece, score]] of pieces.entries()) {
    if (id === 0) continue
    const fid = freq[id] ?? 0
    const alts = alternatives[id] ?? []
    if (fid === 0 && alwaysKeep[id] !== true) continue
    if (alts.length === 0) {
      kept.push([piece, score])
      continue
    }
    let f = 0
    for (const n of inverted[id] ?? []) f += sentences[n]?.[1] ?? 0
    if (f === 0 || Number.isNaN(f)) continue
    f /= vsum
    const logprobSp = Math.log(fid) - logsum
    const logsumAlt = Math.log(sum + fid * (pieces.length - 1))
    let logprobAlt = 0
    for (const n of alts) logprobAlt += Math.log((freq[n] ?? 0) + fid) - logsumAlt
    candidates.push([id, f * (logprobSp - logprobAlt)])
  }
  const desired = Math.floor((vocabSize * 11) / 10)
  const prunedSize = Math.max(desired, Math.floor(pieces.length * shrinking))
  candidates.sort((a, b) => b[1] - a[1])
  for (const [id] of candidates) {
    if (kept.length === prunedSize) break
    const p = pieces[id]
    if (p !== undefined) kept.push([p[0], p[1]])
  }
  return kept
}

/** The final vocabulary (HF's finalize): required characters, the best pieces, sorted by score, specials first. */
function finalize(
  pieces: readonly (readonly [string, number])[],
  required: readonly string[],
  options: UnigramTrainingOptions,
): UnigramModel {
  const vocabSize = options.vocabSize ?? 8000
  const specials = [...(options.specialTokens ?? [])]
  let minScore = Number.POSITIVE_INFINITY
  for (const [, s] of pieces) if (s < minScore) minScore = s
  const existing = new Map(pieces.map(([p, s]) => [p, s]))
  const out: [string, number][] = []
  const inserted = new Set<string>(['<UNK>'])
  let penalty = 0
  for (const c of required) {
    inserted.add(c)
    const s = existing.get(c)
    if (s !== undefined) out.push([c, s])
    else {
      out.push([c, minScore + penalty])
      penalty += 0.0001
    }
  }
  const unk = options.unkToken
  const unkIndex = unk === undefined ? -1 : specials.indexOf(unk)
  const addUnk = unk !== undefined && unkIndex < 0
  const room = vocabSize - specials.length - (addUnk ? 1 : 0)
  for (const [p, s] of pieces) {
    if (inserted.has(p)) continue
    inserted.add(p)
    out.push([p, Number.isNaN(s) ? 0 : s])
    if (out.length === room) break
  }
  // Stable sort by score, descending; ties keep insertion order.
  out.sort((a, b) => b[1] - a[1])
  const head: [string, number][] = specials.map((s) => [s, 0])
  if (addUnk && unk !== undefined) head.unshift([unk, 0])
  return unigramModel([...head, ...out], unk === undefined ? {} : { unkId: addUnk ? 0 : unkIndex })
}

/**
 * Unigram language-model training as an {@link Algorithm} (Kudo 2018), following HF tokenizers' UnigramTrainer:
 * seed pieces from a suffix array (every character, then the substrings occurring at least twice, by frequency times
 * length), then rounds of EM with a digamma (Bayesian) M step that drops pieces with expected count below 0.5,
 * alternating with pruning of the pieces whose removal costs the least likelihood, until the vocabulary is within
 * 1.1 times the target; then the required characters and the best pieces form the final vocabulary.
 *
 * @param wordCounts - Pre-tokenised words (for example with SentencePiece's `▁`) and their counts.
 * @param options - Vocabulary size, specials, unknown token, piece length, EM and pruning parameters.
 * @returns The algorithm; each step is one EM iteration, one pruning or the final step.
 * @throws MdmlError with code `value` when the vocabulary cannot hold the required characters.
 * @concept unigram-lm
 * @example
 * ```ts
 * import { unigramTrainingAlgorithm } from 'mdml-engine/text/subword'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const r = runAlgorithm(unigramTrainingAlgorithm(new Map([['▁hello', 4], ['▁help', 3], ['▁yellow', 2]]), { vocabSize: 12 }), { maxSteps: 50 })
 * [r.status, r.state.model?.pieces.length] // => ['converged', 11]
 * ```
 */
export function unigramTrainingAlgorithm(
  wordCounts: ReadonlyMap<string, number> | Iterable<readonly [string, number]>,
  options: UnigramTrainingOptions = {},
): Algorithm<UnigramTrainingState> {
  const sentences = [...wordCounts].map(([w, c]) => [w, c] as const)
  const vocabSize = options.vocabSize ?? 8000
  const desired = Math.floor((vocabSize * 11) / 10)
  const subIterations = options.nSubIterations ?? 2
  const shrinking = options.shrinkingFactor ?? 0.75
  const maxRounds = options.maxRounds ?? 100
  const required: string[] = []
  const seen = new Set<string>()
  for (const [s] of sentences)
    for (const ch of s)
      if (!seen.has(ch)) {
        seen.add(ch)
        required.push(ch)
      }
  for (const ch of options.initialAlphabet ?? [])
    if (!seen.has(ch)) {
      seen.add(ch)
      required.push(ch)
    }
  if (required.length > vocabSize)
    throw new MdmlError('value', 'trainUnigram: the vocabulary is too small for the required characters')
  return {
    name: 'unigramTraining',
    config: { vocabSize, subIterations, shrinking } as Json,
    init: () => ({
      pieces: [
        ['<UNK>', Number.NaN],
        ...seedPieces(sentences, options.maxPieceLength ?? 16, options.seedSize ?? 1_000_000),
      ],
      phase: 'em',
      subIteration: 0,
      rounds: 0,
      objective: Number.NaN,
    }),
    step: (state) => {
      if (state.phase === 'em') {
        const { pieces, objective } = emStep(state.pieces, sentences)
        const sub = state.subIteration + 1
        if (sub < subIterations) return { ...state, pieces, objective, subIteration: sub }
        if (pieces.length > desired && state.rounds < maxRounds)
          return { ...state, pieces, objective, subIteration: 0, phase: 'prune' }
        return {
          ...state,
          pieces,
          objective,
          subIteration: 0,
          phase: 'done',
          model: finalize(pieces, required, options),
        }
      }
      if (state.phase === 'prune') {
        const pruned = prune(state.pieces, sentences, vocabSize, shrinking)
        // A round that cannot shrink the vocabulary would repeat forever.
        const stuck = pruned.length >= state.pieces.length
        return { ...state, pieces: pruned, phase: 'em', rounds: stuck ? maxRounds : state.rounds + 1 }
      }
      return state
    },
    done: (state) => (state.model === undefined ? undefined : 'converged'),
    observe: (state) => ({ metrics: { pieces: state.pieces.length, objective: state.objective } }),
  }
}

/**
 * Train a Unigram model ({@link unigramTrainingAlgorithm}).
 *
 * @param wordCounts - Pre-tokenised words and their counts.
 * @param options - Training options.
 * @returns The model.
 * @concept unigram-lm
 * @example
 * ```ts
 * import { trainUnigram, unigramTokenize } from 'mdml-engine/text/subword'
 * const m = trainUnigram(new Map([['▁hello', 4], ['▁help', 3], ['▁yellow', 2]]), { vocabSize: 14 })
 * unigramTokenize(m, '▁hello').map((t) => t.token).join('') // => '▁hello'
 * ```
 */
export function trainUnigram(
  wordCounts: ReadonlyMap<string, number> | Iterable<readonly [string, number]>,
  options: UnigramTrainingOptions = {},
): UnigramModel {
  const entries = [...wordCounts]
  const alg = unigramTrainingAlgorithm(entries, options)
  const rounds = options.maxRounds ?? 100
  const result = runAlgorithm(alg, {
    maxSteps: (rounds + 1) * ((options.nSubIterations ?? 2) + 1) + 2,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const model = result.state.model
  if (model === undefined) throw new MdmlError('convergence', 'trainUnigram: training did not finish')
  return model
}
