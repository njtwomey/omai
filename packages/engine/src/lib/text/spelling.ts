/**
 * Spelling correction as a noisy channel (Kernighan, Church and Gale 1990; Norvig 2007): the intended word `w` that
 * maximises `P(typed | w) P(w)`. Candidates within an edit distance come from a BK-tree, the channel `P(typed | w)` is a
 * weighted Damerau-Levenshtein alignment whose operation probabilities (match, deletion, insertion, substitution by
 * character pair, adjacent transposition) are counted from (typed, intended) pairs, and the prior is a word frequency
 * table or an n-gram language model that reads the preceding words.
 */

import { MdmlError } from '../base/index.ts'
import { bkSearch, bkTree } from './approximate-search.ts'
import type { NgramModel } from './language-models.ts'

/** The learned probabilities of the edit operations of a typing channel. */
export interface EditChannel {
  /** `log P(typed | intended)` of the most probable alignment of the two strings. */
  logProbability(typed: string, intended: string): number
  /** The probability a character is typed correctly, deleted, substituted, transposed with the next, or followed by an insertion. */
  readonly rates: {
    readonly match: number
    readonly deletion: number
    readonly substitution: number
    readonly transposition: number
    readonly insertion: number
  }
}

type Op =
  | { readonly kind: 'match' }
  | { readonly kind: 'del'; readonly c: string }
  | { readonly kind: 'ins'; readonly c: string }
  | { readonly kind: 'sub'; readonly from: string; readonly to: string }
  | { readonly kind: 'trans' }

/** The cheapest alignment under per-operation costs, returned as the operations (intended to typed). */
function align(
  typed: readonly string[],
  intended: readonly string[],
  cost: (op: Op) => number,
): { cost: number; ops: Op[] } {
  const n = intended.length
  const m = typed.length
  const dp = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Number.POSITIVE_INFINITY))
  const back: (Op | undefined)[][] = Array.from({ length: n + 1 }, () =>
    new Array<Op | undefined>(m + 1).fill(undefined),
  )
  ;(dp[0] as Float64Array)[0] = 0
  const relax = (i: number, j: number, from: number, to: number, op: Op): void => {
    const c = (dp[from] as Float64Array)[to] as number
    const total = c + cost(op)
    if (total < ((dp[i] as Float64Array)[j] as number)) {
      ;(dp[i] as Float64Array)[j] = total
      ;(back[i] as (Op | undefined)[])[j] = op
    }
  }
  for (let i = 0; i <= n; i++)
    for (let j = 0; j <= m; j++) {
      if (i === 0 && j === 0) continue
      if (i > 0 && j > 0) {
        const a = intended[i - 1] as string
        const b = typed[j - 1] as string
        relax(i, j, i - 1, j - 1, a === b ? { kind: 'match' } : { kind: 'sub', from: a, to: b })
      }
      if (i > 0) relax(i, j, i - 1, j, { kind: 'del', c: intended[i - 1] as string })
      if (j > 0) relax(i, j, i, j - 1, { kind: 'ins', c: typed[j - 1] as string })
      if (
        i > 1 &&
        j > 1 &&
        intended[i - 1] === typed[j - 2] &&
        intended[i - 2] === typed[j - 1] &&
        intended[i - 1] !== intended[i - 2]
      )
        relax(i, j, i - 2, j - 2, { kind: 'trans' })
    }
  const ops: Op[] = []
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    const op = (back[i] as (Op | undefined)[])[j] as Op
    ops.push(op)
    if (op.kind === 'match' || op.kind === 'sub') {
      i--
      j--
    } else if (op.kind === 'del') i--
    else if (op.kind === 'ins') j--
    else {
      i -= 2
      j -= 2
    }
  }
  return { cost: (dp[n] as Float64Array)[m] as number, ops: ops.reverse() }
}

/**
 * Learn a typing channel from (typed, intended) pairs: align each pair by unit-cost Damerau-Levenshtein, count the
 * operations (matches, deletions and insertions by character, substitutions by character pair, transpositions), and
 * turn the counts into probabilities with add-`smoothing` (default 0.5) over the characters seen. The result scores
 * any pair by the probability of its most likely alignment.
 *
 * @param pairs - `[typed, intended]` pairs.
 * @param smoothing - The pseudo-count (default 0.5).
 * @returns The channel.
 * @throws MdmlError with code `value` for no pairs.
 * @concept spelling-correction
 * @example
 * ```ts
 * import { learnChannel } from 'mdml-engine/text'
 * const c = learnChannel([['teh', 'the'], ['recieve', 'receive'], ['cat', 'cat']])
 * c.logProbability('teh', 'the') > c.logProbability('tzz', 'the') // => true
 * ```
 */
export function learnChannel(
  pairs: readonly (readonly [typed: string, intended: string])[],
  smoothing = 0.5,
): EditChannel {
  if (pairs.length === 0) throw new MdmlError('value', 'learnChannel: needs at least one pair')
  let match = 0
  let del = 0
  let ins = 0
  let sub = 0
  let trans = 0
  const alphabet = new Set<string>()
  const subs = new Map<string, number>()
  const dels = new Map<string, number>()
  const inss = new Map<string, number>()
  for (const [typed, intended] of pairs) {
    for (const c of typed) alphabet.add(c)
    for (const c of intended) alphabet.add(c)
    const { ops } = align(Array.from(typed), Array.from(intended), (op) => (op.kind === 'match' ? 0 : 1))
    for (const op of ops) {
      if (op.kind === 'match') match++
      else if (op.kind === 'del') {
        del++
        dels.set(op.c, (dels.get(op.c) ?? 0) + 1)
      } else if (op.kind === 'ins') {
        ins++
        inss.set(op.c, (inss.get(op.c) ?? 0) + 1)
      } else if (op.kind === 'sub') {
        sub++
        subs.set(`${op.from}${op.to}`, (subs.get(`${op.from}${op.to}`) ?? 0) + 1)
      } else trans++
    }
  }
  const A = Math.max(1, alphabet.size)
  const total = match + del + sub + trans
  const s = smoothing
  const events = total + 4 * s
  const rate = {
    match: (match + s) / events,
    deletion: (del + s) / events,
    substitution: (sub + s) / events,
    transposition: (trans + s) / events,
    insertion: (ins + s) / (total + ins + s * 2),
  }
  const subTotal = sub + s * A * A
  const delTotal = del + s * A
  const insTotal = ins + s * A
  const cost = (op: Op): number => {
    switch (op.kind) {
      case 'match':
        return -Math.log(rate.match)
      case 'del':
        return -Math.log((rate.deletion * ((dels.get(op.c) ?? 0) + s)) / delTotal)
      case 'ins':
        return -Math.log((rate.insertion * ((inss.get(op.c) ?? 0) + s)) / insTotal)
      case 'sub':
        return -Math.log((rate.substitution * ((subs.get(`${op.from}${op.to}`) ?? 0) + s)) / subTotal)
      case 'trans':
        return -Math.log(rate.transposition)
    }
  }
  return {
    rates: rate,
    logProbability: (typed, intended) => -align(Array.from(typed), Array.from(intended), cost).cost,
  }
}

/**
 * The channel that charges the same probability `epsilon` for every edit, whatever the characters.
 *
 * @param epsilon - The probability of an edit (default 0.01).
 * @returns The channel.
 * @concept spelling-correction
 * @example
 * ```ts
 * import { uniformChannel } from 'mdml-engine/text'
 * Math.exp(uniformChannel(0.01).logProbability('teh', 'the')) // => 0.01
 * ```
 */
export function uniformChannel(epsilon = 0.01): EditChannel {
  return {
    rates: { match: 1 - epsilon, deletion: epsilon, substitution: epsilon, transposition: epsilon, insertion: epsilon },
    logProbability: (typed, intended) =>
      -align(Array.from(typed), Array.from(intended), (op) => (op.kind === 'match' ? 0 : -Math.log(epsilon))).cost,
  }
}

/** A candidate correction with the two terms of its score. */
export interface SpellingCandidate {
  readonly word: string
  readonly distance: number
  /** `log P(typed | word)`. */
  readonly logChannel: number
  /** `log P(word)` or `log P(word | previous words)`. */
  readonly logPrior: number
  /** The sum, which ranks the candidates. */
  readonly score: number
}

/** A spelling corrector. */
export interface SpellingCorrector {
  /** The dictionary. */
  readonly words: readonly string[]
  /** The candidates within the edit distance, best first (ties by word). */
  candidates(typed: string, context?: readonly string[]): SpellingCandidate[]
  /** The best candidate, or the typed word when none is in range. */
  correct(typed: string, context?: readonly string[]): string
  /** A sentence corrected left to right, each word given the corrected words before it as context. */
  correctSentence(tokens: readonly string[]): string[]
}

/** Options of {@link spellingCorrector}. */
export interface SpellingOptions {
  /** The largest edit distance of a candidate (default 2). */
  readonly maxDistance?: number
  /** The channel (default a unit-cost channel with edit probability 0.01). */
  readonly channel?: EditChannel
  /** Add-k smoothing of the word frequencies (default 0.1). */
  readonly smoothing?: number
  /** A language model over words, which reads the context (default the unigram frequencies). */
  readonly languageModel?: NgramModel
  /** Weight of the prior against the channel (default 1). */
  readonly priorWeight?: number
}

/**
 * A noisy-channel spelling corrector over a dictionary with word counts.
 *
 * @param counts - How often each word of the dictionary occurs in text.
 * @param options - The edit distance, the channel, the smoothing and an optional n-gram language model.
 * @returns The corrector.
 * @throws MdmlError with code `value` for an empty dictionary.
 * @concept spelling-correction
 * @example
 * ```ts
 * import { spellingCorrector } from 'mdml-engine/text'
 * spellingCorrector(new Map([['the', 50], ['then', 5], ['they', 4]])).correct('teh') // => 'the'
 * ```
 */
export function spellingCorrector(
  counts: ReadonlyMap<string, number>,
  options: SpellingOptions = {},
): SpellingCorrector {
  if (counts.size === 0) throw new MdmlError('value', 'spellingCorrector: needs a dictionary')
  const words = [...counts.keys()].sort()
  const tree = bkTree(words)
  const channel = options.channel ?? uniformChannel()
  const k = options.smoothing ?? 0.1
  const total = [...counts.values()].reduce((a, b) => a + b, 0)
  const maxDistance = options.maxDistance ?? 2
  const lm = options.languageModel
  const weight = options.priorWeight ?? 1
  const logPrior = (w: string, context: readonly string[]): number =>
    lm !== undefined
      ? Math.log(lm.prob(w, context))
      : Math.log(((counts.get(w) ?? 0) + k) / (total + k * (words.length + 1)))
  const candidates = (typed: string, context: readonly string[] = []): SpellingCandidate[] =>
    bkSearch(tree, typed, maxDistance)
      .matches.map((m) => {
        const logChannel = channel.logProbability(typed, m.value)
        const prior = logPrior(m.value, context)
        return { word: m.value, distance: m.distance, logChannel, logPrior: prior, score: logChannel + weight * prior }
      })
      .sort((a, b) => b.score - a.score || (a.word < b.word ? -1 : 1))
  return {
    words,
    candidates,
    correct: (typed, context) => candidates(typed, context)[0]?.word ?? typed,
    correctSentence: (tokens) => {
      const out: string[] = []
      for (const t of tokens) out.push(candidates(t, out)[0]?.word ?? t)
      return out
    },
  }
}
