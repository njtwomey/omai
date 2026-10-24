import { type MetricResult, type MetricSpec, maxOf, WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import {
  editCounts,
  lcsTable,
  ngramCounts,
  clippedOverlap as overlap,
  porterStem,
  countTotal as total,
} from '../text/index.ts'
import { terEdits } from './_ter.ts'
import { defineMetric, type MetricFunction, matrix, metricResult } from './metric.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower',
  range: readonly [number, number],
  inputs: MetricSpec['inputs'] = 'labels',
): MetricSpec => ({ name, direction, range, inputs, category: 'text', concept: 'text-evaluation' })

/** One text or a corpus of texts. */
export type Texts = string | readonly string[]

/** The references of each hypothesis: one string each, or several. */
export type References = string | readonly (string | readonly string[])[]

const list = (x: Texts): readonly string[] => (typeof x === 'string' ? [x] : x)

const referenceLists = (name: string, refs: References, n: number): (readonly string[])[] => {
  const r = typeof refs === 'string' ? [refs] : refs
  if (r.length !== n) throw new WhiteboxError('shape', `${name}: ${r.length} references for ${n} hypotheses`)
  return r.map((x) => (typeof x === 'string' ? [x] : x))
}

const sameLength = (name: string, a: readonly unknown[], b: readonly unknown[]): void => {
  if (a.length !== b.length)
    throw new WhiteboxError('shape', `${name}: ${a.length} references for ${b.length} hypotheses`)
}

/** Python's str.split(): whitespace-separated tokens. */
const words = (s: string): string[] => s.split(/\s+/).filter((w) => w.length > 0)

// ---------------------------------------------------------------------------------------------------------------
// Edit rates (jiwer)

interface Alignment {
  readonly hits: number
  readonly substitutions: number
  readonly deletions: number
  readonly insertions: number
  readonly refLength: number
  readonly hypLength: number
}

/** jiwer's default word transform: collapse runs of whitespace, strip, split on spaces. */
const jiwerWords = (s: string): string[] =>
  s
    .replace(/\s\s+/g, ' ')
    .trim()
    .split(' ')
    .filter((w) => w.length > 0)

/** jiwer's default character transform: strip, then characters (code points). */
const jiwerChars = (s: string): string[] => Array.from(s.trim())

function align(name: string, references: Texts, hypotheses: Texts, unit: (s: string) => string[]): Alignment {
  const r = list(references)
  const h = list(hypotheses)
  sameLength(name, r, h)
  let hits = 0
  let substitutions = 0
  let deletions = 0
  let insertions = 0
  let refLength = 0
  let hypLength = 0
  for (const [i, ref] of r.entries()) {
    const a = unit(ref)
    const b = unit(h[i] ?? '')
    const c = editCounts(a, b)
    hits += c.hits
    substitutions += c.substitutions
    deletions += c.deletions
    insertions += c.insertions
    refLength += a.length
    hypLength += b.length
  }
  return { hits, substitutions, deletions, insertions, refLength, hypLength }
}

const counts = (a: Alignment): Readonly<Record<string, number>> => ({
  hits: a.hits,
  substitutions: a.substitutions,
  deletions: a.deletions,
  insertions: a.insertions,
})

const errorRate = (a: Alignment): number =>
  a.refLength === 0
    ? a.insertions
    : (a.substitutions + a.deletions + a.insertions) / (a.hits + a.substitutions + a.deletions)

const preserved = (a: Alignment): number =>
  a.refLength === 0
    ? a.hypLength === 0
      ? 1
      : 0
    : a.hypLength >= 1
      ? (a.hits / a.refLength) * (a.hits / a.hypLength)
      : 0

/**
 * The word error rate (S + D + I) / N over a corpus (jiwer.wer): substitutions, deletions and insertions of the
 * Levenshtein word alignment, summed over sentences, over the number of reference words. Words are separated by
 * whitespace after collapsing runs of it.
 *
 * @param references - Reference transcripts.
 * @param hypotheses - Hypothesis transcripts, one per reference.
 * @returns The metric result, with the alignment counts in the breakdown.
 * @concept text-evaluation
 * @example
 * ```ts
 * import { wordErrorRate } from 'whitebox-engine/metrics'
 * wordErrorRate('the cat sat on the mat', 'the cat sit on mat').estimate.value // => 0.3333333333333333
 * ```
 */
export const wordErrorRate: MetricFunction<[Texts, Texts]> = defineMetric(
  spec('wordErrorRate', 'lower', [0, Infinity]),
  (references: Texts, hypotheses: Texts): MetricResult => {
    const a = align('wordErrorRate', references, hypotheses, jiwerWords)
    return metricResult('wordErrorRate', errorRate(a), a.refLength, counts(a))
  },
)

/**
 * The character error rate (jiwer.cer): the word error rate computed over characters (spaces included) of the stripped
 * texts.
 *
 * @param references - Reference transcripts.
 * @param hypotheses - Hypothesis transcripts, one per reference.
 * @returns The metric result, with the alignment counts in the breakdown.
 * @example
 * ```ts
 * import { characterErrorRate } from 'whitebox-engine/metrics'
 * characterErrorRate('kitten', 'sitting').estimate.value // => 0.5
 * ```
 */
export const characterErrorRate: MetricFunction<[Texts, Texts]> = defineMetric(
  spec('characterErrorRate', 'lower', [0, Infinity]),
  (references: Texts, hypotheses: Texts): MetricResult => {
    const a = align('characterErrorRate', references, hypotheses, jiwerChars)
    return metricResult('characterErrorRate', errorRate(a), a.refLength, counts(a))
  },
)

/**
 * The match error rate (S + D + I) / (H + S + D + I) (jiwer.mer), the probability that a word alignment pair is an
 * error.
 *
 * @param references - Reference transcripts.
 * @param hypotheses - Hypothesis transcripts, one per reference.
 * @returns The metric result, in [0, 1].
 * @example
 * ```ts
 * import { matchErrorRate } from 'whitebox-engine/metrics'
 * matchErrorRate('a b c', 'a x c d').estimate.value // => 0.5
 * ```
 */
export const matchErrorRate: MetricFunction<[Texts, Texts]> = defineMetric(
  spec('matchErrorRate', 'lower', [0, 1]),
  (references: Texts, hypotheses: Texts): MetricResult => {
    const a = align('matchErrorRate', references, hypotheses, jiwerWords)
    const errors = a.substitutions + a.deletions + a.insertions
    const value = a.refLength === 0 ? (a.hypLength === 0 ? 0 : 1) : errors / (a.hits + errors)
    return metricResult('matchErrorRate', value, a.refLength, counts(a))
  },
)

/**
 * The word information preserved (H / N_ref)(H / N_hyp) (jiwer.wip; Morris et al., 2004).
 *
 * @param references - Reference transcripts.
 * @param hypotheses - Hypothesis transcripts, one per reference.
 * @returns The metric result, in [0, 1].
 * @example
 * ```ts
 * import { wordInformationPreserved } from 'whitebox-engine/metrics'
 * wordInformationPreserved('a b c d', 'a b').estimate.value // => 0.5
 * ```
 */
export const wordInformationPreserved: MetricFunction<[Texts, Texts]> = defineMetric(
  spec('wordInformationPreserved', 'higher', [0, 1]),
  (references: Texts, hypotheses: Texts): MetricResult => {
    const a = align('wordInformationPreserved', references, hypotheses, jiwerWords)
    return metricResult('wordInformationPreserved', preserved(a), a.refLength, counts(a))
  },
)

/**
 * The word information lost 1 - (H / N_ref)(H / N_hyp) (jiwer.wil; Morris et al., 2004).
 *
 * @param references - Reference transcripts.
 * @param hypotheses - Hypothesis transcripts, one per reference.
 * @returns The metric result, in [0, 1].
 * @example
 * ```ts
 * import { wordInformationLost } from 'whitebox-engine/metrics'
 * wordInformationLost('a b c d', 'a b').estimate.value // => 0.5
 * ```
 */
export const wordInformationLost: MetricFunction<[Texts, Texts]> = defineMetric(
  spec('wordInformationLost', 'lower', [0, 1]),
  (references: Texts, hypotheses: Texts): MetricResult => {
    const a = align('wordInformationLost', references, hypotheses, jiwerWords)
    return metricResult('wordInformationLost', 1 - preserved(a), a.refLength, counts(a))
  },
)

// ---------------------------------------------------------------------------------------------------------------
// BLEU (sacrebleu)

const PUNCT_13A: readonly [RegExp, string][] = [
  [/([\x7b-\x7e\x5b-\x60\x20-\x26\x28-\x2b\x3a-\x40\x2f])/g, ' $1 '],
  [/([^0-9])([.,])/g, '$1 $2 '],
  [/([.,])([^0-9])/g, ' $1 $2'],
  [/([0-9])(-)/g, '$1 $2 '],
]

/**
 * The 13a tokeniser of mteval-v13a and sacrebleu: unescapes a few XML entities and splits punctuation from words (but
 * not periods and commas inside numbers).
 *
 * @param line - The text.
 * @returns The tokens joined by single spaces.
 * @example
 * ```ts
 * import { tokenize13a } from 'whitebox-engine/metrics'
 * tokenize13a('Hello, world (3.5 kg)!') // => 'Hello , world ( 3.5 kg ) !'
 * ```
 */
export function tokenize13a(line: string): string {
  let s = line.replaceAll('<skipped>', '').replaceAll('-\n', '').replaceAll('\n', ' ')
  if (s.includes('&'))
    s = s.replaceAll('&quot;', '"').replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>')
  s = ` ${s} `
  for (const [re, repl] of PUNCT_13A) s = s.replace(re, repl)
  return words(s).join(' ')
}

/** Options of {@link bleuScore} and {@link sentenceBleu}. */
export interface BleuOptions {
  /** Smoothing (Chen and Cherry, 2014): 'exp' (NIST, the default), 'floor' (0.1), 'add-k' (1) or 'none'. */
  readonly smooth?: 'exp' | 'floor' | 'add-k' | 'none'
  /** The value of 'floor' or 'add-k' smoothing. */
  readonly smoothValue?: number
  /** The largest n-gram order (default 4). */
  readonly maxOrder?: number
  /** Drop n-gram orders with no hypothesis n-grams from the geometric mean (default false; true for sentences). */
  readonly effectiveOrder?: boolean
  readonly lowercase?: boolean
  /** The tokeniser: '13a' (default) or 'none' (split on whitespace). */
  readonly tokenize?: '13a' | 'none'
}

function bleuFrom(stats: readonly number[], order: number, options: BleuOptions): MetricResult {
  const method = options.smooth ?? 'exp'
  const value = options.smoothValue ?? (method === 'floor' ? 0.1 : 1)
  const sysLen = stats[0] ?? 0
  const refLen = stats[1] ?? 0
  const correct = stats.slice(2, 2 + order)
  const totals = stats.slice(2 + order, 2 + 2 * order)
  let bp = 1
  if (sysLen < refLen) bp = sysLen > 0 ? Math.exp(1 - refLen / sysLen) : 0
  const precisions = new Array<number>(order).fill(0)
  const breakdown = (score: number): Readonly<Record<string, number>> => ({
    bp,
    sysLen,
    refLen,
    ...Object.fromEntries(precisions.map((p, i) => [`precision${i + 1}`, p])),
    score,
  })
  if (!correct.some((c) => c !== 0)) return metricResult('bleuScore', 0, sysLen, breakdown(0))
  let mteval = 1
  let eff = order
  for (let n = 1; n <= order; n++) {
    if (method === 'add-k' && n > 1) {
      correct[n - 1] = (correct[n - 1] ?? 0) + value
      totals[n - 1] = (totals[n - 1] ?? 0) + value
    }
    const t = totals[n - 1] ?? 0
    const c = correct[n - 1] ?? 0
    if (t === 0) break
    if (options.effectiveOrder === true) eff = n
    if (c === 0) {
      if (method === 'exp') {
        mteval *= 2
        precisions[n - 1] = 100 / (mteval * t)
      } else if (method === 'floor') precisions[n - 1] = (100 * value) / t
    } else precisions[n - 1] = (100 * c) / t
  }
  const logs = precisions.slice(0, eff).reduce((s, p) => s + (p === 0 ? -9999999999 : Math.log(p)), 0)
  const score = bp * Math.exp(logs / eff)
  return metricResult('bleuScore', score, sysLen, breakdown(score))
}

function bleuStats(refs: readonly string[], hyp: string, order: number, options: BleuOptions): number[] {
  const prep = (s: string): string[] => {
    const t = options.lowercase === true ? s.toLowerCase() : s
    const trimmed = t.trimEnd()
    return words(options.tokenize === 'none' ? trimmed : tokenize13a(trimmed))
  }
  const refNgrams: Map<string, number>[] = Array.from({ length: order }, () => new Map<string, number>())
  const refLens: number[] = []
  for (const ref of refs) {
    const tokens = prep(ref)
    refLens.push(tokens.length)
    for (let n = 1; n <= order; n++) {
      const target = refNgrams[n - 1] as Map<string, number>
      for (const [k, v] of ngramCounts(tokens, n)) target.set(k, Math.max(target.get(k) ?? 0, v))
    }
  }
  const tokens = prep(hyp)
  let closest = -1
  let diff = -1
  for (const len of refLens) {
    const d = Math.abs(tokens.length - len)
    if (diff === -1 || d < diff) {
      diff = d
      closest = len
    } else if (d === diff && len < closest) closest = len
  }
  const correct: number[] = []
  const totals: number[] = []
  for (let n = 1; n <= order; n++) {
    const h = ngramCounts(tokens, n)
    totals.push(total(h))
    correct.push(overlap(h, refNgrams[n - 1] as Map<string, number>))
  }
  return [tokens.length, closest, ...correct, ...totals]
}

function corpusBleu(refs: (readonly string[])[], hyps: readonly string[], options: BleuOptions): MetricResult {
  const order = options.maxOrder ?? 4
  const stats = new Array<number>(2 + 2 * order).fill(0)
  for (const [i, h] of hyps.entries()) {
    const s = bleuStats(refs[i] ?? [], h, order, options)
    for (let k = 0; k < stats.length; k++) stats[k] = (stats[k] ?? 0) + (s[k] ?? 0)
  }
  return bleuFrom(stats, order, options)
}

/**
 * Corpus BLEU (Papineni et al., 2002) as sacrebleu computes it: clipped n-gram counts and the closest reference
 * lengths are summed over the corpus, then the brevity penalty times the geometric mean of the n-gram precisions, on
 * a 0 to 100 scale. Text is tokenised with 13a.
 *
 * @param references - The references of each hypothesis (a string, or a list of strings).
 * @param hypotheses - The hypotheses.
 * @param options - Smoothing, maximum order, effective order, lowercasing and tokeniser.
 * @returns The score, with the brevity penalty, lengths and n-gram precisions in the breakdown.
 * @example
 * ```ts
 * import { bleuScore } from 'whitebox-engine/metrics'
 * Number(bleuScore(['the cat is on the mat'], ['the cat is on the mat']).estimate.value.toFixed(10)) // => 100
 * ```
 */
export const bleuScore: MetricFunction<[References, Texts, BleuOptions?]> = defineMetric(
  spec('bleuScore', 'higher', [0, 100]),
  (references: References, hypotheses: Texts, options: BleuOptions = {}): MetricResult => {
    const hyps = list(hypotheses)
    return corpusBleu(referenceLists('bleuScore', references, hyps.length), hyps, options)
  },
)

/**
 * Sentence BLEU (sacrebleu.sentence_bleu): BLEU of one hypothesis with exponential smoothing and the effective order
 * by default.
 *
 * @param references - One reference or several.
 * @param hypothesis - The hypothesis.
 * @param options - As {@link bleuScore}; effectiveOrder defaults to true.
 * @returns The score in [0, 100].
 * @example
 * ```ts
 * import { sentenceBleu } from 'whitebox-engine/metrics'
 * Number(sentenceBleu('the cat is on the mat', 'the cat sat on the mat').estimate.value.toFixed(4)) // => 37.9918
 * ```
 */
export const sentenceBleu: MetricFunction<[string | readonly string[], string, BleuOptions?]> = defineMetric(
  spec('sentenceBleu', 'higher', [0, 100]),
  (references: string | readonly string[], hypothesis: string, options: BleuOptions = {}): MetricResult =>
    corpusBleu([typeof references === 'string' ? [references] : references], [hypothesis], {
      effectiveOrder: true,
      ...options,
    }),
)

// ---------------------------------------------------------------------------------------------------------------
// chrF (sacrebleu)

const PUNCTUATION = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

/** Options of {@link chrfScore}. */
export interface ChrfOptions {
  /** Character n-gram order (default 6). */
  readonly charOrder?: number
  /** Word n-gram order (default 0; 2 gives chrF++). */
  readonly wordOrder?: number
  /** Recall weight (default 2). */
  readonly beta?: number
  readonly lowercase?: boolean
  /** Keep whitespace in character n-grams (default false). */
  readonly whitespace?: boolean
  /** Average the per-order F-scores with epsilon smoothing, as the chrF++.py original (default false). */
  readonly epsSmoothing?: boolean
}

/** Split leading or trailing punctuation from words, as chrF++ does for word n-grams. */
function chrfWords(s: string): string[] {
  const out: string[] = []
  for (const w of words(s)) {
    const cs = Array.from(w)
    if (cs.length === 1) out.push(w)
    else if (PUNCTUATION.has(cs[cs.length - 1] ?? '')) out.push(cs.slice(0, -1).join(''), cs[cs.length - 1] ?? '')
    else if (PUNCTUATION.has(cs[0] ?? '')) out.push(cs[0] ?? '', cs.slice(1).join(''))
    else out.push(w)
  }
  return out
}

function chrfNgrams(s: string, options: ChrfOptions): Map<string, number>[] {
  const charOrder = options.charOrder ?? 6
  const chars = Array.from(options.whitespace === true ? s : words(s).join(''))
  const out: Map<string, number>[] = []
  for (let n = 1; n <= charOrder; n++) out.push(ngramCounts(chars, n, ''))
  const wordOrder = options.wordOrder ?? 0
  if (wordOrder > 0) {
    const ws = chrfWords(s)
    for (let n = 1; n <= wordOrder; n++) out.push(ngramCounts(ws, n, ' '))
  }
  return out
}

function chrfF(stats: readonly number[], options: ChrfOptions): number {
  const order = (options.charOrder ?? 6) + (options.wordOrder ?? 0)
  const factor = (options.beta ?? 2) ** 2
  const eps = 1e-16
  let score = 0
  let effective = 0
  let avgP = 0
  let avgR = 0
  for (let i = 0; i < order; i++) {
    const nh = stats[3 * i] ?? 0
    const nr = stats[3 * i + 1] ?? 0
    const nm = stats[3 * i + 2] ?? 0
    const p = nh > 0 ? nm / nh : eps
    const r = nr > 0 ? nm / nr : eps
    const denom = factor * p + r
    score += denom > 0 ? ((1 + factor) * p * r) / denom : eps
    if (nh > 0 && nr > 0) {
      avgP += p
      avgR += r
      effective++
    }
  }
  if (options.epsSmoothing === true) return (100 * score) / order
  if (effective === 0) avgP = avgR = 0
  else {
    avgP /= effective
    avgR /= effective
  }
  return avgP + avgR !== 0 ? (100 * ((1 + factor) * avgP * avgR)) / (factor * avgP + avgR) : 0
}

/**
 * chrF (Popovic, 2015) and chrF++ (Popovic, 2017) as sacrebleu computes them: character (and, for chrF++, word)
 * n-gram precision and recall averaged over orders and combined into an F-beta score on a 0 to 100 scale; statistics
 * are summed over the corpus, taking for each hypothesis the reference with the best sentence score.
 *
 * @param references - The references of each hypothesis.
 * @param hypotheses - The hypotheses.
 * @param options - charOrder, wordOrder (2 for chrF++), beta, lowercase, whitespace, epsSmoothing.
 * @returns The score in [0, 100].
 * @example
 * ```ts
 * import { chrfScore } from 'whitebox-engine/metrics'
 * chrfScore(['abc'], ['abc']).estimate.value // => 100
 * ```
 */
export const chrfScore: MetricFunction<[References, Texts, ChrfOptions?]> = defineMetric(
  spec('chrfScore', 'higher', [0, 100]),
  (references: References, hypotheses: Texts, options: ChrfOptions = {}): MetricResult => {
    const hyps = list(hypotheses)
    const refs = referenceLists('chrfScore', references, hyps.length)
    const prep = (s: string): string => (options.lowercase === true ? s.toLowerCase() : s)
    const order = (options.charOrder ?? 6) + (options.wordOrder ?? 0)
    const sums = new Array<number>(3 * order).fill(0)
    for (const [i, h] of hyps.entries()) {
      const hg = chrfNgrams(prep(h), options)
      let best: number[] = []
      let bestF = -1
      for (const ref of refs[i] ?? []) {
        const rg = chrfNgrams(prep(ref), options)
        const stats = hg.flatMap((c, k) => {
          const r = rg[k] ?? new Map<string, number>()
          return [total(c), total(r), overlap(c, r)]
        })
        const f = chrfF(stats, options)
        if (f > bestF) {
          bestF = f
          best = stats
        }
      }
      for (let k = 0; k < sums.length; k++) sums[k] = (sums[k] ?? 0) + (best[k] ?? 0)
    }
    return metricResult('chrfScore', chrfF(sums, options), hyps.length)
  },
)

// ---------------------------------------------------------------------------------------------------------------
// TER (sacrebleu, Tercom)

/** Options of {@link translationEditRate}: Tercom's normalisation flags. */
export interface TerOptions {
  /** Keep case (default false: lowercase). */
  readonly caseSensitive?: boolean
  /** Tercom's normalisation: entities, punctuation tokenisation (default false). */
  readonly normalized?: boolean
  /** Remove punctuation (default false). */
  readonly noPunct?: boolean
  /** With normalized or noPunct, split CJK characters and remove CJK punctuation (default false). */
  readonly asianSupport?: boolean
}

const ASIAN_PUNCT = /([、。〈-】〔-〟｡-･・])/g
const FULL_WIDTH_PUNCT = /([．，？：；！＂（）])/g

function normaliseWestern(sent: string): string {
  let s = sent.replace(/\n-/g, '').replace(/\n/g, ' ')
  s = s
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
  s = ` ${s} `
  s = s.replace(/([\x7b-\x7e\x5b-\x60\x20-\x26\x28-\x2b\x3a-\x40\x2f])/g, ' $1 ')
  s = s.replace(/'s /g, " 's ").replace(/'s$/, " 's")
  s = s.replace(/([^0-9])([.,])/g, '$1 $2 ').replace(/([.,])([^0-9])/g, ' $1 $2')
  return s.replace(/([0-9])(-)/g, '$1 $2 ')
}

function normaliseAsian(sent: string): string {
  let s = sent.replace(/([一-鿿㐀-䶿])/g, ' $1 ')
  s = s.replace(/([㇀-㇯⺀-⻿])/g, ' $1 ')
  s = s.replace(/([㌀-㏿豈-﫿︰-﹏])/g, ' $1 ')
  s = s.replace(/([㈀-㼢])/g, ' $1 ')
  s = s.replace(/(^|^[぀-ゟ])([぀-ゟ]+)(?=$|^[぀-ゟ])/g, '$1 $2 ')
  s = s.replace(/(^|^[゠-ヿ])([゠-ヿ]+)(?=$|^[゠-ヿ])/g, '$1 $2 ')
  s = s.replace(/(^|^[ㇰ-ㇿ])([ㇰ-ㇿ]+)(?=$|^[ㇰ-ㇿ])/g, '$1 $2 ')
  return s.replace(ASIAN_PUNCT, ' $1 ').replace(FULL_WIDTH_PUNCT, ' $1 ')
}

function tercom(sent: string, options: TerOptions): string[] {
  let s = sent.trimEnd()
  if (s.length === 0) return []
  if (options.caseSensitive !== true) s = s.toLowerCase()
  if (options.normalized === true) {
    s = normaliseWestern(s)
    if (options.asianSupport === true) s = normaliseAsian(s)
  }
  if (options.noPunct === true) {
    s = s.replace(/[.,?:;!"()]/g, '')
    if (options.asianSupport === true) s = s.replace(ASIAN_PUNCT, '').replace(FULL_WIDTH_PUNCT, '')
  }
  return words(s)
}

/**
 * The translation edit rate (Snover et al., 2006) as sacrebleu computes it: Tercom's greedy block shifts plus a
 * beam-limited word edit distance, taking the best reference per hypothesis and dividing the summed edits by the
 * summed average reference lengths, on a 0 to 100 scale.
 *
 * @param references - The references of each hypothesis.
 * @param hypotheses - The hypotheses.
 * @param options - Tercom's caseSensitive, normalized, noPunct and asianSupport flags.
 * @returns The rate, with the edits and reference length in the breakdown.
 * @example
 * ```ts
 * import { translationEditRate } from 'whitebox-engine/metrics'
 * // One shift moves "the cat" to the front.
 * translationEditRate(['the cat sat on the mat'], ['sat on the mat the cat']).estimate.value // => 16.666666666666664
 * ```
 */
export const translationEditRate: MetricFunction<[References, Texts, TerOptions?]> = defineMetric(
  spec('translationEditRate', 'lower', [0, Infinity]),
  (references: References, hypotheses: Texts, options: TerOptions = {}): MetricResult => {
    const hyps = list(hypotheses)
    const refs = referenceLists('translationEditRate', references, hyps.length)
    let edits = 0
    let length = 0
    for (const [i, h] of hyps.entries()) {
      const hw = tercom(h, options)
      const rs = refs[i] ?? []
      let best = 1e16
      let lengths = 0
      for (const r of rs) {
        const rw = tercom(r, options)
        best = Math.min(best, terEdits(hw, rw))
        lengths += rw.length
      }
      edits += best
      length += lengths / rs.length
    }
    const value = length > 0 ? edits / length : edits > 0 ? 1 : 0
    return metricResult('translationEditRate', 100 * value, hyps.length, { edits, referenceLength: length })
  },
)

// ---------------------------------------------------------------------------------------------------------------
// ROUGE (rouge-score)

/** Options of the ROUGE scores. */
export interface RougeOptions {
  /**
   * Stem tokens longer than three characters: `true` applies nltk's Porter stemmer (`text.porterStem`), as
   * rouge-score's `use_stemmer`; a function applies that stemmer instead. Default false.
   */
  readonly stem?: boolean | ((token: string) => string)
}

/** rouge-score's tokeniser: lowercase, non-alphanumerics to spaces, split, optionally stem. */
function rougeTokens(text: string, options: RougeOptions): string[] {
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/)
  const stem = options.stem === true ? (t: string): string => porterStem(t) : options.stem
  const stemmed = typeof stem === 'function' ? tokens.map((t) => (t.length > 3 ? stem(t) : t)) : tokens
  return stemmed.filter((t) => /^[a-z0-9]+$/.test(t))
}

interface Prf {
  readonly precision: number
  readonly recall: number
  readonly fmeasure: number
}

const fmeasure = (p: number, r: number): number => (p + r > 0 ? (2 * p * r) / (p + r) : 0)

function rougeResult(
  name: string,
  references: References,
  hypotheses: Texts,
  score: (ref: string, hyp: string) => Prf,
): MetricResult {
  const hyps = list(hypotheses)
  const refs = referenceLists(name, references, hyps.length)
  let p = 0
  let r = 0
  let f = 0
  for (const [i, h] of hyps.entries()) {
    let best: Prf | undefined
    for (const ref of refs[i] ?? []) {
      const s = score(ref, h)
      if (best === undefined || s.fmeasure > best.fmeasure) best = s
    }
    p += best?.precision ?? 0
    r += best?.recall ?? 0
    f += best?.fmeasure ?? 0
  }
  const n = hyps.length
  return metricResult(name, f / n, n, { precision: p / n, recall: r / n, fmeasure: f / n })
}

/**
 * ROUGE-N (Lin, 2004) as rouge-score computes it: clipped n-gram overlap over the hypothesis (precision) and the
 * reference (recall) n-grams, and their F1; the mean F1 over pairs (the best reference of each).
 *
 * @param references - The references of each hypothesis.
 * @param hypotheses - The hypotheses (summaries).
 * @param options - n (default 1) and an optional stemmer.
 * @returns The mean F-measure, with mean precision and recall in the breakdown.
 * @example
 * ```ts
 * import { rougeN } from 'whitebox-engine/metrics'
 * rougeN(['the cat sat'], ['the cat ran']).estimate.value // => 0.6666666666666666
 * ```
 */
export const rougeN: MetricFunction<[References, Texts, (RougeOptions & { readonly n?: number })?]> = defineMetric(
  spec('rougeN', 'higher', [0, 1]),
  (references: References, hypotheses: Texts, options: RougeOptions & { readonly n?: number } = {}): MetricResult =>
    rougeResult('rougeN', references, hypotheses, (ref, hyp) => {
      const n = options.n ?? 1
      const t = ngramCounts(rougeTokens(ref, options), n)
      const h = ngramCounts(rougeTokens(hyp, options), n)
      const hits = overlap(t, h)
      const precision = hits / Math.max(total(h), 1)
      const recall = hits / Math.max(total(t), 1)
      return { precision, recall, fmeasure: fmeasure(precision, recall) }
    }),
)

/**
 * ROUGE-L (Lin, 2004) as rouge-score computes it: the longest common subsequence of the token sequences over the
 * hypothesis (precision) and reference (recall) lengths, and their F1.
 *
 * @param references - The references of each hypothesis.
 * @param hypotheses - The hypotheses.
 * @param options - An optional stemmer.
 * @returns The mean F-measure, with mean precision and recall in the breakdown.
 * @example
 * ```ts
 * import { rougeL } from 'whitebox-engine/metrics'
 * rougeL(['a b c d'], ['a c d']).breakdown?.['recall'] // => 0.75
 * ```
 */
export const rougeL: MetricFunction<[References, Texts, RougeOptions?]> = defineMetric(
  spec('rougeL', 'higher', [0, 1]),
  (references: References, hypotheses: Texts, options: RougeOptions = {}): MetricResult =>
    rougeResult('rougeL', references, hypotheses, (ref, hyp) => {
      const t = rougeTokens(ref, options)
      const h = rougeTokens(hyp, options)
      if (t.length === 0 || h.length === 0) return { precision: 0, recall: 0, fmeasure: 0 }
      const lcs = lcsTable(t, h)[t.length]?.[h.length] ?? 0
      const precision = lcs / h.length
      const recall = lcs / t.length
      return { precision, recall, fmeasure: fmeasure(precision, recall) }
    }),
)

/** One longest common subsequence as indices into `ref` (rouge-score's _backtrack_norec). */
function lcsIndices(ref: readonly string[], can: readonly string[]): number[] {
  const t = lcsTable(ref, can)
  let i = ref.length
  let j = can.length
  const out: number[] = []
  while (i > 0 && j > 0) {
    if (ref[i - 1] === can[j - 1]) {
      out.unshift(i - 1)
      i--
      j--
    } else if ((t[i]?.[j - 1] ?? 0) > (t[i - 1]?.[j] ?? 0)) j--
    else i--
  }
  return out
}

/**
 * ROUGE-Lsum (summary-level ROUGE-L; Lin, 2004, section 3.2) as rouge-score computes it: texts are split into
 * sentences on newlines, and each reference sentence is matched by the union of its LCSs with the hypothesis
 * sentences, counting each token at most as often as it occurs in both summaries.
 *
 * @param references - The references of each hypothesis, sentences separated by newlines.
 * @param hypotheses - The hypotheses, sentences separated by newlines.
 * @param options - An optional stemmer.
 * @returns The mean F-measure, with mean precision and recall in the breakdown.
 * @example
 * ```ts
 * import { rougeLsum } from 'whitebox-engine/metrics'
 * rougeLsum(['a b\nc d'], ['c d\na b']).estimate.value // => 1
 * ```
 */
export const rougeLsum: MetricFunction<[References, Texts, RougeOptions?]> = defineMetric(
  spec('rougeLsum', 'higher', [0, 1]),
  (references: References, hypotheses: Texts, options: RougeOptions = {}): MetricResult =>
    rougeResult('rougeLsum', references, hypotheses, (ref, hyp) => {
      const sentences = (text: string): string[][] =>
        text
          .split('\n')
          .filter((x) => x.length > 0)
          .map((x) => rougeTokens(x, options))
      const rs = sentences(ref)
      const cs = sentences(hyp)
      const m = rs.reduce((s, x) => s + x.length, 0)
      const n = cs.reduce((s, x) => s + x.length, 0)
      if (rs.length === 0 || cs.length === 0 || m === 0 || n === 0) return { precision: 0, recall: 0, fmeasure: 0 }
      const countsOf = (ss: readonly string[][]): Map<string, number> => {
        const c = new Map<string, number>()
        for (const s of ss) for (const t of s) c.set(t, (c.get(t) ?? 0) + 1)
        return c
      }
      const cr = countsOf(rs)
      const cc = countsOf(cs)
      let hits = 0
      for (const r of rs) {
        const union = [...new Set(cs.flatMap((c) => lcsIndices(r, c)))].sort((a, b) => a - b)
        for (const i of union) {
          const t = r[i] ?? ''
          if ((cc.get(t) ?? 0) > 0 && (cr.get(t) ?? 0) > 0) {
            hits++
            cc.set(t, (cc.get(t) ?? 0) - 1)
            cr.set(t, (cr.get(t) ?? 0) - 1)
          }
        }
      }
      const precision = hits / n
      const recall = hits / m
      return { precision, recall, fmeasure: fmeasure(precision, recall) }
    }),
)

// ---------------------------------------------------------------------------------------------------------------
// Question answering (SQuAD)

/** SQuAD's answer normalisation: lowercase, drop punctuation and articles, collapse whitespace. */
function squadNormalise(s: string): string {
  const noPunct = Array.from(s.toLowerCase())
    .filter((c) => !PUNCTUATION.has(c))
    .join('')
  return words(noPunct.replace(/(?<![\p{L}\p{N}_])(a|an|the)(?![\p{L}\p{N}_])/gu, ' ')).join(' ')
}

function squadF1(prediction: string, truth: string): number {
  const t = truth.length === 0 ? [] : words(squadNormalise(truth))
  const p = prediction.length === 0 ? [] : words(squadNormalise(prediction))
  if (t.length === 0 || p.length === 0) return t.length === p.length ? 1 : 0
  const count = (xs: readonly string[]): Map<string, number> => ngramCounts(xs, 1)
  const same = overlap(count(t), count(p))
  if (same === 0) return 0
  const precision = same / p.length
  const recall = same / t.length
  return (2 * precision * recall) / (precision + recall)
}

function squad(
  name: string,
  references: References,
  predictions: Texts,
  score: (prediction: string, truth: string) => number,
): MetricResult {
  const preds = list(predictions)
  const refs = referenceLists(name, references, preds.length)
  let sum = 0
  for (const [i, p] of preds.entries()) sum += maxOf([0, ...(refs[i] ?? []).map((t) => score(p, t))])
  return metricResult(name, (100 * sum) / preds.length, preds.length)
}

/**
 * Exact match of the SQuAD evaluation (torchmetrics' SQuAD): the percentage of predictions equal to some reference
 * answer after normalisation (lowercase, no punctuation or articles, single spaces).
 *
 * @param references - The reference answers of each question.
 * @param predictions - The predicted answers.
 * @returns The percentage in [0, 100].
 * @example
 * ```ts
 * import { exactMatch } from 'whitebox-engine/metrics'
 * exactMatch([['The Eiffel Tower', 'Eiffel tower']], ['eiffel tower!']).estimate.value // => 100
 * ```
 */
export const exactMatch: MetricFunction<[References, Texts]> = defineMetric(
  spec('exactMatch', 'higher', [0, 100]),
  (references: References, predictions: Texts): MetricResult =>
    squad('exactMatch', references, predictions, (p, t) => (squadNormalise(p) === squadNormalise(t) ? 1 : 0)),
)

/**
 * Token F1 of the SQuAD evaluation (torchmetrics' SQuAD): the F1 of the bag-of-tokens overlap of the normalised
 * prediction and reference, the best over references, averaged and given as a percentage.
 *
 * @param references - The reference answers of each question.
 * @param predictions - The predicted answers.
 * @returns The percentage in [0, 100].
 * @example
 * ```ts
 * import { tokenF1 } from 'whitebox-engine/metrics'
 * tokenF1(['the big red dog'], ['a red dog']).estimate.value // => 80
 * ```
 */
export const tokenF1: MetricFunction<[References, Texts]> = defineMetric(
  spec('tokenF1', 'higher', [0, 100]),
  (references: References, predictions: Texts): MetricResult => squad('tokenF1', references, predictions, squadF1),
)

// ---------------------------------------------------------------------------------------------------------------
// Embedding and model-based scores

/** Options of {@link bertScore}. */
export interface BertScoreOptions {
  /** Token importance weights (e.g. idf) of each reference and candidate; uniform when absent. */
  readonly targetWeights?: readonly TensorLike[]
  readonly predWeights?: readonly TensorLike[]
  /** Baseline (precision, recall, F1) for rescaling (s - b) / (1 - b), as bert-score's rescale_with_baseline. */
  readonly baseline?: readonly [number, number, number]
}

/**
 * BERTScore (Zhang et al., 2020) from token embeddings, as torchmetrics computes it once embeddings are extracted:
 * after normalising each token embedding, precision is the weighted mean over candidate tokens of the best cosine
 * similarity to a reference token, recall the same from the reference side, F1 their harmonic mean. whitebox ships no
 * BERT; pass the contextual embeddings of each sentence's tokens.
 *
 * @param targetEmbeddings - The reference token embeddings of each sentence, `[R_i, D]`.
 * @param predEmbeddings - The candidate token embeddings of each sentence, `[P_i, D]`.
 * @param options - Token weights and an optional baseline.
 * @returns The mean F1, with mean precision and recall in the breakdown.
 * @example
 * ```ts
 * import { bertScore } from 'whitebox-engine/metrics'
 * bertScore([[[1, 0], [0, 1]]], [[[1, 0]]]).breakdown?.['precision'] // => 1
 * ```
 */
export const bertScore: MetricFunction<[readonly TensorLike[], readonly TensorLike[], BertScoreOptions?]> =
  defineMetric(
    spec('bertScore', 'higher', [-1, 1], 'values'),
    (
      targetEmbeddings: readonly TensorLike[],
      predEmbeddings: readonly TensorLike[],
      options: BertScoreOptions = {},
    ): MetricResult => {
      sameLength('bertScore', targetEmbeddings, predEmbeddings)
      const unit = (x: TensorLike): { n: number; d: number; data: Float64Array } => {
        const m = matrix(x)
        if (m.shape.length !== 2) throw new WhiteboxError('shape', 'bertScore: embeddings are [tokens, D]')
        const n = m.shape[0] ?? 0
        const d = m.shape[1] ?? 0
        const data = Float64Array.from(m.data)
        for (let i = 0; i < n; i++) {
          let s = 0
          for (let k = 0; k < d; k++) s += (data[i * d + k] ?? 0) ** 2
          const norm = Math.sqrt(s)
          for (let k = 0; k < d; k++) data[i * d + k] = (data[i * d + k] ?? 0) / norm
        }
        return { n, d, data }
      }
      const weights = (w: TensorLike | undefined, n: number): Float64Array => {
        const v = w === undefined ? new Float64Array(n).fill(1) : matrix(w).data
        const s = v.reduce((a, b) => a + b, 0)
        return Float64Array.from(v, (x) => x / s)
      }
      const [bp, br, bf] = options.baseline ?? [0, 0, 0]
      let sp = 0
      let sr = 0
      let sf = 0
      for (const [i, te] of targetEmbeddings.entries()) {
        const t = unit(te)
        const p = unit(predEmbeddings[i] ?? [])
        const wt = weights(options.targetWeights?.[i], t.n)
        const wp = weights(options.predWeights?.[i], p.n)
        const bestP = new Float64Array(p.n).fill(Number.NEGATIVE_INFINITY)
        const bestR = new Float64Array(t.n).fill(Number.NEGATIVE_INFINITY)
        for (let a = 0; a < p.n; a++) {
          for (let b = 0; b < t.n; b++) {
            let dot = 0
            for (let k = 0; k < p.d; k++) dot += (p.data[a * p.d + k] ?? 0) * (t.data[b * t.d + k] ?? 0)
            if (dot > (bestP[a] ?? 0)) bestP[a] = dot
            if (dot > (bestR[b] ?? 0)) bestR[b] = dot
          }
        }
        let precision = bestP.reduce((s, v, a) => s + v * (wp[a] ?? 0), 0)
        let recall = bestR.reduce((s, v, b) => s + v * (wt[b] ?? 0), 0)
        const f = (2 * precision * recall) / (precision + recall)
        let f1 = Number.isNaN(f) ? 0 : f
        if (options.baseline !== undefined) {
          precision = (precision - bp) / (1 - bp)
          recall = (recall - br) / (1 - br)
          f1 = (f1 - bf) / (1 - bf)
        }
        sp += precision
        sr += recall
        sf += f1
      }
      const n = targetEmbeddings.length
      return metricResult('bertScore', sf / n, n, { precision: sp / n, recall: sr / n, f1: sf / n })
    },
  )

/**
 * Perplexity exp(-(1/N) sum log p(target)) of next-token logits (torchmetrics.text.Perplexity), with tokens equal to
 * `ignoreIndex` left out.
 *
 * @param logits - Unnormalised scores `[B, T, V]` (or `[T, V]`).
 * @param targets - Target token ids `[B, T]` (or `[T]`).
 * @param options - ignoreIndex.
 * @returns The metric result.
 * @example
 * ```ts
 * import { perplexity } from 'whitebox-engine/metrics'
 * perplexity([[0, 0], [0, 0]], [0, 1]).estimate.value // => 2
 * ```
 */
export const perplexity: MetricFunction<[TensorLike, TensorLike, { readonly ignoreIndex?: number }?]> = defineMetric(
  spec('perplexity', 'lower', [1, Infinity], 'scores'),
  (logits: TensorLike, targets: TensorLike, options: { readonly ignoreIndex?: number } = {}): MetricResult => {
    const l = matrix(logits)
    const t = matrix(targets).data
    const v = l.shape[l.shape.length - 1] ?? 0
    if (l.data.length !== t.length * v)
      throw new WhiteboxError('shape', 'perplexity: logits are [..., T, V] for targets [..., T]')
    let nll = 0
    let count = 0
    for (const [i, target] of t.entries()) {
      if (options.ignoreIndex !== undefined && target === options.ignoreIndex) continue
      let max = Number.NEGATIVE_INFINITY
      for (let k = 0; k < v; k++) max = Math.max(max, l.data[i * v + k] ?? 0)
      let s = 0
      for (let k = 0; k < v; k++) s += Math.exp((l.data[i * v + k] ?? 0) - max)
      nll -= (l.data[i * v + target] ?? 0) - max - Math.log(s)
      count++
    }
    return metricResult('perplexity', Math.exp(nll / count), count)
  },
)

/**
 * Distinct-n (Li et al., 2016): the number of distinct n-grams over the total number of n-grams in a corpus of
 * generated texts, a measure of lexical diversity. Tokens are whitespace-separated unless a tokeniser is given.
 *
 * @param texts - Generated texts.
 * @param options - n (default 1) and an optional tokeniser.
 * @returns The ratio in [0, 1].
 * @example
 * ```ts
 * import { distinctN } from 'whitebox-engine/metrics'
 * distinctN(['a b a b'], { n: 2 }).estimate.value // => 0.6666666666666666
 * ```
 */
export const distinctN: MetricFunction<
  [Texts, { readonly n?: number; readonly tokenize?: (text: string) => readonly string[] }?]
> = defineMetric(
  spec('distinctN', 'higher', [0, 1]),
  (
    texts: Texts,
    options: { readonly n?: number; readonly tokenize?: (text: string) => readonly string[] } = {},
  ): MetricResult => {
    const n = options.n ?? 1
    const all = new Map<string, number>()
    for (const text of list(texts)) {
      const tokens = options.tokenize === undefined ? words(text) : options.tokenize(text)
      for (const [k, c] of ngramCounts(tokens, n)) all.set(k, (all.get(k) ?? 0) + c)
    }
    const count = total(all)
    return metricResult('distinctN', count === 0 ? 0 : all.size / count, count)
  },
)
