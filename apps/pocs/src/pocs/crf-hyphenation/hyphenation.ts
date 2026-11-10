import { trainTestSplit } from 'omai-engine/data'
import {
  hyphenatedWord,
  hyphenationLabels,
  hyphenationPatternsEnUs,
  hyphenationTables,
  hyphenationWords,
} from 'omai-engine/datasets'
import { f1Score, precisionScore, recallScore, sequenceAccuracyScore } from 'omai-engine/metrics'
import { attributeExtractor, type CrfFitted, type FeatureExtractor } from 'omai-engine/pgm'
import { key } from 'omai-engine/random'
import type { TokenTable } from 'omai-engine/text'
import { hyphenate } from 'omai-engine/text'

/**
 * What the page computes around the library's CRF estimator: the split of the word list, the templates the controls
 * choose, how a hyphenation is scored, and the reading of a fitted model (which templates survive, which features
 * decided one gap). Anything with substance (the estimator, the patterns, the metrics) is the library's.
 */

/** The words of one part of the split, as token tables with their label strings and the word itself. */
export interface Words {
  readonly words: readonly string[]
  readonly tables: readonly TokenTable[]
  readonly labels: readonly (readonly string[])[]
}

const TEST_WORDS = 400

let split: { readonly train: Words; readonly test: Words } | undefined

const wordsOf = (tokens: readonly (readonly string[])[], tags: readonly (readonly number[])[]): Words => ({
  words: tokens.map((t) => t.join('')),
  tables: hyphenationTables(tokens),
  labels: hyphenationLabels(tags),
})

/**
 * The word list split once, shuffled by a fixed key: 400 words held out for testing and the rest to train on (the page
 * takes the first `n` of them). The same split on every page load, so the numbers shown are reproducible.
 *
 * @returns The training and the held-out words.
 */
export function wordSplit(): { readonly train: Words; readonly test: Words } {
  if (split === undefined) {
    const parts = trainTestSplit(hyphenationWords(), { testSize: TEST_WORDS, key: key(7) })
    split = {
      train: wordsOf(parts.train.tokens, parts.train.tags),
      test: wordsOf(parts.test.tokens, parts.test.tags),
    }
  }
  return split
}

/** The first `n` words of a part (their tables and labels). */
export function firstWords(part: Words, n: number): Words {
  return { words: part.words.slice(0, n), tables: part.tables.slice(0, n), labels: part.labels.slice(0, n) }
}

/** The settings of the template window. */
export interface Window {
  /** Letters looked at on each side of the gap-letter: 1 to 3. */
  readonly radius: number
  /** Whether the vowel/consonant classes of the neighbouring letters are templates too. */
  readonly classes: boolean
}

/** The class template over a run of letters: `k[-2]|k[-1]|k[0]|k[1]|k[2]` for a reach of 2. */
const classRun = (reach: number): string => Array.from({ length: 2 * reach + 1 }, (_, i) => `k[${i - reach}]`).join('|')

/**
 * The CRFsuite attribute templates for a window of letters around the one being labelled (`c` is the letter, `k` its
 * class `V` or `C`): every letter in the window, every pair of neighbours in it, triples when the window is wide
 * enough, and with `classes` the patterns of vowels and consonants.
 *
 * @param w - The window.
 * @returns The templates, one per line.
 */
export function templateLines(w: Window): string[] {
  const r = w.radius
  const range = Array.from({ length: 2 * r + 1 }, (_, i) => i - r)
  const lines = range.map((i) => `c[${i}]`)
  for (let i = -r; i < r; i++) lines.push(`c[${i}]|c[${i + 1}]`)
  if (r >= 2) for (let i = -r; i < r - 1; i++) lines.push(`c[${i}]|c[${i + 1}]|c[${i + 2}]`)
  if (w.classes) {
    lines.push('k[-1]|k[0]|k[1]')
    for (const reach of [2, 3]) if (r >= reach) lines.push(classRun(reach))
  }
  return lines
}

/** The feature extractor of a window: its templates, the letter and class columns of `hyphenationTables`, and a bias. */
export function extractorOf(w: Window): FeatureExtractor<TokenTable> {
  return attributeExtractor(templateLines(w).join('\n'), {
    columns: ['c', 'k'],
    bias: true,
  }) as FeatureExtractor<TokenTable>
}

let listed: Map<string, readonly string[]> | undefined

/** The dictionary's labels of every word of the list, by word. */
function dictionary(): Map<string, readonly string[]> {
  if (listed === undefined) {
    const { train, test } = wordSplit()
    const words = [...train.words, ...test.words]
    const labels = [...train.labels, ...test.labels]
    listed = new Map(words.map((w, i) => [w, labels[i] ?? []]))
  }
  return listed
}

let patterns: ReturnType<typeof hyphenationPatternsEnUs> | undefined

/**
 * The labels TeX's US patterns give a word, as `O`/`B` per letter: patterns may break anywhere (minimum fragments of
 * one letter, so the baseline is not handicapped by TeX's two-and-three-letter rule, which the dictionary does not
 * follow).
 *
 * @param word - The word.
 * @returns A label per letter.
 */
export function patternLabels(word: string): string[] {
  patterns ??= hyphenationPatternsEnUs()
  const { points } = hyphenate(word, patterns, { leftMin: 1, rightMin: 1 })
  const out = [...word].map(() => 'O')
  for (const p of points) if (p >= 1 && p < out.length) out[p - 1] = 'B'
  return out
}

/** How a hyphenator did on a list of words. */
export interface Scores {
  /** Share of words hyphenated entirely right. */
  readonly words: number
  /** Of the hyphens it put, the share that are in the dictionary. */
  readonly precision: number
  /** Of the dictionary's hyphens, the share it put. */
  readonly recall: number
  readonly f1: number
}

/**
 * Word accuracy and the precision, recall and F1 of the hyphens (the `B` labels) of a hyphenator, by the library's metrics.
 *
 * @param truth - The dictionary labels of each word.
 * @param predicted - The hyphenator's labels.
 * @returns The scores.
 */
export function scoreLabels(truth: readonly (readonly string[])[], predicted: readonly (readonly string[])[]): Scores {
  const a = truth.flat()
  const b = predicted.flat()
  const opts = { posLabel: 'B' }
  return {
    words: sequenceAccuracyScore(truth, predicted).estimate.value,
    precision: precisionScore(a, b, opts).estimate.value,
    recall: recallScore(a, b, opts).estimate.value,
    f1: f1Score(a, b, opts).estimate.value,
  }
}

/** One template of the extractor and what the fit left of it. */
export interface TemplateSurvival {
  readonly template: string
  /** Its features in the model's vocabulary. */
  readonly features: number
  /** Those with a nonzero weight. */
  readonly nonzero: number
}

/**
 * Which templates survive in a fitted model: for each one, how many of its features have a nonzero weight for some
 * label. With an L1 penalty (OWL-QN) the weights are exactly zero, so a template can vanish altogether.
 *
 * @param fitted - The fitted CRF.
 * @param templates - The template lines the extractor was built from (the bias is added as its own row).
 * @returns The templates in their order.
 */
export function templateSurvival(fitted: CrfFitted<TokenTable>, templates: readonly string[]): TemplateSurvival[] {
  const names = ['bias', ...templates]
  const total = new Map(names.map((n) => [n, 0]))
  const alive = new Map(names.map((n) => [n, 0]))
  const nonzero = fitted.stateFeatures()
  for (const a of fitted.model.attributes) {
    const t = a.includes('=') ? a.slice(0, a.indexOf('=')) : a
    total.set(t, (total.get(t) ?? 0) + 1)
    if (nonzero[a] !== undefined) alive.set(t, (alive.get(t) ?? 0) + 1)
  }
  return names.map((template) => ({
    template,
    features: total.get(template) ?? 0,
    nonzero: alive.get(template) ?? 0,
  }))
}

/** One feature that fired at a letter, and what it pushes towards. */
export interface Evidence {
  readonly template: string
  readonly feature: string
  /** The log-odds it adds for a hyphen after the letter: the weight for `B` minus the weight for `O`. */
  readonly push: number
}

/**
 * The features that decided one gap: those that fire at the letter before it, with the log-odds each adds for a hyphen
 * (the weight for `B` minus that for `O`), the strongest first. Transitions between neighbouring labels come on top of
 * these in the CRF's decision.
 *
 * @param fitted - The fitted CRF.
 * @param extractor - Its extractor.
 * @param table - The word's token table.
 * @param letter - The index of the letter before the gap.
 * @param top - How many to keep.
 * @returns The evidence, by the size of its push.
 */
export function gapEvidence(
  fitted: CrfFitted<TokenTable>,
  extractor: FeatureExtractor<TokenTable>,
  table: TokenTable,
  letter: number,
  top: number,
): Evidence[] {
  const weights = fitted.stateFeatures()
  const found: Evidence[] = []
  for (const o of extractor.explain(table, letter)) {
    const w = weights[o.feature]
    const push = ((w?.['B'] ?? 0) - (w?.['O'] ?? 0)) * o.value
    if (push !== 0) found.push({ template: o.template, feature: o.feature, push })
  }
  return found.sort((x, y) => Math.abs(y.push) - Math.abs(x.push)).slice(0, top)
}

/** What the word box shows of a word. */
export interface WordReading {
  readonly letters: readonly string[]
  /** The CRF's probability of a hyphen after each letter but the last (marginals). */
  readonly probability: readonly number[]
  readonly crf: string
  readonly tex: string
  /** The dictionary's hyphenation, when the word is in the list. */
  readonly dictionary: string | undefined
}

/**
 * The reading of one word: the CRF's hyphen probability in each gap, its hyphenation and the TeX patterns', and the
 * dictionary's when it knows the word.
 *
 * @param fitted - The fitted CRF.
 * @param word - The lowercase word.
 * @returns The reading.
 */
export function readWord(fitted: CrfFitted<TokenTable>, word: string): WordReading {
  const letters = [...word]
  const table = hyphenationTables([word])[0] as TokenTable
  const marginals = fitted.predictMarginalsSingle(table)
  const known = dictionary().get(word)
  return {
    letters,
    probability: marginals.slice(0, -1).map((m) => m['B'] ?? 0),
    crf: hyphenatedWord(letters, fitted.predictSingle(table)),
    tex: hyphenatedWord(letters, patternLabels(word)),
    dictionary: known === undefined ? undefined : hyphenatedWord(letters, known),
  }
}
