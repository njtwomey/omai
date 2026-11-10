/**
 * English word hyphenation as sequence labelling, the classic structured-prediction exercise (Trogkanis and Elkan
 * 2010): a sample of the Moby Hyphenator word list with one tag per letter (a hyphen follows it or not), token tables of
 * letter columns for the CRF template languages, and the TeX patterns of American English as the baseline Liang
 * hyphenator.
 */

import { type SequenceLabellingData, sequenceLabelling } from '../data/index.ts'
import { type HyphenationPatterns, hyphenationPatterns, type TokenTable } from '../text/index.ts'
import { HYPHENATION_EXCEPTIONS_EN_US, HYPHENATION_PATTERNS_EN_US } from './_data/hyphenation-patterns.data.ts'
import {
  HYPHENATION_ELIGIBLE,
  HYPHENATION_SHA256,
  HYPHENATION_STRIDE,
  HYPHENATION_WORDS,
} from './_data/hyphenation-words.data.ts'

/** The tags of {@link hyphenationWords}: `O` (no hyphen after the letter) is 0 and `B` (a hyphen follows) is 1. */
export const HYPHENATION_TAG_NAMES: readonly string[] = ['O', 'B']

/**
 * A deterministic sample of the Moby Hyphenator word list (Grady Ward, Project Gutenberg eBook 3204, public domain):
 * 3984 English words of 4 to 14 lowercase letters with their dictionary hyphenation points, one sentence per word whose
 * tokens are its letters and whose tags say whether a hyphen follows the letter (`B`) or not (`O`; always `O` for the
 * last letter). The sample is every 32nd of the 127482 eligible entries (a word listed with two hyphenations is left
 * out), in the list's alphabetical order; split it with `trainTestSplit` and a key. The list's hyphenation is that of a
 * dictionary (syllable-based, as Webster), not TeX's typesetting hyphenation.
 *
 * @returns The dataset.
 * @index dataset English words with their hyphenation points, one tag per letter (Moby Hyphenator sample).
 * @concept datasets
 * @example
 * ```ts
 * import { hyphenationWords } from 'omai-engine/datasets'
 * const d = hyphenationWords()
 * [d.tokens[2]?.join(''), d.tags[2]?.join('')] // => ['abirritative', '010101010000']
 * ```
 */
export function hyphenationWords(): SequenceLabellingData {
  const tokens: string[][] = []
  const tags: number[][] = []
  for (const entry of HYPHENATION_WORDS) {
    const parts = entry.split('-')
    const letters = parts.join('')
    tokens.push([...letters])
    tags.push([...letters].map((_, i) => (isBreakAfter(parts, i) ? 1 : 0)))
  }
  return sequenceLabelling(tokens, tags, HYPHENATION_TAG_NAMES, {
    name: 'hyphenationWords',
    description: `A sample of ${HYPHENATION_WORDS.length} English words (every ${HYPHENATION_STRIDE}th of ${HYPHENATION_ELIGIBLE}) with their hyphenation points, one tag per letter.`,
    source: 'Moby Hyphenator II, Grady Ward (Project Gutenberg eBook 3204, mhyph.txt)',
    url: 'https://www.gutenberg.org/ebooks/3204',
    citation:
      'Ward (2002), Moby Hyphenator; task as in Trogkanis and Elkan (2010), Conditional random fields for word hyphenation.',
    licence: 'Public domain (by grant of the author)',
    sha256: HYPHENATION_SHA256,
  })
}

function isBreakAfter(parts: readonly string[], letter: number): boolean {
  let end = 0
  for (const p of parts.slice(0, -1)) {
    end += p.length
    if (end - 1 === letter) return true
  }
  return false
}

const VOWELS = new Set([...'aeiouy'])

/**
 * Token tables of words for the CRF template languages: one row per letter with two columns, the letter (column 0) and
 * its class `V` (a, e, i, o, u, y) or `C` (column 1), so `%x[-1,0]/%x[0,0]` is a letter bigram and `%x[0,1]` the class.
 *
 * @param words - The words, as strings or as letter arrays.
 * @returns One table per word.
 * @concept crf-features
 * @example
 * ```ts
 * import { hyphenationTables } from 'omai-engine/datasets'
 * hyphenationTables(['ace']) // => [[['a', 'V'], ['c', 'C'], ['e', 'V']]]
 * ```
 */
export function hyphenationTables(words: readonly (string | readonly string[])[]): TokenTable[] {
  return words.map((w) => [...w].map((c) => [c, VOWELS.has(c) ? 'V' : 'C']))
}

/**
 * The labels of a word's letters as strings for a CRF (`O` and `B`), from the tag ids of {@link hyphenationWords}.
 *
 * @param tags - The tag ids of each word.
 * @returns The label strings of each word.
 * @example
 * ```ts
 * import { hyphenationLabels } from 'omai-engine/datasets'
 * hyphenationLabels([[0, 1, 0]]) // => [['O', 'B', 'O']]
 * ```
 */
export function hyphenationLabels(tags: readonly (readonly number[])[]): string[][] {
  return tags.map((t) => t.map((k) => HYPHENATION_TAG_NAMES[k] ?? 'O'))
}

/**
 * A word written with hyphens at the letters labelled `B` (or tagged 1).
 *
 * @param letters - The letters.
 * @param labels - Their labels: `B` or 1 where a hyphen follows (the last letter's is ignored).
 * @returns The hyphenated word.
 * @example
 * ```ts
 * import { hyphenatedWord } from 'omai-engine/datasets'
 * hyphenatedWord([...'table'], ['O', 'B', 'O', 'O', 'O']) // => 'ta-ble'
 * ```
 */
export function hyphenatedWord(letters: readonly string[], labels: readonly (string | number)[]): string {
  return letters
    .map((c, i) => (i < letters.length - 1 && (labels[i] === 'B' || labels[i] === 1) ? `${c}-` : c))
    .join('')
}

/**
 * The TeX patterns for American English (`hyph-en-us.tex` of hyph-utf8: Liang's patterns with Kuiken's additions and
 * Knuth's exceptions) compiled for `hyphenate`, the baseline against which a learned hyphenator is measured. Copyright
 * (C) 1990, 2004, 2005 Gerard D.C. Kuiken; copying and distribution are permitted in any medium without royalty provided
 * the copyright notice and this notice are preserved. The patterns were built from a different dictionary than the Moby
 * list, and the exception log behind them overlaps with it, so the comparison is not a held-out one for the patterns.
 *
 * @returns The compiled patterns.
 * @concept hyphenation
 * @example
 * ```ts
 * import { hyphenationPatternsEnUs } from 'omai-engine/datasets'
 * import { insertHyphens } from 'omai-engine/text'
 * insertHyphens('hyphenation', hyphenationPatternsEnUs()) // => 'hy-phen-ation'
 * ```
 */
export function hyphenationPatternsEnUs(): HyphenationPatterns {
  return hyphenationPatterns(HYPHENATION_PATTERNS_EN_US, HYPHENATION_EXCEPTIONS_EN_US)
}
