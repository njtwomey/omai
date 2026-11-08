/**
 * Text normalisation: Unicode normal forms (NFC, NFD, NFKC, NFKD), full case folding, accent stripping, whitespace
 * collapsing and a pipeline combining them. Also aligned text, which keeps for every normalised character the range of
 * the original it came from, with the mapping, filtering, replacement and slicing operations that tokenizers need to
 * report offsets.
 */

import { OmaiError } from '../base/index.ts'
import { CASE_FOLDING_EXCEPTIONS, COMBINING_RANGES } from './_unicode.data.ts'

/** A Unicode normalisation form (UAX #15). */
export type NormalForm = 'NFC' | 'NFD' | 'NFKC' | 'NFKD'

/**
 * Unicode normalisation (Python's unicodedata.normalize, String.prototype.normalize): canonical (NFC, NFD) or
 * compatibility (NFKC, NFKD) composition or decomposition.
 *
 * @param text - The text.
 * @param form - The normal form (default NFC).
 * @returns The normalised text.
 * @throws OmaiError with code `value` for an unknown form.
 * @concept unicode-normalisation
 * @example
 * ```ts
 * import { normalizeUnicode } from 'omai-engine/text'
 * normalizeUnicode('é') === 'é' // => true
 * normalizeUnicode('ﬁ', 'NFKC') // => 'fi'
 * ```
 */
export function normalizeUnicode(text: string, form: NormalForm = 'NFC'): string {
  if (form !== 'NFC' && form !== 'NFD' && form !== 'NFKC' && form !== 'NFKD')
    throw new OmaiError('value', `normalizeUnicode: unknown form ${String(form)}`)
  return text.normalize(form)
}

let foldingTable: Map<number, string> | undefined
function folding(): Map<number, string> {
  foldingTable ??= new Map(CASE_FOLDING_EXCEPTIONS)
  return foldingTable
}

/**
 * Full Unicode case folding (Python's str.casefold): lowercase each code point independently, except where the full
 * folding differs (ß to ss, final sigma to sigma, ligatures expanded, Cherokee to upper case).
 *
 * @param text - The text.
 * @returns The folded text, for caseless matching.
 * @concept case-folding
 * @example
 * ```ts
 * import { caseFold } from 'omai-engine/text'
 * caseFold('Straße') // => 'strasse'
 * caseFold('ΌΣΟΣ') === caseFold('όσος') // => true
 * ```
 */
export function caseFold(text: string): string {
  const table = folding()
  let out = ''
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0
    out += table.get(cp) ?? ch.toLowerCase()
  }
  return out
}

/**
 * Whether a code point has a non-zero canonical combining class (Python's unicodedata.combining), the test that
 * scikit-learn's accent stripping uses.
 *
 * @param codePoint - The code point.
 * @returns True for a combining character.
 * @example
 * ```ts
 * import { isCombining } from 'omai-engine/text'
 * [isCombining(0x301), isCombining(0x61)] // => [true, false]
 * ```
 */
export function isCombining(codePoint: number): boolean {
  let lo = 0
  let hi = COMBINING_RANGES.length / 2 - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const start = COMBINING_RANGES[2 * mid] ?? 0
    const end = COMBINING_RANGES[2 * mid + 1] ?? 0
    if (codePoint < start) hi = mid - 1
    else if (codePoint > end) lo = mid + 1
    else return true
  }
  return false
}

const NONSPACING = /\p{Mn}/u

function isAscii(text: string): boolean {
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) > 127) return false
  return true
}

/** How {@link stripAccents} removes accents. */
export type AccentStripping = 'unicode' | 'ascii' | 'nonspacing'

/**
 * Remove accents. `unicode` (default) is scikit-learn's `strip_accents_unicode`: NFKD, then drop characters with a
 * non-zero combining class (ASCII text is returned unchanged). `ascii` is
 * `strip_accents_ascii`: NFKD, then drop every non-ASCII character. `nonspacing` is BERT's (HF tokenizers'
 * BertNormalizer): NFD, then drop non-spacing marks (category Mn).
 *
 * @param text - The text.
 * @param mode - The convention.
 * @returns The text without accents.
 * @concept unicode-normalisation
 * @example
 * ```ts
 * import { stripAccents } from 'omai-engine/text'
 * stripAccents('café naïve') // => 'cafe naive'
 * stripAccents('Ωmega ½', 'ascii') // => 'mega 12'
 * ```
 */
export function stripAccents(text: string, mode: AccentStripping = 'unicode'): string {
  if (mode === 'nonspacing') {
    let out = ''
    for (const ch of text.normalize('NFD')) if (!NONSPACING.test(ch)) out += ch
    return out
  }
  const nfkd = text.normalize('NFKD')
  if (mode === 'ascii') {
    let out = ''
    for (const ch of nfkd) if ((ch.codePointAt(0) ?? 0) < 128) out += ch
    return out
  }
  if (mode !== 'unicode') throw new OmaiError('value', `stripAccents: unknown mode ${String(mode)}`)
  if (isAscii(text)) return text
  let out = ''
  for (const ch of nfkd) if (!isCombining(ch.codePointAt(0) ?? 0)) out += ch
  return out
}

/** Unicode White_Space (Python's and Rust's notion of whitespace; JavaScript's `\s` differs on U+0085 and U+FEFF). */
export const WHITESPACE: RegExp = /[\p{White_Space}]/u

/**
 * Trim and collapse every run of Unicode whitespace to one space.
 *
 * @param text - The text.
 * @returns The collapsed text.
 * @example
 * ```ts
 * import { collapseWhitespace } from 'omai-engine/text'
 * collapseWhitespace('  a \t b\n\nc ') // => 'a b c'
 * ```
 */
export function collapseWhitespace(text: string): string {
  return text
    .replace(/[\p{White_Space}]+/gu, ' ')
    .replace(/^ /, '')
    .replace(/ $/, '')
}

/** Options of {@link normaliseText}, applied in this order. */
export interface NormaliseOptions {
  /** A Unicode normal form. */
  readonly form?: NormalForm
  /** Accent stripping (see {@link stripAccents}). */
  readonly stripAccents?: AccentStripping
  /** Lowercase (per code point, as HF tokenizers). */
  readonly lowercase?: boolean
  /** Full case folding instead of lowercasing. */
  readonly caseFold?: boolean
  /** Trim and collapse whitespace. */
  readonly collapseWhitespace?: boolean
}

/**
 * A normalisation pipeline: normal form, accent stripping, lowercasing or case folding, whitespace collapsing.
 *
 * @param text - The text.
 * @param options - The steps.
 * @returns The normalised text.
 * @example
 * ```ts
 * import { normaliseText } from 'omai-engine/text'
 * normaliseText(' Crème  BRÛLÉE ', { stripAccents: 'unicode', lowercase: true, collapseWhitespace: true }) // => 'creme brulee'
 * ```
 */
export function normaliseText(text: string, options: NormaliseOptions = {}): string {
  let s = options.form === undefined ? text : normalizeUnicode(text, options.form)
  if (options.stripAccents !== undefined) s = stripAccents(s, options.stripAccents)
  if (options.caseFold === true) s = caseFold(s)
  else if (options.lowercase === true) s = lowercaseCodePoints(s)
  if (options.collapseWhitespace === true) s = collapseWhitespace(s)
  return s
}

/** Lowercase each code point on its own (no final-sigma context, as Rust's char::to_lowercase and HF tokenizers). */
function lowercaseCodePoints(text: string): string {
  let out = ''
  for (const ch of text) out += ch.toLowerCase()
  return out
}

// ---------------------------------------------------------------------------------------------------------------
// Aligned text: normalised text that remembers where each character came from (HF tokenizers' NormalizedString).

/**
 * Normalised text with alignments back to the original: for each UTF-16 code unit of `text`, the `[start, end)` range
 * of `original` (in UTF-16 code units) it came from. Offsets of tokens in the normalised text map back to the original
 * with {@link originalSpan}, as HF tokenizers' `NormalizedString` does.
 */
export interface AlignedText {
  readonly original: string
  readonly text: string
  readonly alignments: readonly (readonly [number, number])[]
}

/** The code points of a string with their UTF-16 starting offsets. */
function codePoints(text: string): { readonly ch: string; readonly at: number }[] {
  const out: { ch: string; at: number }[] = []
  let at = 0
  for (const ch of text) {
    out.push({ ch, at })
    at += ch.length
  }
  return out
}

/**
 * Aligned text that is its own original (each character aligned to itself).
 *
 * @param text - The original text.
 * @returns The aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText } from 'omai-engine/text'
 * alignedText('ab').alignments // => [[0, 1], [1, 2]]
 * ```
 */
export function alignedText(text: string): AlignedText {
  const alignments: [number, number][] = []
  for (const { ch, at } of codePoints(text)) for (let k = 0; k < ch.length; k++) alignments.push([at, at + ch.length])
  return { original: text, text, alignments }
}

/** The alignment of a code point starting at `at` in an aligned text. */
function alignmentAt(a: AlignedText, at: number, length: number): [number, number] {
  const first = a.alignments[at] ?? [0, 0]
  const last = a.alignments[at + length - 1] ?? first
  return [first[0], last[1]]
}

/**
 * Map each code point to a replacement string (empty to delete); every character of a replacement is aligned to the
 * code point it replaces (HF tokenizers' per-character transforms: lowercasing, accent removal, byte mapping).
 *
 * @param a - The aligned text.
 * @param f - The mapping of one code point.
 * @returns The new aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, mapAligned } from 'omai-engine/text'
 * const r = mapAligned(alignedText('Aß'), (c) => (c === 'ß' ? 'ss' : c.toLowerCase()))
 * [r.text, r.alignments] // => ['ass', [[0, 1], [1, 2], [1, 2]]]
 * ```
 */
export function mapAligned(a: AlignedText, f: (codePoint: string) => string): AlignedText {
  let text = ''
  const alignments: (readonly [number, number])[] = []
  for (const { ch, at } of codePoints(a.text)) {
    const r = f(ch)
    const range = alignmentAt(a, at, ch.length)
    text += r
    for (let k = 0; k < r.length; k++) alignments.push(range)
  }
  return { original: a.original, text, alignments }
}

/**
 * Keep the code points for which a predicate holds.
 *
 * @param a - The aligned text.
 * @param keep - The predicate on one code point.
 * @returns The filtered aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, filterAligned } from 'omai-engine/text'
 * filterAligned(alignedText('a-b'), (c) => c !== '-').alignments // => [[0, 1], [2, 3]]
 * ```
 */
export function filterAligned(a: AlignedText, keep: (codePoint: string) => boolean): AlignedText {
  return mapAligned(a, (c) => (keep(c) ? c : ''))
}

/**
 * Unicode normalisation of aligned text. The text is cut into pieces before each starter (a character of combining
 * class 0) that neither composes nor reorders with the piece before it, and each piece is normalised on its own;
 * within a piece, characters that decompose are aligned to their source character and composed characters to the
 * whole piece.
 *
 * @param a - The aligned text.
 * @param form - The normal form.
 * @returns The normalised aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, normaliseAligned } from 'omai-engine/text'
 * const r = normaliseAligned(alignedText('ﬁx'), 'NFKD')
 * [r.text, r.alignments] // => ['fix', [[0, 1], [0, 1], [1, 2]]]
 * ```
 */
export function normaliseAligned(a: AlignedText, form: NormalForm): AlignedText {
  const cps = codePoints(a.text)
  let text = ''
  const alignments: (readonly [number, number])[] = []
  let i = 0
  while (i < cps.length) {
    let j = i + 1
    let source = cps[i]?.ch ?? ''
    for (; j < cps.length; j++) {
      const ch = cps[j]?.ch ?? ''
      // A starter begins a new piece unless it composes or reorders with the current one (Hangul jamo, some Indic
      // vowel signs, starters that decompose to non-starters).
      if (
        !isCombining(ch.codePointAt(0) ?? 0) &&
        (source + ch).normalize(form) === source.normalize(form) + ch.normalize(form)
      )
        break
      source += ch
    }
    const piece = cps.slice(i, j)
    const out = source.normalize(form)
    const parts = piece.map((p) => p.ch.normalize(form))
    if (parts.join('') === out) {
      for (const [k, p] of piece.entries()) {
        const range = alignmentAt(a, p.at, p.ch.length)
        for (let u = 0; u < (parts[k] ?? '').length; u++) alignments.push(range)
      }
    } else {
      const first = piece[0]
      const last = piece[piece.length - 1]
      const start = first === undefined ? [0, 0] : alignmentAt(a, first.at, first.ch.length)
      const end = last === undefined ? [0, 0] : alignmentAt(a, last.at, last.ch.length)
      const range: [number, number] = [start[0] ?? 0, end[1] ?? 0]
      for (let u = 0; u < out.length; u++) alignments.push(range)
    }
    text += out
    i = j
  }
  return { original: a.original, text, alignments }
}

/**
 * Replace every match of a pattern (a literal string or a regular expression, always applied globally). As in HF
 * tokenizers' `NormalizedString.replace`, the characters of each replacement are aligned to the last character of the
 * match, and an empty replacement deletes the match.
 *
 * @param a - The aligned text.
 * @param pattern - The pattern.
 * @param replacement - The replacement (literal).
 * @returns The new aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, replaceAligned } from 'omai-engine/text'
 * const r = replaceAligned(alignedText('a  b'), / +/, '_')
 * [r.text, r.alignments] // => ['a_b', [[0, 1], [2, 3], [3, 4]]]
 * ```
 */
export function replaceAligned(a: AlignedText, pattern: string | RegExp, replacement: string): AlignedText {
  const re =
    typeof pattern === 'string'
      ? new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gu')
      : new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`)
  let text = ''
  const alignments: (readonly [number, number])[] = []
  let last = 0
  for (const m of a.text.matchAll(re)) {
    const start = m.index
    const end = start + m[0].length
    if (end === start) continue
    text += a.text.slice(last, start)
    for (let k = last; k < start; k++) alignments.push(a.alignments[k] ?? [0, 0])
    const range = a.alignments[end - 1] ?? [0, 0]
    text += replacement
    for (let k = 0; k < replacement.length; k++) alignments.push(range)
    last = end
  }
  text += a.text.slice(last)
  for (let k = last; k < a.text.length; k++) alignments.push(a.alignments[k] ?? [0, 0])
  return { original: a.original, text, alignments }
}

/**
 * Insert a string before the text (HF tokenizers' `prepend`): its characters are aligned to the first character.
 *
 * @param a - The aligned text.
 * @param prefix - The string to insert.
 * @returns The new aligned text (unchanged when the text is empty).
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, prependAligned } from 'omai-engine/text'
 * prependAligned(alignedText('ab'), '_').alignments // => [[0, 1], [0, 1], [1, 2]]
 * ```
 */
export function prependAligned(a: AlignedText, prefix: string): AlignedText {
  if (a.text.length === 0) return a
  const first = a.alignments[0] ?? [0, 0]
  return {
    original: a.original,
    text: prefix + a.text,
    alignments: [...Array.from({ length: prefix.length }, () => first), ...a.alignments],
  }
}

/**
 * Remove leading and/or trailing whitespace.
 *
 * @param a - The aligned text.
 * @param sides - Which sides to strip (default both).
 * @returns The stripped aligned text.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, stripAligned } from 'omai-engine/text'
 * stripAligned(alignedText(' ab ')).alignments // => [[1, 2], [2, 3]]
 * ```
 */
export function stripAligned(
  a: AlignedText,
  sides: { readonly left?: boolean; readonly right?: boolean } = {},
): AlignedText {
  let start = 0
  let end = a.text.length
  if (sides.left !== false) while (start < end && WHITESPACE.test(a.text[start] ?? '')) start++
  if (sides.right !== false) while (end > start && WHITESPACE.test(a.text[end - 1] ?? '')) end--
  return sliceAligned(a, start, end)
}

/**
 * The part of an aligned text between two offsets of the normalised text (alignments kept).
 *
 * @param a - The aligned text.
 * @param start - Start offset in `a.text`.
 * @param end - End offset in `a.text` (exclusive).
 * @returns The slice.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, sliceAligned } from 'omai-engine/text'
 * sliceAligned(alignedText('abc'), 1, 3).text // => 'bc'
 * ```
 */
export function sliceAligned(a: AlignedText, start: number, end: number): AlignedText {
  return { original: a.original, text: a.text.slice(start, end), alignments: a.alignments.slice(start, end) }
}

/**
 * The range of the original text that a range of the normalised text came from: from the start of the first
 * character's source to the end of the last one's (an empty range maps to an empty range at its position).
 *
 * @param a - The aligned text.
 * @param start - Start offset in `a.text`.
 * @param end - End offset in `a.text` (exclusive).
 * @returns `[start, end)` in `a.original`.
 * @concept text-alignment
 * @example
 * ```ts
 * import { alignedText, normaliseAligned, originalSpan } from 'omai-engine/text'
 * const r = normaliseAligned(alignedText('xﬁ'), 'NFKC')
 * originalSpan(r, 2, 3) // => [1, 2]
 * ```
 */
export function originalSpan(a: AlignedText, start: number, end: number): [number, number] {
  if (end <= start) {
    const at = a.alignments[start]?.[0] ?? a.alignments[start - 1]?.[1] ?? 0
    return [at, at]
  }
  return [a.alignments[start]?.[0] ?? 0, a.alignments[end - 1]?.[1] ?? 0]
}
