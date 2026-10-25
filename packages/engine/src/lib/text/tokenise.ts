/**
 * Basic tokenisation into words, characters and pieces, each with its spans in the text: regular-expression, whitespace
 * and word tokenisers as in nltk and scikit-learn, Unicode grapheme clusters, the pre-tokeniser patterns of well-known
 * models (GPT-2, tiktoken encodings and others), HF tokenizers' `Split` behaviours, and BERT's pre-tokenisation.
 */

import { MdmlError } from '../base/index.ts'
import { boundary, globalRegExp, PY_SPACE, PY_WORD, UNI_SPACE, UNI_WORD } from './_classes.ts'

/** A half-open `[start, end)` range of UTF-16 code units in a text. */
export type Span = readonly [number, number]

/** The text of each span. */
function sliceAll(text: string, spans: readonly Span[]): string[] {
  return spans.map(([a, b]) => text.slice(a, b))
}

/**
 * The spans of a regular-expression tokeniser (nltk's RegexpTokenizer): the matches of the pattern, or with
 * `gaps: true` the text between matches (empty pieces dropped unless `discardEmpty` is false).
 *
 * @param text - The text.
 * @param pattern - The token (or, with gaps, separator) pattern; a string is a literal.
 * @param options - Whether the pattern matches gaps and whether to drop empty pieces.
 * @returns The token spans.
 * @concept tokenisation
 * @example
 * ```ts
 * import { regexSpans } from 'mdml-engine/text'
 * regexSpans('ab, cd', /\w+/) // => [[0, 2], [4, 6]]
 * ```
 */
export function regexSpans(
  text: string,
  pattern: RegExp | string,
  options: { readonly gaps?: boolean; readonly discardEmpty?: boolean } = {},
): Span[] {
  const re = globalRegExp(pattern)
  const out: Span[] = []
  if (options.gaps !== true) {
    for (const m of text.matchAll(re)) out.push([m.index, m.index + m[0].length])
    return out
  }
  let last = 0
  for (const m of text.matchAll(re)) {
    out.push([last, m.index])
    last = m.index + m[0].length
  }
  out.push([last, text.length])
  return options.discardEmpty === false ? out : out.filter(([a, b]) => b > a)
}

/**
 * A regular-expression tokeniser (nltk's `regexp_tokenize`): the matched tokens, or the pieces between separators.
 *
 * @param text - The text.
 * @param pattern - The pattern (a string is a literal).
 * @param options - Gaps and empty-piece handling, as {@link regexSpans}.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { regexTokenize } from 'mdml-engine/text'
 * regexTokenize('a1 b22', /\d+/) // => ['1', '22']
 * regexTokenize('a,b,,c', ',', { gaps: true }) // => ['a', 'b', 'c']
 * ```
 */
export function regexTokenize(
  text: string,
  pattern: RegExp | string,
  options: { readonly gaps?: boolean; readonly discardEmpty?: boolean } = {},
): string[] {
  return sliceAll(text, regexSpans(text, pattern, options))
}

const PY_SPACES = new RegExp(`[${PY_SPACE}]+`, 'gu')

/**
 * Split on runs of whitespace, dropping empty pieces (Python's `str.split()`, nltk's WhitespaceTokenizer).
 *
 * @param text - The text.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { whitespaceTokenize } from 'mdml-engine/text'
 * whitespaceTokenize(' a\tb c\n') // => ['a', 'b', 'c']
 * ```
 */
export function whitespaceTokenize(text: string): string[] {
  return regexTokenize(text, PY_SPACES, { gaps: true })
}

/** scikit-learn's default token pattern `(?u)\b\w\w+\b` (Python `re` word characters). */
export const SKLEARN_TOKEN_PATTERN: RegExp = new RegExp(
  `${boundary(PY_WORD)}[${PY_WORD}]{2,}${boundary(PY_WORD)}`,
  'gu',
)

/** nltk's WordPunctTokenizer pattern `\w+|[^\w\s]+` (Unicode word class of the `regex` module). */
export const WORD_PUNCT_PATTERN: RegExp = new RegExp(`[${UNI_WORD}]+|[^${UNI_WORD}${UNI_SPACE}]+`, 'gu')

/**
 * Word tokens as scikit-learn's vectorisers find them: the matches of a token pattern (default two or more word
 * characters, so punctuation and one-letter words are dropped); when the pattern has one capturing group, its content
 * is the token.
 *
 * @param text - The text.
 * @param pattern - The token pattern (default {@link SKLEARN_TOKEN_PATTERN}).
 * @returns The tokens.
 * @throws MdmlError with code `value` for a pattern with more than one capturing group.
 * @concept tokenisation
 * @example
 * ```ts
 * import { wordTokenize } from 'mdml-engine/text'
 * wordTokenize("I can't stop, naïve café!") // => ['can', 'stop', 'naïve', 'café']
 * ```
 */
export function wordTokenize(text: string, pattern: RegExp = SKLEARN_TOKEN_PATTERN): string[] {
  const re = globalRegExp(pattern)
  const groups = new RegExp(`${re.source}|`, re.flags.replace('g', '')).exec('')?.length ?? 1
  if (groups > 2) throw new MdmlError('value', 'wordTokenize: at most one capturing group')
  const out: string[] = []
  for (const m of text.matchAll(re)) out.push(groups === 2 ? (m[1] ?? '') : m[0])
  return out
}

/**
 * nltk's `wordpunct_tokenize`: runs of word characters and runs of other non-space characters.
 *
 * @param text - The text.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { wordPunctTokenize } from 'mdml-engine/text'
 * wordPunctTokenize('$3.88, ok?!') // => ['$', '3', '.', '88', ',', 'ok', '?!']
 * ```
 */
export function wordPunctTokenize(text: string): string[] {
  return regexTokenize(text, WORD_PUNCT_PATTERN)
}

/**
 * Characters as tokens: one token per Unicode code point (so surrogate pairs stay whole).
 *
 * @param text - The text.
 * @returns The code points.
 * @concept tokenisation
 * @example
 * ```ts
 * import { characterTokenize } from 'mdml-engine/text'
 * characterTokenize('a😀b') // => ['a', '😀', 'b']
 * ```
 */
export function characterTokenize(text: string): string[] {
  return Array.from(text)
}

let graphemes: Intl.Segmenter | undefined

/**
 * Extended grapheme clusters (UAX #29, the `\X` of the `regex` module): user-perceived characters, keeping base
 * letters with their combining marks, emoji ZWJ sequences, flags and Hangul syllable blocks together.
 *
 * @param text - The text.
 * @returns The grapheme clusters.
 * @concept tokenisation
 * @example
 * ```ts
 * import { graphemeTokenize } from 'mdml-engine/text'
 * graphemeTokenize('é👍🏽🇮🇪') // => ['é', '👍🏽', '🇮🇪']
 * ```
 */
export function graphemeTokenize(text: string): string[] {
  graphemes ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' })
  return Array.from(graphemes.segment(text), (s) => s.segment)
}

// ---------------------------------------------------------------------------------------------------------------
// Pre-tokeniser patterns and splitting (HF tokenizers, tiktoken).

const L = '\\p{L}'
const N = '\\p{N}'
const S = UNI_SPACE

/**
 * Pre-tokeniser patterns of well-known tokenizers, as global Unicode regular expressions whose matches are the
 * pieces: `gpt2` (GPT-2's byte-level BPE and HF ByteLevel), `cl100k` (tiktoken's GPT-4 encoding), `o200k`
 * (tiktoken's GPT-4o encoding), `whitespace` (HF Whitespace, `\w+|[^\w\s]+`). Possessive quantifiers and inline
 * case-insensitive groups of the originals are rewritten to equivalent portable forms.
 */
export const PRE_TOKENIZER_PATTERNS: Readonly<Record<'gpt2' | 'cl100k' | 'o200k' | 'whitespace', RegExp>> = {
  gpt2: new RegExp(`'s|'t|'re|'ve|'m|'ll|'d| ?${L}+| ?${N}+| ?[^${S}${L}${N}]+|[${S}]+(?![^${S}])|[${S}]+`, 'gu'),
  cl100k: new RegExp(
    `'(?:[sdmtSDMT\u017f]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n${L}${N}]?${L}+|${N}{1,3}| ?[^${S}${L}${N}]+[\\r\\n]*|[${S}]*[\\r\\n]|[${S}]+(?![^${S}])|[${S}]+`,
    'gu',
  ),
  o200k: (() => {
    const upper = '[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]'
    const lower = '[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]'
    const suffix = "(?:'[sS\u017f]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    return new RegExp(
      [
        `[^\\r\\n${L}${N}]?${upper}*${lower}+${suffix}`,
        `[^\\r\\n${L}${N}]?${upper}+${lower}*${suffix}`,
        `${N}{1,3}`,
        ` ?[^${S}${L}${N}]+[\\r\\n/]*`,
        `[${S}]*[\\r\\n]+`,
        `[${S}]+(?![^${S}])`,
        `[${S}]+`,
      ].join('|'),
      'gu',
    )
  })(),
  whitespace: WORD_PUNCT_PATTERN,
}

/**
 * Split a text into pieces with a pre-tokeniser pattern (the matches; text between matches is dropped, as tiktoken's
 * `findall` and HF's inverted splits do).
 *
 * @param text - The text.
 * @param pattern - A named pattern of {@link PRE_TOKENIZER_PATTERNS} or a regular expression.
 * @returns The pieces.
 * @concept pre-tokenisation
 * @example
 * ```ts
 * import { preTokenize } from 'mdml-engine/text'
 * preTokenize("Hello world's 12345!!", 'gpt2') // => ['Hello', ' world', "'s", ' 12345', '!!']
 * preTokenize('12345 apples', 'cl100k') // => ['123', '45', ' apples']
 * ```
 */
export function preTokenize(text: string, pattern: keyof typeof PRE_TOKENIZER_PATTERNS | RegExp): string[] {
  return regexTokenize(text, typeof pattern === 'string' ? PRE_TOKENIZER_PATTERNS[pattern] : pattern)
}

/** What happens to the delimiters (the matches) when splitting (HF tokenizers' SplitDelimiterBehavior). */
export type SplitBehavior = 'removed' | 'isolated' | 'mergedWithPrevious' | 'mergedWithNext' | 'contiguous'

/** A delimiter: a literal string, a regular expression, or a predicate on single code points. */
export type Delimiter = string | RegExp | ((codePoint: string) => boolean)

/** The segments of a text: matches of the delimiter and the gaps between them, in order, covering the text. */
function segments(text: string, delimiter: Delimiter): (readonly [number, number, boolean])[] {
  if (text.length === 0) return [[0, 0, false]]
  const out: (readonly [number, number, boolean])[] = []
  let prev = 0
  const push = (a: number, b: number): void => {
    if (prev !== a) out.push([prev, a, false])
    out.push([a, b, true])
    prev = b
  }
  if (typeof delimiter === 'function') {
    let at = 0
    for (const ch of text) {
      if (delimiter(ch)) push(at, at + ch.length)
      at += ch.length
    }
  } else {
    for (const m of text.matchAll(globalRegExp(delimiter))) push(m.index, m.index + m[0].length)
  }
  if (prev !== text.length) out.push([prev, text.length, false])
  return out
}

/**
 * Split a text on a delimiter as HF tokenizers' `Split` pre-tokeniser does: the delimiters are removed, isolated,
 * merged with the previous or next piece, or merged when contiguous; `invert` swaps the roles of matches and gaps.
 * Empty pieces are dropped.
 *
 * @param text - The text.
 * @param delimiter - The delimiter.
 * @param behavior - What to do with delimiters (default `removed`).
 * @param invert - Whether the matches are the pieces (default false).
 * @returns The spans of the pieces.
 * @concept pre-tokenisation
 * @example
 * ```ts
 * import { splitSpans } from 'mdml-engine/text'
 * splitSpans('a-b--c', '-', 'mergedWithPrevious') // => [[0, 2], [2, 4], [4, 5], [5, 6]]
 * splitSpans('a-b--c', '-', 'contiguous') // => [[0, 1], [1, 2], [2, 3], [3, 5], [5, 6]]
 * ```
 */
export function splitSpans(
  text: string,
  delimiter: Delimiter,
  behavior: SplitBehavior = 'removed',
  invert = false,
): Span[] {
  const segs = segments(text, delimiter).map(([a, b, m]) => [a, b, m !== invert] as const)
  const kept: [number, number][] = []
  let previous = false
  switch (behavior) {
    case 'removed':
      for (const [a, b, m] of segs) if (!m) kept.push([a, b])
      break
    case 'isolated':
      for (const [a, b] of segs) kept.push([a, b])
      break
    case 'contiguous':
      for (const [a, b, m] of segs) {
        const last = kept[kept.length - 1]
        if (m === previous && last !== undefined) last[1] = b
        else kept.push([a, b])
        previous = m
      }
      break
    case 'mergedWithPrevious':
      for (const [a, b, m] of segs) {
        const last = kept[kept.length - 1]
        if (m && !previous && last !== undefined) last[1] = b
        else kept.push([a, b])
        previous = m
      }
      break
    case 'mergedWithNext':
      for (let i = segs.length - 1; i >= 0; i--) {
        const [a, b, m] = segs[i] ?? [0, 0, false]
        const last = kept[kept.length - 1]
        if (m && !previous && last !== undefined) last[0] = a
        else kept.push([a, b])
        previous = m
      }
      kept.reverse()
      break
    default:
      throw new MdmlError('value', `splitSpans: unknown behavior ${String(behavior)}`)
  }
  return kept.filter(([a, b]) => b > a)
}

const BERT_PUNCTUATION = /[\p{P}!-/:-@[-`{-~]/u
const WHITE = new RegExp(`[${UNI_SPACE}]`, 'u')

/**
 * Whether a code point is punctuation as BERT's pre-tokeniser sees it: ASCII punctuation or symbols, or Unicode
 * punctuation (category P).
 *
 * @param codePoint - One code point.
 * @returns True for punctuation.
 * @example
 * ```ts
 * import { isBertPunctuation } from 'mdml-engine/text'
 * ['$', '«', 'a', '€'].map(isBertPunctuation) // => [true, true, false, false]
 * ```
 */
export function isBertPunctuation(codePoint: string): boolean {
  return BERT_PUNCTUATION.test(codePoint)
}

/**
 * BERT's pre-tokenisation (HF BertPreTokenizer): split on whitespace (removed) and isolate each punctuation character.
 *
 * @param text - The (normalised) text.
 * @returns The spans of the pieces.
 * @concept pre-tokenisation
 * @example
 * ```ts
 * import { bertPreTokenize } from 'mdml-engine/text'
 * const s = "Hey, it's 3.5!"
 * bertPreTokenize(s).map(([a, b]) => s.slice(a, b)) // => ['Hey', ',', 'it', "'", 's', '3', '.', '5', '!']
 * ```
 */
export function bertPreTokenize(text: string): Span[] {
  const out: Span[] = []
  for (const [a, b] of splitSpans(text, (c) => WHITE.test(c), 'removed')) {
    for (const [x, y] of splitSpans(text.slice(a, b), isBertPunctuation, 'isolated')) out.push([a + x, a + y])
  }
  return out
}
