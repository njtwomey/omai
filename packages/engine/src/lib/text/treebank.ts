import { WhiteboxError } from '../base/index.ts'
import { boundary, PY_END, PY_SPACE, PY_WORD, UNI_SPACE, UNI_WORD } from './_classes.ts'
import type { Span } from './tokenise.ts'

/** A rewrite rule: a pattern and its replacement (JavaScript `$1` syntax). */
type Rule = readonly [RegExp, string]

const apply = (text: string, rules: readonly Rule[]): string => rules.reduce((s, [re, by]) => s.replace(re, by), text)

const r = (source: string, flags = 'gu'): RegExp => new RegExp(source, flags)
const D = '\\p{Nd}'
const UB = boundary(UNI_WORD)
const PY_SPLIT = r(`[${PY_SPACE}]+`)

const split = (text: string): string[] => text.split(PY_SPLIT).filter((t) => t.length > 0)

// MacIntyre's contractions. The tokeniser compiles them with stdlib `re` (Python word boundaries, nothing between
// the parts); the detokeniser with the `regex` module and whitespace between the parts.
const CONTRACTIONS2 = [
  ['can', 'not'],
  ['d', "'ye"],
  ['gim', 'me'],
  ['gon', 'na'],
  ['got', 'ta'],
  ['lem', 'me'],
  ['more', "'n"],
] as const
const contractions2 = (separator: string, word: string, space: string, by: string): Rule[] => [
  ...CONTRACTIONS2.map(([a, b]): Rule => [r(`${boundary(word)}(${a})${separator}(${b})${boundary(word)}`, 'giu'), by]),
  [r(`${boundary(word)}(wan)${separator}(na)(?=[${space}])`, 'giu'), by],
]
const contractions3 = (separator: string, word: string, by: string): Rule[] => [
  [r(` ('t)${separator}(is)${boundary(word)}`, 'giu'), by],
  [r(` ('t)${separator}(was)${boundary(word)}`, 'giu'), by],
]
const TOKENISE_CONTRACTIONS: readonly Rule[] = [
  ...contractions2('', PY_WORD, PY_SPACE, ' $1 $2 '),
  ...contractions3('', PY_WORD, ' $1 $2 '),
]

const TREEBANK_STARTING: readonly Rule[] = [
  [r('^"'), '``'],
  [r('(``)'), ' $1 '],
  [r(`([ ([{<])("|'{2})`), '$1 `` '],
]
const TREEBANK_PUNCTUATION: readonly Rule[] = [
  [r(`([:,])([^${D}])`), ' $1 $2'],
  [r(`([:,])${PY_END}`), ' $1 '],
  [r('\\.\\.\\.'), ' ... '],
  [r('[;@#$%&]'), ' $& '],
  [r(`([^.])(\\.)([\\])}>"']*)[${UNI_SPACE}]*$`), '$1 $2$3 '],
  [r('[?!]'), ' $& '],
  [r("([^'])' "), "$1 ' "],
]
const PARENS: Rule = [r('[\\][(){}<>]'), ' $& ']
const CONVERT_PARENTHESES: readonly Rule[] = [
  [r('\\('), '-LRB-'],
  [r('\\)'), '-RRB-'],
  [r('\\['), '-LSB-'],
  [r('\\]'), '-RSB-'],
  [r('\\{'), '-LCB-'],
  [r('\\}'), '-RCB-'],
]
const DOUBLE_DASHES: Rule = [r('--'), ' -- ']
const ENDING_CLITICS: readonly Rule[] = [
  [r("([^' ])('[sS]|'[mM]|'[dD]|') "), '$1 $2 '],
  [r("([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), '$1 $2 '],
]
const TREEBANK_ENDING: readonly Rule[] = [[r("''"), " '' "], [r('"'), " '' "], ...ENDING_CLITICS]

/** Options of the Treebank tokenisers. */
export interface TreebankOptions {
  /** Replace brackets with the PTB symbols `-LRB-`, `-RRB-`, `-LSB-`, `-RSB-`, `-LCB-`, `-RCB-` (default false). */
  readonly convertParentheses?: boolean
}

/**
 * The Penn Treebank tokeniser (nltk's TreebankWordTokenizer, a port of McIntyre's sed script): split standard
 * contractions (`don't` to `do n't`), separate most punctuation, split commas and single quotes followed by
 * whitespace, and separate a final period. Double quotes become `` and ''.
 *
 * @param text - The text (one sentence, or several; only the final period is split).
 * @param options - Bracket conversion.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { treebankTokenize } from 'whitebox-engine/text'
 * treebankTokenize("They'll save $3.88, can't they?") // => ['They', "'ll", 'save', '$', '3.88', ',', 'ca', "n't", 'they', '?']
 * ```
 */
export function treebankTokenize(text: string, options: TreebankOptions = {}): string[] {
  let s = apply(text, TREEBANK_STARTING)
  s = apply(s, TREEBANK_PUNCTUATION)
  s = apply(s, [PARENS])
  if (options.convertParentheses === true) s = apply(s, CONVERT_PARENTHESES)
  s = apply(s, [DOUBLE_DASHES])
  s = ` ${s} `
  s = apply(s, TREEBANK_ENDING)
  s = apply(s, TOKENISE_CONTRACTIONS)
  return split(s)
}

const NLTK_STARTING: readonly Rule[] = [
  [r('([«“‘„]|[`]+)'), ' $1 '],
  [r('^"'), '``'],
  [r('(``)'), ' $1 '],
  [r(`([ ([{<])("|'{2})`), '$1 `` '],
  [r(`(?<![${UNI_WORD}])(')(?!(?:re|ve|ll|m|t|s|d|n)${UB})(?=[${UNI_WORD}])`, 'giu'), '$1 '],
]
const NLTK_ENDING: readonly Rule[] = [
  [r('([»”’])'), ' $1 '],
  [r("''"), " '' "],
  [r('"'), " '' "],
  [r(`[${UNI_SPACE}]+`), ' '],
  ...ENDING_CLITICS,
]
const NLTK_PUNCTUATION: readonly Rule[] = [
  [r(`([^.])(\\.)([\\])}>"'»”’ ]*)[${UNI_SPACE}]*$`), '$1 $2 $3 '],
  [r(`([:,])([^${D}])`), ' $1 $2'],
  [r(`([:,])${PY_END}`), ' $1 '],
  [r('\\.{2,}'), ' $& '],
  [r('[;@#$%&]'), ' $& '],
  [r('[\\u2012-\\u2015]'), ' $& '],
  [r(`([^.])(\\.)([\\])}>"']*)[${UNI_SPACE}]*$`), '$1 $2$3 '],
  [r('[?!]'), ' $& '],
  [r("([^'])' "), "$1 ' "],
  [r('[*]'), ' $& '],
]

/**
 * nltk's improved Treebank tokeniser (NLTKWordTokenizer, what `word_tokenize` applies to each sentence, as with
 * `preserve_line=True`): the Treebank rules plus Unicode quotes, dashes, ellipses of any length, asterisks and
 * opening single quotes.
 *
 * @param text - One sentence.
 * @param options - Bracket conversion.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { nltkWordTokenize } from 'whitebox-engine/text'
 * nltkWordTokenize('“Wait…” — he said.. ok') // => ['“', 'Wait…', '”', '—', 'he', 'said', '..', 'ok']
 * ```
 */
export function nltkWordTokenize(text: string, options: TreebankOptions = {}): string[] {
  let s = apply(text, NLTK_STARTING)
  s = apply(s, NLTK_PUNCTUATION)
  s = apply(s, [PARENS])
  if (options.convertParentheses === true) s = apply(s, CONVERT_PARENTHESES)
  s = apply(s, [DOUBLE_DASHES])
  s = ` ${s} `
  s = apply(s, NLTK_ENDING)
  s = apply(s, TOKENISE_CONTRACTIONS)
  return split(s)
}

/**
 * Spans of Treebank tokens in the original text (nltk's `span_tokenize`): the tokens are located left to right, with
 * the `` and '' produced from double quotes mapped back to the quotes that were there.
 *
 * @param text - The text.
 * @returns The token spans.
 * @throws WhiteboxError with code `value` if a token cannot be found (bracket conversion was used).
 * @concept tokenisation
 * @example
 * ```ts
 * import { treebankSpans } from 'whitebox-engine/text'
 * treebankSpans('He said "no".') // => [[0, 2], [3, 7], [8, 9], [9, 11], [11, 12], [12, 13]]
 * ```
 */
export function treebankSpans(text: string): Span[] {
  const raw = treebankTokenize(text)
  let tokens = raw
  if (text.includes('"') || text.includes("''")) {
    const quotes = Array.from(text.matchAll(/``|'{2}|"/g), (m) => m[0])
    let q = 0
    tokens = raw.map((t) => (t === '"' || t === '``' || t === "''" ? (quotes[q++] ?? t) : t))
  }
  const out: Span[] = []
  let at = 0
  for (const t of tokens) {
    const i = text.indexOf(t, at)
    if (i < 0) throw new WhiteboxError('value', `treebankSpans: token ${t} not found in the text`)
    out.push([i, i + t.length])
    at = i + t.length
  }
  return out
}

const S = `[${UNI_SPACE}]`
const DETOK_CONTRACTIONS: readonly Rule[] = [
  ...contractions3(S, UNI_WORD, '$1$2'),
  ...contractions2(S, UNI_WORD, UNI_SPACE, '$1$2'),
]
const DETOK_ENDING: readonly Rule[] = [
  [r(`([^' ])${S}('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) `), '$1$2 '],
  [r(`([^' ])${S}('[sS]|'[mM]|'[dD]|') `), '$1$2 '],
  [r(`([^'${UNI_SPACE}])${S}('')`), '$1$2'],
  [r(`([,.;:!?'])${S}+("|'')`), '$1$2'],
  [r(`('')${S}([.,:)\\]>};%])`), '$1$2'],
  [r("''"), '"'],
  [r(`([,.;:!?])"(')`), '$1$2"'],
]
const DETOK_PARENS: readonly Rule[] = [
  [r(`([[({<])${S}`), '$1'],
  [r(`${S}([\\])}>])`), '$1'],
  [r(`([\\])}>])${S}([:;,.])`), '$1$2'],
]
const DETOK_PUNCTUATION: readonly Rule[] = [
  [r(`([^'])${S}'${S}`), "$1' "],
  [r(`${S}([?!])`), '$1'],
  [r(`([^.])${S}(\\.)(?!\\.)([\\])}>"']*)`), '$1$2$3'],
  [r(`([#$])${S}`), '$1'],
  [r(`${S}([;%])`), '$1'],
  [r(`${S}\\.\\.\\.${S}`), '...'],
  [r(`${S}([:,])`), '$1'],
]
const DETOK_STARTING: readonly Rule[] = [
  [r(`([ ([{<])${S}\`\``), '$1``'],
  [r(`(\`\`)${S}`), '$1'],
  [r('``'), '"'],
]
const PY_TRIM = r(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`)
const UNCONVERT: readonly Rule[] = [
  [r('-LRB-'), '('],
  [r('-RRB-'), ')'],
  [r('-LSB-'), '['],
  [r('-RSB-'), ']'],
  [r('-LCB-'), '{'],
  [r('-RCB-'), '}'],
]

/**
 * The Treebank detokeniser (nltk's TreebankWordDetokenizer): undo the Treebank tokeniser's padding of punctuation,
 * quotes, brackets and contractions. Original whitespace (newlines, double spaces) cannot be recovered.
 *
 * @param tokens - Treebank tokens.
 * @param options - Whether to turn PTB bracket symbols back into brackets.
 * @returns The text.
 * @concept detokenisation
 * @example
 * ```ts
 * import { treebankDetokenize } from 'whitebox-engine/text'
 * treebankDetokenize(['hello', ',', 'i', 'ca', "n't", 'feel', 'my', 'feet', '!', 'Help', '!', '!']) // => "hello, i can't feel my feet! Help!!"
 * ```
 */
export function treebankDetokenize(tokens: readonly string[], options: TreebankOptions = {}): string {
  let s = ` ${tokens.join(' ')} `
  s = apply(s, DETOK_CONTRACTIONS)
  s = apply(s, DETOK_ENDING)
  s = s.replace(PY_TRIM, '')
  s = s.replace(/ -- /g, '--')
  if (options.convertParentheses === true) s = apply(s, UNCONVERT)
  s = apply(s, DETOK_PARENS)
  s = apply(s, DETOK_PUNCTUATION)
  s = apply(s, DETOK_STARTING)
  return s.replace(PY_TRIM, '')
}
