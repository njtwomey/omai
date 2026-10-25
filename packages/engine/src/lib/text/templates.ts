/**
 * Feature templates for CRF sequence labellers: CRF++ templates with `%x[row,col]` macros and Wapiti patterns, parsed
 * and expanded over a token-by-attribute table into the unigram and bigram feature strings of each position, plus a
 * feature index with frequency cut-off, CRF++'s weight layout, and inspection of which template produced which feature.
 */

import { MdmlError } from '../base/index.ts'

/**
 * A token-by-attribute table: one row per token of a sequence, one column per attribute (word, part of speech, ...),
 * as the columns of a CRF++ or Wapiti training file without the label column.
 */
export type TokenTable = readonly (readonly string[])[]

/** A `%x[row,col]` macro of a CRF++ template: the attribute in column `col` of the token `row` positions away. */
export interface CrfppMacro {
  readonly row: number
  readonly col: number
}

/** One line of a CRF++ template: unigram (`U`) or bigram (`B`), its text and its literal and macro parts. */
export interface CrfppLine {
  readonly kind: 'U' | 'B'
  readonly source: string
  /** The 1-based line number in the template file. */
  readonly line: number
  readonly parts: readonly (string | CrfppMacro)[]
}

/** A parsed CRF++ template file. */
export interface CrfppTemplate {
  readonly unigram: readonly CrfppLine[]
  readonly bigram: readonly CrfppLine[]
}

/** CRF++'s largest context offset (kMaxContextSize). */
const MAX_CONTEXT = 8

function syntaxError(format: string, line: number, column: number, message: string): MdmlError {
  return new MdmlError('value', `${format} template line ${line}, column ${column}: ${message}`, { line, column })
}

/**
 * Parse a CRF++ feature template (Kudo 2005). Each non-empty line not starting with a space or `#` is a unigram
 * template (`U...`, conjoined with the current label) or a bigram template (`B...`, conjoined with the previous and
 * current labels); `%x[row,col]` stands for the attribute in column `col` of the token `row` positions away
 * (|row| <= 8). The rest of a line, including its identifier such as `U01:`, is literal.
 *
 * @param source - The template file's text.
 * @returns The unigram and bigram templates, in file order.
 * @throws MdmlError with code `value` and `{ line, column }` details for a line of unknown type or a malformed
 * macro.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseCrfppTemplate } from 'mdml-engine/text'
 * const t = parseCrfppTemplate('# words\nU00:%x[-1,0]\nU01:%x[0,0]/%x[0,1]\n\nB')
 * [t.unigram.length, t.bigram[0]?.source, t.unigram[1]?.parts] // => [2, 'B', ['U01:', { row: 0, col: 0 }, '/', { row: 0, col: 1 }]]
 * ```
 */
export function parseCrfppTemplate(source: string): CrfppTemplate {
  const unigram: CrfppLine[] = []
  const bigram: CrfppLine[] = []
  for (const [k, raw] of source.split('\n').entries()) {
    const text = raw.endsWith('\r') ? raw.slice(0, -1) : raw
    const line = k + 1
    if (text.length === 0 || text[0] === ' ' || text[0] === '#') continue
    const kind = text[0]
    if (kind !== 'U' && kind !== 'B') throw syntaxError('CRF++', line, 1, `unknown template type ${kind}`)
    const parts: (string | CrfppMacro)[] = []
    let literal = ''
    let i = 0
    while (i < text.length) {
      const ch = text[i] ?? ''
      if (ch !== '%') {
        literal += ch
        i++
        continue
      }
      const m = /^%x\[(-?)(\d*),(\d*)\]/.exec(text.slice(i))
      if (m === null) throw syntaxError('CRF++', line, i + 1, 'expected a macro %x[row,col]')
      const row = (m[1] === '-' ? -1 : 1) * Number(m[2] === '' ? '0' : m[2])
      const col = Number(m[3] === '' ? '0' : m[3])
      if (Math.abs(row) > MAX_CONTEXT) throw syntaxError('CRF++', line, i + 1, `row ${row} is outside [-8, 8]`)
      if (literal.length > 0) parts.push(literal)
      literal = ''
      parts.push({ row: row === 0 ? 0 : row, col })
      i += m[0].length
    }
    if (literal.length > 0) parts.push(literal)
    ;(kind === 'U' ? unigram : bigram).push({ kind, source: text, line, parts })
  }
  return { unigram, bigram }
}

/** CRF++'s boundary values: `_B-k` before the sequence and `_B+k` after it. */
function crfppCell(table: TokenTable, position: number, macro: CrfppMacro, line: CrfppLine): string {
  const width = table[0]?.length ?? 0
  if (macro.col >= width)
    throw new MdmlError('value', `CRF++ template line ${line.line}: column ${macro.col} is out of range`, {
      line: line.line,
    })
  const i = position + macro.row
  if (i < 0) return `_B-${-i}`
  if (i >= table.length) return `_B+${i - table.length + 1}`
  return table[i]?.[macro.col] ?? ''
}

/**
 * Expand one template line at one position of a sequence: macros replaced by table cells, or by the boundary values
 * `_B-1`, `_B-2`, ... and `_B+1`, ... outside the sequence.
 *
 * @param line - A parsed template line.
 * @param table - The token-by-attribute table.
 * @param position - The current token.
 * @returns The feature string (for example `U01:the/DT`).
 * @throws MdmlError with code `value` for a column outside the table.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseCrfppTemplate, expandCrfppLine } from 'mdml-engine/text'
 * const [line] = parseCrfppTemplate('U02:%x[-1,0]|%x[1,1]').unigram
 * line === undefined ? '' : expandCrfppLine(line, [['the', 'DT'], ['cat', 'NN']], 0) // => 'U02:_B-1|NN'
 * ```
 */
export function expandCrfppLine(line: CrfppLine, table: TokenTable, position: number): string {
  let out = ''
  for (const p of line.parts) out += typeof p === 'string' ? p : crfppCell(table, position, p, line)
  return out
}

/** Feature strings of each position of a sequence, split into unigram and bigram features. */
export interface SequenceFeatures {
  /** For each position, the unigram (state) features. */
  readonly unigram: readonly (readonly string[])[]
  /** For each position, the bigram (transition) features of the edge into it (none at position 0). */
  readonly bigram: readonly (readonly string[])[]
}

/**
 * Expand a CRF++ template over a sequence, as CRF++'s `buildFeatures`: every unigram line at every position, and every
 * bigram line at every position from the second (the edge from the previous token).
 *
 * @param template - The parsed template.
 * @param table - The token-by-attribute table.
 * @returns The features of each position.
 * @throws MdmlError with code `value` for a column outside the table.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseCrfppTemplate, crfppFeatures } from 'mdml-engine/text'
 * const t = parseCrfppTemplate('U00:%x[0,0]\nU01:%x[-1,0]/%x[0,0]\nB')
 * crfppFeatures(t, [['He'], ['ran']]) // => { unigram: [['U00:He', 'U01:_B-1/He'], ['U00:ran', 'U01:He/ran']], bigram: [[], ['B']] }
 * ```
 */
export function crfppFeatures(template: CrfppTemplate, table: TokenTable): SequenceFeatures {
  const unigram = table.map((_, t) => template.unigram.map((l) => expandCrfppLine(l, table, t)))
  const bigram = table.map((_, t) => (t === 0 ? [] : template.bigram.map((l) => expandCrfppLine(l, table, t))))
  return { unigram, bigram }
}

// ---------------------------------------------------------------------------------------------------------------
// Wapiti patterns: CRF++'s syntax extended with regular-expression tests and matches.

/** A Wapiti command: `x` extracts the cell, `t` tests it against a pattern (`true`/`false`), `m` gives the match. */
export interface WapitiCommand {
  readonly command: 'x' | 't' | 'm'
  /** An upper-case command letter makes `t` and `m` case-insensitive. */
  readonly caseInsensitive: boolean
  readonly offset: number
  /** An `@` before the offset makes it absolute: a 0-based position from the start, or from the end if negative. */
  readonly absolute: boolean
  readonly col: number
  readonly pattern?: RegExp
}

/** One Wapiti pattern line: unigram (`u`), bigram (`b`) or both (`*`). */
export interface WapitiLine {
  readonly kind: 'u' | 'b' | '*'
  readonly source: string
  readonly line: number
  readonly parts: readonly (string | WapitiCommand)[]
}

/** Wapiti's boundary values: `_x-1` to `_x-4` before the sequence (`_x-#` further), `_x+1` to `_x+4` after. */
function wapitiBoundary(i: number, length: number): string {
  if (i < 0) return -i <= 4 ? `_x-${-i}` : '_x-#'
  const k = i - length + 1
  return k <= 4 ? `_x+${k}` : '_x+#'
}

/**
 * Parse Wapiti patterns (Lavergne et al. 2010): CRF++-style lines whose first letter is `u`, `b` or `*` (unigram,
 * bigram, both; either case), with `%x[off,col]`, `%t[off,col,"regex"]` and `%m[off,col,"regex"]` commands; an
 * upper-case command letter makes the test or match case-insensitive and `@` makes an offset absolute. Regular
 * expressions use JavaScript syntax plus Wapiti's classes `\a` (letters), `\l` (lower), `\u` (upper), `\p`
 * (punctuation) and their negations `\A`, `\L`, `\U`, `\P`.
 *
 * @param source - The pattern file's text.
 * @returns The pattern lines.
 * @throws MdmlError with code `value` and `{ line, column }` details for malformed lines.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseWapitiPatterns } from 'mdml-engine/text'
 * parseWapitiPatterns('*:Cap=%t[0,0,"^\\u"]').map((l) => l.kind) // => ['*']
 * ```
 */
export function parseWapitiPatterns(source: string): WapitiLine[] {
  const out: WapitiLine[] = []
  for (const [k, raw] of source.split('\n').entries()) {
    const text = raw.replace(/\r$/, '').replace(/^[ \t]+/, '')
    const line = k + 1
    if (text.length === 0 || text[0] === '#') continue
    const first = (text[0] ?? '').toLowerCase()
    if (first !== 'u' && first !== 'b' && first !== '*')
      throw syntaxError('Wapiti', line, 1, `unknown pattern type ${text[0]}`)
    const parts: (string | WapitiCommand)[] = []
    let literal = ''
    let i = 0
    while (i < text.length) {
      const ch = text[i] ?? ''
      if (ch !== '%') {
        literal += ch
        i++
        continue
      }
      const m = /^%([xXtTmM])\[(@?)(-?\d+),(\d+)(?:,"((?:[^"\\]|\\.)*)")?\]/.exec(text.slice(i))
      if (m === null) throw syntaxError('Wapiti', line, i + 1, 'expected a command %x[off,col] or %t/%m[off,col,"re"]')
      const letter = m[1] ?? 'x'
      const command = letter.toLowerCase() as 'x' | 't' | 'm'
      if ((command === 'x') !== (m[5] === undefined))
        throw syntaxError('Wapiti', line, i + 1, command === 'x' ? 'x takes no pattern' : `${command} needs a pattern`)
      const caseInsensitive = letter !== command
      let pattern: RegExp | undefined
      if (m[5] !== undefined) {
        try {
          pattern = new RegExp(wapitiRegex(m[5]), caseInsensitive ? 'iu' : 'u')
        } catch {
          throw syntaxError('Wapiti', line, i + 1, `invalid pattern ${m[5]}`)
        }
      }
      if (literal.length > 0) parts.push(literal)
      literal = ''
      const offset = Number(m[3])
      parts.push({
        command,
        caseInsensitive,
        offset: offset === 0 ? 0 : offset,
        absolute: m[2] === '@',
        col: Number(m[4]),
        ...(pattern === undefined ? {} : { pattern }),
      })
      i += m[0].length
    }
    if (literal.length > 0) parts.push(literal)
    out.push({ kind: first, source: text, line, parts })
  }
  return out
}

/** Wapiti's character classes in a JavaScript pattern. */
function wapitiRegex(source: string): string {
  const classes: Readonly<Record<string, string>> = {
    a: '\\p{L}',
    A: '\\P{L}',
    l: '\\p{Ll}',
    L: '\\P{Ll}',
    u: '\\p{Lu}',
    U: '\\P{Lu}',
    p: '\\p{P}',
    P: '\\P{P}',
  }
  return source.replace(/\\(.)/g, (all: string, c: string) => classes[c] ?? (c === '"' ? '"' : all))
}

/**
 * Expand Wapiti patterns over a sequence: unigram lines at every position, bigram lines at every position from the
 * second, `*` lines in both.
 *
 * @param patterns - The parsed patterns.
 * @param table - The token-by-attribute table.
 * @returns The features of each position.
 * @throws MdmlError with code `value` for a column outside the table.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseWapitiPatterns, wapitiFeatures } from 'mdml-engine/text'
 * const p = parseWapitiPatterns('u:w=%x[-1,0]\n*:cap=%t[0,0,"^\\u"]\nU:suf=%m[0,0,"..$"]')
 * wapitiFeatures(p, [['The'], ['cat']]).unigram // => [['u:w=_x-1', '*:cap=true', 'U:suf=he'], ['u:w=The', '*:cap=false', 'U:suf=at']]
 * ```
 */
export function wapitiFeatures(patterns: readonly WapitiLine[], table: TokenTable): SequenceFeatures {
  const width = table[0]?.length ?? 0
  const expand = (l: WapitiLine, t: number): string => {
    let out = ''
    for (const p of l.parts) {
      if (typeof p === 'string') {
        out += p
        continue
      }
      if (p.col >= width)
        throw new MdmlError('value', `Wapiti pattern line ${l.line}: column ${p.col} is out of range`, {
          line: l.line,
        })
      const i = p.absolute ? (p.offset < 0 ? table.length + p.offset : p.offset) : t + p.offset
      const inside = i >= 0 && i < table.length
      const cell = inside ? (table[i]?.[p.col] ?? '') : wapitiBoundary(i, table.length)
      if (p.command === 'x') out += cell
      else if (p.command === 't') out += inside && p.pattern?.test(cell) === true ? 'true' : 'false'
      else out += inside ? (p.pattern?.exec(cell)?.[0] ?? '') : ''
    }
    return out
  }
  const unigram = table.map((_, t) => patterns.filter((l) => l.kind !== 'b').map((l) => expand(l, t)))
  const bigram = table.map((_, t) => (t === 0 ? [] : patterns.filter((l) => l.kind !== 'u').map((l) => expand(l, t))))
  return { unigram, bigram }
}

// ---------------------------------------------------------------------------------------------------------------
// Feature indexes and inspection.

/** A feature vocabulary: features in first-seen order, their frequencies, and the index of each. */
export interface FeatureIndex {
  readonly features: readonly string[]
  readonly counts: readonly number[]
  readonly ids: ReadonlyMap<string, number>
}

/**
 * Index the features of a training set: every firing counts once, and features seen fewer than `minFrequency` times
 * are dropped, keeping first-seen order (CRF++'s `-f` cut-off and id shrinking).
 *
 * @param sequences - For each sequence, the features of each position (feature lists, or attribute records whose keys
 * are the features).
 * @param options - The minimum frequency (default 1).
 * @returns The index.
 * @concept crf-features
 * @example
 * ```ts
 * import { featureIndex } from 'mdml-engine/text'
 * featureIndex([[['a', 'b'], ['a']], [{ a: 1, c: 1 }]], { minFrequency: 2 }).features // => ['a']
 * ```
 */
export function featureIndex(
  sequences: Iterable<Iterable<readonly string[] | Readonly<Record<string, number>>>>,
  options: { readonly minFrequency?: number } = {},
): FeatureIndex {
  const counts = new Map<string, number>()
  for (const s of sequences)
    for (const item of s)
      for (const f of Array.isArray(item) ? (item as readonly string[]) : Object.keys(item))
        counts.set(f, (counts.get(f) ?? 0) + 1)
  const min = options.minFrequency ?? 1
  const kept = [...counts].filter(([, c]) => c >= min)
  return {
    features: kept.map(([f]) => f),
    counts: kept.map(([, c]) => c),
    ids: new Map(kept.map(([f], i) => [f, i])),
  }
}

/**
 * CRF++'s weight offsets: each kept feature, in index order, owns one weight per label if it is a unigram feature
 * (its string starts with `U`) or one per label pair if it is a bigram feature (`B`); the weight of feature f with
 * label y (or labels y', y) is at offset(f) + y (or offset(f) + y' L + y).
 *
 * @param index - The feature index.
 * @param labels - The number of labels L.
 * @returns The offset of each feature and the total number of weights.
 * @concept crf-features
 * @example
 * ```ts
 * import { featureIndex, crfppWeightOffsets } from 'mdml-engine/text'
 * const r = crfppWeightOffsets(featureIndex([[['U00:a', 'B'], ['U00:b', 'B']]]), 3)
 * [[...r.offsets], r.size] // => [[['U00:a', 0], ['B', 3], ['U00:b', 12]], 15]
 * ```
 */
export function crfppWeightOffsets(
  index: FeatureIndex,
  labels: number,
): { readonly offsets: ReadonlyMap<string, number>; readonly size: number } {
  const offsets = new Map<string, number>()
  let next = 0
  for (const f of index.features) {
    offsets.set(f, next)
    next += f.startsWith('B') ? labels * labels : labels
  }
  return { offsets, size: next }
}

/** One template firing, for inspection: which line produced which feature, and its id in an index (-1 if absent). */
export interface FeatureFiring {
  readonly kind: 'U' | 'B'
  readonly line: number
  readonly template: string
  readonly feature: string
  readonly id: number
}

/**
 * The features a CRF++ template fires at one position, with the template line behind each and its id in a feature
 * index (for inspecting what a model sees, or why a feature is missing).
 *
 * @param template - The parsed template.
 * @param table - The token-by-attribute table.
 * @param position - The position.
 * @param index - An optional feature index.
 * @returns The firings, unigram lines first.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseCrfppTemplate, fireFeatures } from 'mdml-engine/text'
 * fireFeatures(parseCrfppTemplate('U00:%x[0,0]\nB'), [['a'], ['b']], 1).map((f) => f.feature) // => ['U00:b', 'B']
 * ```
 */
export function fireFeatures(
  template: CrfppTemplate,
  table: TokenTable,
  position: number,
  index?: FeatureIndex,
): FeatureFiring[] {
  const lines = position === 0 ? template.unigram : [...template.unigram, ...template.bigram]
  return lines.map((l) => {
    const feature = expandCrfppLine(l, table, position)
    return { kind: l.kind, line: l.line, template: l.source, feature, id: index?.ids.get(feature) ?? -1 }
  })
}
