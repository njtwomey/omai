/**
 * Hyphenation by Liang's algorithm, the method of TeX: patterns of letters with digits between them vote on where a
 * word may break, and odd maxima mark the hyphenation points. Compiling patterns in TeX notation or from Hunspell
 * dictionaries (as pyphen uses), hyphenating with a trace of the matching patterns, and inserting hyphens.
 */

import { WhiteboxError } from '../base/index.ts'

/** A compiled set of Liang hyphenation patterns, keyed by their letters, with exceptions. */
export interface HyphenationPatterns {
  /** Pattern letters (with `.` for a word boundary) to the inter-letter values, offset from the pattern's start. */
  readonly patterns: ReadonlyMap<string, { readonly offset: number; readonly values: readonly number[] }>
  /** Words with fixed hyphenation points. */
  readonly exceptions: ReadonlyMap<string, readonly number[]>
  readonly maxLength: number
}

/**
 * Compile hyphenation patterns in TeX's notation (Liang 1983), such as `.ach4` or `hy3ph`: letters with digits
 * between them, `.` marking a word boundary; and exceptions such as `ta-ble`. The source is a whitespace-separated
 * list (lines starting with `%` or `#` are comments; `^^xx` escapes are decoded) or a list of patterns.
 *
 * @param source - The patterns (a string or a list).
 * @param exceptions - Exception words with hyphens (default none).
 * @returns The compiled patterns.
 * @throws WhiteboxError with code `not-implemented` for non-standard (`/`) patterns.
 * @concept hyphenation
 * @example
 * ```ts
 * import { hyphenationPatterns } from 'whitebox-engine/text'
 * hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n').patterns.get('hyph') // => { offset: 2, values: [3] }
 * ```
 */
export function hyphenationPatterns(
  source: string | readonly string[],
  exceptions: readonly string[] = [],
): HyphenationPatterns {
  const lines = typeof source === 'string' ? splitSource(source) : source
  const patterns = new Map<string, { offset: number; values: number[] }>()
  let maxLength = 0
  for (const raw of lines) {
    const pattern = raw
      .trim()
      .replace(/\^\^([0-9a-f]{2})/g, (_: string, h: string) => String.fromCharCode(Number.parseInt(h, 16)))
    if (pattern.length === 0) continue
    if (pattern.includes('/') && pattern.includes('='))
      throw new WhiteboxError('not-implemented', `hyphenationPatterns: non-standard pattern ${pattern}`)
    const letters: string[] = []
    const values: number[] = []
    let pending = 0
    for (const ch of pattern) {
      if (ch >= '0' && ch <= '9') pending = Number(ch)
      else {
        letters.push(ch)
        values.push(pending)
        pending = 0
      }
    }
    values.push(pending)
    let start = 0
    let end = values.length
    while (start < end && values[start] === 0) start++
    while (end > start && values[end - 1] === 0) end--
    if (start === end) continue
    const key = letters.join('')
    patterns.set(key, { offset: start, values: values.slice(start, end) })
    maxLength = Math.max(maxLength, Array.from(key).length)
  }
  const exc = new Map<string, number[]>()
  for (const e of exceptions) {
    const points: number[] = []
    let k = 0
    for (const ch of e) {
      if (ch === '-') points.push(k)
      else k++
    }
    exc.set(e.replaceAll('-', '').toLowerCase(), points)
  }
  return { patterns, exceptions: exc, maxLength }
}

const IGNORED = ['%', '#', 'LEFTHYPHENMIN', 'RIGHTHYPHENMIN', 'COMPOUNDLEFTHYPHENMIN', 'COMPOUNDRIGHTHYPHENMIN']

function splitSource(source: string): string[] {
  return source.split('\n').flatMap((l) => {
    const t = l.trim()
    return t.length === 0 || IGNORED.some((p) => t.startsWith(p)) ? [] : t.split(/\s+/)
  })
}

/**
 * Compile the patterns of a LibreOffice/Hunspell hyphenation dictionary (`hyph_*.dic`, the files pyphen ships): the
 * first line names the encoding and is skipped, as are comments and `LEFTHYPHENMIN`-style settings.
 *
 * @param text - The dictionary's text.
 * @returns The compiled patterns.
 * @throws WhiteboxError with code `not-implemented` for non-standard (`/`) patterns.
 * @concept hyphenation
 * @example
 * ```ts
 * import { hyphenationDictionary } from 'whitebox-engine/text'
 * hyphenationDictionary('UTF-8\nLEFTHYPHENMIN 2\n.ach4\n1ba').patterns.size // => 2
 * ```
 */
export function hyphenationDictionary(text: string): HyphenationPatterns {
  return hyphenationPatterns(splitSource(text.split('\n').slice(1).join('\n')))
}

/** One pattern that matched while hyphenating, and where. */
export interface PatternMatch {
  readonly pattern: string
  /** The position of the match in the word with boundary dots. */
  readonly at: number
  readonly values: readonly number[]
}

/** Options of {@link hyphenate}. */
export interface HyphenateOptions {
  /** The fewest letters before the first hyphen (default 2). */
  readonly leftMin?: number
  /** The fewest letters after the last hyphen (default 2, as pyphen; TeX uses 3). */
  readonly rightMin?: number
}

/**
 * The hyphenation points of a word by Liang's algorithm, with the trace of matching patterns: every pattern that
 * occurs in `.word.` contributes its values, the maximum at each gap wins, and odd values are hyphenation points
 * (respecting leftMin and rightMin; exceptions override the patterns). This is pyphen's `positions`.
 *
 * @param word - The word.
 * @param patterns - The compiled patterns.
 * @param options - Minimum fragment lengths.
 * @returns The points (indices of the code points the hyphens precede), the parts, and the matched patterns.
 * @concept hyphenation
 * @example
 * ```ts
 * import { hyphenationPatterns, hyphenate } from 'whitebox-engine/text'
 * const p = hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n')
 * hyphenate('hyphenation', p).parts // => ['hy', 'phen', 'ation']
 * ```
 */
export function hyphenate(
  word: string,
  patterns: HyphenationPatterns,
  options: HyphenateOptions = {},
): { readonly points: number[]; readonly parts: string[]; readonly matches: PatternMatch[] } {
  const chars = Array.from(word)
  const lower = Array.from(word.toLowerCase())
  const left = options.leftMin ?? 2
  const right = chars.length - (options.rightMin ?? 2)
  const matches: PatternMatch[] = []
  let points: number[]
  const exception = patterns.exceptions.get(word.toLowerCase())
  if (exception !== undefined) points = [...exception]
  else {
    const dotted = ['.', ...lower, '.']
    const refs = new Array<number>(dotted.length + 1).fill(0)
    for (let i = 0; i < dotted.length - 1; i++) {
      const stop = Math.min(i + patterns.maxLength, dotted.length)
      for (let j = i + 1; j <= stop; j++) {
        const key = dotted.slice(i, j).join('')
        const p = patterns.patterns.get(key)
        if (p === undefined) continue
        matches.push({ pattern: key, at: i, values: p.values })
        for (const [k, v] of p.values.entries()) {
          const at = i + p.offset + k
          if (v > (refs[at] ?? 0)) refs[at] = v
        }
      }
    }
    points = []
    for (const [i, v] of refs.entries()) if (v % 2 === 1) points.push(i - 1)
  }
  points = points.filter((p) => p >= left && p <= right)
  const parts: string[] = []
  let last = 0
  for (const p of points) {
    parts.push(chars.slice(last, p).join(''))
    last = p
  }
  parts.push(chars.slice(last).join(''))
  return { points, parts, matches }
}

/**
 * A word with hyphens inserted at its hyphenation points (pyphen's `inserted`).
 *
 * @param word - The word.
 * @param patterns - The compiled patterns.
 * @param options - Minimum fragment lengths and the hyphen (default `-`).
 * @returns The hyphenated word.
 * @concept hyphenation
 * @example
 * ```ts
 * import { hyphenationPatterns, insertHyphens } from 'whitebox-engine/text'
 * insertHyphens('Table', hyphenationPatterns([], ['ta-ble'])) // => 'Ta-ble'
 * ```
 */
export function insertHyphens(
  word: string,
  patterns: HyphenationPatterns,
  options: HyphenateOptions & { readonly hyphen?: string } = {},
): string {
  return hyphenate(word, patterns, options).parts.join(options.hyphen ?? '-')
}
