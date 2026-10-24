/**
 * Character classes of the regular-expression engines the reference libraries use, as JavaScript `u`-mode class
 * bodies (to splice into `[...]`). JavaScript's own `\w`, `\d`, `\b` are ASCII-only and its `\s` differs from Unicode
 * White_Space on U+0085 and U+FEFF, so patterns are written with these instead.
 */

/** Python `re`'s `\w` on str patterns: alphanumerics (str.isalnum) and underscore. */
export const PY_WORD = '\\p{L}\\p{N}_'

/** Python `re`'s `\s` on str patterns (str.isspace): Unicode White_Space plus the separators U+001C to U+001F. */
export const PY_SPACE = '\\p{White_Space}\\x1c-\\x1f'

/** The Unicode word class of the `regex` module, Rust regex and Oniguruma (UTS #18): `\w`. */
export const UNI_WORD = '\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}'

/** Unicode White_Space: `\s` of the `regex` module, Rust regex and Oniguruma. */
export const UNI_SPACE = '\\p{White_Space}'

/** A word boundary `\b` for a word class body. */
export function boundary(word: string): string {
  return `(?:(?<=[${word}])(?![${word}])|(?<![${word}])(?=[${word}]))`
}

/** Python `re`'s `$` without MULTILINE: the end of the text, or just before a final newline. */
export const PY_END = '(?=\\n?$)'

/** Escape a literal string for use in a regular expression (outside a character class, valid in `u` mode). */
export function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')
}

/** A global copy of a regular expression, or of a literal string as a Unicode-mode pattern. */
export function globalRegExp(re: RegExp | string): RegExp {
  if (typeof re === 'string') return new RegExp(escapeRegExp(re), 'gu')
  const flags = new Set(re.flags.split(''))
  flags.add('g')
  flags.delete('y')
  return new RegExp(re.source, [...flags].join(''))
}

/** Compare two strings by code point (Python's string order; JavaScript's `<` compares UTF-16 code units). */
export function compareCodePoints(a: string, b: string): number {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    const x = a.codePointAt(i) ?? 0
    const y = b.codePointAt(i) ?? 0
    if (x !== y) return x - y
    if (x > 0xffff) i++
  }
  return a.length - b.length
}
