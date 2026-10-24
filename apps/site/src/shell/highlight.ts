/** A run of source text with the kind of token it is (`undefined` for plain text). */
export interface Token {
  readonly text: string
  readonly kind?: 'keyword' | 'string' | 'number' | 'comment' | 'type'
}

const KEYWORDS = new Set(
  (
    'as async await break case catch class const continue default do else export extends false finally for from ' +
    'function if import in instanceof interface let new null of readonly return switch this throw true try type ' +
    'typeof undefined while yield'
  ).split(' '),
)

const PATTERN =
  /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|('(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`)|(\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_$][\w$]*)/g

/**
 * Split TypeScript (and TSX) into tokens for display: comments, strings, numbers, keywords and capitalised names
 * (types and components). A display aid, not a parser: it is right for the code the sites show.
 *
 * @param source - The code.
 * @returns The tokens, covering the source exactly.
 */
export function highlight(source: string): Token[] {
  const out: Token[] = []
  let last = 0
  for (const m of source.matchAll(PATTERN)) {
    const at = m.index ?? 0
    const [text, comment, string, number, word] = m
    let kind: Token['kind']
    if (comment !== undefined) kind = 'comment'
    else if (string !== undefined) kind = 'string'
    else if (number !== undefined) kind = 'number'
    else if (word !== undefined && KEYWORDS.has(word)) kind = 'keyword'
    else if (word !== undefined && /^[A-Z]/.test(word)) kind = 'type'
    if (kind === undefined) continue
    if (at > last) out.push({ text: source.slice(last, at) })
    out.push({ text, kind })
    last = at + text.length
  }
  if (last < source.length) out.push({ text: source.slice(last) })
  return out
}
