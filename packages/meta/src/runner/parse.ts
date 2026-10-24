/**
 * Reading an example: its lines, the modules it imports and its `// =>` expectations. No JavaScript parser is needed:
 * an expectation's expression is found by walking back from its marker until the brackets balance.
 */
import type { Expectation, ExpectationKind } from '../schema/examples.ts'

/** A line split into its code and its trailing `//` comment (the first `//` outside a string). */
export interface SplitLine {
  readonly code: string
  /** The comment's text after `//`, or `null` when the line has none. */
  readonly comment: string | null
}

/** An example read into lines, imports and expectations. */
export interface ParsedExample {
  readonly lines: readonly string[]
  /** The module specifiers its `import` statements name, in order, without repeats. */
  readonly imports: readonly string[]
  readonly expectations: readonly ParsedExpectation[]
}

/** An expectation with what instrumenting it needs: the declaration it initialises and the marker's own line. */
export interface ParsedExpectation extends Expectation {
  /** A `const x = ` (or `let`, `var`) before the expression on its first line, kept outside the instrumentation. */
  readonly prefix: string
  /** The line the `// =>` marker is written on (later than `line` for a marker on a line of its own). */
  readonly markerLine: number
}

const QUOTES = new Set(["'", '"', '`'])

/**
 * Split a line into code and its trailing `//` comment, skipping `//` inside string and template literals.
 *
 * @param line - One line of code.
 * @returns The code (right-trimmed) and the comment text, or `null` for no comment.
 */
export function splitComment(line: string): SplitLine {
  let quote: string | null = null
  for (let i = 0; i < line.length; i++) {
    const c = line[i] as string
    if (quote !== null) {
      if (c === '\\') i++
      else if (c === quote) quote = null
    } else if (QUOTES.has(c)) quote = c
    else if (c === '/' && line[i + 1] === '/') return { code: line.slice(0, i).trimEnd(), comment: line.slice(i + 2) }
  }
  return { code: line.trimEnd(), comment: null }
}

/**
 * The bracket depth a piece of code changes by: `(`, `[` and `{` add one, their partners take one away. Brackets
 * inside string literals do not count.
 *
 * @param code - Code without comments.
 * @returns The net change in depth.
 */
export function bracketBalance(code: string): number {
  let depth = 0
  let quote: string | null = null
  for (let i = 0; i < code.length; i++) {
    const c = code[i] as string
    if (quote !== null) {
      if (c === '\\') i++
      else if (c === quote) quote = null
    } else if (QUOTES.has(c)) quote = c
    else if (c === '(' || c === '[' || c === '{') depth++
    else if (c === ')' || c === ']' || c === '}') depth--
  }
  return depth
}

// A line that continues the expression above it (`.map(...)`, `?.x`, a binary operator).
const CONTINUES = /^\s*(\.|\?\.|\?\?|\|\||&&|[+\-*/%]\s|[<>]=?\s|===?\s|!==?\s|\?\s|:\s)/
// A line that leaves its expression unfinished (`a +`, `const x =`).
const UNFINISHED = /(=>|[=+\-*/%,?:]|\|\||&&|\?\?)\s*$/
const DECLARATION = /^(\s*(?:export\s+)?(?:const|let|var)\s+[^=]+?=(?![=>])\s*)(.*)$/

function kindOf(expected: string): { kind: ExpectationKind; expected: string } {
  const throws = /^throws\b\s*([\w.-]*)/.exec(expected)
  if (throws !== null) return { kind: 'throws', expected: (throws[1] ?? '').replace(/[.:]+$/, '') }
  if (/^figure\b/.test(expected)) return { kind: 'figure', expected: '' }
  if (/^trace\b/.test(expected)) return { kind: 'trace', expected: '' }
  return { kind: 'value', expected }
}

/**
 * The module specifiers an example imports (static `import ... from '...'` and `import '...'`).
 *
 * @param code - The example's code.
 * @returns The specifiers, in order of appearance, without repeats.
 */
export function importsOf(code: string): string[] {
  const out: string[] = []
  for (const m of code.matchAll(/^\s*import\s+(?:[^'";]*?\s+from\s+)?['"]([^'"]+)['"]/gm)) {
    const spec = m[1] as string
    if (!out.includes(spec)) out.push(spec)
  }
  return out
}

/**
 * Read an example: its lines, imports and `// =>` expectations. A marker ends the expression on its line, or, on a
 * line of its own, the expression on the code line above; the expression starts where its brackets balance.
 *
 * @param code - The example's code.
 * @returns The parsed example.
 */
export function parseExample(code: string): ParsedExample {
  const lines = code.split('\n')
  const split = lines.map(splitComment)
  const expectations: ParsedExpectation[] = []
  for (let i = 0; i < lines.length; i++) {
    const comment = split[i]?.comment
    if (comment === null || comment === undefined) continue
    const marker = /^\s*=>\s?(.*)$/.exec(comment)
    if (marker === null) continue
    let end = i
    if ((split[i]?.code ?? '').trim() === '') {
      end = i - 1
      while (end >= 0 && (split[end]?.code ?? '').trim() === '') end--
      if (end < 0) continue
    }
    let start = end
    let depth = bracketBalance(split[end]?.code ?? '')
    while (start > 0) {
      const here = split[start]?.code ?? ''
      const above = split[start - 1]?.code ?? ''
      if (depth < 0 || CONTINUES.test(here) || (above.trim() !== '' && UNFINISHED.test(above))) {
        start--
        depth += bracketBalance(above)
      } else break
    }
    const body = split.slice(start, end + 1).map((s) => s.code)
    const first = body[0] ?? ''
    const declaration = DECLARATION.exec(first)
    const prefix = declaration === null ? (/^\s*;?\s*/.exec(first)?.[0] ?? '') : (declaration[1] as string)
    body[0] = first.slice(prefix.length)
    body[body.length - 1] = (body[body.length - 1] ?? '').replace(/;\s*$/, '')
    const { kind, expected } = kindOf((marker[1] ?? '').trim())
    expectations.push({
      kind,
      startLine: start,
      line: end,
      markerLine: i,
      expression: body.join('\n'),
      expected,
      prefix,
    })
  }
  return { lines, imports: importsOf(code), expectations }
}
