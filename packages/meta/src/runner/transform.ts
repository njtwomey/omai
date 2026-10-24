/**
 * Turning an example into JavaScript the runner can call: each expectation's expression is wrapped in a call that
 * records its value (or the error it throws), then sucrase strips the types and rewrites the imports into `require`
 * calls. Line numbers are kept, so errors point at the example's own lines.
 */
import { transform } from 'sucrase'
import { type ParsedExample, parseExample } from './parse.ts'

/** The names the instrumented code calls; the runner passes functions under these names. */
export const HOOKS: readonly ['require', '__at', '__throws', 'console'] = ['require', '__at', '__throws', 'console']

/** An example as JavaScript, with what parsing found. */
export interface TransformedExample {
  readonly parsed: ParsedExample
  /** The body of an async function taking {@link HOOKS} as parameters. */
  readonly js: string
}

/**
 * Wrap every expectation of a parsed example in a hook call, keeping every line where it was. A value expectation
 * becomes `__at(k, (expr), () => (expected))`, `figure` and `trace` become `__at(k, (expr))`, and `throws` becomes
 * `await __throws(k, async () => (expr))`.
 *
 * @param parsed - The example, from {@link parseExample}.
 * @returns The instrumented TypeScript.
 */
export function instrument(parsed: ParsedExample): string {
  const lines = [...parsed.lines]
  parsed.expectations.forEach((e, k) => {
    const body = e.expression.split('\n')
    const open = e.kind === 'throws' ? `await __throws(${k}, async () => (` : `__at(${k}, (`
    const close = e.kind === 'value' ? `), () => (${e.expected}))` : '))'
    body[0] = `${e.prefix}${open}${body[0] ?? ''}`
    body[body.length - 1] = `${body[body.length - 1] ?? ''}${close}`
    for (let j = 0; j < body.length; j++) lines[e.startLine + j] = body[j] ?? ''
    if (e.markerLine !== e.line) lines[e.markerLine] = ''
  })
  return lines.join('\n')
}

/**
 * Parse, instrument and compile an example: types stripped, imports turned into `require` calls.
 *
 * @param code - The example's TypeScript.
 * @returns The JavaScript body and the parse.
 * @throws SyntaxError (from sucrase, with `(line:column)` in its message) for code that does not parse.
 */
export function transformExample(code: string): TransformedExample {
  const parsed = parseExample(code)
  const js = transform(instrument(parsed), {
    transforms: ['typescript', 'imports'],
    production: true,
    disableESTransforms: true,
  }).code
  return { parsed, js }
}
