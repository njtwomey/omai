/**
 * The `@index` tag of the agent index: `@index <kind> <description>`. The first word is the kind (a closed vocabulary,
 * `INDEX_KINDS`), the rest a one-line description. This file parses and validates the tags `parseDocComment` collects
 * and reports problems (unknown kind, empty description, too long, duplicate) with their file and line.
 */
import { INDEX_KINDS, type IndexKind, type IndexProblem, type IndexTag } from '../schema/agent-index.ts'

/** A description longer than this many words is reported as a warning. */
export const INDEX_WORDS_WARN = 16
/** A description longer than this many words is an error and the tag is dropped. */
export const INDEX_WORDS_MAX = 24

const KINDS: readonly string[] = INDEX_KINDS

/** The number of whitespace-separated words in a text. */
function words(text: string): number {
  return text.split(/\s+/).filter((w) => w !== '').length
}

/**
 * Parse the text after `@index` into a tag.
 *
 * @param text - The tag text: a kind, then the description.
 * @returns The tag, or the reason it is not one (unknown kind, no description, more than {@link INDEX_WORDS_MAX} words).
 * @example
 * ```ts
 * import { parseIndexTag } from 'omai-meta/extract'
 * parseIndexTag('function Solve a linear system.') // => { tag: { kind: 'function', description: 'Solve a linear system.' } }
 * ```
 */
export function parseIndexTag(text: string): { readonly tag: IndexTag } | { readonly error: string } {
  const trimmed = text.replace(/\s+/g, ' ').trim()
  const space = trimmed.indexOf(' ')
  const kind = space < 0 ? trimmed : trimmed.slice(0, space)
  const description = space < 0 ? '' : trimmed.slice(space + 1).trim()
  if (kind === '') return { error: `@index needs a kind and a description (kinds: ${INDEX_KINDS.join(', ')})` }
  if (!KINDS.includes(kind)) return { error: `unknown @index kind '${kind}' (kinds: ${INDEX_KINDS.join(', ')})` }
  if (description === '') return { error: `@index ${kind} needs a description` }
  const n = words(description)
  if (n > INDEX_WORDS_MAX) return { error: `@index description has ${n} words; at most ${INDEX_WORDS_MAX}, aim for 12` }
  return { tag: { kind: kind as IndexKind, description } }
}

/**
 * Validate the `@index` tags of one doc comment (at most one is allowed) and turn the first into a tag.
 *
 * @param tags - The raw tags of `ParsedDoc.indexTags`.
 * @param file - The repository-relative path of the file the comment is in.
 * @param commentLine - The 1-based file line the comment text starts on, so reported lines are file lines.
 * @returns The tag (absent when the comment has none or the tag is invalid) and the problems found.
 * @example
 * ```ts
 * import { readIndexTags } from 'omai-meta/extract'
 * const r = readIndexTags([{ text: 'verb Do it.', line: 2 }], 'a.ts', 10)
 * r.problems[0]?.line // => 12
 * ```
 */
export function readIndexTags(
  tags: readonly { readonly text: string; readonly line: number }[],
  file: string,
  commentLine: number,
): { readonly index?: IndexTag; readonly problems: readonly IndexProblem[] } {
  const problems: IndexProblem[] = []
  const first = tags[0]
  if (first === undefined) return { problems }
  for (const extra of tags.slice(1))
    problems.push({
      severity: 'error',
      file,
      line: commentLine + extra.line,
      message: 'duplicate @index tag (one per declaration or file)',
    })
  const parsed = parseIndexTag(first.text)
  if ('error' in parsed) {
    problems.push({ severity: 'error', file, line: commentLine + first.line, message: parsed.error })
    return { problems }
  }
  const n = words(parsed.tag.description)
  if (n > INDEX_WORDS_WARN)
    problems.push({
      severity: 'warning',
      file,
      line: commentLine + first.line,
      message: `@index description has ${n} words; aim for at most 12`,
    })
  return { index: parsed.tag, problems }
}
