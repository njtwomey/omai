/**
 * TSDoc comments as data: the summary and description, `@param`, `@returns`, `@throws`, `@example`, `@concept`,
 * `@see`, `@reference`, `@index` (kept as raw text; `index-tags.ts` validates it), `@deprecated`, `@remarks` and the
 * modifier tags (`@internal`, `@alpha`, `@beta`, `@experimental`). Plain string work on the comment text, so it runs
 * anywhere.
 */
import type { Example } from '../schema/examples.ts'

/** A doc comment read into its parts. */
export interface ParsedDoc {
  readonly summary: string
  readonly description: string
  /** `@param` descriptions in order, by name (`options.field` for a field of an options object). */
  readonly params: readonly { readonly name: string; readonly description: string }[]
  readonly typeParams: readonly { readonly name: string; readonly description: string }[]
  readonly returns: string
  readonly throws: readonly string[]
  /** The examples; `line` in each `source` is the 0-based line of the comment text on which its code starts. */
  readonly examples: readonly Example[]
  readonly concepts: readonly string[]
  readonly see: readonly string[]
  readonly references: readonly string[]
  /** Every `@index` tag as written (text after the tag, on one line) with the 0-based comment line it starts on. */
  readonly indexTags: readonly { readonly text: string; readonly line: number }[]
  readonly deprecated?: string
  readonly defaultValue?: string
  /** Modifier tags present, without the `@`. */
  readonly modifiers: readonly string[]
}

const MODIFIERS = ['internal', 'alpha', 'beta', 'experimental', 'public', 'hidden', 'packageDocumentation', 'sealed']
const MODIFIER = new RegExp(`(^|\\s)@(${MODIFIERS.join('|')})\\b`, 'g')
const FENCE = /^\s*(```|~~~)/

/**
 * The text of a `/** … *\/` comment without its frame, one output line per comment line (so line numbers carry over).
 *
 * @param comment - The comment, frame included.
 * @returns The text, with the leading ` * ` of each line removed.
 */
export function unframeComment(comment: string): string {
  return comment
    .replace(/^\/\*\*/, '')
    .replace(/\*\/$/, '')
    .split('\n')
    .map((line) => line.replace(/^\s*\* ?/, '').replace(/^\s+$/, ''))
    .join('\n')
}

function squash(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function paragraphs(text: string): string[] {
  const out: string[] = []
  let current: string[] = []
  let fenced = false
  for (const line of text.split('\n')) {
    if (FENCE.test(line)) fenced = !fenced
    if (!fenced && line.trim() === '') {
      if (current.length > 0) out.push(current.join('\n'))
      current = []
    } else current.push(line)
  }
  if (current.length > 0) out.push(current.join('\n'))
  return out
}

interface Block {
  readonly tag: string
  readonly lines: string[]
  readonly start: number
}

function exampleOf(block: Block): Example | undefined {
  const [head = '', ...rest] = block.lines
  const fenceAt = rest.findIndex((l) => FENCE.test(l))
  if (fenceAt >= 0) {
    const closeAt = rest.findIndex((l, i) => i > fenceAt && FENCE.test(l))
    const code = rest.slice(fenceAt + 1, closeAt < 0 ? undefined : closeAt).join('\n')
    const before = squash(rest.slice(0, fenceAt).join(' '))
    const title = squash(head) || before
    if (code.trim() === '') return undefined
    return { title, code, source: { file: '', line: block.start + 1 + fenceAt + 1 } }
  }
  // Unfenced: a one-line `@example code` has no title; otherwise the tag's line is the title.
  const titled = rest.some((l) => l.trim() !== '')
  const lines = titled ? rest : [head]
  const first = titled ? block.start + 1 : block.start
  const lead = lines.findIndex((l) => l.trim() !== '')
  const code = lines.slice(Math.max(lead, 0)).join('\n').replace(/\s+$/, '')
  if (code.trim() === '') return undefined
  return { title: titled ? squash(head) : '', code, source: { file: '', line: first + Math.max(lead, 0) } }
}

function named(text: string): { name: string; description: string } {
  const m = /^(?:\{[^}]*\}\s*)?(\[?[\w$.]+(?:=[^\]]*)?\]?)\s*(?:-\s*)?([\s\S]*)$/.exec(text.trim())
  const name = (m?.[1] ?? '').replace(/^\[|\]$/g, '').replace(/=.*$/, '')
  return { name, description: squash(m?.[2] ?? '') }
}

/**
 * Read a doc comment's text (from {@link unframeComment}) into its parts. Tags start at the beginning of a line; a
 * fenced code block never ends a tag. The summary is the first paragraph, the description the rest of the prose and
 * any `@remarks`.
 *
 * @param text - The unframed comment text.
 * @returns The parsed comment.
 */
export function parseDocComment(text: string): ParsedDoc {
  const lines = text.split('\n')
  const prose: string[] = []
  const blocks: Block[] = []
  let fenced = false
  let block: Block | null = null
  lines.forEach((line, i) => {
    const tag = fenced ? null : /^\s*@(\w+)\b\s?(.*)$/.exec(line)
    if (FENCE.test(line)) fenced = !fenced
    if (tag !== null && !MODIFIERS.includes(tag[1] as string)) {
      block = { tag: tag[1] as string, lines: [tag[2] ?? ''], start: i }
      blocks.push(block)
    } else if (block !== null) (block as Block).lines.push(line)
    else prose.push(line)
  })
  const modifiers = new Set<string>()
  for (const m of text.matchAll(MODIFIER)) {
    // A modifier inside a code fence or an example is not a tag; tags in prose and on their own lines count.
    modifiers.add(m[2] as string)
  }
  const clean = (s: string): string => s.replace(MODIFIER, '$1').replace(/[ \t]+$/gm, '')
  const parts = paragraphs(clean(prose.join('\n')).trim())
  const params: { name: string; description: string }[] = []
  const typeParams: { name: string; description: string }[] = []
  const throws: string[] = []
  const examples: Example[] = []
  const concepts: string[] = []
  const see: string[] = []
  const references: string[] = []
  const indexTags: { text: string; line: number }[] = []
  const remarks: string[] = []
  let returns = ''
  let deprecated: string | undefined
  let defaultValue: string | undefined
  for (const b of blocks) {
    const body = clean(b.lines.join('\n'))
    switch (b.tag) {
      case 'param':
        params.push(named(body))
        break
      case 'typeParam':
      case 'template':
        typeParams.push(named(body))
        break
      case 'returns':
      case 'return':
        returns = squash(body)
        break
      case 'throws':
        throws.push(squash(body))
        break
      case 'example': {
        const e = exampleOf(b)
        if (e !== undefined) examples.push(e)
        break
      }
      case 'concept':
        concepts.push(...body.split(/[\s,]+/).filter((c) => c !== ''))
        break
      case 'see':
        see.push(squash(body))
        break
      case 'reference':
        references.push(squash(body))
        break
      case 'index':
        indexTags.push({ text: squash(body), line: b.start })
        break
      case 'remarks':
        remarks.push(body.trim())
        break
      case 'deprecated':
        deprecated = squash(body)
        break
      case 'defaultValue':
      case 'default':
        defaultValue = squash(body)
        break
      default:
        break
    }
  }
  return {
    summary: squash(parts[0] ?? ''),
    description: [...parts.slice(1), ...remarks].join('\n\n'),
    params,
    typeParams,
    returns,
    throws,
    examples,
    concepts,
    see,
    references,
    indexTags,
    ...(deprecated === undefined ? {} : { deprecated }),
    ...(defaultValue === undefined ? {} : { defaultValue }),
    modifiers: [...modifiers],
  }
}
