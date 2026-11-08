/**
 * CRFsuite-style attribute templates for sequence labelling: each template joins the values of chosen columns at chosen
 * offsets around a token (such as `w[-1]|w[0]`, the previous and current words) into one feature. Template parsing and
 * expansion over a token sequence, and MALLET's offset conjunctions.
 */

import { OmaiError } from '../base/index.ts'

/** One token of a sequence as named columns (word, part of speech, ...), the input of attribute templates. */
export type Token = Readonly<Record<string, string>>

/** The attributes of one item of a sequence, with values (1 for an indicator, as CRFsuite). */
export type ItemAttributes = Readonly<Record<string, number>>

/** A CRFsuite attribute template: (column, offset) pairs joined into one attribute, as `w[-1]|w[0]`. */
export type AttributeTemplate = readonly (readonly [string, number])[]

/** Options of {@link expandAttributeTemplates}. */
export interface AttributeTemplateOptions {
  /** Cells outside the sequence: `skip` the template there (CRFsuite, default) or `pad` with `_B-k` / `_B+k` (CRF++). */
  readonly boundary?: 'skip' | 'pad'
  /** Add a `bias` attribute at every position. */
  readonly bias?: boolean
  /** Add `__BOS__` at the first and `__EOS__` at the last position (CRFsuite's crfutils). */
  readonly bosEos?: boolean
}

/**
 * Expand CRFsuite-style attribute templates over a sequence of tokens (crfutils' `apply_templates`): each template
 * joins the values of its (column, offset) cells into one attribute named like `w[-1]|w[0]=the|cat`. Cells outside
 * the sequence skip the template (`boundary: 'skip'`) or are padded with CRF++'s `_B-k` / `_B+k` (`'pad'`).
 *
 * @param tokens - The tokens.
 * @param templates - The templates.
 * @param options - Boundary handling, the bias attribute and sentence markers.
 * @returns The attributes of each position (value 1).
 * @concept crf-features
 * @example
 * ```ts
 * import { expandAttributeTemplates } from 'omai-engine/text'
 * const tokens = [{ w: 'the' }, { w: 'cat' }]
 * expandAttributeTemplates(tokens, [[['w', 0]], [['w', -1], ['w', 0]]]) // => [{ 'w[0]=the': 1 }, { 'w[0]=cat': 1, 'w[-1]|w[0]=the|cat': 1 }]
 * ```
 */
export function expandAttributeTemplates(
  tokens: readonly Token[],
  templates: readonly AttributeTemplate[],
  options: AttributeTemplateOptions = {},
): Record<string, number>[] {
  const pad = options.boundary === 'pad'
  return tokens.map((_, t) => {
    const out: Record<string, number> = {}
    if (options.bias === true) out['bias'] = 1
    for (const template of templates) {
      const values: string[] = []
      let ok = true
      for (const [column, offset] of template) {
        const i = t + offset
        const v =
          i < 0
            ? pad
              ? `_B${i}`
              : undefined
            : i >= tokens.length
              ? pad
                ? `_B+${i - tokens.length + 1}`
                : undefined
              : tokens[i]?.[column]
        if (v === undefined) {
          ok = false
          break
        }
        values.push(v)
      }
      if (!ok) continue
      out[`${template.map(([c, o]) => `${c}[${o}]`).join('|')}=${values.join('|')}`] = 1
    }
    if (options.bosEos === true) {
      if (t === 0) out['__BOS__'] = 1
      if (t === tokens.length - 1) out['__EOS__'] = 1
    }
    return out
  })
}

/**
 * Parse CRFsuite attribute templates written as text: one template per line (or `;`), cells `column[offset]` joined
 * by `|`, so `w[-1]|w[0]` is the pair of the previous and current words.
 *
 * @param source - The templates.
 * @returns The templates.
 * @throws OmaiError with code `value` and the line number for a malformed cell.
 * @concept crf-features
 * @example
 * ```ts
 * import { parseAttributeTemplates } from 'omai-engine/text'
 * parseAttributeTemplates('w[0]\nw[-1]|pos[0]') // => [[['w', 0]], [['w', -1], ['pos', 0]]]
 * ```
 */
export function parseAttributeTemplates(source: string): AttributeTemplate[] {
  const out: AttributeTemplate[] = []
  for (const [k, raw] of source.split(/[\n;]/).entries()) {
    const text = raw.trim()
    if (text.length === 0 || text.startsWith('#')) continue
    out.push(
      text.split('|').map((cell): readonly [string, number] => {
        const m = /^([^[\]|]+)\[(-?\d+)\]$/.exec(cell.trim())
        if (m === null)
          throw new OmaiError('value', `attribute template ${k + 1}: malformed cell ${cell}`, { line: k + 1 })
        const offset = Number(m[2])
        return [m[1] ?? '', offset === 0 ? 0 : offset]
      }),
    )
  }
  return out
}

/**
 * Offset conjunctions as MALLET's `OffsetConjunctions` pipe builds them: for each conjunction (a list of offsets), the
 * products of the features at those offsets, named `f@-1&g` (offset 0 is unmarked) with the product of their values;
 * a conjunction is skipped where any offset falls outside the sequence, and pairs at the same offset are taken once
 * (upper triangle). Original features are kept unless `includeOriginalSingletons` is false.
 *
 * @param items - The features of each position.
 * @param conjunctions - The offset lists (for example `[[-1], [1], [-1, 0]]`).
 * @param options - Whether to keep the original features (default true).
 * @returns The conjoined features of each position.
 * @throws OmaiError with code `value` for a single offset 0 while original features are kept.
 * @concept crf-features
 * @example
 * ```ts
 * import { offsetConjunctions } from 'omai-engine/text'
 * offsetConjunctions([{ 'W=a': 1 }, { 'W=b': 1 }], [[-1], [-1, 0]]) // => [{ 'W=a': 1 }, { 'W=b': 1, 'W=a@-1': 1, 'W=a@-1&W=b': 1 }]
 * ```
 */
export function offsetConjunctions(
  items: readonly ItemAttributes[],
  conjunctions: readonly (readonly number[])[],
  options: { readonly includeOriginalSingletons?: boolean } = {},
): Record<string, number>[] {
  const keep = options.includeOriginalSingletons !== false
  for (const c of conjunctions)
    if (c.length === 1 && c[0] === 0 && keep)
      throw new OmaiError('value', 'offsetConjunctions: original singletons are already included')
  const lists = items.map((f) => Object.entries(f))
  return items.map((f, i) => {
    const out: Record<string, number> = keep ? { ...f } : {}
    for (const conj of conjunctions) {
      if (conj.some((o) => i + o < 0 || i + o >= items.length)) continue
      const choose = (k: number, picked: readonly (readonly [string, number, number])[]): void => {
        if (k === conj.length) {
          const key = picked.map(([name, , o]) => (o === 0 ? name : `${name}@${o}`)).join('&')
          const value = picked.reduce((p, [, v]) => p * v, 1)
          out[key] = (out[key] ?? 0) + value
          return
        }
        const o = conj[k] ?? 0
        const list = lists[i + o] ?? []
        // Equal offsets in one conjunction take each unordered pair once (MALLET's upper triangle).
        let start = 0
        for (let j = k - 1; j >= 0; j--) {
          if (conj[j] === o) {
            start = indexOf(list, picked[j]) + 1
            break
          }
        }
        for (let q = start; q < list.length; q++) {
          const [name, value] = list[q] ?? ['', 0]
          choose(k + 1, [...picked, [name, value, o]])
        }
      }
      choose(0, [])
    }
    return out
  })
}

function indexOf(
  list: readonly (readonly [string, number])[],
  picked: readonly [string, number, number] | undefined,
): number {
  return picked === undefined ? -1 : list.findIndex(([n]) => n === picked[0])
}
