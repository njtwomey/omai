/**
 * Derivations as typeset maths: a list of steps (from `symbolic` simplifications or `foundations` proofs, or written by
 * hand) becomes one `aligned` TeX block, a step per row aligned at the relation, with each step's note set as text
 * beside it.
 */

/** One step of a derivation: TeX (`lhs &= rhs`, or a right-hand side continuing the previous line) and a note. */
export interface DerivationStep {
  readonly tex: string
  readonly note?: string
}

/** TeX's special characters, as `\text{}` accepts them. */
const TEXT_ESCAPES: Readonly<Record<string, string>> = {
  '\\': '\\textbackslash{}',
  '{': '\\{',
  '}': '\\}',
  '#': '\\#',
  '%': '\\%',
  '&': '\\&',
  _: '\\_',
  '^': '\\textasciicircum{}',
  '~': '\\textasciitilde{}',
}

/**
 * A plain-text note as TeX: text in `\text{}` with TeX's special characters escaped, and `$...$` segments kept as
 * inline maths (`'add and subtract $(b/2a)^2$'`).
 *
 * @param note - The note.
 * @returns The TeX.
 * @example
 * ```ts
 * import { noteTex } from 'omai-render/maths'
 * noteTex('add and subtract $(b/2a)^2$') // => '\\text{add and subtract }(b/2a)^2'
 * noteTex('50% of x_1') // => '\\text{50\\% of x\\_1}'
 * ```
 */
export function noteTex(note: string): string {
  return note
    .split(/(\$[^$]*\$)/g)
    .filter((part) => part !== '')
    .map((part) =>
      part.length >= 2 && part.startsWith('$') && part.endsWith('$')
        ? part.slice(1, -1)
        : `\\text{${part.replace(/[\\{}#%&_^~]/g, (c) => TEXT_ESCAPES[c] ?? c)}}`,
    )
    .join('')
}

/**
 * Steps of a derivation (from `symbolic` simplifications or `foundations` proofs, or written by hand) as one
 * `aligned` TeX block: one step per row, aligned at the relation, notes set as text on the right (plain text, with
 * `$...$` for inline maths; see {@link noteTex}).
 *
 * @param steps - The steps.
 * @returns The TeX.
 * @example
 * ```ts
 * import { derivationTex } from 'omai-render/maths'
 * const t = derivationTex([{ tex: 'x &= 1' }, { tex: '2 - 1', note: 'rewrite' }])
 * t // => '\\begin{aligned}x &= 1 \\\\ &= 2 - 1 && \\text{rewrite}\\end{aligned}'
 * ```
 */
export function derivationTex(steps: readonly DerivationStep[]): string {
  const rows = steps.map((s) => {
    const row = s.tex.includes('&') ? s.tex : `&= ${s.tex}`
    return s.note === undefined ? row : `${row} && ${noteTex(s.note)}`
  })
  return `\\begin{aligned}${rows.join(' \\\\ ')}\\end{aligned}`
}
