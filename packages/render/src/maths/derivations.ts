/** One step of a derivation: TeX (`lhs &= rhs`, or a right-hand side continuing the previous line) and a note. */
export interface DerivationStep {
  readonly tex: string
  readonly note?: string
}

/**
 * Steps of a derivation (from `symbolic` simplifications or `foundations` proofs, or written by hand) as one
 * `aligned` TeX block: one step per row, aligned at the relation, notes set as text on the right.
 *
 * @param steps - The steps.
 * @returns The TeX.
 * @example
 * ```ts
 * import { derivationTex } from 'whitebox-render/maths'
 * const t = derivationTex([{ tex: 'x &= 1' }, { tex: '2 - 1', note: 'rewrite' }])
 * t // => '\\begin{aligned}x &= 1 \\\\ &= 2 - 1 && \\text{rewrite}\\end{aligned}'
 * ```
 */
export function derivationTex(steps: readonly DerivationStep[]): string {
  const rows = steps.map((s) => {
    const row = s.tex.includes('&') ? s.tex : `&= ${s.tex}`
    return s.note === undefined ? row : `${row} && \\text{${s.note.replace(/[{}\\]/g, '')}}`
  })
  return `\\begin{aligned}${rows.join(' \\\\ ')}\\end{aligned}`
}
