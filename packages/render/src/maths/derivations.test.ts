import { describe, expect, it } from 'vitest'
import { derivationTex, noteTex } from './derivations.ts'
import { texError, texToHtml } from './tex.ts'

describe('derivation notes', () => {
  it('escapes TeX specials in text and keeps $...$ as maths', () => {
    expect(noteTex('add and subtract $(b/2a)^2$')).toBe('\\text{add and subtract }(b/2a)^2')
    expect(noteTex('50% of x_1 & {y}^2 ~ #3 \\')).toBe(
      '\\text{50\\% of x\\_1 \\& \\{y\\}\\textasciicircum{}2 \\textasciitilde{} \\#3 \\textbackslash{}}',
    )
  })

  it('always gives TeX that parses', () => {
    const tex = derivationTex([
      { tex: 'ax^2 + bx &= a(x^2 + (b/a) x)', note: 'factor out a' },
      { tex: 'a(x^2 + (b/a)x + (b/2a)^2 - (b/2a)^2)', note: 'add and subtract (b/2a)^2' },
      { tex: 'a(x + b/2a)^2 - b^2/4a', note: 'complete the square, $h = b/2a$' },
    ])
    expect(texError(tex)).toBeUndefined()
  })

  it('shows a compact marker for TeX that does not parse', () => {
    const html = texToHtml('\\frac{1}')
    expect(html).toContain('wb-tex-error')
    expect(html.endsWith('>⚠ TeX</span>')).toBe(true)
  })
})
