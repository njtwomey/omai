import { describe, expect, it } from 'vitest'
import { derivationTex } from './derivations.ts'
import { bindTex, live, numberTex, tex, toTex } from './live.ts'
import { extendMacros, macros } from './macros.ts'
import { texError, texRenderer, texToHtml } from './tex.ts'

describe('maths', () => {
  it('has the shared macros and extends them without mutation', () => {
    expect(macros['\\Sigmamat']).toBe('\\boldsymbol{\\Sigma}')
    expect(macros['\\Dcal']).toBe('\\mathcal{D}')
    const more = extendMacros({ '\\R': '\\mathbb{R}' })
    expect(more['\\R']).toBe('\\mathbb{R}')
    expect(macros['\\R']).toBeUndefined()
  })
  it('typesets with macros in every output, never throwing', () => {
    expect(texToHtml('\\norm{\\xvec}', { output: 'html' })).toContain('katex')
    expect(texToHtml('\\frac{1}', { display: true })).toContain('KaTeX parse error')
    expect(texToHtml('\\htmlClass{wb-slot}{2}', { output: 'html' })).toContain('wb-slot')
    expect(texError('\\Gauss(\\muvec, \\Sigmamat)')).toBeUndefined()
    const r = texRenderer({ output: 'html' })
    expect(r('x', true)).toContain('katex-display')
  })
  it('binds live values', () => {
    expect(toTex(tex`a = ${'\\pi'}, b = ${live(1 / 3, { digits: 2 })}`)).toBe(
      'a = \\htmlClass{wb-slot}{\\pi}, b = \\htmlClass{wb-slot}{0.33}',
    )
    expect(numberTex(Number.NaN)).toBe('\\text{NaN}')
    expect(numberTex(2.5e7)).toBe('2.5 \\times 10^{7}')
    expect(bindTex('{{ k }} and {{missing}}', { k: 'x' })).toBe('\\htmlClass{wb-slot}{x} and missing')
    expect(derivationTex([{ tex: 'a', note: '{bad}' }])).toContain('\\text{bad}')
  })
})
