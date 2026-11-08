import { describe, expect, it } from 'vitest'
import { derivationTex } from './derivations.ts'
import { bindTex, live, numberTex, tex, toTex } from './live.ts'
import { extendMacros, macros } from './macros.ts'
import { texError, texExtent, texRenderer, texToHtml } from './tex.ts'

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
    expect(derivationTex([{ tex: 'a', note: '{bad}' }])).toContain('\\text{\\{bad\\}}')
  })
})

describe('texExtent', () => {
  // The widths of KaTeX's HTML output at 100 px in Chrome with KaTeX's stylesheet (`getBoundingClientRect`).
  const measured: readonly (readonly [string, number])[] = [
    ['q(x_t | x_0)', 409.6],
    ['q(\\mathbf x_t \\mid \\mathbf x_0)', 485.2],
    ['\\mathbf{x}_t', 110.1],
    ['\\frac{a}{b}', 73.8],
    ['\\epsilon_\\theta(\\mathbf{x}_t, t)', 399],
    ['\\lVert \\epsilon - \\epsilon_\\theta \\rVert^2', 463.7],
    ['\\textbf{TRAINING}', 739],
    ['\\textsf{\\textbf{TRAINING}}', 623.4],
    ['\\approx \\mathcal{N}(\\mathbf{0}, \\mathbf{I})', 515.1],
    ['\\sqrt{x}', 170],
    ['\\sum_{i=1}^n x_i', 396],
    ['p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)', 632.4],
    ['12.5 \\sin x', 473.3],
    ['\\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}', 491],
  ]
  it('measures typeset TeX within 5% of the browser, not by its source length', () => {
    for (const [source, width] of measured) {
      const got = texExtent(source, 100).width
      expect(got / width, source).toBeGreaterThan(0.95)
      expect(got / width, source).toBeLessThan(1.05)
    }
    // The source of `\\mathbf{x}_t` is 13 characters, the formula is a bold x with a small t.
    expect(texExtent('\\mathbf{x}_t', 14).width).toBeLessThan(20)
  })
  it('scales with the size, has a height, and falls back to the source for what does not parse', () => {
    expect(texExtent('x^2', 20).width).toBeCloseTo(2 * texExtent('x^2', 10).width, 6)
    expect(texExtent('\\frac{a}{b}', 10).height).toBeGreaterThan(texExtent('a', 10).height)
    const bad = texExtent('\\frac{1', 10)
    expect(bad.width).toBeGreaterThan(0)
    expect(texExtent('x', 10, { '\\y': 'z' }).width).toBeGreaterThan(0)
  })
})
