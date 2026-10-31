import { equationFit } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { equationFitView, fittedTex, residualStripView, termCounts, truthTex } from './equation-fit.ts'

const r = equationFit({
  target: 'y',
  terms: [
    { tex: '1', estimate: 0.31, truth: 0.3 },
    { tex: 'x', estimate: 0.4, truth: 0 },
    { tex: 'x^2', estimate: -0.42, truth: -0.4 },
    { tex: 'x^3', estimate: 0, truth: 0 },
  ],
  x: [0, 1, 2],
  fit: [0, 1, 2],
  truth: [0, 1, 1.5],
  observed: { x: [0, 1], y: [0.1, 0.9] },
})

describe('the equation against the truth', () => {
  it('typesets the fit with a coloured coefficient per status, dropping the terms at zero', () => {
    const plain = fittedTex(r, 'light', false)
    expect(plain).toBe('y = 0.31 + 0.4\\,x - 0.42\\,x^2')
    const coloured = fittedTex(r, 'light')
    expect(coloured).toContain('\\textcolor{')
    expect(coloured.match(/\\textcolor/g)?.length).toBe(3)
    expect(fittedTex(equationFit({ expression: '\\sqrt{x}', x: [0, 1], fit: [0, 1] }))).toBe('y = \\sqrt{x}')
  })

  it('typesets the truth from the terms that are in the law, or the given expression', () => {
    expect(truthTex(r)).toBe('y = 0.3 - 0.4\\,x^2')
    expect(truthTex(equationFit({ terms: [{ tex: 'x', estimate: 1 }], x: [0, 1], fit: [0, 1] }))).toBeUndefined()
    expect(
      truthTex(equationFit({ expression: 'x', truthExpression: 'x + 1', target: 'f(x)', x: [0, 1], fit: [0, 1] })),
    ).toBe('f(x) = x + 1')
  })

  it('counts the terms by status', () => {
    expect(termCounts(r)).toEqual({ right: 3, off: 0, missing: 0, extra: 1, unknown: 0 })
  })

  it('draws the truth, the fit and the observations, and the residual strip', () => {
    expect((equationFitView(r) as { series: unknown[] }).series.length).toBe(3)
    expect((residualStripView(r) as { series: unknown[] }).series.length).toBe(2)
  })
})
