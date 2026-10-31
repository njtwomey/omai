import { describe, expect, it } from 'vitest'
import { coefficientErrors, equationFit, fitResidual } from './equation-fit.ts'
import { isExposition } from './record.ts'

const x = [0, 1, 2]
describe('the equation fit record', () => {
  it('classifies each coefficient against its truth', () => {
    const r = equationFit({
      tolerance: 0.1,
      terms: [
        { tex: '1', estimate: 0.31, truth: 0.3 },
        { tex: 'x', estimate: 0.4, truth: 0 },
        { tex: 'x^2', estimate: 0, truth: -0.4 },
        { tex: '\\sin x', estimate: 1.2, truth: 1.5 },
        { tex: '\\cos x', estimate: 0, truth: 0 },
        { tex: 'x^3', estimate: 3 },
      ],
      x,
      fit: [0, 1, 2],
    })
    expect(coefficientErrors(r).map((e) => e.status)).toEqual(['right', 'extra', 'missing', 'off', 'right', 'unknown'])
    expect(coefficientErrors(r)[3]?.relative).toBeCloseTo(0.2, 12)
    expect(isExposition(r)).toBe(true)
  })

  it('checks the curves, the observations and that there is a model to show', () => {
    expect(() => equationFit({ terms: [], x: [0], fit: [0] })).toThrow(/at least two/)
    expect(() => equationFit({ terms: [], x, fit: [0, 1] })).toThrow(/fit has 2/)
    expect(() => equationFit({ terms: [], x, fit: [0, 1, 2], truth: [0] })).toThrow(/truth has 1/)
    expect(() => equationFit({ x, fit: [0, 1, 2] })).toThrow(/terms or the expression/)
    expect(() => equationFit({ terms: [{ tex: 'x', estimate: Number.NaN }], x, fit: [0, 1, 2] })).toThrow(/terms\[0\]/)
    expect(() => equationFit({ expression: 'x', x, fit: [0, 1, 2], observed: { x: [0], y: [] } })).toThrow(/differ/)
  })

  it('takes the residual against the truth, or against the observations when there is none', () => {
    expect(fitResidual(equationFit({ terms: [], x, fit: [1, 3, 5], truth: [1, 2, 3] })).values).toEqual([0, 1, 2])
    const r = fitResidual(equationFit({ terms: [], x, fit: [0, 2, 4], observed: { x: [0.5, 2], y: [1, 3] } }))
    expect(r.x).toEqual([0.5, 2])
    expect(r.values).toEqual([0, 1])
  })
})
