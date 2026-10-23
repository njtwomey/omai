import { describe, expect, it } from 'vitest'
import { problem } from '../../testing/optim-problems.ts'
import * as O from './index.ts'

// The test functions of More and Thuente (1994), section 6, with their derivatives.
const MT: Readonly<Record<string, (a: number) => O.LinePoint>> = {
  // phi(a) = -a / (a^2 + 2)
  rational: (a) => ({ value: -a / (a * a + 2), slope: (a * a - 2) / (a * a + 2) ** 2 }),
  // phi(a) = (a + 0.004)^5 - 2 (a + 0.004)^4
  quintic: (a) => {
    const t = a + 0.004
    return { value: t ** 5 - 2 * t ** 4, slope: 5 * t ** 4 - 8 * t ** 3 }
  },
  // phi0(a) + 2 (1 - b) / (l pi) sin(l pi a / 2), b = 0.01, l = 39, phi0 = 1 - a for a <= 1 - b, ...
  wiggly: (a) => {
    const b = 0.01
    const l = 39
    let v: number
    let d: number
    if (a <= 1 - b) {
      v = 1 - a
      d = -1
    } else if (a >= 1 + b) {
      v = a - 1
      d = 1
    } else {
      v = (a - 1) ** 2 / (2 * b) + b / 2
      d = (a - 1) / b
    }
    return {
      value: v + ((2 * (1 - b)) / (l * Math.PI)) * Math.sin((l * Math.PI * a) / 2),
      slope: d + (1 - b) * Math.cos((l * Math.PI * a) / 2),
    }
  },
}

const searches: Readonly<Record<string, O.LineSearch>> = {
  strongWolfe: O.strongWolfe({ c1: 1e-3, c2: 0.1, maxIterations: 40, zoomIterations: 40 }),
  moreThuente: O.moreThuente({ c1: 1e-3, c2: 0.1, alphaMax: 1e10 }),
  hagerZhang: O.hagerZhang({ delta: 1e-3, sigma: 0.1 }),
}

describe('Wolfe line searches on the More-Thuente test functions', () => {
  for (const [fname, f] of Object.entries(MT)) {
    for (const [sname, search] of Object.entries(searches)) {
      for (const alpha of [1e-3, 1e-1, 10, 1000]) {
        it(`${sname} on ${fname} from ${alpha}`, () => {
          const start = f(0)
          const r = search({ at: f }, { value: start.value, slope: start.slope, alpha })
          expect(r.success).toBe(true)
          const p = f(r.alpha)
          expect(p.value).toBeCloseTo(r.value, 12)
          // Sufficient decrease (or the approximate Wolfe conditions for Hager-Zhang) and the curvature condition.
          const armijo = p.value <= start.value + 1e-3 * r.alpha * start.slope
          const approx = p.value <= start.value + 1e-6 * Math.abs(start.value) && p.slope <= (2e-3 - 1) * start.slope
          expect(armijo || (sname === 'hagerZhang' && approx)).toBe(true)
          if (sname === 'hagerZhang') expect(p.slope).toBeGreaterThanOrEqual(0.1 * start.slope - 1e-12)
          else expect(Math.abs(p.slope)).toBeLessThanOrEqual(0.1 * Math.abs(start.slope) + 1e-12)
          expect(r.trials.length).toBe(r.evaluations)
          expect(r.trials.at(-1)?.alpha).toBeDefined()
        })
      }
    }
  }
})

describe('backtracking', () => {
  const phi = { at: (a: number) => ({ value: Math.exp(a) - 2 * a, slope: Math.exp(a) - 2 }) }
  it('halving multiplies by rho until the Armijo condition holds', () => {
    const r = O.backtracking({ interpolation: 'halving', rho: 0.5 })(phi, { value: 1, slope: -1, alpha: 8 })
    expect(r.success).toBe(true)
    expect(r.trials.map((t) => t.alpha)).toEqual([8, 4, 2, 1])
  })
  it('uses value-only evaluations when given them', () => {
    let full = 0
    const counted = {
      at: (a: number) => {
        full += 1
        return phi.at(a)
      },
      valueAt: (a: number) => phi.at(a).value,
    }
    const r = O.backtracking()(counted, { value: 1, slope: -1, alpha: 8 })
    expect(r.success).toBe(true)
    expect(full).toBe(0)
    expect(r.gradEvaluations).toBe(0)
    expect(Number.isNaN(r.slope)).toBe(true)
  })
  it('rejects ascent directions', () => {
    expect(O.backtracking()(phi, { value: 1, slope: 1 }).success).toBe(false)
    expect(O.strongWolfe()(phi, { value: 1, slope: 1 }).success).toBe(false)
    expect(O.moreThuente()(phi, { value: 1, slope: 1 }).success).toBe(false)
    expect(O.hagerZhang()(phi, { value: 1, slope: 1 }).success).toBe(false)
    expect(() => O.lineSearch(problem('rosen'), [-1.2, 1], [-1, -0.4])).toThrow(/descent/)
  })
  it('validates its constants', () => {
    expect(() => O.strongWolfe({ c1: 0.5, c2: 0.1 })).toThrow()
  })
})

describe('combinators and the vector front', () => {
  it('falls back and accumulates evaluations', () => {
    const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
    const failing: O.LineSearch = (_p, s) => ({
      alpha: Number.NaN,
      value: s.value,
      slope: Number.NaN,
      success: false,
      evaluations: 3,
      gradEvaluations: 3,
      trials: [],
      message: 'gave up',
    })
    const r = O.withFallback(failing, O.strongWolfe())(phi, { value: 0.0625, slope: -0.5 })
    expect(r.success).toBe(true)
    expect(r.evaluations).toBe(5)
    expect(r.message).toMatch(/after: gave up/)
  })
  it('lineSearch returns the new point and the gradient there', () => {
    const r = O.lineSearch(problem('rosen'), [-1.2, 1], [1, 0.4])
    expect(r.success).toBe(true)
    expect(r.trials.length).toBe(r.evaluations)
    expect(Array.from(r.x)).toEqual([-1.2 + r.alpha, 1 + 0.4 * r.alpha])
    expect(r.grad).toHaveLength(2)
  })
  it('Hager-Zhang works as the line search of a vector front', () => {
    const r = O.lineSearch(problem('beale'), [1, 1], [0, -1], { method: O.hagerZhang() })
    expect(r.success).toBe(true)
    expect(r.value).toBeLessThan(14.203125)
  })
})
