// Beta outcomes with two random shapes compiled with the shapes packed (`packScalars`): the two log shapes are one Gaussian vector
// (a `gaussian-mv-stack` factor) and each outcome is a `likelihood-pair-row` on it, whose message carries the correlation between the
// shapes (the data fix their ratio better than their size), so the spreads are the exact ones where one message per shape gave 0.65 to
// 0.8 of them. Against the exact posterior of the two shapes on a grid (the same model and data as `observed-pairs.test.ts`).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../../message-passing/index.ts'
import { lgammaOf } from '../../../message-passing/numerics/_scalar.ts'
import * as d from '../dists/index.ts'
import { defineModel, toFactorGraph } from '../index.ts'

const logNormalPdf = (x: number, m: number, s: number): number => -0.5 * ((x - m) / s) ** 2 - Math.log(s)

/** The mean and sd of each log shape, and of each shape, of the exact posterior on a grid. */
function exact(y: readonly number[], prior: readonly [number, number, number, number]) {
  const [ma, sa, mb, sb] = prior
  const n = 500
  const [x0, x1] = [-2, 3.5]
  const [y0, y1] = [-1.5, 4.5]
  const hx = (x1 - x0) / n
  const hy = (y1 - y0) / n
  let z = 0
  const m = { a: 0, b: 0, aa: 0, bb: 0, ea: 0, eb: 0, eaa: 0, ebb: 0 }
  for (let i = 0; i < n; i++) {
    const ta = x0 + (i + 0.5) * hx
    for (let j = 0; j < n; j++) {
      const tb = y0 + (j + 0.5) * hy
      const a = Math.exp(ta)
      const b = Math.exp(tb)
      let lp = logNormalPdf(ta, ma, sa) + logNormalPdf(tb, mb, sb)
      for (const v of y)
        lp += (a - 1) * Math.log(v) + (b - 1) * Math.log1p(-v) + lgammaOf(a + b) - lgammaOf(a) - lgammaOf(b)
      const w = Math.exp(lp)
      z += w
      m.a += w * ta
      m.b += w * tb
      m.aa += w * ta * ta
      m.bb += w * tb * tb
      m.ea += w * a
      m.eb += w * b
      m.eaa += w * a * a
      m.ebb += w * b * b
    }
  }
  const sd = (s1: number, s2: number): number => Math.sqrt(s2 / z - (s1 / z) ** 2)
  return {
    logMean: [m.a / z, m.b / z],
    logSd: [sd(m.a, m.aa), sd(m.b, m.bb)],
    mean: [m.ea / z, m.eb / z],
    sd: [sd(m.ea, m.eaa), sd(m.eb, m.ebb)],
  }
}

describe('beta outcomes with two random shapes on a packed vector', () => {
  const y = [0.12, 0.3, 0.22, 0.5, 0.18, 0.27, 0.41, 0.09]
  const model = defineModel('beta-outcomes-packed', (b) => {
    const a = b.latent('a', d.logNormal(0.5, 0.7))
    const c = b.latent('b', d.logNormal(1, 0.7))
    b.observe('y', d.beta(a, c), b.data('y', { dims: ['i'], role: 'response' }))
  })

  it('compiles to one stacked vector of the two shapes and a likelihood-pair-row per outcome, with the exact spreads', () => {
    const { c, g, r } = runEp(model.bind({ y }), {
      tol: 1e-9,
      damping: 0.3,
      maxSteps: 400,
      verdict: true,
      converged: true,
      packScalars: true,
    })
    const operators = c.graph?.factors.map((f) => f.operator) ?? []
    expect(operators.filter((o) => o === 'likelihood-pair-row')).toHaveLength(y.length)
    expect(operators.filter((o) => o === 'gaussian-mv-stack')).toHaveLength(1)
    expect(operators).not.toContain('likelihood-pair')
    const marginals = marginalsOf(g, r.state as MpState) as unknown as Record<
      string,
      { mean: number; variance: number }
    >
    const ref = exact(y, [0.5, 0.7, 1, 0.7])
    for (const [k, name] of ['a', 'b'].entries()) {
      const got = marginals[name] as { mean: number; variance: number }
      const sdRef = ref.sd[k] as number
      expect(Math.abs(got.mean - (ref.mean[k] as number)) / sdRef, `${name} mean`).toBeLessThan(0.1)
      const ratio = Math.sqrt(got.variance) / sdRef
      expect(ratio, `${name} sd`).toBeGreaterThan(0.93)
      expect(ratio, `${name} sd`).toBeLessThan(1.07)
    }
  }, 60_000)

  it('without packScalars the shapes keep one message each (likelihood-pair), and a gamma shape is not packed', () => {
    const unpacked = toFactorGraph(model.bind({ y }))
    expect(unpacked.graph?.factors.some((f) => f.operator === 'likelihood-pair-row')).toBe(false)
    const gammaModel = defineModel('beta-outcomes-gamma', (b) => {
      const a = b.latent('a', d.gamma(2, 1))
      const c = b.latent('b', d.gamma(3, 1))
      b.observe('y', d.beta(a, c), b.data('y', { dims: ['i'], role: 'response' }))
    })
    const compiled = toFactorGraph(gammaModel.bind({ y }), { packScalars: true })
    const operators = compiled.graph?.factors.map((f) => f.operator) ?? []
    expect(operators).toContain('likelihood-pair')
    expect(operators).not.toContain('likelihood-pair-row')
  })
})
