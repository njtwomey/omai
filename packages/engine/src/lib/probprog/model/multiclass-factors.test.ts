// A categorical outcome with softmax probabilities of Gaussian class scores (a multinomial logit classifier), as a model, under EP,
// against the exact posterior of the three class slopes on a grid.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphVerdict, toFactorGraph } from './index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0]) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-7, damping: 0.3 }), { maxSteps: 400 })
  return {
    c,
    status: r.status,
    marginals: marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>>,
  }
}

describe('a multinomial logit classifier: three classes, one slope each', () => {
  const x = [-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.5, -0.3, 0.8]
  const y = [0, 0, 0, 2, 1, 1, 1, 1, 2, 2]
  const s = 2
  const model = defineModel('softmax-slopes', (m) => {
    const row = m.range('row')
    const cls = m.range('class', { size: 3 })
    const w = m.latent('w', d.normal(0, s), { dims: [cls] })
    const logits = m.data('x', { dims: [row], role: 'covariate' }).mul(w)
    m.observe('y', d.categorical({ logits }, { over: cls }), m.data('y', { dims: [row], support: 'count' }))
  })
  // The three slopes share every observation, so EP's spreads are 0.78 to 0.86 of the exact ones and its means within 0.06 of an exact sd.
  it('compiles to one multiclass-site per observation, and the slopes agree with the exact posterior', () => {
    const { c, status, marginals } = ep(model.bind({ x, y }))
    expect(['converged', 'completed']).toContain(status)
    expect(c.graph?.factors.filter((f) => f.operator === 'multiclass-site').length).toBe(x.length)
    // The exact posterior of (w0, w1, w2) on a grid.
    const n = 45
    const lo = -6
    const hi = 6
    const h = (hi - lo) / n
    const axis = Array.from({ length: n }, (_, i) => lo + (i + 0.5) * h)
    let z = 0
    const m1 = [0, 0, 0]
    const m2 = [0, 0, 0]
    for (const a of axis)
      for (const b of axis)
        for (const e of axis) {
          const w = [a, b, e]
          let lp = (-0.5 * (a * a + b * b + e * e)) / (s * s)
          for (const [i, xi] of x.entries()) {
            const logits = w.map((v) => v * xi)
            const top = Math.max(...logits)
            lp += (logits[y[i] as number] as number) - top - Math.log(logits.reduce((p, v) => p + Math.exp(v - top), 0))
          }
          const p = Math.exp(lp)
          z += p
          for (let k = 0; k < 3; k++) {
            m1[k] = (m1[k] as number) + p * (w[k] as number)
            m2[k] = (m2[k] as number) + p * (w[k] as number) ** 2
          }
        }
    for (let k = 0; k < 3; k++) {
      const mean = (m1[k] as number) / z
      const sd = Math.sqrt((m2[k] as number) / z - mean * mean)
      const got = marginals[`w[${k}]`] as { mean: number; variance: number }
      expect(Math.abs(got.mean - mean) / sd, `slope ${k} mean`).toBeLessThan(0.15)
      expect(Math.sqrt(got.variance) / sd, `slope ${k} sd`).toBeGreaterThan(0.7)
      expect(Math.sqrt(got.variance) / sd, `slope ${k} sd`).toBeLessThan(1.2)
    }
  })
})
