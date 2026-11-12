// Stochastic volatility with the persistence and the innovation scale inferred: a Gaussian transition with a random slope (the
// `product` factor) and a gamma precision on the latent log-variance, under EP, against the means of a long NUTS run.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../../message-passing/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphVerdict, ops, toFactorGraph } from '../index.ts'

const T = 40
const unit = (i: number): number => Math.abs((Math.sin(i * 12.9898) * 43758.5453) % 1)
const ys: number[] = []
let h = -1
for (let t = 0; t < T; t++) {
  h = -1 + 0.9 * (h + 1) + 0.3 * 3.4 * (unit(t) - 0.5)
  ys.push(Math.exp(h / 2) * 3.4 * (unit(t + 100) - 0.5))
}

describe('stochastic volatility with inferred parameters', () => {
  const model = defineModel('sv-parameters', (b) => {
    const y = b.data('y', { dims: ['time'] })
    const phi = b.latent('phi', d.normal(0.8, 0.3))
    const tau = b.latent('tau', d.gamma(3, 0.3))
    b.scan('h', {
      over: 'time',
      init: d.normal(-1, 1),
      step: ({ prev }) => d.normal(prev.add(1).mul(phi).sub(1), tau.pow(-0.5)),
      emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(state.mul(0.5))), y),
    })
  })

  it('compiles to product, precision and log-scale factors, and EP agrees with NUTS on the parameters and the states', () => {
    const c = toFactorGraph(model.bind({ y: ys }))
    expect(c.issues).toEqual([])
    const operators = new Set(c.graph?.factors.map((f) => f.operator))
    for (const name of ['product', 'gaussian-gamma-precision', 'gaussian-log-scale'])
      expect(operators.has(name)).toBe(true)
    const g = compileGraph(c.graph as never, factorOperators())
    expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-8, damping: 0.5 }), { maxSteps: 400 })
    const m = marginalsOf(g, r.state as MpState) as Record<string, Record<string, number>>
    // Posterior means and sds of a NUTS run (three chains, 500 draws after 500 warmup).
    const phi = m['phi'] as { mean: number; variance: number }
    expect(Math.abs(phi.mean - 0.886)).toBeLessThan(0.6 * 0.125)
    const tau = m['tau'] as { shape: number; rate: number }
    expect(Math.abs(tau.shape / tau.rate - 10.53)).toBeLessThan(0.6 * 5.82)
    for (const [site, mean, sd] of [
      ['h[10]', -1.236, 0.508],
      ['h[30]', -0.081, 0.55],
    ] as const) {
      const q = m[site] as { mean: number; variance: number }
      expect(Math.abs(q.mean - mean)).toBeLessThan(0.6 * sd)
      expect(Math.sqrt(q.variance)).toBeGreaterThan(0.6 * sd)
      expect(Math.sqrt(q.variance)).toBeLessThan(1.4 * sd)
    }
  })
})
