// A Gaussian process classifier written with a dense `multivariateNormal` prior on the latent function and a probit Bernoulli
// per row, run by expectation propagation through the compiled factor graph (one `gaussian-mv` variable with a dense prior),
// against the gp domain's own EP classifier (the same algorithm written for this one model); the whitened form of the same classifier is checked against NUTS in the example method's tests.
import { describe, expect, it } from 'vitest'
import { gpClassifier } from '../../gp/index.ts'
import { gram, rbfKernel, scaleKernel } from '../../kernels/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../model/index.ts'
import { epSpecMethod } from './mp.ts'

const x = [[-2.1], [-1.5], [-0.9], [-0.3], [0.2], [0.8], [1.4], [2.0]]
const y = [0, 0, 0, 1, 0, 1, 1, 1]
const kernel = scaleKernel(rbfKernel({ lengthScale: 1 }), 3)
const K = Array.from({ length: x.length }, (_, i) =>
  Array.from({ length: x.length }, (_, j) => toFlat(gram(kernel, x, { jitter: 1e-6 }))[i * x.length + j] ?? 0),
)
const definition = defineModel('gp-dense', (b) => {
  const row = b.range('row')
  const obs = b.data('y', { dims: [row], support: 'binary' })
  const f = b.latent('f', d.multivariateNormal(new Array<number>(x.length).fill(0), { covariance: K }), { dims: [row] })
  b.observe('y', d.bernoulli({ probs: ops.probit(f) }), obs)
})
const bound = definition.bind({ y })

describe('a Gaussian process prior written as a dense multivariateNormal runs under expectation propagation', () => {
  it('compiles to one gaussian-mv variable with a dense prior and a probit likelihood per row', () => {
    const graph = toFactorGraph(bound)
    expect(graph.issues).toEqual([])
    expect(graph.sites.find((s) => s.site === 'f')?.layout).toBe('packed')
  })

  it('EP agrees with the gp domain EP classifier to a few thousandths (the means and spreads of the latent function)', () => {
    const ep = epSpecMethod(bound)
    const run = session(ep, { config: { tol: 1e-10 }, seed: 1, steps: 300, mode: 'eager' })
    expect(run.status).toBe('converged')
    const rows =
      ep
        .posterior?.(run.state, run.config)
        ?.summary()
        .filter((r) => r.site === 'f') ?? []
    expect(rows.length).toBe(x.length)
    const reference = gpClassifier
      .with({ kernel, method: 'ep', likelihood: 'probit', optimise: false, tol: 1e-10 })
      .fit({ x, y }, { explain: false })
    const { mean, variance } = reference.latent(x)
    const m = toFlat(mean)
    const v = toFlat(variance)
    for (const [i, r] of rows.entries()) {
      expect(Math.abs(r.mean - (m[i] ?? 0)), `mean ${i}`).toBeLessThan(5e-3)
      expect(Math.abs(r.sd - Math.sqrt(v[i] ?? 1)), `sd ${i}`).toBeLessThan(5e-3)
    }
  })
})
