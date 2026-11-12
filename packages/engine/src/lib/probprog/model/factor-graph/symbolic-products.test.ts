// A product of two random values is the product factor (EP lands on the grid posterior), also plated.
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph } from '../index.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

describe('a product of two random values is the product factor (expectation propagation)', () => {
  const prodModel = defineModel('prod', (b) => {
    const a = b.latent('a', d.normal(1, 1))
    const c = b.latent('c', d.normal(2, 1))
    b.observe('y', d.normal(ops.mul(a, c), 0.5), b.data('y', { dims: ['row'] }))
  })

  it('compiles to a product factor on an auxiliary variable, and EP lands on the grid posterior of the means', () => {
    const c = toFactorGraph(prodModel.bind({ y: [2.5] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'product', 'gaussian'])
    expect(c.auxiliary).toEqual(['y#0[0]'])
    const g = graphOf(c.graph)
    const { state } = run(g)
    const post = factorGraphPosterior(c, g, state)
    // The exact posterior mean of a and c by a two-dimensional grid.
    const logp = (a: number, cc: number): number => -0.5 * (a - 1) ** 2 - 0.5 * (cc - 2) ** 2 - 2 * (2.5 - a * cc) ** 2
    let z = 0
    let ma = 0
    let mc = 0
    for (let i = -80; i <= 80; i++)
      for (let j = -80; j <= 80; j++) {
        const a = i * 0.1
        const cc = j * 0.1 + 2
        const w = Math.exp(logp(a, cc))
        z += w
        ma += w * a
        mc += w * cc
      }
    expect(toFlat(post.mean('a'))[0]).toBeCloseTo(ma / z, 1)
    expect(toFlat(post.mean('c'))[0]).toBeCloseTo(mc / z, 1)
    // The product defines its output under the mean field, so variational message passing runs it too.
    expect(factorGraphVerdict(c, g, 'vmp')).toMatchObject({ ok: true })
  })

  it('a plated product maps per element', () => {
    const plated = defineModel('plated', (b) => {
      const n = b.range('row')
      const a = b.latent('a', d.normal(1, 1), { dims: [n] })
      const c = b.latent('c', d.normal(2, 1), { dims: [n] })
      b.observe('y', d.normal(ops.mul(a, c), 0.5), b.data('y', { dims: [n] }))
    })
    const c = toFactorGraph(plated.bind({ y: [2.5, 1, 3] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'product').length).toBe(3)
    expect(c.auxiliary.length).toBe(3)
  })
})
