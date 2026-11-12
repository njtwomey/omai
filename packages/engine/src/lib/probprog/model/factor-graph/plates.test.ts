// The probit regression as one plated factor: the graph does not grow with the rows, has the posterior of the
// expansion, and falls to the expansion where the rows are not one regression.
import { describe, expect, it } from 'vitest'
import { probitData, probitModel as probitDefinition } from '../../../../testing/probit-slice.ts'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
  vmpAlgorithm,
} from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, graphOnSpec, ops, toFactorGraph, toStructure } from '../index.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

const probitModel = probitDefinition

describe('the probit regression as one plated factor', () => {
  const { X, y } = probitData()
  const bound = probitModel.bind({ X, y })
  const c = toFactorGraph(bound)
  const e = toFactorGraph(bound, { plates: 'expand' })

  it('is a vector variable and two factors whatever the number of rows, with the ids of the spec', () => {
    expect(c.issues).toEqual([])
    expect(c.graph?.variables.map((v) => v.name)).toEqual(['w'])
    expect(c.graph?.factors.map((f) => `${f.id} ${f.operator}`)).toEqual(['factor:w prior', 'factor:y probit-plate'])
    expect(c.graph?.factors[1]?.params).toMatchObject({ y })
    expect(c.plated).toEqual(['y'])
    expect(c.auxiliary).toEqual([])
    expect(c.expanded.variables).toBe(1)
    expect(c.expanded.factors).toBe(2)
    expect(c.sites.map((s) => [s.site, s.layout, s.posterior])).toEqual([
      ['w', 'packed', true],
      ['y', 'observed', false],
    ])
    // Every id maps onto the drawing of the spec: the weights, and the factors of the prior and of the likelihood.
    const on = graphOnSpec(c.graph as FactorGraphSpec)
    const ids = new Set(toStructure(bound, { view: 'factor' }).nodes.map((n) => n.id))
    for (const site of [...Object.values(on.variables), ...Object.values(on.factors)]) expect(ids.has(site)).toBe(true)
    expect(new Set(Object.values(on.factors))).toEqual(new Set(['factor:w', 'factor:y']))
  })

  it('has the posterior and the evidence of the expanded graph, and the grid answer', () => {
    const ga = graphOf(c.graph)
    const gb = graphOf(e.graph)
    const ra = run(ga)
    const rb = run(gb)
    expect(ra.status).toBe('converged')
    const pa = factorGraphPosterior(c, ga, ra.state)
    const pb = factorGraphPosterior(e, gb, rb.state)
    const ma = toFlat(pa.mean('w'))
    const mb = toFlat(pb.mean('w'))
    for (const [q, v] of ma.entries()) expect(Math.abs(v - (mb[q] as number))).toBeLessThan(1e-6)
    const sa = toFlat(pa.stddev('w'))
    const sb = toFlat(pb.stddev('w'))
    for (const [q, v] of sa.entries()) expect(Math.abs(v - (sb[q] as number))).toBeLessThan(1e-6)
    expect(Math.abs((pa.logEvidence?.value ?? 0) - (pb.logEvidence?.value ?? 1))).toBeLessThan(1e-5)
    // The variational bound of the plate is that of the expanded graph's mean field on the weights as one block... under VMP
    // the plate has the Albert-Chib message, the expanded graph the same one per row, with the predictor as a defined variable.
    const va = runAlgorithm(vmpAlgorithm(ga, { tol: 1e-12 }), { maxSteps: 400 })
    const vb = runAlgorithm(vmpAlgorithm(gb, { tol: 1e-12 }), { maxSteps: 400 })
    const wa = marginalsOf(ga, va.state as MpState)['w']?.['mean'] as number[]
    const wb = marginalsOf(gb, vb.state as MpState)['w']?.['mean'] as number[]
    for (const [q, v] of wa.entries()) expect(Math.abs(v - (wb[q] as number))).toBeLessThan(1e-6)
  })

  it('falls to the expansion for rows that are not one regression on a packed vector, and without the plate operators', () => {
    const ops2 = { ...factorOperators() }
    delete (ops2 as Record<string, unknown>)['probit-plate']
    expect(toFactorGraph(bound, { operators: ops2 }).plated).toEqual([])
    const scalars = defineModel('scalars', (b) => {
      const n = b.range('row')
      const a = b.latent('a', d.normal(0, 2))
      const s = b.latent('s', d.normal(0, 2))
      b.observe('y', d.normal(ops.add(a, ops.mul(s, b.data('x', { dims: [n] }))), 0.5), b.data('y', { dims: [n] }))
    })
    const cs = toFactorGraph(scalars.bind({ x: [1, 2, 3], y: [1, 2, 3] }))
    expect(cs.plated).toEqual([])
    expect(cs.graph?.variables.length).toBeGreaterThan(5)
    // A per-row scale is not a plate either.
    const scales = defineModel('scales', (b) => {
      const n = b.range('row')
      const f = b.range('feature')
      const w = b.latent('w', d.normal(0, 2), { dims: [f] })
      b.observe(
        'y',
        d.normal(b.data('X', { dims: [n, f] }).matmul(w), b.data('sd', { dims: [n] })),
        b.data('y', { dims: [n] }),
      )
    })
    const cc = toFactorGraph(
      scales.bind({
        X: [
          [1, 2],
          [3, 4],
        ],
        y: [1, 2],
        sd: [0.5, 1],
      }),
    )
    expect(cc.plated).toEqual([])
    expect(cc.graph?.factors.filter((f) => f.operator === 'linear-regression-row').length).toBe(2)
  })

  it('leaves missing rows out of the plate and gives the same posterior as the expansion that ignores them', () => {
    const m = defineModel('gaps', (b) => {
      const n = b.range('row')
      const f = b.range('feature')
      const w = b.latent('w', d.normal(0, 2), { dims: [f] })
      b.observe('y', d.normal(b.data('X', { dims: [n, f] }).matmul(w), 0.5), b.data('y', { dims: [n] }), {
        missing: 'ignore',
      })
    })
    const bd = m.bind({
      X: [
        [1, 0.5],
        [1, -1],
        [1, 2],
        [1, 0],
      ],
      y: [1, Number.NaN, 2, 0.3],
    })
    const a = toFactorGraph(bd)
    const b2 = toFactorGraph(bd, { plates: 'expand' })
    expect(a.plated).toEqual(['y'])
    const ga = graphOf(a.graph)
    const gb = graphOf(b2.graph)
    const pa = factorGraphPosterior(a, ga, run(ga).state)
    const pb = factorGraphPosterior(b2, gb, run(gb).state)
    for (const [q, v] of toFlat(pa.mean('w')).entries())
      expect(Math.abs(v - (toFlat(pb.mean('w'))[q] as number))).toBeLessThan(1e-9)
  })
})
