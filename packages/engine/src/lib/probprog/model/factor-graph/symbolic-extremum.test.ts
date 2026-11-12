// The larger and the smaller of two random values are the max and min factors; a known operand makes a rectifier.
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
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from '../index.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

const _N = 40

describe('the larger and the smaller of two random values are the max and min factors', () => {
  it('compile, run under EP, and a known operand makes a rectifier', () => {
    const m = defineModel('hinge', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const c = b.latent('c', d.normal(1, 1))
      b.observe('y', d.normal(ops.maximum(a, c), 0.5), b.data('y', { dims: ['row'] }))
    })
    const c = toFactorGraph(m.bind({ y: [2] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'max', 'gaussian'])
    const g = graphOf(c.graph)
    const { state } = run(g)
    // The exact posterior mean of a by a grid over (a, c), y = max(a, c) observed through N(2, 1/4).
    let z = 0
    let ma = 0
    for (let i = -70; i <= 70; i++)
      for (let j = -70; j <= 70; j++) {
        const a = i * 0.1
        const cc = j * 0.1
        const w = Math.exp(-0.5 * a * a - 0.5 * (cc - 1) ** 2 - 2 * (2 - Math.max(a, cc)) ** 2)
        z += w
        ma += w * a
      }
    expect(toFlat(factorGraphPosterior(c, g, state).mean('a'))[0]).toBeCloseTo(ma / z, 1)
    const low = defineModel('low', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const cc = b.latent('c', d.normal(1, 1))
      b.observe('y', d.normal(ops.minimum(a, cc), 0.5), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(low.bind({ y: [0] })).graph?.factors.some((f) => f.operator === 'min')).toBe(true)
    const clip = defineModel('clip', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      b.observe('y', d.normal(ops.maximum(a, 0), 0.5), b.data('y', { dims: ['row'] }))
    })
    // max(a, 0) is a rectifier: the apply factor of the engine, on a new variable.
    const rectified = toFactorGraph(clip.bind({ y: [1] }))
    expect(rectified.issues).toEqual([])
    expect(rectified.graph?.factors.some((f) => f.operator === 'apply')).toBe(true)
  })
})
