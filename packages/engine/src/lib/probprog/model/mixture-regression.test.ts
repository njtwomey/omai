// A mixture of two regressions on a shared design: a class per row (a categorical latent), a coefficient vector per class and the
// prediction of the row's class chosen by `ops.pick` (the model layer's gate over branches), compiled to one engine gate per row whose
// branches are Gaussian likelihoods of the classes' predictions. The reference is the exact posterior by enumeration of the 2^n
// class assignments: given the classes each coefficient vector is Gaussian (a conjugate regression on its rows), so the posterior
// of a coefficient is a mixture of Gaussians over the assignments.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { mixtureOfRegressions } from '../validation/index.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toFactorGraph } from './index.ts'
import { defaultOperators, operatorShape } from './operators.ts'

const SD = 0.6
const PRIOR = 1.5
// The prior mean of each coefficient (an intercept and a slope per class): the classes are told apart softly, which fixes the labels.
const MEAN = [
  [1, -1],
  [1.5, -1.5],
]
// Two lines: y = 1 + 2 x and y = -1 - x, six rows with their classes not told.
const X = [-1.5, -0.5, 0.2, 1.1, -0.9, 0.7]
const Y = [-2.1, 0.2, 1.4, 3.3, -0.1, -1.6]

const model = defineModel('mixture-of-regressions', (m) => {
  const row = m.range('row')
  const feature = m.range('feature', { size: 2 })
  const component = m.range('component', { size: 2 })
  const x = m.data('X', { dims: [row, feature] })
  const y = m.data('y', { dims: [row], role: 'response' })
  const b = m.latent('B', d.normal(m.const(MEAN, [feature, component]), PRIOR), { dims: [feature, component] })
  const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
  const mean = ops.pick(ops.matmul(x, b), z)
  m.observe('y', d.normal(mean, SD), y)
})

describe('a mixture of regressions chosen by pick', () => {
  const bound = model.bind({ X: X.map((x) => [1, x]), y: Y })
  const c = toFactorGraph(bound)

  it('compiles to one gate per row over the two classes’ predictions', () => {
    expect(c.issues).toEqual([])
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(X.length)
    expect(gates[0]?.args[0]).toBe('z[0]')
  })

  it('agrees with the enumeration of the class assignments: means within a fifth of a sd, spreads within 15 percent', () => {
    const g = compileGraph(c.graph as never, factorOperators())
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, damping: 0.3 }), { maxSteps: 600 })
    expect(r.status).toBe('converged')
    const marginals = marginalsOf(g, r.state as MpState)
    const ref = mixtureOfRegressions({
      X: X.map((x) => [1, x]),
      y: Y,
      priorMean: [
        [MEAN[0]?.[0] as number, MEAN[1]?.[0] as number],
        [MEAN[0]?.[1] as number, MEAN[1]?.[1] as number],
      ],
      priorSd: PRIOR,
      noiseSd: SD,
    })
    expect(ref.assignments).toBe(64)
    for (let k = 0; k < 2; k++)
      for (let f = 0; f < 2; f++) {
        const mg = marginals[`B[${f * 2 + k}]`] as { mean: number; variance: number }
        const t = ref.coefficients[k]?.[f] as { mean: number; sd: number }
        expect(Math.abs(mg.mean - t.mean) / t.sd).toBeLessThan(0.2)
        expect(Math.sqrt(mg.variance) / t.sd).toBeGreaterThan(0.85)
        expect(Math.sqrt(mg.variance) / t.sd).toBeLessThan(1.15)
      }
    // Each row's class probability against the exact one (the loop over the shared coefficients is what EP approximates).
    for (let i = 0; i < X.length; i++) {
      const probs = (marginals[`z[${i}]`] as { probs: number[] }).probs
      expect(Math.abs((probs[1] as number) - ((ref.membership[i] as number[])[1] as number))).toBeLessThan(0.12)
    }
  })

  it('is a plain regression when the class is known: the gather of one branch', () => {
    const known = defineModel('known-class', (m) => {
      const row = m.range('row')
      const feature = m.range('feature', { size: 2 })
      const component = m.range('component', { size: 2 })
      const x = m.data('X', { dims: [row, feature] })
      const klass = m.data('k', { dims: [row], support: 'index', indexes: component })
      const y = m.data('y', { dims: [row], role: 'response' })
      const b = m.latent('B', d.normal(0, 2), { dims: [feature, component] })
      m.observe('y', d.normal(ops.pick(ops.matmul(x, b), klass), SD), y)
    })
    const bound = known.bind({ X: X.map((x) => [1, x]), y: Y, k: [1, 1, 0, 0, 1, 0] })
    const graph = toFactorGraph(bound)
    expect(graph.issues).toEqual([])
    expect(graph.graph?.factors.some((f) => f.operator === 'gate')).toBe(false)
  })
})

describe('the pick operator', () => {
  const table = defaultOperators()
  it('takes along the last dimension the entry its index names, for every element of the other dimensions', () => {
    const values = {
      value: tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      dims: ['row', 'class'],
    }
    const index = { value: tensor([2, 0]), dims: ['row'] }
    const out = table['pick']?.evaluate([values, index], {})
    expect(Array.from(toFlat(out?.value ?? tensor(0)))).toEqual([3, 4])
    expect(out?.dims).toEqual(['row'])
    expect(
      operatorShape(table, 'pick', [
        { dims: ['row', 'class'], eventRank: 0 },
        { dims: ['row'], eventRank: 0 },
      ]),
    ).toEqual({ dims: ['row'], eventRank: 0 })
  })
})
