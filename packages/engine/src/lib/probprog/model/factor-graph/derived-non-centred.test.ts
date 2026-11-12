// The derived node of a non-centred vector: `z / S(x)` with a scale of its own for each coefficient (ARD) is `apply-product` scalars
// assembled into one Gaussian vector by one `gaussian-mv-stack` factor, so each variable has one definer; the shared scale is the dense
// function-scale factor (tested with the Bayesian GAM), and a derived value that is not a bare quotient stays scalar.
import { describe, expect, it } from 'vitest'
import { compileGraph, defaultOperators } from '../../../message-passing/index.ts'
import * as d from '../dists/index.ts'
import { toFactorGraph } from '../factors.ts'
import { defineModel } from '../index.ts'

const data = {
  X: [
    [1, 0, 0.5],
    [0, 1, 0.5],
    [1, 1, 0],
    [0.3, 0.2, 1],
  ],
  y: [0.1, 0.2, 0.3, 0.4],
}

const ard = (scale: 'sqrt' | 'exp', offset = false) =>
  defineModel(`ard-${scale}`, (m) => {
    const row = m.range('row')
    const coef = m.range('coef')
    const X = m.data('X', { dims: [row, coef] })
    const y = m.data('y', { dims: [row] })
    const a = m.latent('a', scale === 'sqrt' ? d.gamma(2, 2) : d.normal(0, 1), { dims: [coef] })
    const z = m.latent('z', d.normal(0, 1), { dims: [coef] })
    const quotient = z.div(scale === 'sqrt' ? a.sqrt() : a.exp().sqrt())
    const beta = m.derived('beta', offset ? quotient.add(1) : quotient)
    m.observe('y', d.normal(X.matmul(beta), 0.3), y)
  })

describe('a non-centred vector with a scale for each coefficient', () => {
  for (const scale of ['sqrt', 'exp'] as const) {
    it(`is one apply-product per coefficient assembled by one gaussian-mv-stack (a ${scale === 'sqrt' ? 'gamma' : 'Gaussian'} latent for each)`, () => {
      const c = toFactorGraph(ard(scale).bind(data))
      expect(c.issues).toEqual([])
      const factors = c.graph?.factors ?? []
      const operators = factors.map((f) => f.operator)
      expect(operators.filter((o) => o === 'apply-product')).toHaveLength(3)
      expect(operators.filter((o) => o === 'gaussian-mv-stack')).toHaveLength(1)
      // The vector is read whole by the rows, not as scalar coordinates tied one by one.
      expect(operators).not.toContain('dot-constant')
      expect(operators).not.toContain('gaussian-mv-function-scale')
      const beta = (c.graph?.variables ?? []).find((v) => v.name === 'beta')
      expect(beta?.family).toEqual({ kind: 'gaussian-mv', size: 3 })
    })
  }

  it('puts the vector first and the three products after it, each product defining its own scalar', () => {
    const graph = toFactorGraph(ard('sqrt').bind(data)).graph
    const stack = graph?.factors.find((f) => f.operator === 'gaussian-mv-stack')
    expect(stack?.params).toEqual({ n: 3 })
    const [vector, ...scalars] = stack?.args ?? []
    expect(vector).toBe('beta')
    expect(scalars).toHaveLength(3)
    const products = (graph?.factors ?? []).filter((f) => f.operator === 'apply-product')
    expect(products.map((f) => f.args[0])).toEqual(scalars)
  })

  it('has exactly one definer for every variable, in the engine as well (the graph compiles)', () => {
    const graph = toFactorGraph(ard('sqrt').bind(data)).graph
    if (graph === undefined) throw new Error('no graph')
    const compiled = compileGraph(graph, defaultOperators())
    const defined = compiled.variables.filter((v) => v.definer !== undefined).map((v) => v.name)
    // The vector by the stack and each scalar by its product; the latents and the observed rows are not defined.
    expect(defined.sort()).toEqual(['beta', 'beta#1[0]', 'beta#1[1]', 'beta#1[2]'])
    expect(compiled.hasDefined).toBe(true)
  })

  it('leaves a value that is not a bare quotient as scalars (no vector to assemble)', () => {
    const c = toFactorGraph(ard('sqrt', true).bind(data))
    expect(c.issues).toEqual([])
    const operators = (c.graph?.factors ?? []).map((f) => f.operator)
    expect(operators).not.toContain('gaussian-mv-stack')
    expect(operators).toContain('apply-product')
  })
})
