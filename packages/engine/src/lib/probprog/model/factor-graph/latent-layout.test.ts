// The site layout the compiler records for latents made one variable per element (`finishLatent`): a scalar variable for a node
// with no dimensions, a variable per element for a node over a dimension, and the symbols later nodes read.
import { describe, expect, it } from 'vitest'
import * as d from '../dists/index.ts'
import { defineModel, toFactorGraph } from '../index.ts'

describe('the layout of a latent with one variable per element', () => {
  const model = defineModel('layout', (b) => {
    const n = b.range('n')
    const y = b.data('y', { dims: [n] })
    const mu = b.latent('mu', d.normal(0, 3))
    const w = b.latent('w', d.normal(0, 1), { dims: [n] })
    const rate = b.latent('rate', d.gamma(2, 1))
    b.observe('y', d.normal(mu.add(w), rate.pow(-0.5)), y)
  })
  const compiled = toFactorGraph(model.bind({ y: [1, 2, 3] }))
  const site = (name: string) => compiled.sites.find((s) => s.site === name)

  it('compiles', () => {
    expect(compiled.issues).toEqual([])
  })

  it('lays a node with no dimensions out as one scalar variable', () => {
    expect(site('mu')).toMatchObject({
      kind: 'latent',
      family: 'gaussian',
      layout: 'scalar',
      variables: ['mu'],
      posterior: true,
    })
    expect(site('rate')).toMatchObject({ family: 'gamma', layout: 'scalar', variables: ['rate'] })
  })

  it('lays a node over a dimension out as a variable per element in row-major order', () => {
    expect(site('w')).toMatchObject({
      kind: 'latent',
      family: 'gaussian',
      layout: 'elements',
      dims: ['n'],
      shape: [3],
      variables: ['w[0]', 'w[1]', 'w[2]'],
      posterior: true,
    })
  })

  it('has a variable and a prior factor for each element', () => {
    const names = (compiled.graph?.variables ?? []).map((v) => v.name)
    expect(names).toEqual(expect.arrayContaining(['mu', 'w[0]', 'w[1]', 'w[2]', 'rate']))
    expect((compiled.graph?.factors ?? []).filter((f) => f.id?.startsWith('factor:w['))).toHaveLength(3)
  })
})
