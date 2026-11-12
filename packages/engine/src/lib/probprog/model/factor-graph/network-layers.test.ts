// A network layer is refused by EP with the reason, never run into a state that is not a posterior: the units of a layer (an `apply` of an
// affine map of weights that rows share, multiplied by a weight that rows share) are found in the compiled graph, and nothing else is
// taken for one (a hierarchy's `exp` of a scale, a nonlinear regression, a nonlinear state space).
import { describe, expect, it } from 'vitest'
import { compileGraph, defaultOperators as factorOperators } from '../../../message-passing/index.ts'
import { key, normal } from '../../../random/index.ts'
import { epRoute } from '../../inference/ep-route.ts'
import { fourierFeatures, gplvmFourierModel } from '../../models/gplvm.ts'
import { benchmarkModel } from '../../models/suite.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphVerdict, networkLayerUnits, ops, toFactorGraph } from '../index.ts'

type Builder = Parameters<Parameters<typeof defineModel>[1]>[0]

/** A network `y_i ~ N(sum_h w2_h g(w1_h x_i + b1_h) + b2, 0.2)` with `width` hidden units. */
function network(width: number, g: (u: never) => never): ReturnType<typeof defineModel> {
  return defineModel(`net-${width}`, (m: Builder) => {
    const row = m.range('row')
    const hidden = m.range('hidden', { size: width })
    const x = m.data('x', { dims: [row] })
    const w1 = m.latent('w1', d.normal(0, 1), { dims: [hidden] })
    const b1 = m.latent('b1', d.normal(0, 1), { dims: [hidden] })
    const w2 = m.latent('w2', d.normal(0, 1), { dims: [hidden] })
    const b2 = m.latent('b2', d.normal(0, 1))
    const unit = g(x.mul(w1).add(b1) as never)
    m.observe('y', d.normal((unit as never as typeof x).mul(w2).sum(hidden).add(b2), 0.2), m.data('y', { dims: [row] }))
  })
}

const DATA = { x: [-1.5, -0.7, 0, 0.6, 1.4], y: [-0.9, -0.6, 0, 0.5, 0.9] }

const graphOf = (model: ReturnType<typeof defineModel>): ReturnType<typeof compileGraph> => {
  const c = toFactorGraph(model.bind(DATA), { packScalars: true })
  return compileGraph(c.graph as never, factorOperators())
}

describe('networkLayerUnits', () => {
  it('finds one unit per hidden unit and row of the benchmark network (not at the input 0, where the first layer has no weight)', () => {
    const g = graphOf(network(3, (u) => ops.tanh(u) as never))
    expect(networkLayerUnits(g)).toBe(3 * (DATA.x.length - 1))
  })

  it('finds the units of a network of one hidden unit and of any activation', () => {
    expect(networkLayerUnits(graphOf(network(1, (u) => ops.tanh(u) as never)))).toBeGreaterThan(0)
    expect(networkLayerUnits(graphOf(network(2, (u) => ops.sigmoid(u) as never)))).toBeGreaterThan(0)
  })

  it('takes nothing else for a layer: a hierarchy, a funnel and a nonlinear regression have none', () => {
    for (const name of [
      'eight-schools-non-centred',
      'funnel-non-centred',
      'radon',
      'banana',
      'negative-binomial-regression',
    ]) {
      const c = toFactorGraph(benchmarkModel(name).bound(), { packScalars: true })
      if (c.graph === undefined) continue
      expect(networkLayerUnits(compileGraph(c.graph, factorOperators())), name).toBe(0)
    }
    const curve = defineModel('curve', (m: Builder) => {
      const row = m.range('row')
      const x = m.data('x', { dims: [row] })
      const a = m.latent('a', d.normal(0, 1))
      const b = m.latent('b', d.normal(0, 1.5))
      m.observe('y', d.normal(ops.tanh(x.mul(b).add(a)) as never, 0.2), m.data('y', { dims: [row] }))
    })
    expect(networkLayerUnits(graphOf(curve))).toBe(0)
  })
})

describe('networkLayerUnits and the latent coordinates of a feature model', () => {
  it('does not take the features of a GPLVM for a layer: the units that read one coordinate multiply weights of their own', () => {
    const options = { features: 4, coords: 1, outputs: 3 }
    const { frequencies, phases } = fourierFeatures(key(0), options)
    const bound = gplvmFourierModel(options).bind({ omega: frequencies, phase: phases, y: normal(key(1), [5, 3]) })
    const g = compileGraph(toFactorGraph(bound, { packScalars: true }).graph as never, factorOperators())
    expect(g.factors.filter((f) => f.operator.name === 'apply').length).toBe(5 * 4)
    expect(networkLayerUnits(g)).toBe(0)
  })
})

describe('the verdict of EP on a network layer', () => {
  it('refuses the benchmark network with the reason and the missing feature, and the route is none', () => {
    const bound = benchmarkModel('bayesian-neural-network').bound()
    const route = epRoute(bound)
    expect(route.via).toBe('none')
    expect(route.method).toBeUndefined()
    expect(route.verdict.applies).toBe(false)
    expect(route.verdict.reason).toMatch(/neural-network layer.*sign and permutation symmetries.*does not converge/)
    expect(route.verdict.missing).toEqual(['beliefs that hold the symmetric modes of a hidden layer'])
  })

  it('refuses it in the verdict of the compiler for EP only: the variational rule is not asked', () => {
    const c = toFactorGraph(benchmarkModel('bayesian-neural-network').bound(), { packScalars: true })
    const g = compileGraph(c.graph as never, factorOperators())
    expect(factorGraphVerdict(c, g, 'ep').ok).toBe(false)
    expect(factorGraphVerdict(c, g, 'ep').reason).toMatch(/neural-network layer/)
    expect(factorGraphVerdict(c, g, 'vmp').reason).not.toMatch(/neural-network layer/)
  })

  it('still runs a unary function of a random value as a regression (the unit is not a layer)', () => {
    const curve = defineModel('curve-route', (m: Builder) => {
      const row = m.range('row')
      const x = m.data('x', { dims: [row] })
      const a = m.latent('a', d.normal(0, 1))
      const b = m.latent('b', d.normal(0, 1.5))
      m.observe('y', d.normal(ops.tanh(x.mul(b).add(a)) as never, 0.2), m.data('y', { dims: [row] }))
    })
    expect(epRoute(curve.bind(DATA)).via).toBe('ep')
  })
})
