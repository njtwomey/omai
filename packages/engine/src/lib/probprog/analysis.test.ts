import { describe, expect, it } from 'vitest'
import { DISCRETE, MODELS } from '../../testing/probprog.ts'
import { bernoulli, beta, categorical, exponential, gamma, normal, poisson } from '../dists/index.ts'
import { exp, item, mul, take, tensor, type Value } from '../tensor/index.ts'
import {
  analyse,
  deterministic,
  type Model,
  markov,
  observe,
  param,
  plate,
  sample,
  simulator,
  toStructure,
} from './index.ts'

describe('analyse', () => {
  it('describes sites, plates, dependencies and conjugacy of a hierarchical program', () => {
    const a = analyse(MODELS['schools'] as Model, undefined)
    expect(a.latent).toEqual(['mu', 'tau', 'theta'])
    expect(a.observed).toEqual(['y'])
    expect(a.plates.map((p) => [p.name, p.size, p.sites])).toEqual([['J', 4, ['theta', 'y']]])
    expect(a.edges).toEqual([
      { from: 'mu', to: 'theta' },
      { from: 'tau', to: 'theta' },
      { from: 'theta', to: 'y' },
    ])
    expect(a.dependenciesExact).toBe(true)
    expect(a.continuousDim).toBe(6)
    expect(a.differentiable).toBe(true)
    // mu and tau are co-parents of theta: married in the moral graph, a triangle.
    expect(a.graphShape).toBe('loopy')
    expect(a.fullyConjugate).toBe(false)
    expect(a.linearGaussian.linear).toBe(false)
    expect(a.sites.find((s) => s.name === 'theta')?.elements).toBe(4)
  })

  it('finds conjugate structures and linear-Gaussian programs', () => {
    const coin = analyse(MODELS['gammaPoisson'] as Model, undefined)
    expect(coin.conjugate.map((c) => [c.prior, c.priorFamily, c.likelihood, c.parameter])).toEqual([
      ['rate', 'gamma', 'poisson', 'rate'],
    ])
    expect(coin.fullyConjugate).toBe(true)
    expect(coin.deterministic).toEqual(['double'])
    const lg = analyse(MODELS['mvn'] as Model, undefined)
    expect(lg.linearGaussian.linear).toBe(true)
    const nonlinear: Model = () => {
      const x = sample('x', normal(0, 1))
      observe('y', normal(exp(x), 1), 0.2)
    }
    expect(analyse(nonlinear, undefined).linearGaussian.reason).toMatch(/not affine/)
    const scaled: Model = () => {
      const x = sample('x', normal(0, 1))
      observe('y', normal(0, exp(x)), 0.2)
    }
    expect(analyse(scaled, undefined).linearGaussian.reason).toMatch(/scale/)
  })

  it('counts discrete states and recognises chains', () => {
    const hmm = analyse(DISCRETE['hmm'] as Model, undefined)
    expect(hmm.discrete.sites).toEqual(['z0', 'z1', 'z2', 'z3'])
    expect(hmm.discrete.jointSize).toBe(16)
    expect(hmm.graphShape).toBe('chain')
    expect(hmm.markov.map((m) => [m.name, m.length, m.sites.length])).toEqual([['time', 4, 8]])
    const mixture = analyse(DISCRETE['mixture'] as Model, undefined)
    expect(mixture.discrete.cardinalities).toEqual({ z: 2 })
    expect(mixture.discrete.jointSize).toBe(8)
    const counts: Model = () => {
      sample('k', poisson(3))
    }
    expect(analyse(counts, undefined).discrete.unbounded).toEqual(['k'])
  })

  it('tells simulators from likelihoods and records params', () => {
    const sim: Model = () => {
      const m = sample('m', normal(0, 1))
      observe(
        'y',
        simulator('sim', (k) => normal(m, 1).sample(k)),
        0.3,
      )
    }
    const a = analyse(sim, undefined)
    expect(a.hasLikelihood).toBe(false)
    expect(a.simulators).toEqual(['y'])
    const withParam: Model = () => {
      const s = param('s', 1, { support: { kind: 'positive' } })
      const r = sample('r', exponential(s))
      observe('t', exponential(r), 2)
    }
    const b = analyse(withParam, undefined)
    expect(b.params).toEqual(['s'])
    expect(b.edges.some((e) => e.from === 's' && e.to === 'r')).toBe(true)
  })

  it('follows dependencies through indexing and deterministic sites', () => {
    const model: Model = () => {
      const p = sample('p', beta(1, 1))
      const c = sample('c', bernoulli({ probs: p }))
      const loc = deterministic('loc', take(tensor([-1, 1]), c))
      observe('y', normal(mul(loc, 2), 1), 0.4)
    }
    const a = analyse(model, undefined)
    expect(a.edges).toEqual([
      { from: 'p', to: 'c' },
      { from: 'c', to: 'loc' },
      { from: 'loc', to: 'y' },
    ])
    expect(a.conjugate).toEqual([])
  })
})

describe('toStructure', () => {
  it('draws a program like a graphical model, with plates and markov loops as groups', () => {
    const s = toStructure(MODELS['nested'] as Model, undefined)
    expect(s.nodes.map((n) => [n.id, n.role, n.group])).toEqual([
      ['x', 'latent', 'plate:b'],
      ['y', 'observed', 'plate:b'],
    ])
    expect(s.groups.map((g) => [g.id, g.parent, g.count])).toEqual([
      ['plate:a', undefined, 2],
      ['plate:b', 'plate:a', 3],
    ])
    expect(s.edges).toEqual([{ from: 'x', to: 'y', directed: true, kind: 'dependency' }])
    const walk: Model = () => {
      markov('t', 2, (t, x: Value | number) => sample(`x${t}`, normal(x, 1)), 0)
    }
    const w = toStructure(walk, undefined)
    expect(w.groups).toEqual([{ id: 'markov:t', label: 't', kind: 'repeat', count: 2 }])
    expect(w.nodes.map((n) => n.group)).toEqual(['markov:t', 'markov:t'])
  })

  it('labels params, factors and discrete states', () => {
    const model: Model = () => {
      const g = param('g', 2)
      const z = plate('n', 3, () => sample('z', categorical({ probs: [0.5, 0.5] })))
      void z
      observe('y', gamma(g, 1), 1)
    }
    const s = toStructure(model, undefined)
    expect(s.nodes.find((n) => n.id === 'g')?.role).toBe('parameter')
    expect(s.nodes.find((n) => n.id === 'z')?.badges).toEqual({ shape: '[3]', states: 2 })
  })
})
