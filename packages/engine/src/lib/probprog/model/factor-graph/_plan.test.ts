// The layout plan of the model compiler: what each of the four detections finds in a spec, in their fixed order.
import { describe, expect, it } from 'vitest'
import * as d from '../dists/index.ts'
import { defineModel, ops } from '../index.ts'
import { defaultOperators } from '../operators.ts'
import { knownConstants } from './_known.ts'
import { type LayoutPlan, layoutPlan } from './_plan.ts'

const planOf = (
  model: ReturnType<typeof defineModel>,
  data: Parameters<ReturnType<typeof defineModel>['bind']>[0],
  options: { readonly denseScales?: boolean; readonly packScalars?: boolean } = {},
): LayoutPlan => {
  const bound = model.bind(data)
  const modelOps = defaultOperators()
  return layoutPlan(bound, knownConstants(bound, modelOps).known, {
    modelOps,
    denseScales: options.denseScales ?? true,
    packScalars: options.packScalars ?? false,
  })
}

describe('findPacked', () => {
  const regression = defineModel('packed', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const X = m.data('X', { dims: [row, feature] })
    const w = m.latent('w', d.normal(0, 1), { dims: [feature] })
    m.latent('free', d.normal(0, 1), { dims: [feature] })
    m.observe('y', d.normal(X.matmul(w), 1), m.data('y', { dims: [row] }))
  })
  const data = {
    X: [
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    y: [0, 1, 2],
  }

  it('keeps a vector latent that a matmul reads as one vector, and records what the matmul reads', () => {
    const plan = planOf(regression, data)
    expect([...plan.packed]).toEqual(['w'])
    expect([...plan.matmulRead].sort()).toEqual(['X', 'w'])
  })

  it('does not pack a normal whose scale is a random value of its own (ARD)', () => {
    const ard = defineModel('ard', (m) => {
      const row = m.range('row')
      const feature = m.range('feature')
      const X = m.data('X', { dims: [row, feature] })
      const tau = m.latent('tau', d.gamma(2, 1), { dims: [feature] })
      const w = m.latent('w', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [feature] })
      m.observe('y', d.normal(X.matmul(w), 1), m.data('y', { dims: [row] }))
    })
    expect([...planOf(ard, data).packed]).toEqual([])
  })

  it('packs it when the scale is the precision of one shared scalar gamma and the scales are dense, not when they are elements', () => {
    const shared = defineModel('shared', (m) => {
      const row = m.range('row')
      const feature = m.range('feature')
      const X = m.data('X', { dims: [row, feature] })
      const tau = m.latent('tau', d.gamma(2, 1))
      const w = m.latent('w', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [feature] })
      m.observe('y', d.normal(X.matmul(w), 1), m.data('y', { dims: [row] }))
    })
    expect([...planOf(shared, data).packed]).toEqual(['w'])
    expect([...planOf(shared, data, { denseScales: false }).packed]).toEqual([])
  })
})

describe('findNonCentred', () => {
  const spline = defineModel('non-centred', (m) => {
    const row = m.range('row')
    const k = m.range('k')
    const B = m.data('B', { dims: [row, k] })
    const dv = m.data('dv', { dims: [k] })
    const x = m.latent('x', d.normal(0, 1))
    const z = m.latent('z', d.normal(0, 1), { dims: [k] })
    const D = m.derived('D', ops.div(z, ops.sqrt(ops.add(ops.mul(ops.exp(x), dv), 1))))
    m.observe('y', d.normal(B.matmul(D), 1), m.data('y', { dims: [row] }))
  })
  const data = {
    B: [
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    dv: [1, 2],
    y: [0, 1, 2],
  }

  it('finds a derived vector z / S(x) that a matmul reads, whose numerator is standard normal and read by nothing else', () => {
    expect([...planOf(spline, data).nonCentred]).toEqual([['D', 'z']])
  })

  it('finds none when the scales are elements (the dense factor is not used)', () => {
    expect(planOf(spline, data, { denseScales: false }).nonCentred.size).toBe(0)
  })
})

describe('findGroups', () => {
  const scalars = defineModel('scalars', (m) => {
    const row = m.range('row')
    const x = m.data('x', { dims: [row] })
    const a = m.latent('a', d.normal(0, 2))
    const b = m.latent('b', d.normal(0, 2))
    m.latent('lonely', d.normal(0, 2))
    m.observe('y', d.normal(a.add(b.mul(x)), 1), m.data('y', { dims: [row] }))
  })
  const data = { x: [0, 1, 2], y: [0, 1, 2] }

  it('groups the scalar normals an observation reads together, in the order of the spec, only with packScalars', () => {
    const plan = planOf(scalars, data, { packScalars: true })
    expect([...plan.groups.keys()]).toEqual(['a', 'b'])
    expect(plan.groups.get('a')).toEqual({ vector: 'a#8', members: ['a', 'b'] })
    expect(plan.groups.get('b')).toEqual(plan.groups.get('a'))
    expect(planOf(scalars, data).groups.size).toBe(0)
  })
})

describe('findGroups: tied groups', () => {
  const grouped = (
    model: ReturnType<typeof defineModel>,
    data: Parameters<ReturnType<typeof defineModel>['bind']>[0],
  ) => planOf(model, data, { packScalars: true }).groups
  const counties = { y: [1, 2, 3, 4], floor: [0, 1, 0, 1], county: [0, 0, 1, 1], size: [2, 2] }

  it('ties a vector with a prior of its own to the scalar beside it, as one joint vector of their coordinates', () => {
    const hierarchy = defineModel('centred', (m) => {
      const county = m.range('county')
      const house = m.range('house')
      const code = m.data('county', { dims: [house], support: 'index', indexes: county })
      const floor = m.data('floor', { dims: [house] })
      m.data('size', { dims: [county] })
      const mu = m.latent('mu', d.normal(0, 5))
      const tau = m.latent('tau', d.halfNormal(1))
      const beta = m.latent('beta', d.normal(0, 2))
      const alpha = m.latent('alpha', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), { dims: [county] })
      m.observe('y', d.normal(alpha.at(code).add(beta.mul(floor)), 1), m.data('y', { dims: [house] }))
    })
    const groups = grouped(hierarchy, counties)
    expect([...groups.keys()]).toEqual(['beta', 'alpha'])
    expect(groups.get('beta')).toEqual({ vector: 'beta#8', members: ['beta', 'alpha'], tied: true, sizes: [1, 2] })
    expect(groups.get('alpha')).toEqual(groups.get('beta'))
  })

  it('leaves a vector alone when a row reads it through a product with a random scale (the non-centred form)', () => {
    const centreless = defineModel('non-centred', (m) => {
      const county = m.range('county')
      const house = m.range('house')
      const code = m.data('county', { dims: [house], support: 'index', indexes: county })
      const floor = m.data('floor', { dims: [house] })
      m.data('size', { dims: [county] })
      const mu = m.latent('mu', d.normal(0, 5))
      const tau = m.latent('tau', d.halfNormal(1))
      const beta = m.latent('beta', d.normal(0, 2))
      const z = m.latent('z', d.normal(0, 1), { dims: [county] })
      const alpha = m.derived('alpha', mu.add(tau.mul(z)))
      m.observe('y', d.normal(alpha.at(code).add(beta.mul(floor)), 1), m.data('y', { dims: [house] }))
    })
    const groups = grouped(centreless, counties)
    expect([...groups.keys()]).toEqual(['mu', 'beta'])
    expect(groups.get('mu')?.tied).toBeUndefined()
  })

  it('groups one vector that a row reads at two places (the two players of a game), through the link of the likelihood', () => {
    const tournament = defineModel('tournament', (m) => {
      const player = m.range('player', { size: 3 })
      const game = m.range('game')
      const a = m.data('a', { dims: [game], support: 'index', indexes: player })
      const b = m.data('b', { dims: [game], support: 'index', indexes: player })
      const skill = m.latent('skill', d.normal(0, 1), { dims: [player] })
      const gap = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
      m.observe('win', d.bernoulli({ probs: ops.probit(gap) }), m.data('win', { dims: [game] }))
    })
    const groups = grouped(tournament, { a: [0, 1, 2], b: [1, 2, 0], win: [1, 0, 1] })
    expect(groups.get('skill')).toEqual({ vector: 'skill#8', members: ['skill'], sizes: [3] })
  })

  it('does not group a vector that each row reads once (one coordinate per row: nothing is shared)', () => {
    const elementwise = defineModel('elementwise', (m) => {
      const school = m.range('school')
      const theta = m.latent('theta', d.normal(0, 1), { dims: [school] })
      m.observe('y', d.normal(theta, 1), m.data('y', { dims: [school] }))
    })
    expect(grouped(elementwise, { y: [1, 2, 3] }).size).toBe(0)
  })
})

describe('findNormalGamma', () => {
  const pair = (readers: 'one' | 'two') =>
    defineModel('nig', (m) => {
      const sigma2 = m.latent('sigma2', d.inverseGamma(3, 2))
      const mu = m.latent('mu', d.normal(0.2, ops.sqrt(ops.div(sigma2, 0.5))))
      m.observe('y', d.normal(mu, ops.sqrt(sigma2)), m.data('y', { dims: ['i'] }))
      if (readers === 'two') m.derived('shifted', ops.add(mu, 1))
    })

  it('finds the pair of an inverse-gamma variance and the mean that shares it, with their observations', () => {
    const found = planOf(pair('one'), { y: [1, 2, 3] }).normalGamma.get('sigma2')
    expect(found?.pair?.plan).toEqual({
      precision: 'sigma2',
      mean: 'mu',
      variable: 'mu',
      inverse: true,
      kappa: 0.5,
      observed: ['y'],
    })
    expect(found?.pair?.multiples.get('y')).toBe(1)
  })

  it('has an entry with no pair when something else reads the mean, and none for a node that is not a gamma', () => {
    const plan = planOf(pair('two'), { y: [1, 2, 3] })
    expect(plan.normalGamma.get('sigma2')?.pair).toBeUndefined()
    expect(plan.normalGamma.has('mu')).toBe(false)
  })
})
