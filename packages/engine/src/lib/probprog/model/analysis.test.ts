// Static analysis of specs: dependencies, plates, conjugate pairs and the conditions that make them usable, the
// linear-Gaussian check, opaque nodes and the feature list a method's check reads.
import { describe, expect, it } from 'vitest'
import { exponentialFamily } from '../../dists/index.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import { distributionFamilies } from './dists/families.ts'
import * as d from './dists/index.ts'
import { analyseSpec, defineModel, type NamedValue, ops } from './index.ts'

describe('conjugate pairs', () => {
  it('beta and Bernoulli, binomial, and gamma and Poisson', () => {
    const coin = defineModel('coin', (b) => {
      b.observe('y', d.bernoulli({ probs: b.latent('p', d.beta(1, 1)) }), b.data('y', { dims: ['flip'] }))
    })
    const c = analyseSpec(coin).conjugate
    expect(c).toHaveLength(1)
    expect(c[0]).toMatchObject({
      prior: 'p',
      priorFamily: 'beta',
      likelihood: 'y',
      likelihoodFamily: 'bernoulli',
      parameter: 'probs',
      via: 'direct',
      known: true,
      hyperparametersKnown: true,
      exclusive: true,
    })
    const counts = defineModel('counts', (b) => {
      b.observe('k', d.poisson(b.latent('rate', d.gamma(2, 1))), b.data('k', { dims: ['n'] }))
    })
    expect(analyseSpec(counts).conjugate[0]?.pair.prior).toBe('gamma')
    const bin = defineModel('bin', (b) => {
      b.observe('k', d.binomial(10, b.latent('p', d.beta(1, 2))), b.data('k', { dims: ['n'] }))
    })
    expect(analyseSpec(bin).conjugate[0]).toMatchObject({ likelihoodFamily: 'binomial', known: true })
  })

  it('normal-normal needs the scale known; a latent scale breaks it', () => {
    const known = defineModel('known', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(known).conjugate[0]).toMatchObject({ parameter: 'loc', known: true })
    const unknown = defineModel('unknown', (b) => {
      const mu = b.latent('mu', d.normal(0, 2))
      const s = b.latent('s', d.halfNormal(1))
      b.observe('y', d.normal(mu, s), b.data('y', { dims: ['n'] }))
    })
    const c = analyseSpec(unknown).conjugate.find((x) => x.prior === 'mu')
    expect(c?.known).toBe(false)
  })

  it('an inverse-gamma prior on a variance, through the square root', () => {
    const m = defineModel('variance', (b) => {
      const v = b.latent('v', d.inverseGamma(2, 1))
      b.observe('y', d.normal(0, ops.sqrt(v)), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(m).conjugate[0]).toMatchObject({
      prior: 'v',
      priorFamily: 'inverseGamma',
      parameter: 'variance',
      known: true,
    })
  })

  it('Dirichlet and categorical', () => {
    const m = defineModel('cat', (b) => {
      b.observe('z', d.categorical({ probs: b.latent('p', d.dirichlet([1, 1, 1])) }), b.data('z', { dims: ['n'] }))
    })
    expect(analyseSpec(m).conjugate[0]?.pair.likelihood).toBe('categorical')
  })

  it('sees a pair through an index lookup (a group effect), and reports unknown hyperparameters', () => {
    const m = defineModel('groups', (b) => {
      const g = b.range('group')
      const idx = b.data('g', { dims: ['row'], support: 'index', indexes: g })
      const mu = b.latent('mu', d.normal(0, 1))
      const a = b.latent('a', d.normal(mu, 1), { dims: [g] })
      b.observe('y', d.normal(a.at(idx), 0.5), b.data('y', { dims: ['row'] }))
    })
    const c = analyseSpec(m).conjugate
    expect(c.map((x) => [x.prior, x.via, x.hyperparametersKnown])).toEqual([['a', 'index', false]])
  })

  it('is not exclusive when something else depends on the latent', () => {
    const m = defineModel('shared', (b) => {
      const p = b.latent('p', d.beta(1, 1))
      b.observe('y', d.bernoulli({ probs: p }), b.data('y', { dims: ['n'] }))
      b.latent('q', d.normal(p, 1))
    })
    expect(analyseSpec(m).conjugate[0]?.exclusive).toBe(false)
  })

  it('finds none where the parameter is transformed', () => {
    const m = defineModel('transformed', (b) => {
      const p = b.latent('p', d.beta(1, 1))
      b.observe('y', d.bernoulli({ probs: ops.sqrt(p) }), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(m).conjugate).toEqual([])
  })
})

describe('structure and features', () => {
  const regression = defineModel('regression', (b) => {
    const n = b.range('row')
    const f = b.range('feature')
    const X = b.data('X', { dims: [n, f] })
    const y = b.data('y', { dims: [n] })
    const w = b.latent('w', d.normal(0, 1), { dims: [f] })
    const eta = b.derived('eta', X.matmul(w))
    b.observe('y', d.normal(eta, 0.5), y)
    b.generated('yRep', d.normal(eta, 0.5))
  })

  it('lists nodes by kind with parents, children, dims and operators', () => {
    const a = analyseSpec(regression)
    expect(a.data).toEqual(['X', 'y'])
    expect(a.latent).toEqual(['w'])
    expect(a.derived).toEqual(['eta'])
    expect(a.observed).toEqual(['y'])
    expect(a.generated).toEqual(['yRep'])
    const eta = a.nodes.find((n) => n.name === 'eta')
    expect(eta).toMatchObject({
      id: 'eta',
      parents: ['data:X', 'w'],
      children: ['y', 'yRep'],
      operators: ['matmul'],
      dims: ['row'],
    })
    expect(a.dims.map((x) => [x.name, x.nodes])).toEqual([
      ['row', ['data:X', 'data:y', 'eta', 'y', 'yRep']],
      ['feature', ['data:X', 'w']],
    ])
    expect(a.edges).toContainEqual({ from: 'eta', to: 'y' })
  })

  it('knows the sizes of a bound model', () => {
    const bound = regression.bind({
      X: [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      y: [1, 2, 3],
    })
    expect(analyseSpec(bound).dims.map((x) => [x.name, x.size])).toEqual([
      ['row', 3],
      ['feature', 2],
    ])
  })

  it('recognises a linear-Gaussian model, and says why a model is not one', () => {
    expect(analyseSpec(regression).linearGaussian.holds).toBe(true)
    const nonlinear = defineModel('nonlinear', (b) => {
      const w = b.latent('w', d.normal(0, 1))
      b.observe('y', d.normal(ops.exp(w), 1), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(nonlinear).linearGaussian).toMatchObject({
      holds: false,
      reason: expect.stringContaining('not affine'),
    })
    const logistic = defineModel('logistic', (b) => {
      b.observe('y', d.bernoulli({ logits: b.latent('w', d.normal(0, 1)) }), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(logistic).linearGaussian.reason).toContain('bernoulli')
    const scaled = defineModel('scaled', (b) => {
      const s = b.latent('s', d.normal(1, 0.1))
      b.observe('y', d.normal(0, s), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(scaled).linearGaussian.reason).toContain('scale')
  })

  it('marks custom calls and families without an exponential-family form as opaque', () => {
    const m = defineModel('opaque', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      const c = b.custom('f', ([v]) => (v as NamedValue).value, [x])
      b.observe('y', d.normal(c, 1), 0)
      b.latent('t', d.cauchy(0, 1))
    })
    const a = analyseSpec(m)
    expect(a.opaque).toEqual(['y', 't'])
    expect(a.features).toContain('opaque nodes')
    expect(a.features).not.toContain('exponential-family factors')
  })

  it('reports the features a method reads', () => {
    const discrete = defineModel('discrete', (b) => {
      const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: ['row'] })
      b.latent('k', d.poisson(3))
      b.observe('y', d.normal(z, 1), b.data('y', { dims: ['row'] }))
    })
    const a = analyseSpec(discrete)
    expect(a.finiteDiscrete).toEqual(['z'])
    expect(a.unboundedDiscrete).toEqual(['k'])
    expect(a.continuous).toEqual([])
    expect(a.features).toEqual(
      expect.arrayContaining(['finite discrete latents', 'unbounded discrete latents', 'likelihood', 'plates']),
    )
    const h = analyseSpec(
      defineModel('h', (b) => {
        const mu = b.latent('mu', d.normal(0, 1))
        b.latent('x', d.normal(mu, 1))
      }),
    )
    expect(h.features).toContain('hierarchical')
    expect(h.hasLikelihood).toBe(false)
  })
})

describe('the family table', () => {
  const families = distributionFamilies()

  it('builds the library distribution of every family from its example', () => {
    for (const [name, f] of Object.entries(families)) {
      const args = Object.fromEntries(Object.entries(f.example).map(([k, v]) => [k, tensor(v as number)]))
      const dist = f.make(args)
      expect(dist.eventShape.length, name).toBe(f.eventRank)
      expect(
        Number.isFinite(toFlat(dist.mean())[0] as number) || name === 'cauchy' || name === 'halfCauchy',
        name,
      ).toBe(true)
    }
  })

  it('flags exponential families as the library sees them', () => {
    for (const [name, f] of Object.entries(families)) {
      const args = Object.fromEntries(Object.entries(f.example).map(([k, v]) => [k, tensor(v as number)]))
      let isFamily = true
      try {
        exponentialFamily(f.make(args))
      } catch {
        isFamily = false
      }
      expect(f.exponentialFamily, `${name}: ${isFamily}`).toBe(isFamily)
    }
  })

  it('rejects unknown families and parameters, and missing or doubled alternatives', () => {
    expect(() => d.family('nope', {})).toThrow(/unknown distribution family/)
    expect(() => d.family('normal', { mean: 0 })).toThrow(/unknown parameter 'mean'/)
    expect(() => d.family('gamma', {})).toThrow(/'concentration' is missing/)
    expect(() => d.family('bernoulli', {})).toThrow(/exactly one of probs, logits/)
    expect(() => d.family('bernoulli', { probs: 0.5, logits: 0 })).toThrow(/exactly one/)
  })
})
