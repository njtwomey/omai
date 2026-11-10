// The structure records of analyseSpec: conjugacy through plates, lookups, affine maps, scaled rates, hierarchies and joint
// priors, each claimed closed form checked against the model's own log joint (the log joint along the site minus the claimed
// log density is a constant); the linear-Gaussian blocks (a block's log joint is quadratic in the block's variables); the
// blankets, components, discrete width and minibatch axes; the report and its text.
import { describe, expect, it } from 'vitest'
import { type Distribution, gamma, inverseGamma, normal } from '../../dists/index.ts'
import { item, tensor } from '../../tensor/index.ts'
import { logJoint } from '../density.ts'
import * as d from './dists/index.ts'
import { analyseSpec, type BoundModel, defineModel, ops, structureReport, toProgram } from './index.ts'

const lj = (bound: BoundModel, values: Record<string, number | number[]>): number =>
  item(logJoint(toProgram(bound), undefined, values).logJoint)

/** The log joint along a scalar site minus a claimed log conditional density: constant when the claim is right. */
function conditionalGap(
  bound: BoundModel,
  values: Record<string, number | number[]>,
  at: (x: number) => Record<string, number | number[]>,
  claimed: Distribution,
  xs: readonly number[],
): number {
  const gaps = xs.map((x) => lj(bound, { ...values, ...at(x) }) - item(claimed.logProb(tensor(x))))
  return Math.max(...gaps) - Math.min(...gaps)
}

describe('conjugacy through plates and affine maps', () => {
  it('a plate sum: one latent with many observations, the posterior summing over the plate', () => {
    const m = defineModel('plate', (b) => {
      const y = b.data('y', { dims: ['n'] })
      b.observe('y', d.normal(b.latent('x', d.normal(1, 2)), 0.5), y)
    })
    const bound = m.bind({ y: [1.2, 0.3, 2.2, 1.7] })
    const [r] = analyseSpec(bound).conjugacy
    expect(r).toMatchObject({
      kind: 'pair',
      sites: ['x'],
      pairs: ['normal-normal loc'],
      given: [],
      sumsOver: ['n'],
      observations: 4,
      holds: true,
      collapse: { applies: true },
    })
    expect(r?.links[0]).toMatchObject({ node: 'y', via: 'direct', parameter: 'loc' })
    const prec = 1 / 4 + 4 / 0.25
    const mean = (1 / 4 + (1.2 + 0.3 + 2.2 + 1.7) / 0.25) / prec
    expect(
      conditionalGap(bound, {}, (x) => ({ x }), normal(mean, Math.sqrt(1 / prec)), [-1, 0.2, 1, 2.4]),
    ).toBeLessThan(1e-10)
  })

  it('an affine map: a known coefficient and an offset that is another latent, closed form given the offset', () => {
    const m = defineModel('affine', (b) => {
      const y = b.data('y', { dims: ['n'] })
      const off = b.latent('b', d.normal(0.5, 1))
      const x = b.latent('x', d.normal(1, 2))
      b.observe('y', d.normal(x.mul(2).add(off), 0.5), y)
    })
    const bound = m.bind({ y: [1.2, 0.3, 2.2, 1.7] })
    const r = analyseSpec(bound).conjugacy.find((c) => c.sites[0] === 'x')
    expect(r).toMatchObject({ pairs: ['normal-normal loc'], given: ['b'], holds: true })
    expect(r?.links[0]?.via).toBe('affine')
    expect(r?.collapse.applies).toBe(false)
    expect(r?.collapse.reason).toMatch(/affine/)
    // 1/s^2 = 1/s0^2 + sum c^2 / sigma^2 with c = 2, and the mean takes c (y - b) / sigma^2.
    const off = 0.4
    const ys = [1.2, 0.3, 2.2, 1.7]
    const prec = 1 / 4 + (4 * 4) / 0.25
    const mean = (1 / 4 + ys.reduce((a, v) => a + (2 * (v - off)) / 0.25, 0)) / prec
    expect(
      conditionalGap(bound, { b: off }, (x) => ({ x }), normal(mean, Math.sqrt(1 / prec)), [-0.5, 0.4, 1.1, 2]),
    ).toBeLessThan(1e-10)
  })

  it('a design matrix: the weights of a linear regression are conjugate through matmul', () => {
    const m = defineModel('lin', (b) => {
      const X = b.data('X', { dims: ['row', 'f'] })
      const y = b.data('y', { dims: ['row'] })
      b.observe('y', d.normal(X.matmul(b.latent('w', d.normal(0, 1), { dims: ['f'] })), 0.3), y)
    })
    const r = analyseSpec(m).conjugacy[0]
    expect(r?.links[0]?.via).toBe('affine')
    expect(r?.sumsOver).toEqual(['row'])
  })

  it('a group effect through an index, and a hierarchy: the group means have a conjugate hyperprior', () => {
    const m = defineModel('groups', (b) => {
      const g = b.range('group', { size: 3 })
      const idx = b.data('g', { dims: ['row'], support: 'index', indexes: g })
      const y = b.data('y', { dims: ['row'] })
      const mu = b.latent('mu', d.normal(0, 3))
      const a = b.latent('a', d.normal(mu, 1.5), { dims: [g] })
      b.observe('y', d.normal(a.at(idx), 0.5), y)
    })
    const gs = [0, 0, 1, 2, 2, 2]
    const ys = [1, 1.4, -0.5, 2, 2.3, 1.9]
    const bound = m.bind({ g: gs, y: ys })
    const recs = analyseSpec(bound).conjugacy
    const a = recs.find((c) => c.sites[0] === 'a')
    const mu = recs.find((c) => c.sites[0] === 'mu')
    expect(a).toMatchObject({ kind: 'pair', given: ['mu'], sumsOver: ['row'], observations: 6, holds: true })
    expect(a?.links[0]?.via).toBe('index')
    expect(mu).toMatchObject({ kind: 'hierarchical', given: [], holds: true, collapse: { applies: false } })
    expect(mu?.links[0]).toMatchObject({ node: 'a', nodeKind: 'latent' })
    // mu given the group means: the means are its data, with scale 1.5.
    const av = [0.9, -0.4, 2.1]
    const prec = 1 / 9 + 3 / 2.25
    const mean = av.reduce((s, v) => s + v / 2.25, 0) / prec
    expect(
      conditionalGap(bound, { a: av }, (x) => ({ mu: x }), normal(mean, Math.sqrt(1 / prec)), [-1, 0, 1.2, 2.5]),
    ).toBeLessThan(1e-10)
    // a[0] given mu and the rows of its group: two rows.
    const prec0 = 1 / 1.5 ** 2 + 2 / 0.25
    const mean0 = (0.7 / 1.5 ** 2 + (1 + 1.4) / 0.25) / prec0
    expect(
      conditionalGap(
        bound,
        { mu: 0.7 },
        (x) => ({ a: [x, av[1] as number, av[2] as number] }),
        normal(mean0, Math.sqrt(1 / prec0)),
        [-1, 0.3, 1, 1.8],
      ),
    ).toBeLessThan(1e-10)
  })

  it('a known multiple of a gamma rate: an exposure on a Poisson rate', () => {
    const m = defineModel('exposure', (b) => {
      const k = b.data('k', { dims: ['n'], support: 'count' })
      const w = b.data('w', { dims: ['n'], support: 'positive' })
      b.observe('k', d.poisson(b.latent('r', d.gamma(2, 1)).mul(w)), k)
    })
    const bound = m.bind({ k: [3, 1, 4, 2], w: [1.5, 0.5, 2, 1] })
    const r = analyseSpec(bound).conjugacy[0]
    expect(r?.links[0]?.via).toBe('scaled')
    expect(r?.posterior).toBe('Gamma(shape + sum y, rate + sum w)')
    expect(conditionalGap(bound, {}, (x) => ({ r: x }), gamma(2 + 10, 1 + 5), [0.4, 1, 2.2, 3.5])).toBeLessThan(1e-10)
  })

  it('a mixture mean is conjugate given the component labels', () => {
    const m = defineModel('mix', (b) => {
      const k = b.range('k', { size: 2 })
      const y = b.data('y', { dims: ['row'] })
      const mu = b.latent('mu', d.normal(0, 3), { dims: [k] })
      const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: ['row'] })
      b.observe('y', d.normal(mu.at(z), 1), y)
    })
    const r = analyseSpec(m).conjugacy.find((c) => c.sites[0] === 'mu')
    expect(r).toMatchObject({ given: ['z'], holds: true })
    expect(r?.collapse.applies).toBe(false)
  })

  it('a variance through the square root', () => {
    const m = defineModel('variance', (b) => {
      const v = b.latent('v', d.inverseGamma(2, 1))
      b.observe('y', d.normal(0, ops.sqrt(v)), b.data('y', { dims: ['n'] }))
    })
    const r = analyseSpec(m).conjugacy[0]
    expect(r?.links[0]?.via).toBe('variance')
    expect(r?.collapse.applies).toBe(true)
    const bound = m.bind({ y: [0.5, -1, 2] })
    const ss = 0.25 + 1 + 4
    expect(
      conditionalGap(bound, {}, (x) => ({ v: x }), inverseGamma(2 + 1.5, 1 + ss / 2), [0.3, 0.9, 2, 5]),
    ).toBeLessThan(1e-10)
  })

  it('says why a site is not exact: a factor term, a use that is not a conjugate form, a known scale that depends on it', () => {
    const factor = defineModel('factor', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe('y', d.normal(x, 1), b.data('y', { dims: ['n'] }))
      b.factor('pen', x.square().neg())
    })
    const r = analyseSpec(factor).conjugacy[0]
    expect(r?.holds).toBe(false)
    expect(r?.reason).toMatch(/factor 'pen'/)
    expect(r?.conditions.map((c) => c.holds)).toEqual([true, true, false])

    const curved = defineModel('curved', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe('y', d.normal(x, 1), b.data('y', { dims: ['n'] }))
      b.observe('z', d.normal(ops.exp(x), 1), b.data('z', { dims: ['n'] }))
    })
    const c = analyseSpec(curved).conjugacy[0]
    expect(c?.holds).toBe(false)
    expect(c?.conditions[0]?.holds).toBe(false)
    expect(c?.reason).toMatch(/exp\(x\)/)

    const scaled = defineModel('scale', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe('y', d.normal(x, ops.exp(x)), b.data('y', { dims: ['n'] }))
    })
    const s = analyseSpec(scaled).conjugacy[0]
    expect(s?.holds).toBe(false)
    expect(s?.conditions[1]?.holds).toBe(false)
  })

  it('the normal and inverse-gamma joint prior: the joint posterior is closed form', () => {
    const m = defineModel('nig', (b) => {
      const y = b.data('y', { dims: ['n'] })
      const v = b.latent('v', d.inverseGamma(2.5, 1.5))
      const mu = b.latent('mu', d.normal(0.5, ops.sqrt(v.div(2))))
      b.observe('y', d.normal(mu, ops.sqrt(v)), y)
    })
    const ys = [0.9, 1.4, 0.2, 1.1, 0.7]
    const bound = m.bind({ y: ys })
    const r = analyseSpec(bound).conjugacy.find((c) => c.kind === 'joint-prior')
    expect(r).toMatchObject({ sites: ['mu', 'v'], holds: true, observations: 5, collapse: { applies: false } })
    expect(r?.posterior).toMatch(/NormalInverseGamma/)
    // The claim: (mu, v) | y ~ NIG(m_n, k_n, a_n, b_n). The log joint minus its log density is constant on a grid.
    const n = ys.length
    const k0 = 2
    const m0 = 0.5
    const ybar = ys.reduce((a, v) => a + v, 0) / n
    const ss = ys.reduce((a, v) => a + (v - ybar) ** 2, 0)
    const kn = k0 + n
    const mn = (k0 * m0 + n * ybar) / kn
    const an = 2.5 + n / 2
    const bn = 1.5 + 0.5 * (ss + (k0 * n * (ybar - m0) ** 2) / kn)
    const gaps: number[] = []
    for (const v of [0.3, 0.8, 1.7])
      for (const mu of [-0.4, 0.5, 1.3]) {
        const claimed =
          item(inverseGamma(an, bn).logProb(tensor(v))) + item(normal(mn, Math.sqrt(v / kn)).logProb(tensor(mu)))
        gaps.push(lj(bound, { mu, v }) - claimed)
      }
    expect(Math.max(...gaps) - Math.min(...gaps)).toBeLessThan(1e-10)
  })

  it('the joint prior is refused when an observation has another scale or the variance is used elsewhere', () => {
    const m = defineModel('nig-bad', (b) => {
      const y = b.data('y', { dims: ['n'] })
      const v = b.latent('v', d.inverseGamma(2.5, 1.5))
      const mu = b.latent('mu', d.normal(0.5, ops.sqrt(v.div(2))))
      b.observe('y', d.normal(mu, ops.sqrt(v).mul(2)), y)
    })
    const r = analyseSpec(m).conjugacy.find((c) => c.kind === 'joint-prior')
    expect(r?.holds).toBe(false)
    expect(r?.reason).toMatch(/not Normal\(mu, sqrt\(v\)\)/)
  })
})

describe('linear-Gaussian blocks', () => {
  /** The log joint along a random line in the block's variables has constant second differences (it is quadratic). */
  function quadratic(bound: BoundModel, block: readonly string[], rest: Record<string, number | number[]>): number {
    const sizes: Record<string, number> = {}
    for (const n of bound.spec.nodes)
      if (n.kind === 'latent') sizes[n.name] = n.dims.reduce((p, dim) => p * (bound.sizes[dim] ?? 1), 1)
    const point = (t: number, dir: number): Record<string, number | number[]> => {
      const out: Record<string, number | number[]> = { ...rest }
      let k = 0
      for (const s of block) {
        const size = sizes[s] as number
        const vals = Array.from(
          { length: size },
          (_, i) => 0.3 * Math.sin(1 + dir + k + i) + t * Math.cos(2 + dir + k + i),
        )
        k += size
        out[s] =
          size === 1 && bound.spec.nodes.find((n) => n.name === s)?.dims.length === 0 ? (vals[0] as number) : vals
      }
      return out
    }
    const f = (t: number): number => lj(bound, point(t, 1))
    const h = 0.7
    const second = (t: number): number => f(t + h) - 2 * f(t) + f(t - h)
    return Math.abs(second(0) - second(0.9)) + Math.abs(second(0) - second(-1.3))
  }

  it('a hierarchy is a block given its scales: a quadratic log joint in the block, not in the scales', () => {
    const m = defineModel('hier', (b) => {
      const j = b.range('group')
      const y = b.data('y', { dims: [j, 'i'] })
      const x = b.data('x', { dims: [j, 'i'] })
      const mu = b.latent('mu', d.normal(0, 5))
      const beta = b.latent('beta', d.normal(0, 2))
      const tau = b.latent('tau', d.halfNormal(1))
      const sigma = b.latent('sigma', d.halfNormal(1))
      const a = b.latent('a', d.normal(mu, tau), { dims: [j] })
      b.observe('y', d.normal(a.add(beta.mul(x)), sigma), y)
    })
    const bound = m.bind({
      y: [
        [1, 2, 1.5],
        [0.2, 0.1, 0.5],
        [2, 2.5, 2.2],
      ],
      x: [
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 1],
      ],
    })
    const [blk] = analyseSpec(bound).linearGaussianBlocks
    expect(blk).toMatchObject({
      sites: ['mu', 'beta', 'a'],
      given: ['tau', 'sigma'],
      observed: ['y'],
      dimension: 5,
      whole: false,
    })
    const rest = { tau: 0.8, sigma: 0.6 }
    expect(quadratic(bound, ['mu', 'beta', 'a'], rest)).toBeLessThan(1e-9)
    // The scales are not in the block: the log joint is not quadratic in tau.
    const f = (t: number): number => lj(bound, { mu: 0.2, beta: -0.3, a: [0.1, 0.4, -0.2], tau: t, sigma: 0.6 })
    const second = (t: number): number => f(t + 0.2) - 2 * f(t) + f(t - 0.2)
    expect(Math.abs(second(0.8) - second(1.6))).toBeGreaterThan(1e-3)
  })

  it('the whole model is a block when nothing is left over', () => {
    const m = defineModel('whole', (b) => {
      const y = b.data('y', { dims: ['n'] })
      const mu = b.latent('mu', d.normal(0, 1))
      const s = b.latent('s', d.normal(0, 1))
      b.observe('y', d.normal(mu.add(s), 0.5), y)
    })
    const a = analyseSpec(m)
    expect(a.linearGaussian.holds).toBe(true)
    expect(a.linearGaussianBlocks[0]).toMatchObject({ sites: ['mu', 's'], given: [], whole: true })
  })

  it('a product of two block latents leaves the later one out; a non-Gaussian child or an exp leaves its latents out', () => {
    const bilinear = defineModel('bilinear', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const c = b.latent('c', d.normal(0, 1))
      b.observe('y', d.normal(a.mul(c), 1), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(bilinear).linearGaussianBlocks[0]).toMatchObject({ sites: ['a'], given: ['c'] })

    const curved = defineModel('curved', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      b.observe('y', d.normal(ops.exp(a), 1), b.data('y', { dims: ['n'] }))
    })
    expect(analyseSpec(curved).linearGaussianBlocks).toEqual([])

    const logistic = defineModel('logistic', (b) => {
      const X = b.data('X', { dims: ['row', 'f'] })
      b.observe(
        'y',
        d.bernoulli({ logits: X.matmul(b.latent('w', d.normal(0, 1), { dims: ['f'] })) }),
        b.data('y', { dims: ['row'] }),
      )
    })
    expect(analyseSpec(logistic).linearGaussianBlocks).toEqual([])
  })

  it('two independent regressions are two blocks', () => {
    const m = defineModel('two', (b) => {
      b.observe('y1', d.normal(b.latent('a', d.normal(0, 1)), 1), b.data('y1', { dims: ['n'] }))
      b.observe('y2', d.normal(b.latent('c', d.normal(0, 1)), 1), b.data('y2', { dims: ['m'] }))
    })
    expect(analyseSpec(m).linearGaussianBlocks.map((x) => x.sites)).toEqual([['a'], ['c']])
    expect(structureReport(m).components.map((c) => c.latents)).toEqual([['a'], ['c']])
  })
})

describe('graph structure', () => {
  const grouped = defineModel('grouped', (b) => {
    const j = b.range('group')
    const y = b.data('y', { dims: [j, 'i'] })
    const mu = b.latent('mu', d.normal(0, 5))
    const tau = b.latent('tau', d.halfNormal(1))
    const a = b.latent('a', d.normal(mu, tau), { dims: [j] })
    b.observe('y', d.normal(a, 1), y)
  })
  const bound = grouped.bind({
    y: [
      [1, 2],
      [3, 4],
      [5, 6],
    ],
  })

  it('Markov blankets: parents, children and the other parents of the children, per element along plates', () => {
    const blankets = Object.fromEntries(structureReport(bound).blankets.map((x) => [x.site, x]))
    expect(blankets['a']).toMatchObject({
      parents: ['mu', 'tau'],
      children: ['y'],
      coParents: [],
      perElement: ['group'],
    })
    expect(blankets['mu']).toMatchObject({ parents: [], children: ['a'], coParents: ['tau'] })
    expect(blankets['mu']?.blanket).toEqual(['a', 'tau'])
  })

  it('minibatch axes: observed dimensions that nothing reduces or looks up', () => {
    expect(structureReport(bound).minibatchAxes).toEqual([
      { dim: 'group', size: 3, observed: ['y'], local: ['a'], shared: ['mu', 'tau'] },
      { dim: 'i', size: 2, observed: ['y'], local: [], shared: ['mu', 'tau', 'a'] },
    ])
    // A sum over the rows ties them together; an index lookup of a dimension blocks it as an axis.
    const coupled = defineModel('coupled', (b) => {
      const x = b.latent('x', d.normal(0, 1), { dims: ['row'] })
      b.observe('y', d.normal(ops.sum(x, 'row'), 1), b.data('y', { dims: [] }))
      b.observe('z', d.normal(x, 1), b.data('z', { dims: ['row'] }))
    })
    expect(structureReport(coupled).minibatchAxes).toEqual([])
  })

  it('the discrete part: independent labels have width 0; a chain has width 1; labels meeting in one observation have width 2', () => {
    const mixture = defineModel('mixture', (b) => {
      const mu = b.latent('mu', d.normal(0, 3), { dims: [b.range('k', { size: 3 })] })
      const z = b.latent('z', d.categorical({ probs: [0.2, 0.3, 0.5] }), { dims: ['row'] })
      b.observe('y', d.normal(mu.at(z), 1), b.data('y', { dims: ['row'] }))
    })
    const mb = mixture.bind({ y: [0.1, 1.2, 2.2, 0.4] })
    expect(structureReport(mb).discretePart).toMatchObject({
      sites: ['z'],
      width: 0,
      components: [{ sites: ['z'], width: 0, cliqueStates: 3, elementwise: true, elements: 4, cardinality: { z: 3 } }],
    })
    const chain = defineModel('chain', (b) => {
      const z1 = b.latent('z1', d.bernoulli({ probs: 0.5 }))
      const z2 = b.latent('z2', d.bernoulli({ probs: ops.where(z1, 0.9, 0.2) }))
      const z3 = b.latent('z3', d.bernoulli({ probs: ops.where(z2, 0.9, 0.2) }))
      const z4 = b.latent('z4', d.bernoulli({ probs: ops.where(z3, 0.9, 0.2) }))
      for (const [i, z] of [z1, z2, z3, z4].entries())
        b.observe(`y${i}`, d.bernoulli({ probs: ops.where(z, 0.8, 0.3) }), b.data(`y${i}`))
    })
    const cp = structureReport(chain).discretePart
    expect(cp.width).toBe(1)
    expect(cp.components[0]?.cliqueStates).toBe(4)
    expect(cp.orderHint).toHaveLength(4)
    const meeting = defineModel('meeting', (b) => {
      const [p, q, r] = ['a', 'b', 'c'].map((n) => b.latent(n, d.bernoulli({ probs: 0.5 })))
      b.observe('y', d.bernoulli({ probs: ops.sigmoid(ops.add(ops.add(p ?? 0, q ?? 0), r ?? 0)) }), b.data('y'))
    })
    expect(structureReport(meeting).discretePart).toMatchObject({ width: 2, components: [{ cliqueStates: 8 }] })
  })

  it('a sum over the plate of a discrete site couples its elements', () => {
    const m = defineModel('count', (b) => {
      const z = b.latent('z', d.bernoulli({ probs: 0.5 }), { dims: ['row'] })
      b.observe('y', d.normal(ops.sum(z, 'row'), 1), b.data('y'))
    })
    const c = structureReport(m).discretePart.components[0]
    expect(c).toMatchObject({ elementwise: false, coupledAlong: ['row'] })
    expect(structureReport(m).suggestions.find((s) => s.kind === 'enumerate')?.text).toMatch(
      /cannot be summed out cheaply/,
    )
  })
})

describe('the structure report', () => {
  it('names the method or handler, its knobs and the reason, and prints', () => {
    const m = defineModel('hier', (b) => {
      const j = b.range('group')
      const y = b.data('y', { dims: [j, 'i'] })
      const mu = b.latent('mu', d.normal(0, 5))
      const tau = b.latent('tau', d.halfNormal(1))
      const a = b.latent('a', d.normal(mu, tau), { dims: [j] })
      b.observe('y', d.normal(a, 1), y, { dims: ['i', j] })
    })
    const bound = m.bind({
      y: [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    })
    const r = structureReport(bound)
    const collapse = r.suggestions.find((s) => s.kind === 'collapse')
    expect(collapse).toMatchObject({
      sites: ['a'],
      use: [{ kind: 'handler', name: 'collapse', knobs: { sites: ['a'] } }],
    })
    const reparam = r.suggestions.find((s) => s.kind === 'reparameterise')
    expect(reparam?.use[0]).toMatchObject({ name: 'reparam', knobs: { a: 'locScale(0)' } })
    // Declared with the group dimension first the handler cannot line the site up, and the report says so.
    const first = defineModel('hier-first', (b) => {
      const j = b.range('group')
      const a = b.latent('a', d.normal(0, 1), { dims: [j] })
      b.observe('y', d.normal(a, 1), b.data('y', { dims: [j, 'i'] }))
    }).bind({ y: [[1, 2], [3, 4], [5, 6]] })
    const conditional = structureReport(first).suggestions.find((s) => s.kind === 'conditional')
    expect(conditional?.text).toMatch(/trailing dimensions/)
    const text = analyseSpec(bound).explain()
    expect(text).toBe(r.explain())
    expect(text).toMatch(/Structure of hier/)
    expect(text).toMatch(/collapse can integrate it out/)
    expect(text).toMatch(/Gaussian block conditional on \{tau\}/)
  })

  it('leaves the analysis a plain value: the methods are not enumerated or serialised', () => {
    const m = defineModel('coin', (b) => {
      b.observe('y', d.bernoulli({ probs: b.latent('p', d.beta(1, 1)) }), b.data('y', { dims: ['flip'] }))
    })
    const a = analyseSpec(m)
    expect(Object.keys(a)).not.toContain('explain')
    expect(() => JSON.stringify(a)).not.toThrow()
    expect(JSON.parse(JSON.stringify(a)).conjugacy[0].sites).toEqual(['p'])
    expect(JSON.parse(JSON.stringify(structureReport(m))).suggestions[0].kind).toBe('closed-form')
  })
})
