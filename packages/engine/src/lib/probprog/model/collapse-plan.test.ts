// collapsePlan against the handler: a site is in the plan exactly when `collapse` accepts it, the collapsed program is the
// marginal likelihood (checked against the closed form), recover draws the removed sites back, and the plan says why not for
// the rest.
import { describe, expect, it } from 'vitest'
import { item, tensor, toFlat } from '../../tensor/index.ts'
import { collapse } from '../collapse.ts'
import { logJoint } from '../density.ts'
import * as d from './dists/index.ts'
import { applyCollapse, type BoundModel, collapsePlan, defineModel, ops, toProgram } from './index.ts'

function accepted(bound: BoundModel, site: string): boolean {
  try {
    collapse(toProgram(bound), undefined, { sites: [site] })
    return true
  } catch {
    return false
  }
}

describe('collapsePlan agrees with the handler', () => {
  const school = 'school'
  const cases: Record<string, BoundModel> = {
    'plate sum': defineModel('plate', (b) => {
      b.observe('y', d.normal(b.latent('x', d.normal(1, 2)), 0.5), b.data('y', { dims: ['n'] }))
    }).bind({ y: [1.2, 0.3, 2.2, 1.7] }),
    'hierarchy: group effects over a plate of houses': defineModel('houses', (b) => {
      const h = b.range('house', { size: 2 })
      const j = b.range('county', { size: 3 })
      const mu = b.latent('mu', d.normal(0, 3))
      const tau = b.latent('tau', d.halfNormal(1))
      const a = b.latent('alpha', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), { dims: [j] })
      b.observe('y', d.normal(a, 0.5), b.data('y', { dims: [h, j] }), { dims: [h, j] })
    }).bind({
      y: [
        [1, 0.1, 2],
        [1.2, -0.2, 2.4],
      ],
    }),
    'the replicated dimension is declared after the site: the handler cannot line it up': defineModel('order', (b) => {
      const j = b.range('county', { size: 3 })
      const h = b.range('house', { size: 2 })
      const a = b.latent('alpha', d.normal(0, 1), { dims: [j] })
      b.observe('y', d.normal(a, 0.5), b.data('y', { dims: [j, h] }))
    }).bind({
      y: [
        [1, 1.2],
        [0.1, -0.2],
        [2, 2.4],
      ],
    }),
    'affine map': defineModel('affine', (b) => {
      const x = b.latent('x', d.normal(1, 2))
      b.observe('y', d.normal(x.mul(2).add(0.5), 0.5), b.data('y', { dims: ['n'] }))
    }).bind({ y: [1.2, 0.3, 2.2] }),
    'index lookup': defineModel('index', (b) => {
      const g = b.range('group', { size: 2 })
      const idx = b.data('g', { dims: ['row'], support: 'index', indexes: g })
      b.observe(
        'y',
        d.normal(b.latent('a', d.normal(0, 1), { dims: [g] }).at(idx), 0.5),
        b.data('y', { dims: ['row'] }),
      )
    }).bind({ g: [0, 1, 1], y: [0.2, 1, 1.4] }),
    'a recorded derived node reads the site': defineModel('derived', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.derived('x2', x.mul(2))
      b.observe('y', d.normal(x, 1), b.data('y', { dims: ['n'] }))
    }).bind({ y: [0.1, 0.5] }),
    'a factor term reads the site': defineModel('factor', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe('y', d.normal(x, 1), b.data('y', { dims: ['n'] }))
      b.factor('pen', x.square().neg())
    }).bind({ y: [0.1, 0.5] }),
    'beta and gamma sites': defineModel('two', (b) => {
      b.observe('c', d.bernoulli({ probs: b.latent('p', d.beta(1, 1)) }), b.data('c', { dims: ['n'] }))
      b.observe('k', d.poisson(b.latent('r', d.gamma(2, 1))), b.data('k', { dims: ['m'] }))
    }).bind({ c: [1, 0, 1, 1], k: [2, 3, 1] }),
    'a mean and a variance that depend on each other': defineModel('mv', (b) => {
      const v = b.latent('v', d.inverseGamma(3, 2))
      const mu = b.latent('mu', d.normal(0, 3))
      b.observe('y', d.normal(mu, ops.sqrt(v)), b.data('y', { dims: ['n'] }))
    }).bind({ y: [0.5, 1.4, 0.9] }),
    'a scale that is a latent: eight schools': defineModel('schools', (b) => {
      const s = b.range(school, { size: 3 })
      const mu = b.latent('mu', d.normal(0, 5))
      const tau = b.latent('tau', d.halfCauchy(5))
      const theta = b.latent('theta', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), { dims: [s] })
      b.observe('y', d.normal(theta, b.data('sigma', { dims: [s] })), b.data('y', { dims: [s] }))
    }).bind({ y: [28, 8, -3], sigma: [15, 10, 16] }),
  }
  for (const [name, bound] of Object.entries(cases))
    it(name, () => {
      const plan = collapsePlan(bound)
      const latents = bound.spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name)
      // A site is in the plan when the handler takes it alone, and the plan never contains a site the handler refuses.
      for (const s of plan.sites)
        expect(accepted(bound, s), `${s} is in the plan but the handler refuses it`).toBe(true)
      const alone = latents.filter((s) => accepted(bound, s))
      for (const s of alone)
        expect(
          plan.sites.includes(s) || plan.refused.some((r) => r.site === s),
          `${s} is neither planned nor refused`,
        ).toBe(true)
      // What the handler cannot take is never planned; the exceptions are the sites the plan leaves out because they depend on a planned one.
      for (const r of plan.refused.filter((x) => !x.reason.includes('depend on each other')))
        expect(alone).not.toContain(r.site)
      if (plan.sites.length > 0) expect(() => applyCollapse(bound, plan)).not.toThrow()
    })

  it('names the sites and the reasons', () => {
    expect(collapsePlan(cases['plate sum'] as BoundModel).sites).toEqual(['x'])
    const hier = collapsePlan(cases['hierarchy: group effects over a plate of houses'] as BoundModel)
    expect(hier).toMatchObject({ sites: ['alpha'], remaining: ['mu', 'tau'], dimensionBefore: 5, dimensionAfter: 2 })
    expect(hier.entries[0]).toMatchObject({
      pair: 'normal-normal loc',
      given: ['mu', 'tau'],
      removes: 3,
      sumsOver: ['house'],
    })
    expect(hier.refused.map((r) => r.site)).toEqual(['mu'])
    expect(hier.text).toMatch(/leaving 2 of 5 unconstrained coordinates/)
    expect(collapsePlan(cases['affine map'] as BoundModel).refused[0]?.reason).toMatch(/affine/)
    expect(collapsePlan(cases['a recorded derived node reads the site'] as BoundModel).refused[0]?.reason).toMatch(
      /derived node x2/,
    )
    expect(collapsePlan(cases['a factor term reads the site'] as BoundModel).sites).toEqual([])
    const mv = collapsePlan(cases['a mean and a variance that depend on each other'] as BoundModel)
    expect(mv.sites).toEqual(['v'])
    expect(mv.refused.find((r) => r.site === 'mu')?.reason).toMatch(/depend on each other/)
    expect(
      collapsePlan(
        cases['the replicated dimension is declared after the site: the handler cannot line it up'] as BoundModel,
      ).refused[0]?.reason,
    ).toMatch(/trailing dimensions/)
  })

  it('a subsampled child is refused', () => {
    const m = defineModel('batch', (b) => {
      b.observe('y', d.normal(b.latent('x', d.normal(0, 1)), 1), b.data('y', { dims: ['n'] }))
    })
    const bound = m.bind({ y: [0.1, 0.2, 0.3, 0.4] }, { batch: { dim: 'n', size: 2 } })
    expect(collapsePlan(bound).sites).toEqual([])
    expect(collapsePlan(bound).refused[0]?.reason).toMatch(/subsampled/)
  })
})

describe('applyCollapse', () => {
  const m = defineModel('houses', (b) => {
    const h = b.range('house', { size: 2 })
    const j = b.range('county', { size: 3 })
    const mu = b.latent('mu', d.normal(0, 3))
    const tau = b.latent('tau', d.halfNormal(1))
    const a = b.latent('alpha', d.normal(mu, tau), { dims: [j] })
    b.observe('y', d.normal(a, 0.5), b.data('y', { dims: [h, j] }), { dims: [h, j] })
  })
  // [house][county]: the two houses of each of three counties.
  const bound = m.bind({
    y: [
      [1, 0.1, 2],
      [1.2, -0.2, 2.4],
    ],
  })
  const ys = [
    [1, 1.2],
    [0.1, -0.2],
    [2, 2.4],
  ]

  it('gives the marginal likelihood: the collapsed log joint is the closed form with the effects integrated out', () => {
    const c = applyCollapse(bound, collapsePlan(bound))
    expect(c.sites).toEqual(['alpha'])
    // y_j = (y_j1, y_j2) given mu, tau is N(mu 1, 0.25 I + tau^2 11').
    const point = { mu: 0.6, tau: 0.9 }
    const marginal = (mu: number, tau: number): number => {
      let total = 0
      for (const row of ys) {
        const n = row.length
        const ybar = row.reduce((s, v) => s + v, 0) / n
        const ss = row.reduce((s, v) => s + (v - ybar) ** 2, 0)
        const s2 = 0.25
        // log N(y; mu 1, s2 I + tau^2 11') by the matrix determinant lemma.
        const quad = ss / s2 + (n * (ybar - mu) ** 2) / (s2 + n * tau * tau)
        total +=
          -0.5 * n * Math.log(2 * Math.PI) -
          0.5 * (n - 1) * Math.log(s2) -
          0.5 * Math.log(s2 + n * tau * tau) -
          0.5 * quad
      }
      return total
    }
    const collapsed = item(logJoint(c.model, undefined, point).logJoint)
    // The collapsed program also holds the priors of mu and tau (halfNormal(1) on tau): add them to the closed form.
    const prior = (mu: number, tau: number): number =>
      -0.5 * Math.log(2 * Math.PI * 9) - (mu * mu) / 18 + Math.log(2 / Math.sqrt(2 * Math.PI)) - (tau * tau) / 2
    expect(collapsed).toBeCloseTo(marginal(point.mu, point.tau) + prior(point.mu, point.tau), 9)
  })

  it('recovers the removed sites from their closed-form posteriors given the rest', () => {
    const c = applyCollapse(bound, collapsePlan(bound))
    const post = c.posteriors({ mu: 0.6, tau: 0.9 }, undefined)
    // alpha_j | mu, tau, y: precision 1/tau^2 + 2/0.25, mean from mu/tau^2 and the two observations.
    const prec = 1 / 0.81 + 2 / 0.25
    const mean0 = (0.6 / 0.81 + (1 + 1.2) / 0.25) / prec
    const mean = toFlat(post['alpha']?.mean() ?? tensor([]))
    expect(mean[0]).toBeCloseTo(mean0, 10)
    expect(toFlat(post['alpha']?.variance() ?? tensor([]))[0]).toBeCloseTo(1 / prec, 10)
  })

  it('refuses an empty plan with the reasons', () => {
    const none = defineModel('none', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe('y', d.normal(ops.exp(x), 1), b.data('y', { dims: ['n'] }))
    }).bind({ y: [1, 2] })
    expect(() => applyCollapse(none, collapsePlan(none))).toThrow(/no site to collapse/)
  })
})
