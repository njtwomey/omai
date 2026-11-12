// What cannot be mapped is refused with the node and the reason, and the verdict names it.
import { describe, expect, it } from 'vitest'
import { analyseSpec } from '../analysis.ts'
import * as d from '../dists/index.ts'
import { compileFactorGraph, defineModel, factorGraphVerdict, ops, toFactorGraph } from '../index.ts'
import type { NamedValue } from '../operators.ts'

describe('what cannot be mapped is refused with the node and the reason', () => {
  const issuesOf = (
    spec: ReturnType<typeof defineModel>,
    data: Record<string, number[] | number>,
  ): ReturnType<typeof toFactorGraph> => toFactorGraph(spec.bind(data as never))

  it('an opaque custom node', () => {
    const m = defineModel('custom', (b) => {
      const mu = b.latent('mu', d.normal(0, 1))
      const warped = b.custom('warp', ([a]) => (a as NamedValue).value, [mu])
      b.observe('y', d.normal(warped, 1), b.data('y', { dims: ['row'] }))
    })
    const c = toFactorGraph(m.bind({ y: [1, 2] }))
    expect(c.graph).toBeUndefined()
    expect(c.opaque).toEqual(['y'])
    expect(c.issues.find((i) => i.node === 'y')).toMatchObject({ kind: 'opaque' })
    expect(c.issues[0]?.reason).toMatch(/custom function 'warp' is opaque/)
    expect(analyseSpec(m).opaque).toEqual(['y'])
  })

  it('a random scale, a non-affine operator, a family without a message family, a constraint, a param and a log-density term', () => {
    const scale = defineModel('scale', (b) => {
      const s = b.latent('s', d.gamma(2, 2))
      b.observe('y', d.normal(0, s), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(scale, { y: [1, 2] }).issues[0]).toMatchObject({ node: 'y', kind: 'operator' })
    expect(issuesOf(scale, { y: [1, 2] }).issues[0]?.reason).toMatch(/scale of 'y' depends on a random value/)

    const nonlinear = defineModel('log', (b) => {
      const w = b.latent('w', d.normal(0, 1))
      b.observe('y', d.normal(ops.log(w), 1), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(nonlinear, { y: [1, 2] }).issues[0]?.reason).toMatch(/operator 'log' applied to a random value/)

    const family = defineModel('cauchy', (b) => {
      b.observe('y', d.normal(b.latent('s', d.cauchy(0, 1)), 1), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(family, { y: [1] }).issues[0]).toMatchObject({ node: 's', kind: 'family' })

    const constrained = defineModel('cons', (b) => {
      b.observe(
        'y',
        d.normal(b.latent('x', d.normal(0, 1), { constraint: { kind: 'simplex' } }), 1),
        b.data('y', { dims: ['row'] }),
      )
    })
    expect(issuesOf(constrained, { y: [1] }).issues[0]).toMatchObject({ node: 'x', kind: 'constraint' })
    // A lower bound on a Gaussian is a truncated Gaussian: a comparison factor with an observed true outcome.
    const bounded = defineModel('bounded', (b) => {
      b.observe(
        'y',
        d.normal(b.latent('x', d.normal(0, 1), { constraint: { lower: 0 } }), 1),
        b.data('y', { dims: ['row'] }),
      )
    })
    const compiled = issuesOf(bounded, { y: [1] })
    expect(compiled.issues).toEqual([])
    expect(compiled.graph?.factors.some((f) => f.operator === 'greater-than')).toBe(true)

    const other = defineModel('other', (b) => {
      b.param('theta', 1)
      const x = b.latent('x', d.normal(0, 1))
      b.factor('penalty', ops.neg(ops.square(x)))
    })
    expect(toFactorGraph(other.bind({})).issues.map((i) => [i.node, i.kind])).toEqual([
      ['theta', 'node'],
      ['penalty', 'node'],
    ])
  })

  it('unbound data, missing values that are not ignored, and a product of two random values', () => {
    const m = defineModel('missing', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }), { missing: 'impute' })
    })
    expect(toFactorGraph(m.bind({ y: [1, Number.NaN, 2] })).issues[0]).toMatchObject({ node: 'y', kind: 'data' })
    const skip = defineModel('skip', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }), { missing: 'ignore' })
    })
    const c = toFactorGraph(skip.bind({ y: [1, Number.NaN, 2] }))
    expect(c.issues).toEqual([])
    expect(c.sites.find((s) => s.site === 'y')?.variables).toEqual(['y[0]', null, 'y[2]'])
    const square = defineModel('square', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      b.observe('y', d.normal(ops.mul(a, a), 1), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(square.bind({ y: [1] })).issues[0]?.reason).toMatch(/times itself/)
  })

  it('the verdict names the reason, the operators without a rule, and the graph without issues', () => {
    const m = defineModel('p', (b) => {
      const n = b.range('row')
      const X = b.data('X', { dims: [n, 'f'] })
      const w = b.latent('w', d.normal(0, 1), { dims: ['f'] })
      b.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), b.data('y', { dims: [n], support: 'binary' }))
    })
    const c = toFactorGraph(m.bind({ X: [[1, 2]], y: [1] }))
    const g = compileFactorGraph(c)
    expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
    // The probit link has a variational rule (the Albert-Chib augmentation), the affine map too.
    expect(factorGraphVerdict(c, g, 'vmp').ok).toBe(true)
    // A hard comparison has none, and says why.
    const hard = defineModel('hard', (b) => {
      const n = b.range('row')
      const X = b.data('X', { dims: [n, 'f'] })
      const w = b.latent('w', d.normal(0, 1), { dims: ['f'] })
      b.observe('y', d.bernoulli({ probs: ops.gt(X.matmul(w), 0) }), b.data('y', { dims: [n], support: 'binary' }))
    })
    const ch = toFactorGraph(hard.bind({ X: [[1, 2]], y: [1] }))
    const gh = compileFactorGraph(ch)
    expect(factorGraphVerdict(ch, gh, 'ep').ok).toBe(true)
    expect(factorGraphVerdict(ch, gh, 'vmp')).toMatchObject({ ok: false, missing: ['vmp message rules'] })
    expect(factorGraphVerdict(ch, gh, 'vmp').reason).toMatch(
      /greater-than has no variational.*KL\(q \|\| p\) is infinite/,
    )
    const bad = toFactorGraph(
      defineModel('bad', (b) => {
        b.observe(
          'y',
          d.normal(b.latent('mu', d.normal(0, 1)), b.latent('s', d.gamma(1, 1))),
          b.data('y', { dims: ['r'] }),
        )
      }).bind({ y: [1] }),
    )
    expect(factorGraphVerdict(bad, undefined, 'ep')).toMatchObject({ ok: false })
    expect(() => compileFactorGraph(bad)).toThrow(/does not compile to a factor graph/)
  })
})

// ----- what a product of two random values, a random precision and a gate map to -----
