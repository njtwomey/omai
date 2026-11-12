// A likelihood chosen by an index among branches of different families (`d.gate`): the family of the model language, its density
// in the other engines (the library's mixture with the one-hot weights of the index), its compilation to one engine gate per
// element whose branch k is the factor its family maps to alone, the weights of branches with no random parameter, the known
// index, what is refused, and EP through `runEp` against the grid posterior of the same model with the index summed out.
import { describe, expect, it } from 'vitest'
import { mixture } from '../../../dists/compose/index.ts'
import { type Distribution, negativeBinomial, normal, poisson } from '../../../dists/index.ts'
import { add, div, mul, tensor, exp as texp, toFlat } from '../../../tensor/index.ts'
import { logJoint } from '../../density.ts'
import { runEp } from '../../ep/index.ts'
import { epRoute } from '../../inference/ep-route.ts'
import { observe, sample } from '../../primitives.ts'
import { gridPosterior } from '../../validation/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph, toProgram } from '../index.ts'

const R = 2

interface Branch {
  readonly operator?: string
  readonly args: readonly number[]
  readonly params?: Readonly<Record<string, number | string>>
}
/** The branches of a gate factor, and the values of a table factor, as the graph carries them. */
const branchesOf = (factor: { readonly params?: unknown } | undefined): readonly Branch[] =>
  ((factor?.params ?? {}) as { readonly branches?: readonly Branch[] }).branches ?? []
const valuesOf = (factor: { readonly params?: unknown } | undefined): readonly number[] =>
  ((factor?.params ?? {}) as { readonly values?: readonly number[] }).values ?? []
/** The negative binomial in the NB2 idiom with a known dispersion: mean `exp(log mean)`, `R` successes. */
const overdispersed = (logMean: Parameters<typeof ops.exp>[0]): d.DistExpr =>
  d.negativeBinomial(R, ops.div(R, ops.add(R, ops.exp(logMean))))

// A few low counts and a few high, overdispersed ones: which regime a row came from is not told.
const Y = [0, 1, 2, 1, 9, 6, 12, 3, 1, 7]

const regimes = defineModel('count-regimes', (m) => {
  const row = m.range('row')
  const y = m.data('y', { dims: [row], support: 'count' })
  const a = m.latent('a', d.normal(0.3, 0.5))
  const b = m.latent('b', d.normal(2, 0.5))
  const z = m.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [row] })
  m.observe('y', d.gate(z, [d.poisson(ops.exp(a)), overdispersed(b)]), y)
})

describe('the gate family of the model language', () => {
  it('is named by its branches, with their parameters flattened beside the index', () => {
    const dist = d.gate(1, [d.poisson(3), d.negativeBinomial(2, 0.4)])
    expect(dist.spec.family).toBe('gate(poisson,negativeBinomial)')
    expect(Object.keys(dist.spec.params)).toEqual(['index', '0.rate', '1.n', '1.probs'])
    expect(d.gateBranches(dist.spec.family)).toEqual(['poisson', 'negativeBinomial'])
    expect(d.gateBranches('poisson')).toBeUndefined()
    expect(d.distributionFamilies()[dist.spec.family]?.params).toEqual(['index', '0.rate', '1.n', '1.probs'])
  })

  it('has the density of the branch the index names, in the library’s own distributions', () => {
    const family = d.distributionFamilies()['gate(poisson,negativeBinomial)']
    const args = (index: number): Record<string, ReturnType<typeof tensor>> => ({
      index: tensor(index),
      '0.rate': tensor(3),
      '1.n': tensor(2),
      '1.probs': tensor(0.4),
    })
    for (const y of [0, 2, 7]) {
      const first = toFlat(family?.make(args(0)).logProb(tensor(y)) ?? tensor(0))[0]
      const second = toFlat(family?.make(args(1)).logProb(tensor(y)) ?? tensor(0))[0]
      expect(first).toBeCloseTo(toFlat(poisson(3).logProb(tensor(y)))[0] as number, 12)
      expect(second).toBeCloseTo(toFlat(negativeBinomial(2, 0.4).logProb(tensor(y)))[0] as number, 12)
    }
    // an index per row picks per row
    const both = family?.make({ ...args(0), index: tensor([0, 1, 1]) }).logProb(tensor([1, 1, 4]))
    expect(toFlat(both ?? tensor(0)).length).toBe(3)
  })

  it('is read by the other engines as the program of the model: the log joint at fixed values is the branch each row’s index names', () => {
    const rows = [3, 8]
    const joint = logJoint(toProgram(regimes.bind({ y: rows })), undefined, { a: 0.2, b: 1.5, z: [0, 1] }).logJoint
    const at = (dist: Distribution, y: number): number => toFlat(dist.logProb(tensor(y)))[0] as number
    const expected =
      at(normal(0.3, 0.5), 0.2) +
      at(normal(2, 0.5), 1.5) +
      2 * Math.log(0.5) +
      at(poisson(Math.exp(0.2)), 3) +
      at(negativeBinomial(R, R / (R + Math.exp(1.5))), 8)
    expect(toFlat(joint)[0]).toBeCloseTo(expected, 10)
  })

  it('refuses branches that are not scalar-event families of the table, one branch, or discrete mixed with continuous', () => {
    expect(() => d.gate(0, [d.poisson(3)])).toThrow(/at least two branches/)
    expect(() => d.gate(0, [d.poisson(3), d.dirichlet([1, 2])])).toThrow(/event axis/)
    expect(() => d.gate(0, [d.poisson(3), d.normal(0, 1)])).toThrow(/all discrete or all continuous/)
    expect(() => d.gate(0, [d.poisson(3), d.gate(0, [d.poisson(1), d.poisson(2)])])).toThrow(
      /not a family a gate takes/,
    )
  })
})

describe('a gate over different families compiled to the engine gate', () => {
  const c = toFactorGraph(regimes.bind({ y: Y }))

  it('is one gate per row on the index, branch 0 the Poisson site of a and branch 1 the negative binomial site of b', () => {
    expect(c.issues).toEqual([])
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(Y.length)
    const first = gates[0]
    expect(first?.args).toEqual(['z[0]', 'a', 'b'])
    const branches = branchesOf(first)
    expect(branches.map((br) => br.operator)).toEqual(['likelihood-site', 'likelihood-site'])
    expect(branches.map((br) => br.args)).toEqual([[0], [1]])
    expect(branches.map((br) => br.params?.['likelihood'])).toEqual(['poisson-log', 'negative-binomial-log'])
    expect(branches.every((br) => br.params?.['y'] === Y[0])).toBe(true)
  })

  it('keeps no likelihood factor of the branches outside the gates: the graph is the priors and the gates', () => {
    expect([...new Set(c.graph?.factors.map((f) => f.operator))].sort()).toEqual(['gate', 'prior'])
  })

  it('reads a categorical index of three states, a regression predictor and a Gaussian branch', () => {
    const model = defineModel('three-regimes', (m) => {
      const row = m.range('row')
      const x = m.data('x', { dims: [row] })
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(0, 1))
      const s = m.latent('s', d.normal(0, 1))
      const z = m.latent('z', d.categorical({ probs: [0.2, 0.5, 0.3] }), { dims: [row] })
      m.observe(
        'y',
        d.gate(z, [d.poisson(ops.exp(a)), d.poisson(ops.exp(ops.add(a, ops.mul(s, x)))), overdispersed(a)]),
        y,
      )
    })
    const compiled = toFactorGraph(model.bind({ y: [0, 3, 1], x: [0.1, -0.4, 0.8] }))
    expect(compiled.issues).toEqual([])
    const gates = compiled.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(3)
    expect(branchesOf(gates[0]).length).toBe(3)
  })

  it('turns a branch with no random parameter into the weight of its state, a table over the index beside the gate', () => {
    const model = defineModel('known-branch', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(1, 0.5))
      const z = m.latent('z', d.bernoulli({ probs: 0.3 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.poisson(2), d.poisson(ops.exp(a))]), y)
    })
    const compiled = toFactorGraph(model.bind({ y: [3, 0] }))
    expect(compiled.issues).toEqual([])
    const tables = compiled.graph?.factors.filter((f) => f.operator === 'table') ?? []
    expect(tables.length).toBe(2)
    // Poisson(2) at 3 and at 0, and one for the random branch
    const weights = valuesOf(tables[0])
    expect(weights[0]).toBeCloseTo(Math.exp(-2) * (2 ** 3 / 6), 12)
    expect(weights[1]).toBe(1)
    expect(valuesOf(tables[1])[0]).toBeCloseTo(Math.exp(-2), 12)
    const gates = compiled.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(branchesOf(gates[0])[0]).toEqual({})
  })

  it('is only a table over the index when every branch is known', () => {
    const model = defineModel('all-known', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const z = m.latent('z', d.bernoulli({ probs: 0.3 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.poisson(2), d.poisson(9)]), y)
    })
    const compiled = toFactorGraph(model.bind({ y: [3] }))
    expect(compiled.issues).toEqual([])
    expect(compiled.graph?.factors.map((f) => f.operator).sort()).toEqual(['prior', 'table'])
  })

  it('is the branch itself when the index is known: no gate, an ordinary Poisson site', () => {
    const model = defineModel('known-index', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const which = m.data('which', { dims: [row], support: 'index' })
      const a = m.latent('a', d.normal(1, 0.5))
      const b = m.latent('b', d.normal(2, 0.5))
      m.observe('y', d.gate(which, [d.poisson(ops.exp(a)), overdispersed(b)]), y)
    })
    const compiled = toFactorGraph(model.bind({ y: [3, 8], which: [0, 1] }))
    expect(compiled.issues).toEqual([])
    const operators = compiled.graph?.factors.map((f) => f.operator) ?? []
    expect(operators).not.toContain('gate')
    expect(compiled.graph?.factors.filter((f) => f.operator === 'likelihood-site').map((f) => f.args)).toEqual([
      ['a'],
      ['b'],
    ])
  })

  it('maps a Gaussian regression branch against a constant one: a row factor on the weights', () => {
    const model = defineModel('regression-or-constant', (m) => {
      const row = m.range('row')
      const feature = m.range('feature', { size: 2 })
      const x = m.data('X', { dims: [row, feature] })
      const y = m.data('y', { dims: [row] })
      const w = m.latent('w', d.normal(0, 1), { dims: [feature] })
      const z = m.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.normal(x.matmul(w), 0.5), d.normal(0, 2)]), y)
    })
    const compiled = toFactorGraph(
      model.bind({
        X: [
          [1, 0.2],
          [1, -0.7],
        ],
        y: [0.4, 1.1],
      }),
    )
    expect(compiled.issues).toEqual([])
    const gates = compiled.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(2)
    const branches = branchesOf(gates[0])
    expect(branches.map((b) => b.operator)).toEqual(['linear-regression-row', undefined])
  })
})

describe('what a gate over families still refuses, with the node, the branch and the reason', () => {
  const issueOf = (model: ReturnType<typeof defineModel>, data: Record<string, unknown>) =>
    toFactorGraph(model.bind(data as never)).issues[0]

  it('an index that is a computed random value', () => {
    const model = defineModel('index-random', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(0, 1))
      m.observe('y', d.gate(ops.exp(a), [d.poisson(ops.exp(a)), overdispersed(a)]), y)
    })
    expect(issueOf(model, { y: [1] })).toMatchObject({ node: 'y', code: 'wrong-form' })
    expect(issueOf(model, { y: [1] })?.reason).toMatch(
      /index of the gate 'y' must be known or a discrete or Bernoulli latent/,
    )
  })

  it('an index with more states than the gate has branches', () => {
    const model = defineModel('states', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(0, 1))
      const z = m.latent('z', d.categorical({ probs: [0.2, 0.3, 0.5] }), { dims: [row] })
      m.observe('y', d.gate(z, [d.poisson(ops.exp(a)), overdispersed(a)]), y)
    })
    expect(issueOf(model, { y: [1] })).toMatchObject({ node: 'y', code: 'support' })
    expect(issueOf(model, { y: [1] })?.reason).toMatch(/3 states and the gate 2 branches/)
  })

  it('a known index outside the branches', () => {
    const model = defineModel('outside', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const which = m.data('which', { dims: [row], support: 'index' })
      const a = m.latent('a', d.normal(0, 1))
      m.observe('y', d.gate(which, [d.poisson(ops.exp(a)), d.poisson(ops.exp(a))]), y)
    })
    expect(issueOf(model, { y: [1], which: [2] })).toMatchObject({ node: 'y', code: 'support' })
  })

  it('a branch the mapping of its own family refuses, with that reason', () => {
    const model = defineModel('branch-refused', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(0, 1))
      const z = m.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [row] })
      // a Poisson whose rate is not exposure times exp of an affine value
      m.observe('y', d.gate(z, [d.poisson(ops.square(a)), overdispersed(a)]), y)
    })
    const issue = issueOf(model, { y: [1] })
    expect(issue).toMatchObject({ node: 'y' })
    expect(issue?.reason).toMatch(/branch 0 \(poisson\) of the gate 'y' is refused/)
  })

  it('a branch that is not one likelihood factor: a hurdle', () => {
    const model = defineModel('hurdle-branch', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(0, 1))
      const z = m.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.hurdlePoisson(ops.exp(a), 0.3), d.poisson(ops.exp(a))]), y)
    })
    const issue = issueOf(model, { y: [2] })
    expect(issue).toMatchObject({ node: 'y', code: 'wrong-form' })
    expect(issue?.reason).toMatch(/branch 0 \(hurdlePoisson\) of the gate 'y' does not map to one likelihood factor/)
  })
})

describe('expectation propagation on gates over families, against the grid posterior with the index summed out', () => {
  /** EP through `runEp` against the grid of the same model written with the library's mixture: the means and spreads of the sites named. */
  function agree(
    bound: ReturnType<ReturnType<typeof defineModel>['bind']>,
    reference: () => void,
    names: readonly string[],
    tol: { readonly mean: number; readonly sd: readonly [number, number]; readonly evidence: number },
  ): void {
    expect(epRoute(bound).via).toBe('ep')
    const run = runEp(bound, { light: true })
    expect(run.status).toBe('ran')
    if (run.status !== 'ran') return
    expect(run.converged).toBe(true)
    const grid = gridPosterior(reference, undefined, { points: names.length === 2 ? 41 : 33, maxRefinements: 0 })
    for (const name of names) {
      const ref = grid.marginal(name)
      const belief = run.beliefs[name]
      expect(Math.abs((belief?.mean ?? 0) - ref.mean) / ref.sd, `mean of ${name}`).toBeLessThan(tol.mean)
      expect((belief?.sd ?? 0) / ref.sd, `sd of ${name}`).toBeGreaterThan(tol.sd[0])
      expect((belief?.sd ?? 0) / ref.sd, `sd of ${name}`).toBeLessThan(tol.sd[1])
    }
    expect(Math.abs((run.evidence ?? Number.NaN) - grid.logEvidence), 'log evidence').toBeLessThan(tol.evidence)
  }

  it('a Poisson against a negative binomial of another mean, the regime of each row not told', () => {
    agree(
      regimes.bind({ y: Y }),
      () => {
        const a = sample('a', normal(0.3, 0.5))
        const b = sample('b', normal(2, 0.5))
        const second = negativeBinomial(R, div(R, add(texp(b), R)))
        for (const [i, y] of Y.entries())
          observe(`y${i}`, mixture({ probs: [0.5, 0.5] }, [poisson(texp(a)), second]), y)
      },
      ['a', 'b'],
      { mean: 0.1, sd: [0.9, 1.1], evidence: 0.05 },
    )
  })

  it('a known Poisson against an unknown one: the known branch is the weight of its state', () => {
    const counts = [3, 0, 2, 5, 1, 4, 9, 11]
    const model = defineModel('known-or-learned', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row], support: 'count' })
      const a = m.latent('a', d.normal(1, 0.7))
      const z = m.latent('z', d.bernoulli({ probs: 0.3 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.poisson(2), d.poisson(ops.exp(a))]), y)
    })
    agree(
      model.bind({ y: counts }),
      () => {
        const a = sample('a', normal(1, 0.7))
        for (const [i, y] of counts.entries())
          observe(`y${i}`, mixture({ probs: [0.7, 0.3] }, [poisson(2), poisson(texp(a))]), y)
      },
      ['a'],
      { mean: 0.1, sd: [0.9, 1.1], evidence: 0.05 },
    )
  })

  it('a regression against a constant (a Gaussian row factor on two weights against a known density)', () => {
    const X = [
      [1, -1],
      [1, -0.5],
      [1, 0],
      [1, 0.4],
      [1, 0.9],
      [1, 1.4],
    ]
    const y = [-0.6, -0.1, 0.1, 0.5, 0.9, 1.5]
    const model = defineModel('regression-or-constant', (m) => {
      const row = m.range('row')
      const feature = m.range('feature', { size: 2 })
      const x = m.data('X', { dims: [row, feature] })
      const response = m.data('y', { dims: [row] })
      const w = m.latent('w', d.normal(0, 1), { dims: [feature] })
      const z = m.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [row] })
      m.observe('y', d.gate(z, [d.normal(x.matmul(w), 0.5), d.normal(0, 2)]), response)
    })
    agree(
      model.bind({ X, y }),
      () => {
        const w0 = sample('w[0]', normal(0, 1))
        const w1 = sample('w[1]', normal(0, 1))
        X.forEach((row, i) => {
          const mean = add(mul(w0, row[0] as number), mul(w1, row[1] as number))
          observe(`y${i}`, mixture({ probs: [0.5, 0.5] }, [normal(mean, 0.5), normal(0, 2)]), y[i] as number)
        })
      },
      ['w[0]', 'w[1]'],
      { mean: 0.1, sd: [0.8, 1.1], evidence: 0.1 },
    )
  })
})
