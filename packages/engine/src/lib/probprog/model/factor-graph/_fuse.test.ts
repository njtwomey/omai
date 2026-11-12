// The conjugate pairs inside gates: a mixture whose components have a mean and a precision of their own, with the mean's variance a
// multiple of the precision's reciprocal, compiles (with `conjugatePairs`) to one normal-gamma variable per component and a
// `gaussian-normal-gamma` branch per observation, and EP on it agrees with the exact posterior (every assignment of the points to the
// components enumerated, each a conjugate update). What is not a conjugate pair is left as it was.
import { describe, expect, it } from 'vitest'
import { marginalsOfRun, runEp } from '../../../../testing/factor-graph.ts'
import type { MpState } from '../../../message-passing/index.ts'
import { lgammaOf } from '../../../message-passing/numerics/_scalar.ts'
import { linspace, toFlat } from '../../../tensor/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from '../index.ts'

/** Two clusters of three points each, in two coordinates. */
const LENGTH = [-1.3, -1.1, -1.4, 0.8, 0.6, 0.7]
const WAIT = [-1.2, -1.4, -1.0, 0.5, 0.9, 0.6]
const PIECES = 2
const KAPPA = 0.2
const [SHAPE, RATE] = [2, 0.5]

/** The mixture of the story: a label per row, a gamma precision and a centre per group and coordinate, the centre as wide as its group. */
function mixture(
  options: { readonly conjugate?: boolean; readonly shared?: boolean; readonly ordered?: boolean } = {},
) {
  const conjugate = options.conjugate ?? true
  return defineModel('diagonal-mixture', (m) => {
    const row = m.range('row')
    const group = m.range('group', { size: PIECES })
    const length = m.data('length', { dims: [row] })
    const wait = m.data('wait', { dims: [row] })
    const start = m.data('start', { dims: [group] })
    const which = m.latent('which', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    const make = (name: string, observed: typeof length): void => {
      const precision = m.latent(`precision${name}`, d.gamma(SHAPE, RATE), options.shared ? {} : { dims: [group] })
      const sd = options.shared
        ? ops.div(1, ops.sqrt(precision))
        : ops.div(1, ops.sqrt((precision as typeof length).at(which)))
      const spread = options.shared ? ops.div(1, ops.sqrt(ops.mul(precision, KAPPA))) : 0
      const prior = conjugate
        ? d.normal(start, options.shared ? spread : ops.div(1, ops.sqrt(ops.mul(precision as typeof length, KAPPA))))
        : d.normal(start, 1.5)
      const centre = m.latent(`centre${name}`, prior, {
        dims: [group],
        ...(options.ordered && name === 'Length' ? { constraint: { kind: 'ordered' as const } } : {}),
      })
      m.observe(name.toLowerCase(), d.normal(centre.at(which), sd), observed)
    }
    make('Length', length)
    make('Wait', wait)
  })
}

const data = { length: LENGTH, wait: WAIT, start: toFlat(linspace(-1, 1, PIECES)) }

describe('a mixture of conjugate pairs compiles to normal-gamma variables and gaussian-normal-gamma branches', () => {
  const plain = toFactorGraph(mixture().bind(data))
  const fused = toFactorGraph(mixture().bind(data), { conjugatePairs: true })

  it('is left as it is unless asked: a gaussian-gamma-precision prior per centre and branch per observation', () => {
    expect(plain.issues).toEqual([])
    const operators = new Set(plain.graph?.factors.map((f) => f.operator))
    expect(operators.has('gaussian-gamma-precision')).toBe(true)
    expect(plain.graph?.variables.some((v) => v.family.kind === 'normal-gamma')).toBe(false)
  })

  it('makes one normal-gamma variable per component and coordinate, with the prior of the pair', () => {
    expect(fused.issues).toEqual([])
    const pairs = fused.graph?.variables.filter((v) => v.family.kind === 'normal-gamma') ?? []
    expect(pairs.map((v) => v.name)).toEqual(['centreLength[0]', 'centreLength[1]', 'centreWait[0]', 'centreWait[1]'])
    // the precisions have no variable of their own now
    expect(fused.graph?.variables.some((v) => v.name.startsWith('precision'))).toBe(false)
    const priors = fused.graph?.factors.filter((f) => f.operator === 'prior' && f.args[0] === 'centreLength[0]') ?? []
    // normalGammaEta(-1, 0.2, 2, 0.5): the centre starts at -1 with the precision a fifth of the group's, the precision is Gamma(2, 0.5)
    expect(priors[0]?.params?.['eta']).toEqual([-0.2, -0.1, -0.1 - 0.5, 1.5])
  })

  it('reads every observation as a gate over the pair variables of the components', () => {
    const gates = fused.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates).toHaveLength(2 * LENGTH.length)
    const first = gates[0] as NonNullable<(typeof gates)[number]>
    expect(first.args).toEqual(['which[0]', 'centreLength[0]', 'centreLength[1]'])
    const branches = (first.params as { branches: { operator: string; args: unknown[]; params: object }[] }).branches
    expect(branches.map((b) => b.operator)).toEqual(['gaussian-normal-gamma', 'gaussian-normal-gamma'])
    expect(branches[1]?.args).toEqual([{ const: LENGTH[0] }, 1])
    expect(Object.keys(branches[1]?.params ?? {})).toEqual([])
  })

  it('reads the centre as a Student t and the precision as a gamma of the same variable', () => {
    const run = runEp(mixture().bind(data), { conjugatePairs: true, tol: 1e-9, converged: true })
    const sites = Object.fromEntries(run.c.sites.map((s) => [s.site, s]))
    expect(sites['centreLength']).toMatchObject({ family: 'normal-gamma', part: 'mean' })
    expect(sites['precisionLength']).toMatchObject({
      family: 'normal-gamma',
      part: 'precision',
      variables: ['centreLength[0]', 'centreLength[1]'],
    })
  })
})

describe('what is not a conjugate pair is left alone', () => {
  const kinds = (bound: Parameters<typeof toFactorGraph>[0]): string[] =>
    toFactorGraph(bound, { conjugatePairs: true }).graph?.variables.map((v) => v.family.kind) ?? []

  it('a centre whose prior does not depend on the precision', () => {
    expect(kinds(mixture({ conjugate: false }).bind(data))).not.toContain('normal-gamma')
  })

  it('one precision shared by the components', () => {
    expect(kinds(mixture({ shared: true }).bind(data))).not.toContain('normal-gamma')
  })

  it('a centre with another factor on it (the order of the centres)', () => {
    const bound = mixture({ ordered: true }).bind(data)
    const c = toFactorGraph(bound, { conjugatePairs: true })
    expect(c.issues).toEqual([])
    // the length pairs have the order constraint on the centre; the wait pairs are still fused
    const pairs = c.graph?.variables.filter((v) => v.family.kind === 'normal-gamma').map((v) => v.name)
    expect(pairs).toEqual(['centreWait[0]', 'centreWait[1]'])
  })
})

/** log of the marginal likelihood of the points `x` under the normal-gamma prior `(m, kappa, a, b)` with unit precision multiple. */
function logMarginal(x: readonly number[], m: number, kappa: number, a: number, b: number): number {
  const n = x.length
  if (n === 0) return 0
  const bar = x.reduce((s, v) => s + v, 0) / n
  const scatter = x.reduce((s, v) => s + (v - bar) ** 2, 0)
  const kn = kappa + n
  const an = a + n / 2
  const bn = b + scatter / 2 + (kappa * n * (bar - m) ** 2) / (2 * kn)
  return (
    lgammaOf(an) -
    lgammaOf(a) +
    a * Math.log(b) -
    an * Math.log(bn) +
    0.5 * (Math.log(kappa) - Math.log(kn)) -
    (n / 2) * Math.log(2 * Math.PI)
  )
}

/** The posterior `(m, kappa, a, b)` of the same pair after the points `x`. */
function updated(x: readonly number[], m: number, kappa: number, a: number, b: number) {
  const n = x.length
  const bar = n === 0 ? 0 : x.reduce((s, v) => s + v, 0) / n
  const scatter = x.reduce((s, v) => s + (v - bar) ** 2, 0)
  const kn = kappa + n
  return {
    m: (kappa * m + n * bar) / kn,
    a: a + n / 2,
    b: n === 0 ? b : b + scatter / 2 + (kappa * n * (bar - m) ** 2) / (2 * kn),
  }
}

describe('expectation propagation on the fused graph against the exact posterior (every assignment enumerated)', () => {
  const starts = toFlat(linspace(-1, 1, PIECES))
  // The exact posterior: a weight 1/2 per point and component, each assignment a conjugate update in each coordinate.
  const n = LENGTH.length
  let evidence = 0
  const expected = {
    centreLength: [0, 0],
    centreWait: [0, 0],
    precisionLength: [0, 0],
    precisionWait: [0, 0],
  }
  const weights: number[] = []
  const posteriors: { z: number; length: ReturnType<typeof updated>[]; wait: ReturnType<typeof updated>[] }[] = []
  for (let z = 0; z < 2 ** n; z++) {
    const inside = (k: number): number[] => Array.from({ length: n }, (_, i) => i).filter((i) => ((z >> i) & 1) === k)
    const parts = (xs: readonly number[]): ReturnType<typeof updated>[] =>
      [0, 1].map((k) =>
        updated(
          inside(k).map((i) => xs[i] as number),
          starts[k] as number,
          KAPPA,
          SHAPE,
          RATE,
        ),
      )
    const logp =
      n * Math.log(0.5) +
      [0, 1].reduce(
        (s, k) =>
          s +
          logMarginal(
            inside(k).map((i) => LENGTH[i] as number),
            starts[k] as number,
            KAPPA,
            SHAPE,
            RATE,
          ) +
          logMarginal(
            inside(k).map((i) => WAIT[i] as number),
            starts[k] as number,
            KAPPA,
            SHAPE,
            RATE,
          ),
        0,
      )
    weights.push(logp)
    posteriors.push({ z, length: parts(LENGTH), wait: parts(WAIT) })
  }
  const top = Math.max(...weights)
  const total = weights.reduce((s, w) => s + Math.exp(w - top), 0)
  evidence = top + Math.log(total)
  posteriors.forEach((p, i) => {
    const w = Math.exp((weights[i] as number) - top) / total
    for (const k of [0, 1]) {
      expected.centreLength[k] = (expected.centreLength[k] as number) + w * (p.length[k] as { m: number }).m
      expected.centreWait[k] = (expected.centreWait[k] as number) + w * (p.wait[k] as { m: number }).m
      expected.precisionLength[k] =
        (expected.precisionLength[k] as number) +
        (w * (p.length[k] as { a: number }).a) / (p.length[k] as { b: number }).b
      expected.precisionWait[k] =
        (expected.precisionWait[k] as number) + (w * (p.wait[k] as { a: number }).a) / (p.wait[k] as { b: number }).b
    }
  })

  it('finds the centres and the precisions of the exact posterior', () => {
    const run = runEp(mixture().bind(data), { conjugatePairs: true, tol: 1e-10, converged: true })
    const marginals = marginalsOfRun(run)
    for (const k of [0, 1]) {
      for (const [site, coordinate] of [
        ['centreLength', 'Length'],
        ['centreWait', 'Wait'],
      ] as const) {
        const q = marginals[`${site}[${k}]`] as { m: number; a: number; b: number }
        // the centre within 0.02 of the exact mean (the clusters are 2 apart), the precision within 5%
        expect(Math.abs(q.m - (expected[site][k] as number))).toBeLessThan(0.02)
        const precision = q.a / q.b
        const exact = expected[`precision${coordinate}` as 'precisionLength'][k] as number
        expect(Math.abs(precision / exact - 1)).toBeLessThan(0.05)
      }
    }
  })

  it('reads the sites as the posterior of the model: a Student t for a centre, a gamma for a precision', () => {
    const run = runEp(mixture().bind(data), { conjugatePairs: true, tol: 1e-10, converged: true })
    const post = factorGraphPosterior(run.c, run.g, run.r.state as MpState)
    for (const k of [0, 1]) {
      expect(
        Math.abs((toFlat(post.mean('centreLength'))[k] as number) - (expected.centreLength[k] as number)),
      ).toBeLessThan(0.02)
      expect(
        Math.abs((toFlat(post.mean('precisionWait'))[k] as number) / (expected.precisionWait[k] as number) - 1),
      ).toBeLessThan(0.05)
    }
    expect(post.sites).toContain('centreLength')
    // the log evidence of the graph is the exact one of the model, to within a few hundredths of a nat
    expect(Number.isFinite(evidence)).toBe(true)
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - evidence)).toBeLessThan(0.05)
  })
})

describe('the conjugate route against the same model through the quadrature route', () => {
  it('agrees on the centres (the unfused graph runs its precision grids)', () => {
    const fused = marginalsOfRun(runEp(mixture().bind(data), { conjugatePairs: true, tol: 1e-9, converged: true }))
    const plain = marginalsOfRun(runEp(mixture().bind(data), { tol: 1e-9, damping: 0.3, maxSteps: 600 }))
    for (const k of [0, 1]) {
      const a = fused[`centreWait[${k}]`] as { m: number }
      const b = plain[`centreWait[${k}]`] as { mean: number }
      expect(Math.abs(a.m - b.mean)).toBeLessThan(0.05)
    }
  })
})
