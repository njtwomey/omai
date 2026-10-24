import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'
import * as C from './index.ts'

// Pearl's firing squad: court order U -> captain C -> riflemen A, B -> death D; A also fires on a nervous whim W.
const squad = C.scm([
  { kind: 'discrete', name: 'u', states: ['no', 'yes'], parents: [], noise: [0.4, 0.6], assign: [0, 1] },
  { kind: 'discrete', name: 'w', states: ['no', 'yes'], parents: [], noise: [0.9, 0.1], assign: [0, 1] },
  { kind: 'discrete', name: 'c', states: ['no', 'yes'], parents: ['u'], noise: [1], assign: [0, 1] },
  { kind: 'discrete', name: 'a', states: ['no', 'yes'], parents: ['c', 'w'], noise: [1], assign: [0, 1, 1, 1] },
  { kind: 'discrete', name: 'b', states: ['no', 'yes'], parents: ['c'], noise: [1], assign: [0, 1] },
  { kind: 'discrete', name: 'd', states: ['alive', 'dead'], parents: ['a', 'b'], noise: [1], assign: [0, 1, 1, 1] },
])

describe('structural causal models', () => {
  it('induce Bayes networks, sample, and describe themselves', () => {
    const bn = C.scmToBayesNet(squad)
    expect(P.factorValues(P.variableElimination(bn, ['d']).posterior)[1]).toBeCloseTo(0.6 + 0.4 * 0.1, 12)
    const s = C.scmSample(squad, key(1), 3000)
    const dead = s.filter((r) => r['d'] === 'dead').length / 3000
    expect(Math.abs(dead - 0.64)).toBeLessThan(0.04)
    expect(C.scmStructure(squad).nodes.filter((n) => n.role === 'latent')).toHaveLength(6)
    expect(() =>
      C.scm([{ kind: 'discrete', name: 'x', states: ['0'], parents: ['y'], noise: [1], assign: [0] }]),
    ).toThrow(/unknown parent/)
  })

  it('answer counterfactuals by twin networks and by abduction, action and prediction', () => {
    // The prisoner is dead. Had rifleman A not fired, would he be dead? Yes: B fired too (the court ordered it).
    const twin = C.counterfactual(squad, { d: 'dead' }, { a: 'no' }, ['d'])
    const steps = C.abductActPredict(squad, { d: 'dead' }, { a: 'no' }, ['d'])
    const pDeadGivenDead = 0.6 / 0.64
    expect(P.factorValues(twin)[1]).toBeCloseTo(pDeadGivenDead, 12)
    expect(P.factorValues(steps)[1]).toBeCloseTo(pDeadGivenDead, 12)
    expect(P.factorValues(C.abduction(squad, { d: 'alive' }))[0]).toBeCloseTo(1, 12)
    const pr = C.causationProbabilities(
      squad,
      { name: 'a', state: 'yes', alternative: 'no' },
      { name: 'd', state: 'dead', alternative: 'alive' },
    )
    // Had A not fired, the prisoner would have lived only without a court order: 0.04 / 0.64.
    expect(pr.necessity).toBeCloseTo(0.04 / 0.64, 12)
    expect(pr.sufficiency).toBeCloseTo(1, 12)
  })

  it('linear counterfactuals propagate the abducted noise', () => {
    const m = C.scm([
      {
        kind: 'linear',
        name: 'x',
        parents: [],
        weights: [],
        intercept: 0,
        noise: { distribution: 'normal', scale: 1 },
      },
      {
        kind: 'linear',
        name: 'z',
        parents: ['x'],
        weights: [1],
        intercept: 0,
        noise: { distribution: 'normal', scale: 1 },
      },
      {
        kind: 'linear',
        name: 'y',
        parents: ['x', 'z'],
        weights: [0.5, 2],
        intercept: 1,
        noise: { distribution: 'normal', scale: 1 },
      },
    ])
    // Fully observed: the noises are determined, so the counterfactual is a point.
    const r = C.linearCounterfactual(m, { x: 1, z: 2.5, y: 7 }, { x: 0 }, ['z', 'y'])
    const uz = 1.5
    const uy = 7 - 1 - 0.5 - 2 * 2.5
    expect(r.mean[0]).toBeCloseTo(uz, 8)
    expect(r.mean[1]).toBeCloseTo(1 + 2 * uz + uy, 8)
    expect(Math.abs(r.covariance[1]?.[1] ?? 1)).toBeLessThan(1e-8)
    // Interventional means agree with the linear-Gaussian network.
    const g = C.scmToLinearGaussian(m)
    expect(C.linearInterventionalMoments(g, ['y'], { z: 1 }).mean[0]).toBeCloseTo(1 + 2, 9)
  })
})

describe('interventions and identification', () => {
  const bn = P.bayesNet([
    { name: 'z', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
    { name: 'x', states: ['0', '1'], parents: ['z'], cpt: [0.8, 0.2, 0.2, 0.8] },
    { name: 'y', states: ['0', '1'], parents: ['x', 'z'], cpt: [0.9, 0.1, 0.5, 0.5, 0.6, 0.4, 0.2, 0.8] },
  ])

  it('the back-door formula equals the mutilated network, unlike conditioning', () => {
    const doX = P.factorValues(C.interventionalDistribution(bn, ['y'], { x: '1' }))
    const adjusted = P.factorValues(C.backdoorAdjustment(bn, 'x', 'y', ['z'], '1'))
    const seen = P.factorValues(P.variableElimination(bn, ['y'], { evidence: { x: '1' } }).posterior)
    expect(adjusted[1]).toBeCloseTo(doX[1] ?? 0, 12)
    expect(Math.abs((seen[1] ?? 0) - (doX[1] ?? 0))).toBeGreaterThan(0.05)
    expect(C.averageCausalEffect(bn, 'x', 'y', { treated: '1', control: '0', outcome: '1' })).toBeCloseTo(0.6 - 0.3, 12)
    expect(C.identifyEffect(bn, 'x', 'y')).toMatchObject({ identifiable: true, method: 'backdoor', set: ['z'] })
    const cut = C.doIntervention(bn, { x: '0' }) as P.BayesNet
    expect(P.bayesNode(cut, 'x').parents).toEqual([])
  })

  it('front-door adjustment, unidentifiable effects and instruments', () => {
    const g = P.dag([
      ['u', 'x'],
      ['u', 'y'],
    ])
    expect(C.identifyEffect(g, 'x', 'y', { latent: ['u'] }).identifiable).toBe(false)
    const iv = P.dag([
      ['z', 'x'],
      ['u', 'x'],
      ['u', 'y'],
      ['x', 'y'],
      ['w', 'z'],
      ['w', 'y'],
    ])
    expect(C.instrumentalVariables(iv, 'x', 'y', { latent: ['u'] })).toEqual([])
    expect(C.instrumentalVariables(iv, 'x', 'y', { latent: ['u'], given: ['w'] })).toEqual(['z'])
  })
})

describe('NOTEARS', () => {
  it('recovers a linear-Gaussian DAG with decreasing acyclicity violation', () => {
    const n = 300
    const z = normal(key(4), [n, 4]).data
    const rows = Array.from({ length: n }, (_, i) => {
      const a = z[4 * i] ?? 0
      // Equal noise variances, the setting in which the least-squares score identifies the DAG.
      const b = 1.2 * a + (z[4 * i + 1] ?? 0)
      const c = -1.0 * b + (z[4 * i + 2] ?? 0)
      const d = 0.9 * a + (z[4 * i + 3] ?? 0)
      return { a, b, c, d }
    })
    const rec = recorder()
    const r = C.notears(rows, { tracer: rec })
    expect(r.h).toBeLessThan(1e-6)
    const edges = P.dagEdges(r.dag)
      .map((e) => e.join('->'))
      .sort()
    expect(edges).toEqual(['a->b', 'a->d', 'b->c'])
    const h = metricSeries(rec.run(), 'h').values.filter((v) => v >= 0)
    expect(h[h.length - 1] ?? 1).toBeLessThan(h[1] ?? 0)
  })
})
