import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { key } from '../../random/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'
import * as L from './index.ts'

type Rows = L.DataRow[]
const expectCpts = (got: P.BayesNet, want: P.BayesNode[], tol: number): void => {
  for (const w of want) {
    const n = P.bayesNode(got, w.name)
    expect(n.parents).toEqual(w.parents)
    for (const [i, v] of w.cpt.entries()) expect(Math.abs((n.cpt[i] ?? 0) - v)).toBeLessThan(tol)
  }
}

describe('parameter learning against pgmpy', () => {
  for (const c of loadFixture('pgm/learning/parameters').cases) {
    it(c.name, () => {
      const data = c.params['data'] as Rows
      const states = c.params['states'] as Record<string, string[]>
      const want = (c.params['network'] ?? []) as P.BayesNode[]
      const structure = P.dag(
        want.flatMap((n) => n.parents.map((p): [string, string] => [p, n.name])),
        want.map((n) => n.name),
      )
      // Use the fixture's (sorted) parent order for the CPT layout.
      const ordered: P.DirectedModel = { nodes: want.map((n) => ({ name: n.name, parents: n.parents })) }
      switch (c.fn) {
        case 'mle':
          expectCpts(L.maximumLikelihood(ordered, data, { states }), want, 1e-12)
          expect(structure.nodes).toHaveLength(want.length)
          return
        case 'bayesian': {
          const prior = c.params['prior'] as string
          const options: L.BayesianEstimateOptions =
            prior === 'BDeu'
              ? { prior: 'BDeu', equivalentSampleSize: 5, states }
              : prior === 'K2'
                ? { prior: 'K2', states }
                : { prior: 'dirichlet', pseudoCounts: 2, states }
          expectCpts(L.bayesianEstimate(ordered, data, options), want, 1e-12)
          return
        }
        case 'em': {
          const init = P.bayesNet(c.params['init'] as P.BayesNode[])
          const r = L.expectationMaximisation(ordered, data, {
            init,
            latent: { h: ['0', '1'] },
            states: Object.fromEntries(init.nodes.filter((n) => n.name !== 'h').map((n) => [n.name, n.states])),
            tol: 0,
            maxIter: c.params['iterations'] as number,
          })
          expect(r.iterations).toBe(8)
          expectCpts(r.network, want, 1e-10)
          return
        }
        case 'linearGaussian': {
          const nodes = c.params['nodes'] as { name: string; parents: string[]; beta: number[]; std: number }[]
          const g = L.fitLinearGaussian({ nodes: nodes.map((n) => ({ name: n.name, parents: n.parents })) }, data)
          for (const n of nodes) {
            const fitted = g.nodes.find((x) => x.name === n.name)
            expect(fitted?.intercept).toBeCloseTo(n.beta[0] ?? 0, 10)
            for (const [k, w] of (fitted?.weights ?? []).entries()) expect(w).toBeCloseTo(n.beta[k + 1] ?? 0, 10)
            expect(Math.sqrt(fitted?.variance ?? 0)).toBeCloseTo(n.std, 10)
          }
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('parameter learning', () => {
  const truth = P.bayesNet([
    { name: 'a', states: ['0', '1'], parents: [], cpt: [0.3, 0.7] },
    { name: 'b', states: ['0', '1', '2'], parents: ['a'], cpt: [0.6, 0.3, 0.1, 0.1, 0.2, 0.7] },
    { name: 'c', states: ['0', '1'], parents: ['b'], cpt: [0.9, 0.1, 0.5, 0.5, 0.2, 0.8] },
  ])
  const rows = P.ancestralSample(truth, key(4), 600).map((s) =>
    Object.fromEntries(Object.entries(s).map(([k, v]) => [k, String(v)])),
  )

  it('EM with missing values never decreases the likelihood and handles weights', () => {
    const holes = rows.map((r, i) => (i % 3 === 0 ? { ...r, b: null } : i % 5 === 0 ? { ...r, c: null } : r))
    const rec = recorder()
    const r = L.expectationMaximisation(truth, holes, { key: key(9), tracer: rec, maxIter: 200 })
    expect(r.converged).toBe(true)
    const ll = metricSeries(rec.run(), 'logLikelihood').values.filter((v) => v !== -1)
    for (let i = 1; i < ll.length; i++) expect(ll[i] ?? 0).toBeGreaterThanOrEqual((ll[i - 1] ?? 0) - 1e-9)
    // The fitted parameters are close to the truth.
    expect(Math.abs((P.bayesNode(r.network, 'a').cpt[0] ?? 0) - 0.3)).toBeLessThan(0.06)
    expect(r.logLikelihood).toBeCloseTo(L.dataLogLikelihood(r.network, holes), 9)
    // Duplicated rows are the same as weights.
    const twice = L.maximumLikelihood(truth, [...rows.slice(0, 50), ...rows.slice(0, 50)])
    const weighted = L.maximumLikelihood(truth, rows.slice(0, 50), { weights: new Array(50).fill(2) })
    expect(weighted.nodes.map((n) => n.cpt)).toEqual(twice.nodes.map((n) => n.cpt))
  })

  it('counts families and validates inputs', () => {
    const counts = L.stateCounts(rows.slice(0, 10), 'b', ['a'], { states: { a: ['0', '1'], b: ['0', '1', '2'] } })
    expect(counts.counts.flat().reduce((x, y) => x + y, 0)).toBe(10)
    expect(() => L.maximumLikelihood(truth, [{ a: '3', b: '0', c: '0' }], { states: { a: ['0', '1'] } })).toThrow(
      /outside/,
    )
    expect(() => L.bayesianEstimate(truth, rows, { prior: 'dirichlet', pseudoCounts: { a: [[1, 1, 1]] } })).toThrow(
      /pseudo-counts/,
    )
  })
})
