import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { tensor } from '../../tensor/index.ts'
import { messages, recorder, tracePolicy } from '../../trace/index.ts'
import * as P from '../index.ts'

type Params = Readonly<Record<string, unknown>>
interface NetworkJson {
  readonly variables: P.DiscreteVariable[]
  readonly potentials: { variables: string[]; cardinality: number[]; values: number[] }[]
}
const bayesOf = (p: Params): P.BayesNet => P.bayesNet(p['network'] as P.BayesNode[])
const markovOf = (p: Params): P.MarkovNetwork => {
  const n = p['network'] as NetworkJson
  return P.markovNetwork(
    n.variables,
    n.potentials.map((f) => P.factor(f.variables, f.cardinality, f.values)),
  )
}

describe('junction trees against pgmpy', () => {
  for (const c of loadFixture('pgm/inference/junction_tree').cases) {
    it(c.name, () => {
      const query = (c.params['query'] ?? []) as string[]
      const evidence = (c.params['evidence'] ?? {}) as Record<string, string | number>
      switch (c.fn) {
        case 'posterior':
        case 'markovPosterior': {
          const model = c.fn === 'posterior' ? bayesOf(c.params) : markovOf(c.params)
          for (const method of ['shaferShenoy', 'hugin'] as const) {
            const calibrated = P.calibrate(model, { evidence, method })
            expectMatches(
              tensor(P.factorValues(P.junctionTreeQuery(calibrated, query))),
              c.output,
              1e-12,
              1e-15,
              c.name,
            )
            expect(calibrated.logPartition).toBeCloseTo(P.logPartition(model, { evidence }), 12)
          }
          return
        }
        case 'map': {
          const bn = bayesOf(c.params)
          expect(P.junctionTreeMap(bn).assignment).toEqual(c.params['assignment'])
          expect(P.junctionTreeMap(bn, { method: 'hugin' }).assignment).toEqual(c.params['assignment'])
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('junction-tree propagation', () => {
  const sprinkler = P.bayesNet([
    { name: 'cloudy', states: ['no', 'yes'], parents: [], cpt: [0.5, 0.5] },
    { name: 'sprinkler', states: ['off', 'on'], parents: ['cloudy'], cpt: [0.5, 0.5, 0.9, 0.1] },
    { name: 'rain', states: ['no', 'yes'], parents: ['cloudy'], cpt: [0.8, 0.2, 0.2, 0.8] },
    { name: 'wet', states: ['no', 'yes'], parents: ['sprinkler', 'rain'], cpt: [1, 0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99] },
  ])

  it('calibrates: neighbouring cliques agree on their separators', () => {
    for (const method of ['shaferShenoy', 'hugin'] as const) {
      const t = P.calibrate(sprinkler, { method, evidence: { wet: 'yes' } })
      for (const [k, e] of t.tree.edges.entries()) {
        const fromSide = P.factorMarginal(t.cliqueBeliefs[e.from] as P.Factor, e.separator)
        const toSide = P.factorMarginal(t.cliqueBeliefs[e.to] as P.Factor, e.separator)
        const sep = t.separatorBeliefs[k] as P.Factor
        for (let i = 0; i < sep.logValues.length; i++) {
          expect(fromSide.logValues[i]).toBeCloseTo(toSide.logValues[i] ?? 0, 12)
          expect(sep.logValues[i]).toBeCloseTo(toSide.logValues[i] ?? 0, 12)
        }
      }
    }
  })

  it('gives every marginal and agrees with elimination', () => {
    const m = P.junctionTreeMarginals(sprinkler, { evidence: { wet: 'yes' } })
    expect(Object.keys(m)).toEqual(['cloudy', 'sprinkler', 'rain'])
    for (const v of ['cloudy', 'sprinkler', 'rain']) {
      const ve = P.variableElimination(sprinkler, [v], { evidence: { wet: 'yes' } }).posterior
      expect(P.factorValues(m[v] as P.Factor)[1]).toBeCloseTo(P.factorValues(ve)[1] ?? 0, 12)
    }
  })

  it('records one message per tree edge and direction', () => {
    const rec = recorder()
    const t = P.calibrate(sprinkler, { tracer: rec, policy: tracePolicy({ messages: true }) })
    const run = rec.run()
    expect(messages(run, 'collect')).toHaveLength(t.tree.edges.length)
    expect(messages(run, 'distribute')).toHaveLength(t.tree.edges.length)
  })

  it('rejects queries on observed variables', () => {
    const t = P.calibrate(sprinkler, { evidence: { wet: 'yes' } })
    expect(() => P.junctionTreeQuery(t, ['wet'])).toThrow(/observed/)
    expect(() => P.junctionTreeQuery(P.calibrate(sprinkler, { mode: 'max' }), ['rain'])).toThrow(/sum-calibrated/)
  })
})
