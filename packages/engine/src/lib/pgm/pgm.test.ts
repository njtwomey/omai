import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import { messages, recorder, tracePolicy } from '../trace/index.ts'
import * as P from './index.ts'

type Params = Readonly<Record<string, unknown>>
const networkOf = (p: Params): P.BayesNet => P.bayesNet(p['network'] as P.BayesNode[])

describe('Bayes networks against pgmpy', () => {
  for (const c of loadFixture('pgm/networks').cases) {
    it(c.name, () => {
      const bn = networkOf(c.params)
      const evidence = (c.params['evidence'] ?? {}) as Record<string, string>
      const query = (c.params['query'] ?? []) as string[]
      switch (c.fn) {
        case 'posterior': {
          const { posterior } = P.variableElimination(bn, query, { evidence })
          expectMatches(tensor(P.factorValues(posterior)), c.output, 1e-12, 1e-15, c.name)
          // Enumeration agrees with elimination.
          const indices = Object.fromEntries(
            Object.entries(evidence).map(([k, v]) => [k, P.bayesNode(bn, k).states.indexOf(v)]),
          )
          expectMatches(
            tensor(P.factorValues(P.enumerate(bn, query, indices))),
            c.output,
            1e-12,
            1e-15,
            `${c.name} (enumeration)`,
          )
          return
        }
        case 'map':
          expect(P.mapQuery(bn, query, { evidence }).assignment).toEqual(c.params['assignment'])
          return
        case 'joint': {
          const a = c.params['assignment'] as Record<string, string>
          const idx = Object.fromEntries(Object.entries(a).map(([k, v]) => [k, P.bayesNode(bn, k).states.indexOf(v)]))
          expectMatches(P.jointProbability(bn, idx), c.output, 1e-12, 0, c.name)
          return
        }
        case 'blanket':
          expect(P.markovBlanket(bn, c.params['node'] as string)).toEqual(c.params['blanket'])
          return
        case 'dsep': {
          const separated = P.dSeparated(
            bn,
            [c.params['x'] as string],
            [c.params['y'] as string],
            c.params['z'] as string[],
          )
          expectMatches(separated ? 1 : 0, c.output, 0, 0, c.name)
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('factors and elimination', () => {
  const rainy = P.bayesNet([
    { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
    { name: 'sprinkler', states: ['off', 'on'], parents: ['rain'], cpt: [0.6, 0.4, 0.99, 0.01] },
    { name: 'wet', states: ['no', 'yes'], parents: ['sprinkler', 'rain'], cpt: [1, 0, 0.2, 0.8, 0.1, 0.9, 0.01, 0.99] },
  ])

  it('report the evidence probability and emit one message per eliminated variable', () => {
    const rec = recorder()
    const r = P.variableElimination(rainy, ['rain'], {
      evidence: { wet: 'yes' },
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    const pWet = [0, 1]
      .flatMap((ra) => [0, 1].map((s) => P.jointProbability(rainy, { rain: ra, sprinkler: s, wet: 1 })))
      .reduce((a, b) => a + b)
    expect(Math.exp(r.logEvidence)).toBeCloseTo(pWet, 14)
    const events = messages(rec.run(), 'eliminate')
    expect(events.map((e) => (e.payload as { variable: string }).variable)).toEqual(['sprinkler'])
  })

  it('ancestral samples follow the joint distribution', () => {
    const samples = P.ancestralSample(rainy, key(0), 20000)
    const freq = samples.filter((s) => s['wet'] === 1).length / samples.length
    const exact = P.factorValues(P.enumerate(rainy, ['wet']))[1] ?? 0
    expect(Math.abs(freq - exact)).toBeLessThan(0.015)
  })

  it('describe themselves as structures', () => {
    const s = P.toStructure(rainy, ['wet'])
    expect(s.nodes.find((n) => n.id === 'wet')?.role).toBe('observed')
    expect(s.edges).toHaveLength(3)
  })

  it('validate their tables', () => {
    expect(() => P.bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.6] }])).toThrow(/distribution/)
    expect(() =>
      P.bayesNet([
        { name: 'a', states: ['0'], parents: ['b'], cpt: [1] },
        { name: 'b', states: ['0'], parents: ['a'], cpt: [1] },
      ]),
    ).toThrow(/cycle/)
  })
})
