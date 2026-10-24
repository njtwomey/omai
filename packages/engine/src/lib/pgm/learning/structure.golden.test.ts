import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { messages, recorder, tracePolicy } from '../../trace/index.ts'
import * as P from '../index.ts'
import * as L from './index.ts'

const sorted = (es: readonly (readonly [string, string])[]): string[][] =>
  es.map((e) => [...e]).sort((a, b) => a.join().localeCompare(b.join()))

describe('structure learning against pgmpy', () => {
  for (const c of loadFixture('pgm/learning/structure').cases) {
    it(c.name, () => {
      const data = c.params['data'] as L.DataRow[]
      switch (c.fn) {
        case 'chowLiu':
          expect(sorted(P.dagEdges(L.chowLiu(data, { root: c.params['root'] as string })))).toEqual(c.params['edges'])
          return
        case 'tan': {
          const g = L.chowLiu(data, {
            root: c.params['root'] as string,
            classVariable: c.params['classVariable'] as string,
          })
          expect(sorted(P.dagEdges(g))).toEqual(c.params['edges'])
          return
        }
        case 'pc': {
          const r = L.pc(data, { test: c.params['test'] as L.CiTest, sepsetRule: 'union' })
          expect(sorted(r.cpdag.directed)).toEqual(c.params['directed'])
          expect(sorted(r.cpdag.undirected)).toEqual(c.params['undirected'])
          return
        }
        case 'hillClimb': {
          const states = c.params['states'] as Record<string, string[]>
          const score = c.params['score'] as L.DiscreteScore
          const theirs = P.dag(c.params['edges'] as [string, string][], Object.keys(states))
          // The structure score agrees with pgmpy's on pgmpy's result.
          expect(L.structureScore(theirs, data, score, { states })).toBeCloseTo(
            (c.output as { data: number[] }).data[0] ?? 0,
            8,
          )
          // Ours is a local optimum at least as good (ties between moves are broken differently).
          const ours = L.hillClimb(data, { score, states })
          expect(ours.score).toBeCloseTo(L.structureScore(ours.dag, data, score, { states }), 8)
          expect(ours.score).toBeGreaterThanOrEqual(((c.output as { data: number[] }).data[0] ?? 0) - 1e-8)
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('structure learning', () => {
  const truth = P.bayesNet([
    { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
    { name: 'b', states: ['0', '1'], parents: [], cpt: [0.4, 0.6] },
    { name: 'c', states: ['0', '1'], parents: ['a', 'b'], cpt: [0.9, 0.1, 0.3, 0.7, 0.2, 0.8, 0.05, 0.95] },
    { name: 'd', states: ['0', '1'], parents: ['c'], cpt: [0.85, 0.15, 0.25, 0.75] },
  ])

  it('PC recovers the equivalence class of a collider network and traces removals', async () => {
    const { key } = await import('../../random/index.ts')
    const rows = P.ancestralSample(truth, key(2), 3000).map((s) =>
      Object.fromEntries(Object.entries(s).map(([k, v]) => [k, String(v)])),
    )
    const rec = recorder()
    const r = L.pc(rows, { tracer: rec, policy: tracePolicy({ messages: true }) })
    const expected = L.cpdag(truth)
    expect(sorted(r.cpdag.directed)).toEqual(sorted(expected.directed))
    expect(sorted(r.cpdag.undirected)).toEqual(sorted(expected.undirected))
    expect(messages(rec.run(), 'removeEdge').length).toBe(3)
    // Greedy search can stop at a local optimum; started at the truth it stays in its equivalence class.
    const hc = L.hillClimb(rows, { score: 'bic' })
    expect(hc.score).toBeGreaterThan(L.structureScore(P.dag([], ['a', 'b', 'c', 'd']), rows, 'bic'))
    expect(L.isMarkovEquivalent(L.hillClimb(rows, { score: 'bic', start: truth }).dag, truth)).toBe(true)
    expect(L.hillClimb(rows, { score: 'bic', maxIndegree: 1 }).dag.nodes.every((n) => n.parents.length <= 1)).toBe(true)
  })
})
