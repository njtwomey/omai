import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import * as P from '../index.ts'

const networkOf = (p: Readonly<Record<string, unknown>>): P.BayesNet => P.bayesNet(p['network'] as P.BayesNode[])

describe('Bayes-network graph queries against pgmpy', () => {
  for (const c of loadFixture('pgm/models/bayes_net').cases) {
    it(c.name, () => {
      const bn = networkOf(c.params)
      switch (c.fn) {
        case 'trail': {
          const nodes = P.activeTrailNodes(bn, c.params['start'] as string, c.params['observed'] as string[])
          expect(nodes).toEqual(c.params['nodes'])
          // Every reachable node is d-connected to the start, every other unobserved node d-separated.
          const observed = c.params['observed'] as string[]
          for (const n of bn.nodes) {
            if (n.name === c.params['start'] || observed.includes(n.name)) continue
            const connected = !P.dSeparated(bn, [c.params['start'] as string], [n.name], observed)
            expect(connected).toBe(nodes.includes(n.name))
          }
          return
        }
        case 'ancestors':
          expect(P.ancestors(bn, c.params['of'] as string[])).toEqual(c.params['nodes'])
          return
        case 'local':
          expect(P.localIndependencies(bn)).toEqual(c.params['statements'])
          return
        case 'moral':
          expect(P.graphEdges(P.moralGraph(bn))).toEqual(c.params['edges'])
          expect(P.moralise(bn).edges).toEqual(c.params['edges'])
          return
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})
