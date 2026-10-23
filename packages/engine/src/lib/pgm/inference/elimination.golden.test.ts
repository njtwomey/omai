import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import * as P from '../index.ts'

const networkOf = (p: Readonly<Record<string, unknown>>): P.BayesNet => P.bayesNet(p['network'] as P.BayesNode[])

describe('variable elimination against pgmpy', () => {
  for (const c of loadFixture('pgm/inference/elimination').cases) {
    it(c.name, () => {
      const bn = networkOf(c.params)
      const evidence = c.params['evidence'] as Record<string, string>
      switch (c.fn) {
        case 'mpe': {
          const best = P.maxProductElimination(bn, { evidence })
          expect(best.assignment).toEqual(c.params['assignment'])
          // mapQuery over every unobserved variable takes the same path and reports P(x | e).
          const hidden = Object.keys(c.params['assignment'] as object)
          const q = P.mapQuery(bn, hidden, { evidence })
          expect(q.assignment).toEqual(c.params['assignment'])
          const e = P.evidenceIndices(bn, evidence)
          const joint = P.jointProbability(bn, { ...best.indices, ...e })
          expect(best.logValue).toBeCloseTo(Math.log(joint), 12)
          return
        }
        case 'logEvidence':
          expectMatches(P.logPartition(bn, { evidence }), c.output, 1e-12, 0, c.name)
          expectMatches(P.variableElimination(bn, [], { evidence }).logEvidence, c.output, 1e-12, 0, c.name)
          return
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})
