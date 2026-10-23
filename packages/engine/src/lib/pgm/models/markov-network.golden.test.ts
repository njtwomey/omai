import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { tensor } from '../../tensor/index.ts'
import * as P from '../index.ts'

type Params = Readonly<Record<string, unknown>>
interface NetworkJson {
  readonly variables: P.DiscreteVariable[]
  readonly potentials: { variables: string[]; cardinality: number[]; values: number[] }[]
}
const networkOf = (p: Params): P.MarkovNetwork => {
  const n = p['network'] as NetworkJson
  return P.markovNetwork(
    n.variables,
    n.potentials.map((f) => P.factor(f.variables, f.cardinality, f.values)),
  )
}
const latticeOf = (p: Params): P.MarkovNetwork => {
  const options = p['options'] as P.LatticeOptions & { field?: number; q?: number }
  return p['kind'] === 'potts' ? P.pottsModel(options) : P.isingModel(options)
}

describe('Markov networks against pgmpy', () => {
  for (const c of loadFixture('pgm/models/markov_network').cases) {
    it(c.name, () => {
      const query = (c.params['query'] ?? []) as string[]
      switch (c.fn) {
        case 'logZ':
          expectMatches(P.logPartition(networkOf(c.params)), c.output, 1e-12, 0, c.name)
          return
        case 'posterior': {
          const mn = networkOf(c.params)
          const evidence = c.params['evidence'] as Record<string, number>
          const { posterior } = P.variableElimination(mn, query, { evidence })
          expectMatches(tensor(P.factorValues(posterior)), c.output, 1e-12, 1e-15, c.name)
          expectMatches(tensor(P.factorValues(P.enumerate(mn, query, evidence))), c.output, 1e-12, 1e-15, c.name)
          return
        }
        case 'blanket':
          expect(P.markovNeighbours(networkOf(c.params), c.params['node'] as string)).toEqual(c.params['blanket'])
          return
        case 'mpe':
          expect(P.maxProductElimination(networkOf(c.params)).assignment).toEqual(c.params['assignment'])
          return
        case 'latticeLogZ':
          expectMatches(P.logPartition(latticeOf(c.params)), c.output, 1e-12, 0, c.name)
          return
        case 'latticeMarginal': {
          const { posterior } = P.variableElimination(latticeOf(c.params), query)
          expectMatches(tensor(P.factorValues(posterior)), c.output, 1e-12, 1e-15, c.name)
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})
