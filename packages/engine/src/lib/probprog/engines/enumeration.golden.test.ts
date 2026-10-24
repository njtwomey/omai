import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { categorical } from '../../dists/index.ts'
import { add, item, mul, reshape, take, tensor, type Value } from '../../tensor/index.ts'
import { condition } from '../handlers.ts'
import { type Model, sample } from '../primitives.ts'
import { enumerationPosterior } from './enumeration.ts'

interface Node {
  readonly name: string
  readonly states: readonly string[]
  readonly parents: readonly string[]
  readonly cpt: readonly number[]
}

/** A Bayes network as a program: one categorical site per node, its row of the CPT picked by the parents' states. */
function network(nodes: readonly Node[]): Model {
  return () => {
    const values = new Map<string, Value>()
    const cards = new Map(nodes.map((n) => [n.name, n.states.length]))
    for (const n of nodes) {
      const k = n.states.length
      const rows = n.cpt.length / k
      const table = reshape(tensor([...n.cpt]), [rows, k])
      let index: Value = tensor(0)
      for (const p of n.parents) index = add(mul(index, cards.get(p) ?? 1), values.get(p) ?? tensor(0))
      values.set(
        n.name,
        sample(n.name, categorical({ probs: n.parents.length === 0 ? reshape(table, [k]) : take(table, index) })),
      )
    }
  }
}

describe('probprog/engines/enumeration against pgmpy', () => {
  for (const c of loadFixture('probprog/engines/enumeration').cases) {
    it(c.name, () => {
      const model = condition(network(c.params['network'] as Node[]), c.params['evidence'] as Record<string, number>)
      const post = enumerationPosterior(model, undefined)
      if (c.fn === 'evidence') expectMatches(post.logEvidence?.value ?? Number.NaN, c.output, c.rtol, c.atol, c.name)
      else {
        const m = post.marginal(c.params['query'] as string)
        const k = (c.params['network'] as Node[]).find((n) => n.name === c.params['query'])?.states.length ?? 0
        expectMatches(tensor(Array.from({ length: k }, (_, i) => item(m.prob(i)))), c.output, c.rtol, c.atol, c.name)
      }
    })
  }
})
