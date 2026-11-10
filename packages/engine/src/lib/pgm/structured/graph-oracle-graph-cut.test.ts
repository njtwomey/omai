import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { graphCutMap } from '../inference/graph-cut-map.ts'
import { graphParameterNames } from './graph-crf.ts'
import { graphOracle } from './graph-oracle.ts'
import { gridCrf, gridInput } from './graph-structures.ts'

const nodes = (n: number, seed: number): Record<string, number>[] => {
  const v = toFlat(normal(key(seed), [n * 2]))
  return Array.from({ length: n }, (_, i) => ({ f: v[2 * i] ?? 0, g: v[2 * i + 1] ?? 0 }))
}

/** Random weights, except that agreeing neighbours score more than disagreeing ones (a submodular grid CRF). */
function submodularWeights(names: readonly string[], seed: number): Float64Array {
  const noise = toFlat(normal(key(seed), [names.length]))
  return Float64Array.from(names, (name, i) => {
    const [kind, , first, second] = name.split(/[:|]/)
    if (kind !== 'pair') return noise[i] ?? 0
    return first === second ? Math.abs(noise[i] ?? 0) + 0.2 : 0
  })
}

describe('graphOracle with graph-cut MAP', () => {
  const model = gridCrf(['a', 'b'], ['f', 'g'])
  const x = gridInput(nodes(12, 5), 3, 4)
  const w = submodularWeights(graphParameterNames(model), 9)
  const exact = graphOracle(model)
  const cut = graphOracle(model, { map: (fg) => graphCutMap(fg) })

  it('decodes a binary grid CRF to the same score as the exact default', () => {
    const a = exact.decode(w, x)
    const b = cut.decode(w, x)
    expect(b.score).toBeCloseTo(a.score, 9)
  })

  it('decodes with cost augmentation to the same augmented optimum', () => {
    const gold = Array.from({ length: 12 }, (_, i) => (i % 3 === 0 ? 'a' : 'b'))
    const a = exact.decode(w, x, { against: gold })
    const b = cut.decode(w, x, { against: gold })
    // The cost-augmented objective is score + loss; the two decoders reach the same value of it.
    expect(b.score + cut.loss(gold, b.y)).toBeCloseTo(a.score + exact.loss(gold, a.y), 9)
    // Augmenting changes the answer: it is at least as good on the augmented objective as the plain decoding.
    const plain = cut.decode(w, x)
    expect(b.score + cut.loss(gold, b.y)).toBeGreaterThanOrEqual(plain.score + cut.loss(gold, plain.y) - 1e-9)
  })

  it('refuses a CRF whose pair terms are not submodular, through the oracle', () => {
    const anti = Float64Array.from(graphParameterNames(model), (name) => (/pair:grid\|a\|a/.test(name) ? -2 : 0.1))
    expect(() => cut.decode(anti, x)).toThrow(/not submodular/)
  })
})
