import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { logSumExpOf } from '../../base/index.ts'
import { type GraphInput, graphCrf } from './graph-crf.ts'
import { graphOracle } from './graph-oracle.ts'
import { graphPseudoLikelihood } from './graph-pseudo-likelihood.ts'

/** A random small graph: node values, typed and attributed edges (parallel ones and both directions included). */
const arbitraryInput = fc
  .integer({ min: 1, max: 5 })
  .chain((n) =>
    fc.record({
      nodes: fc.array(fc.record({ f: fc.double({ min: -2, max: 2, noNaN: true }) }), { minLength: n, maxLength: n }),
      edges: fc.array(
        fc.record({
          from: fc.integer({ min: 0, max: n - 1 }),
          to: fc.integer({ min: 0, max: n - 1 }),
          type: fc.constantFrom('p', 'q'),
          attributes: fc.record({ c: fc.double({ min: -1, max: 1, noNaN: true }) }),
        }),
        { maxLength: 8 },
      ),
    }),
  )
  .map((g): GraphInput => ({ nodes: g.nodes, edges: g.edges.filter((e) => e.from !== e.to) }))

const model = graphCrf(['a', 'b', 'c'], ['f'], { edgeTypes: ['p', 'q'], symmetric: ['q'], edgeAttributes: ['c'] })
const oracle = graphOracle(model)
const weights = Float64Array.from({ length: model.layout.size }, (_, i) => Math.sin(1 + 2.3 * i) * 0.8)

describe('graph CRF (properties)', () => {
  it('on any small graph the automatically chosen exact inference matches enumeration', () => {
    fc.assert(
      fc.property(arbitraryInput, (x) => {
        const scored: { phi: Float64Array; score: number }[] = []
        const n = x.nodes.length
        for (let code = 0; code < 3 ** n; code++) {
          const y = Array.from({ length: n }, (_, i) => Math.floor(code / 3 ** i) % 3)
          const phi = oracle.features(x, y)
          scored.push({ phi, score: phi.reduce((s, v, i) => s + v * (weights[i] ?? 0), 0) })
        }
        const logZ = logSumExpOf(scored.map((s) => s.score))
        const e = oracle.expectations(weights, x)
        expect(e.exact).toBe(true)
        expect(e.logPartition).toBeCloseTo(logZ, 8)
        for (let i = 0; i < weights.length; i++) {
          const expected = scored.reduce((s, r) => s + Math.exp(r.score - logZ) * (r.phi[i] ?? 0), 0)
          expect(e.counts[i]).toBeCloseTo(expected, 7)
        }
        // the marginals sum to one at every node and the decoded score is the best one
        for (const row of oracle.marginals(weights, x).nodes) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
        expect(oracle.decode(weights, x).score).toBeCloseTo(Math.max(...scored.map((s) => s.score)), 8)
      }),
      { numRuns: 40 },
    )
  })

  it('the pseudo-likelihood of a graph is positive and its gradient has a zero sum over the labels of a bias-free node block', () => {
    fc.assert(
      fc.property(arbitraryInput, (x) => {
        const r = graphPseudoLikelihood(model, [{ x, y: x.nodes.map((_, i) => i % 3) }], weights)
        expect(r.value).toBeGreaterThan(-1e-12)
        // each node's conditional is a softmax, so the node-weight gradient of an attribute sums to zero over the labels
        expect(Array.from(r.gradient.slice(3, 6)).reduce((a, b) => a + b, 0)).toBeCloseTo(0, 9)
      }),
      { numRuns: 40 },
    )
  })
})
