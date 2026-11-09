import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { graphEdges } from './evaluation.ts'
import { neighbourhoodSelection } from './neighbourhood.ts'

const data = fc.array(fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 4, maxLength: 4 }), {
  minLength: 10,
  maxLength: 30,
})

describe('neighbourhood selection', () => {
  it('the "and" graph is a subgraph of the "or" graph, for any data and penalty', () => {
    fc.assert(
      fc.property(data, fc.double({ min: 0.01, max: 0.6, noNaN: true }), (x, penalty) => {
        // A constant column has no variance to regress on or to regress: skip degenerate draws.
        fc.pre([0, 1, 2, 3].every((j) => new Set(x.map((r) => r[j])).size > 1))
        const nb = neighbourhoodSelection(x, { penalty })
        for (const [a, b] of graphEdges(nb.and)) expect(nb.or[a]?.[b]).toBe(true)
        expect(graphEdges(nb.and).length).toBeLessThanOrEqual(graphEdges(nb.or).length)
      }),
      { numRuns: 40 },
    )
  })

  it('a larger penalty never selects more coefficients than a smaller one', () => {
    fc.assert(
      fc.property(data, (x) => {
        fc.pre([0, 1, 2, 3].every((j) => new Set(x.map((r) => r[j])).size > 1))
        const nonzeros = (penalty: number): number =>
          neighbourhoodSelection(x, { penalty, tol: 1e-10 })
            .coefficients.flat()
            .filter((v) => v !== 0).length
        expect(nonzeros(0.5)).toBeLessThanOrEqual(nonzeros(0.05))
      }),
      { numRuns: 30 },
    )
  })
})
