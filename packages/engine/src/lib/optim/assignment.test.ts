import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { key, uniform } from '../random/index.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

describe('assignment against scipy.optimize.linear_sum_assignment', () => {
  for (const c of loadFixture('optim/assignment').cases) {
    const cost = decode(c.inputs[0]) as Tensor
    const maximize = c.params['maximize'] as boolean
    const out = c.output as Readonly<Record<string, unknown>>
    it(`${c.name} by the Hungarian algorithm`, () => {
      const r = O.linearSumAssignment(cost, { maximize })
      expect(r.rowInd).toEqual(toFlat(decode(out['rowInd']) as Tensor))
      expect(r.colInd).toEqual(toFlat(decode(out['colInd']) as Tensor))
      expectMatches(r.cost, out['cost'], 1e-12, 1e-12, c.name)
    })
    it(`${c.name} by auction`, () => {
      const r = O.auction(cost, { maximize })
      expect(r.rowInd).toEqual(toFlat(decode(out['rowInd']) as Tensor))
      expect(r.colInd).toEqual(toFlat(decode(out['colInd']) as Tensor))
      expectMatches(r.cost, out['cost'], 1e-9, 1e-9, c.name)
    })
  }
})

describe('Hungarian algorithm', () => {
  it('keeps dual feasibility and closes the duality gap', () => {
    const cost = toArray(uniform(key(1), [5, 5])) as number[][]
    const rec = recorder()
    const r = O.linearSumAssignment(cost, { tracer: rec })
    const dual = metricSeries(rec.run(), 'dual').values
    expect(dual.at(-1)).toBeCloseTo(r.cost, 12)
    for (let i = 1; i < dual.length; i++) expect(dual[i] ?? 0).toBeGreaterThanOrEqual((dual[i - 1] ?? 0) - 1e-12)
  })

  it('respects forbidden pairs and rejects infeasible matrices', () => {
    const r = O.linearSumAssignment([
      [Infinity, 1],
      [2, Infinity],
    ])
    expect([r.colInd, r.cost]).toEqual([[1, 0], 3])
    expect(() =>
      O.linearSumAssignment([
        [Infinity, 1],
        [Infinity, 2],
      ]),
    ).toThrow()
    expect(O.linearSumAssignment([[]] as number[][]).rowInd).toEqual([])
  })
})

describe('auction', () => {
  it('is exact for integer benefits once epsilon < 1/n, and records bids by phase', () => {
    const benefit = [
      [7, 2, 1, 9],
      [3, 8, 6, 2],
      [5, 4, 9, 1],
      [8, 1, 3, 4],
    ]
    const rec = recorder()
    const r = O.auction(benefit, {
      maximize: true,
      epsilonMin: 0.2,
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    expect(r.cost).toBe(O.linearSumAssignment(benefit, { maximize: true }).cost)
    const run = rec.run()
    const eps = metricSeries(run, 'epsilon').values
    expect(eps[0]).toBe(4.5)
    expect(eps.at(-1)).toBe(0.2)
    expect(messages(run, 'bid').length).toBeGreaterThan(4)
  })
})

describe('stable matching', () => {
  it('gives the proposer-optimal stable matching on random preferences', () => {
    const n = 8
    const perm = (seed: number): number[] => {
      const u = toFlat(uniform(key(seed), [n]))
      return Array.from({ length: n }, (_, i) => i).sort((a, b) => (u[a] ?? 0) - (u[b] ?? 0))
    }
    const proposers = Array.from({ length: n }, (_, i) => perm(10 + i))
    const receivers = Array.from({ length: n }, (_, i) => perm(100 + i))
    const r = O.stableMatching(proposers, receivers)
    expect(O.blockingPairs(r.proposerMatch, proposers, receivers)).toEqual([])
    expect([...r.proposerMatch].sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i))
    // Proposer-optimal: no stable matching gives any proposer a better partner, so the receiver-proposing run is
    // weakly worse for every proposer.
    const reverse = O.stableMatching(receivers, proposers)
    for (let i = 0; i < n; i++) {
      const theirs = reverse.receiverMatch[i] ?? -1
      const list = proposers[i] ?? []
      expect(list.indexOf(r.proposerMatch[i] ?? -1)).toBeLessThanOrEqual(list.indexOf(theirs))
    }
  })

  it('handles incomplete lists and unequal sides, traced proposal by proposal', () => {
    const rec = recorder()
    const r = O.stableMatching([[0], [0, 1], [1]], [[1, 0], [2]], {
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    expect(r.proposerMatch).toEqual([-1, 0, 1])
    expect(r.receiverMatch).toEqual([1, 2])
    expect(messages(rec.run(), 'proposal')).toHaveLength(r.proposals)
    expect(O.blockingPairs(r.proposerMatch, [[0], [0, 1], [1]], [[1, 0], [2]])).toEqual([])
  })
})
