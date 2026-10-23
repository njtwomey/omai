import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

function programOf(inputs: readonly unknown[]): O.MixedIntegerProgram {
  const [c, aUb, bUb, aEq, bEq, bounds, integrality] = inputs.map((x) => decode(x) as Tensor) as Tensor[]
  const b = toArray(bounds as Tensor) as number[][]
  return {
    c: c as Tensor,
    ...((aUb as Tensor).shape[0] === 0 ? {} : { aUb: aUb as Tensor, bUb: bUb as Tensor }),
    ...((aEq as Tensor).shape[0] === 0 ? {} : { aEq: aEq as Tensor, bEq: bEq as Tensor }),
    bounds: b.map(([lo, hi]) => [Number.isFinite(lo) ? (lo ?? 0) : null, Number.isFinite(hi) ? (hi ?? 0) : null]),
    integrality: toFlat(integrality as Tensor),
  }
}

const fixture = loadFixture('optim/integer')

describe('milp against scipy (HiGHS)', () => {
  for (const c of fixture.cases.filter((k) => k.fn === 'milp')) {
    for (const options of [
      { strategy: 'best-first' as const },
      { strategy: 'depth-first' as const },
      { strategy: 'best-first' as const, cutRounds: 3 },
    ]) {
      it(`${c.name} ${options.strategy}${'cutRounds' in options ? ' with cuts' : ''}`, () => {
        const out = c.output as Out
        const r = O.milp(programOf(c.inputs), options)
        expect(r.status).toBe(out['status'])
        if (out['status'] !== 'optimal') return
        expectMatches(r.fun, out['fun'], 1e-9, 1e-9, `${c.name} fun`)
        if (c.name === 'textbook') expectMatches(r.x, out['x'], 1e-9, 1e-9, `${c.name} x`)
        expect(r.gap).toBeLessThanOrEqual(1e-9)
      })
    }
  }

  it('knapsacks by dynamic programming and branch and bound', () => {
    for (const c of fixture.cases.filter((k) => k.fn === 'knapsack')) {
      const [values, weights] = c.inputs.map((x) => toFlat(decode(x) as Tensor))
      const out = c.output as Out
      const capacity = c.params['capacity'] as number
      for (const method of ['dynamic-programming', 'branch-and-bound'] as const) {
        const r = O.knapsack(values ?? [], weights ?? [], capacity, { method })
        expectMatches(r.value, out['value'], 1e-12, 1e-9, `${c.name} ${method}`)
        expect(r.items).toEqual(toFlat(decode(out['items']) as Tensor))
        expect(r.weight).toBeLessThanOrEqual(capacity)
      }
    }
  })

  it('set covers: exact by milp, greedy within the H(n) bound', () => {
    for (const c of fixture.cases.filter((k) => k.fn === 'setCover')) {
      const incidence = toArray(decode(c.inputs[0]) as Tensor) as number[][]
      const costs = toFlat(decode(c.inputs[1]) as Tensor)
      const sets = (incidence[0] ?? []).map((_, j) => incidence.flatMap((row, e) => ((row[j] ?? 0) > 0 ? [e] : [])))
      const exact = O.setCover(sets, { costs, method: 'exact' })
      expectMatches(exact.cost, (c.output as Out)['cost'], 1e-9, 1e-9, c.name)
      const greedy = O.setCover(sets, { costs })
      const h = Array.from({ length: incidence.length }, (_, k) => 1 / (k + 1)).reduce((a, b) => a + b, 0)
      expect(greedy.cost).toBeGreaterThanOrEqual(exact.cost - 1e-9)
      expect(greedy.cost).toBeLessThanOrEqual(h * exact.cost + 1e-9)
      for (let e = 0; e < incidence.length; e++) expect(greedy.chosen.some((j) => sets[j]?.includes(e))).toBe(true)
    }
  })
})

describe('branch and bound', () => {
  const textbook: O.MixedIntegerProgram = {
    c: [-5, -4],
    aUb: [
      [6, 4],
      [1, 2],
    ],
    bUb: [24, 6],
  }

  it('records the search tree, incumbent and bound', () => {
    const rec = recorder()
    const r = O.milp(textbook, { tracer: rec, policy: tracePolicy({ messages: true }) })
    const run = rec.run()
    expect(messages(run, 'node')).toHaveLength(r.nodes)
    expect(r.tree[0]).toMatchObject({ id: 0, parent: -1, outcome: 'branched' })
    expect(r.tree.filter((n) => n.outcome === 'integral').length).toBeGreaterThan(0)
    const incumbent = metricSeries(run, 'incumbent').values
    expect(incumbent.at(-1)).toBe(-20)
    // Each node's relaxation is a child run.
    expect(run.events.some((e) => e.type === 'run' && e.name === 'simplex')).toBe(true)
  })

  it('tightens the root with Gomory cuts that keep every integer point', () => {
    const r = O.milp(textbook, { cutRounds: 2 })
    expect(r.cuts.length).toBeGreaterThan(0)
    for (let x1 = 0; x1 <= 4; x1++) {
      for (let x2 = 0; x2 <= 3; x2++) {
        if (6 * x1 + 4 * x2 > 24 || x1 + 2 * x2 > 6) continue
        for (const cut of r.cuts) expect((cut.a[0] ?? 0) * x1 + (cut.a[1] ?? 0) * x2).toBeLessThanOrEqual(cut.b + 1e-9)
      }
    }
    // The fractional root optimum (3, 1.5) is cut off.
    expect(r.cuts.some((cut) => (cut.a[0] ?? 0) * 3 + (cut.a[1] ?? 0) * 1.5 > cut.b + 1e-9)).toBe(true)
  })

  it('handles continuous variables, unbounded relaxations and node limits', () => {
    const mixed = O.milp({ c: [-1, -1], aUb: [[2, 2]], bUb: [3], integrality: [1, 0] })
    expect(mixed.fun).toBeCloseTo(-1.5, 12)
    expect(O.milp({ c: [-1], integrality: 1 }).status).toBe('unbounded')
    const limited = O.milp({ c: [-1, -1, -1], aUb: [[2, 2, 2]], bUb: [5] }, { strategy: 'depth-first', maxNodes: 1 })
    expect(limited.status).toBe('node limit')
  })

  it('traces the knapsack search node by node', () => {
    const rec = recorder()
    const alg = O.knapsackAlgorithm([60, 100, 120, 30], [10, 20, 30, 15], 50)
    const r = runAlgorithm(alg, { maxSteps: 1000, tracer: rec, policy: tracePolicy({ messages: true }) })
    expect(r.state.bestValue).toBe(220)
    expect(messages(rec.run(), 'node')).toHaveLength(r.state.nodes)
    expect(metricSeries(rec.run(), 'incumbent').values.at(-1)).toBe(220)
  })

  it('knapsack edge cases', () => {
    expect(O.knapsack([], [], 10)).toEqual({ value: 0, weight: 0, items: [], nodes: 0 })
    expect(O.knapsack([5, 4], [2.5, 1.5], 3).items).toEqual([0])
    expect(() => O.knapsack([1], [1.5], 2, { method: 'dynamic-programming' })).toThrow()
    expect(() => O.setCover([[0], [2]])).toThrow()
  })
})
