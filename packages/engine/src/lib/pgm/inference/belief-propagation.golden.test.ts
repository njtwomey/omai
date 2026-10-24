import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { messages, recorder, tracePolicy } from '../../trace/index.ts'
import * as P from '../index.ts'

type Params = Readonly<Record<string, unknown>>
interface GraphJson {
  readonly variables: P.DiscreteVariable[]
  readonly factors: { variables: string[]; cardinality: number[]; values: number[] }[]
}
const graphOf = (p: Params): P.FactorGraph => {
  const g = p['network'] as GraphJson
  return P.factorGraph(
    g.variables,
    g.factors.map((f) => P.factor(f.variables, f.cardinality, f.values)),
  )
}

function expectMarginals(
  actual: Readonly<Record<string, P.Factor>>,
  expected: Readonly<Record<string, number[]>>,
  tol: number,
): void {
  for (const [v, values] of Object.entries(expected)) {
    const got = P.factorValues(actual[v] as P.Factor)
    for (const [i, x] of values.entries()) expect(Math.abs((got[i] ?? 0) - x)).toBeLessThan(tol)
  }
}

describe('belief propagation against pgmpy and pyAgrum', () => {
  for (const c of loadFixture('pgm/inference/belief_propagation').cases) {
    it(c.name, () => {
      const evidence = c.params['evidence'] as Record<string, string | number>
      const expected = c.params['marginals'] as Record<string, number[]>
      if (c.fn === 'tree') {
        const fg = graphOf(c.params)
        for (const schedule of ['tree', 'flooding', 'sequential', 'residual'] as const) {
          const r = P.beliefPropagation(fg, { evidence, schedule })
          expect(r.converged).toBe(true)
          expectMarginals(r.marginals, expected, 1e-12)
          // Exact on trees: the Bethe free energy gives log Z.
          expect(r.logPartition).toBeCloseTo(P.logPartition(fg, { evidence }), 10)
        }
        return
      }
      // Loopy: the fixed point of sum-product on the network's factor graph.
      const bn = P.bayesNet(c.params['network'] as P.BayesNode[])
      for (const schedule of ['flooding', 'sequential', 'residual'] as const) {
        const r = P.beliefPropagation(bn, { evidence, schedule, tol: 1e-13, maxIter: 2000 })
        expect(r.converged).toBe(true)
        // pyAgrum stops at its own epsilon, so the agreement is to iterative precision.
        expectMarginals(r.marginals, expected, 1e-8)
      }
    })
  }
})

describe('loopy belief propagation', () => {
  const grid = P.isingModel({ rows: 3, cols: 3, coupling: 0.25, field: { s_0_0: 0.3, s_2_1: -0.2 } })

  it('approximates the exact marginals and log Z on a weakly coupled grid', () => {
    const r = P.beliefPropagation(grid)
    expect(r.converged).toBe(true)
    const exact = P.junctionTreeMarginals(grid)
    for (const v of Object.keys(exact)) {
      expect(
        Math.abs((P.factorValues(r.marginals[v] as P.Factor)[1] ?? 0) - (P.factorValues(exact[v] as P.Factor)[1] ?? 0)),
      ).toBeLessThan(0.01)
    }
    expect(Math.abs(r.logPartition - P.logPartition(grid))).toBeLessThan(0.02)
    // The free energy of the returned beliefs matches betheFreeEnergy on the same beliefs.
    expect(P.betheFreeEnergy(grid, r.marginals, r.factorBeliefs)).toBeCloseTo(r.betheFreeEnergy, 10)
  })

  it('reaches the same fixed point with damping and any schedule', () => {
    const base = P.beliefPropagation(grid, { tol: 1e-13 })
    for (const options of [{ damping: 0.5 }, { schedule: 'sequential' as const }, { schedule: 'residual' as const }]) {
      const r = P.beliefPropagation(grid, { ...options, tol: 1e-13, maxIter: 1000 })
      expect(r.converged).toBe(true)
      for (const v of Object.keys(base.marginals)) {
        expect(P.factorValues(r.marginals[v] as P.Factor)[0]).toBeCloseTo(
          P.factorValues(base.marginals[v] as P.Factor)[0] ?? 0,
          10,
        )
      }
    }
  })

  it('max-product on a tree gives max-marginals whose argmax is the MPE', () => {
    const chain = P.isingModel({
      rows: 1,
      cols: 5,
      coupling: { 's_0_0|s_0_1': 1, 's_0_1|s_0_2': -0.5, 's_0_2|s_0_3': 0.3, 's_0_3|s_0_4': 0.8 },
      field: { s_0_0: 0.4, s_0_3: -0.9 },
    })
    const r = P.beliefPropagation(chain, { mode: 'max', schedule: 'tree' })
    const mpe = P.maxProductElimination(chain).indices
    for (const [v, f] of Object.entries(r.marginals)) expect(P.factorArgmax(f)[v]).toBe(mpe[v])
    expect(Number.isNaN(r.logPartition)).toBe(true)
  })

  it('records factor-to-variable messages and residuals', () => {
    const rec = recorder()
    P.beliefPropagation(grid, { tracer: rec, policy: tracePolicy({ messages: true }), maxIter: 3 })
    const run = rec.run()
    expect(messages(run, 'factorToVariable').length).toBeGreaterThan(0)
    expect(() => P.beliefPropagation(grid, { schedule: 'tree' })).toThrow(/factor tree/)
    expect(() => P.beliefPropagation(grid, { damping: 1 })).toThrow(/damping/)
  })
})
