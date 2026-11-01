import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { factorFromJson, factorToJson, factorValues } from '../factor.ts'
import { bayesNet } from '../models/bayes-net.ts'
import { isingModel } from '../models/markov-network.ts'
import { eliminationHeuristics } from '../triangulation-options.ts'
import {
  eliminationRunContract,
  logPartition,
  mapQuery,
  maxProductElimination,
  variableElimination,
  variableEliminationAlgorithm,
} from './elimination.ts'
import { eliminationSemirings } from './elimination-options.ts'
import { enumerate } from './enumerate.ts'

const rainy = bayesNet([
  { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
  { name: 'sprinkler', states: ['off', 'on'], parents: ['rain'], cpt: [0.6, 0.4, 0.99, 0.01] },
  { name: 'wet', states: ['no', 'yes'], parents: ['sprinkler', 'rain'], cpt: [1, 0, 0.2, 0.8, 0.1, 0.9, 0.01, 0.99] },
])
const ising = isingModel({ rows: 2, cols: 3 })
const wetYes = { evidence: { wet: 'yes' } }

describe('variableEliminationAlgorithm', () => {
  it('is variableElimination, logPartition and maxProductElimination: the numbers of before', () => {
    const r = variableElimination(rainy, ['rain'], wetYes)
    expect(factorValues(r.posterior)).toEqual([0.6423123243677238, 0.3576876756322761])
    expect(r.logEvidence).toBe(-0.802114191811883)
    expect([r.order, r.largestFactor]).toEqual([['sprinkler'], 4])
    expect(logPartition(ising)).toBeCloseTo(7.820215857724743, 13)
    const big = variableElimination(ising, [])
    expect(big.order).toEqual(['s_0_0', 's_1_0', 's_0_1', 's_0_2', 's_1_1', 's_1_2'])
    expect(big.largestFactor).toBe(8)
    const m = maxProductElimination(rainy, wetYes)
    expect(m.assignment).toEqual({ rain: 'no', sprinkler: 'on' })
    expect(m.logValue).toBe(-1.2447947988461912)
    expect(maxProductElimination(ising).logValue).toBe(7)
    expect(mapQuery(rainy, ['rain'], wetYes).probability).toBe(0.6423123243677238)
  })

  it('eliminates one variable per step, and the final factors are those the function used', () => {
    const alg = variableEliminationAlgorithm(ising, [])
    const end = runAlgorithm(alg, { maxSteps: 6 })
    expect(end.steps).toBe(6)
    expect(end.status).toBe('converged')
    expect(end.state.eliminated).toEqual(end.state.order)
    expect(end.state.factors.every((f) => f.variables.length === 0)).toBe(true)
  })

  it('resumes from a JSON checkpoint to an identical end, zeros (minus infinity) included', () => {
    for (const [model, query, options, semiring] of [
      [rainy, ['rain'], wetYes, 'sum'],
      [rainy, [], wetYes, 'max'],
      [ising, [], {}, 'max'],
    ] as const) {
      const alg = variableEliminationAlgorithm(model, query, { ...options, semiring })
      const whole = runAlgorithm(alg, { maxSteps: 10 })
      const k = Math.max(1, Math.floor(whole.steps / 2))
      const half = runAlgorithm(alg, { maxSteps: k })
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
      const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: k, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    }
    const f = factorToJson({
      variables: ['a'],
      cardinality: [2],
      logValues: Float64Array.of(Number.NEGATIVE_INFINITY, 0),
    })
    expect(JSON.parse(JSON.stringify(f))).toEqual(f)
    expect(factorFromJson(JSON.parse(JSON.stringify(f))).logValues[0]).toBe(Number.NEGATIVE_INFINITY)
  })

  it('logs its contract: a message per elimination with the new factor, and the size, width and count', () => {
    const rec = recorder()
    runAlgorithm(variableEliminationAlgorithm(rainy, []), {
      maxSteps: 3,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, messages: true },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('ve/factorSize')).toHaveLength(3)
    expect(metric('ve/width')).toHaveLength(3)
    expect(metric('ve/factors').at(-1)).toBe(1)
    const msgs = rec.events.flatMap((e) => (e.type === 'message' && e.kind === 've/eliminated' ? [e.payload] : []))
    expect(msgs).toHaveLength(3)
    expect((msgs[0] as { scope: string[] }).scope.length).toBeGreaterThan(0)
    expect(Object.keys(eliminationRunContract)).toEqual(['ve/eliminated', 've/factorSize', 've/width', 've/factors'])
  })

  it('agrees with brute-force enumeration for any elimination order', () => {
    const exact = factorValues(enumerate(ising, ['s_0_0']))
    const total = exact.reduce((a, b) => a + b, 0)
    const hidden = ['s_0_1', 's_0_2', 's_1_0', 's_1_1', 's_1_2']
    fc.assert(
      fc.property(fc.shuffledSubarray(hidden, { minLength: 5, maxLength: 5 }), (order) => {
        const r = variableElimination(ising, ['s_0_0'], { order: [...order] })
        return factorValues(r.posterior).every((p, i) => Math.abs(p - (exact[i] ?? 0) / total) < 1e-12)
      }),
      { numRuns: 30 },
    )
  })

  it('runs every option of the semiring and heuristic tables', () => {
    for (const name of eliminationSemirings.names) {
      const alg = variableEliminationAlgorithm(rainy, [], {
        ...wetYes,
        semiring: eliminationSemirings.make({ kind: name }),
      })
      const st = runAlgorithm(alg, { maxSteps: 10 }).state
      expect(st.eliminated.length).toBeGreaterThan(0)
      expect(st.products.length > 0).toBe(eliminationSemirings.properties({ kind: name }).answers === 'mode')
    }
    for (const name of eliminationHeuristics.names) {
      const heuristic = eliminationHeuristics.make({ kind: name })
      const r = variableElimination(ising, [], { heuristic })
      expect(r.logEvidence).toBeCloseTo(7.820215857724743, 12)
    }
  })

  it('rejects an incomplete order and a query that is observed', () => {
    expect(() => variableEliminationAlgorithm(rainy, ['rain'], { ...wetYes, order: [] })).toThrow(/order/)
    expect(() => variableEliminationAlgorithm(rainy, ['wet'], wetYes)).toThrow(/observed/)
  })
})
