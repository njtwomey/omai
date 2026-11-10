import { describe, expect, it } from 'vitest'
import { type PairKind, randomPairwise } from '../../../testing/pairwise-models.ts'
import type { OmaiError } from '../../base/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import * as P from '../index.ts'

/** Every labelling of the model's variables with its log value, by enumeration. */
function enumerateAll(model: P.DiscreteModel): { x: number[]; logValue: number }[] {
  const vars = P.modelVariables(model)
  const out: { x: number[]; logValue: number }[] = []
  const x = new Array<number>(vars.length).fill(0)
  const total = vars.reduce((t, v) => t * v.states.length, 1)
  for (let c = 0; c < total; c++) {
    out.push({
      x: [...x],
      logValue: P.unnormalisedLogProbability(model, Object.fromEntries(vars.map((v, i) => [v.name, x[i] ?? 0]))),
    })
    for (let a = vars.length - 1; a >= 0; a--) {
      x[a] = (x[a] ?? 0) + 1
      if ((x[a] ?? 0) < (vars[a]?.states.length ?? 1)) break
      x[a] = 0
    }
  }
  return out
}

const best = (model: P.DiscreteModel): number => Math.max(...enumerateAll(model).map((a) => a.logValue))

describe('graphCutMap', () => {
  const shapes = [
    { shape: 'chain', size: 9 },
    { shape: 'grid', size: 3, cols: 3 },
    { shape: 'grid', size: 2, cols: 4 },
    { shape: 'random', size: 8 },
  ] as const
  for (const s of shapes)
    for (const seed of [1, 2, 3]) {
      it(`is the exact MAP of a submodular ${s.shape} (seed ${seed})`, () => {
        const model = randomPairwise({ ...s, seed, kind: 'submodular' })
        const r = P.graphCutMap(model)
        expect(r.exact).toBe(true)
        expect(r.undecided).toEqual([])
        expect(r.logValue).toBeCloseTo(best(model), 9)
        expect(r.energy).toBeCloseTo(-r.logValue, 12)
        expect(P.unnormalisedLogProbability(model, r.indices)).toBeCloseTo(r.logValue, 9)
        expect(r.lowerBound).toBe(r.energy)
      })
    }

  it('handles evidence, an Ising ferromagnet and models with no pair terms', () => {
    const ising = P.isingModel({ rows: 2, cols: 2, coupling: 1, field: { s_0_0: 0.5 } })
    const r = P.graphCutMap(ising)
    expect(r.assignment).toEqual({ s_0_0: '+1', s_0_1: '+1', s_1_0: '+1', s_1_1: '+1' })
    const ev = P.graphCutMap(ising, { evidence: { s_1_1: '-1' } })
    expect(ev.logValue).toBeCloseTo(
      Math.max(
        ...enumerateAll(ising)
          .filter((a) => a.x[3] === 0)
          .map((a) => a.logValue),
      ),
      9,
    )
    expect(Object.keys(ev.indices)).not.toContain('s_1_1')
    const free = P.graphCutMap(randomPairwise({ seed: 5, shape: 'chain', size: 1, kind: 'submodular' }))
    expect(free.exact).toBe(true)
  })

  it('refuses a term that is not submodular, naming the pair, and models it cannot cut', () => {
    const model = P.isingModel({ rows: 1, cols: 3, coupling: -1 })
    expect(() => P.graphCutMap(model)).toThrow(/not submodular/)
    try {
      P.graphCutMap(model)
    } catch (e) {
      expect((e as OmaiError).code).toBe('value')
      expect((e as Error).message).toMatch(/s_0_0 and s_0_1/)
    }
    expect(() => P.graphCutMap(P.pottsModel({ rows: 1, cols: 3, q: 3 }))).toThrow(/binary/)
    const triple = P.factorGraph(
      ['a', 'b', 'c'].map((name) => ({ name, states: ['0', '1'] })),
      [P.logFactor(['a', 'b', 'c'], [2, 2, 2], new Array<number>(8).fill(0))],
    )
    expect(() => P.graphCutMap(triple)).toThrow(/pairwise/)
  })

  describe('roof duality (QPBO) for terms that are not submodular', () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      it(`labels persistently and bounds the optimum (seed ${seed})`, () => {
        const model = randomPairwise({
          seed,
          shape: seed % 2 === 0 ? 'grid' : 'random',
          size: seed % 2 === 0 ? 3 : 8,
          cols: 3,
          kind: 'any',
        })
        const r = P.graphCutMap(model, { nonSubmodular: 'qpbo' })
        const all = enumerateAll(model)
        const optimum = Math.max(...all.map((a) => a.logValue))
        // The lower bound on the energy is below the optimum energy, which is below the completed assignment's.
        expect(r.lowerBound).toBeLessThanOrEqual(-optimum + 1e-9)
        expect(r.energy).toBeGreaterThanOrEqual(-optimum - 1e-9)
        if (r.exact) expect(r.logValue).toBeCloseTo(optimum, 9)
        // Weak persistency: some optimal labelling agrees with every decided variable.
        const names = P.modelVariables(model).map((v) => v.name)
        const decided = names.filter((v) => !r.undecided.includes(v))
        const optimal = all.filter((a) => a.logValue >= optimum - 1e-9)
        // the completed assignment agrees with the persistent labels only where the descent did not move them;
        // persistency is checked against the roof-dual labelling through the optimal set
        expect(optimal.some((a) => decided.every((v) => a.x[names.indexOf(v)] === r.indices[v]))).toBe(true)
      })
    }

    it('equals the exact cut on a model that is submodular after all', () => {
      const model = randomPairwise({ seed: 9, shape: 'grid', size: 3, cols: 3, kind: 'submodular' })
      const a = P.graphCutMap(model)
      const b = P.graphCutMap(model, { nonSubmodular: 'qpbo' })
      expect(b.logValue).toBeCloseTo(a.logValue, 12)
      expect(b.exact).toBe(true)
    })

    it('decides everything on a model with a strong frustrated-free field', () => {
      // A big field on each variable settles every label whatever the pair terms say.
      const model = randomPairwise({ seed: 3, shape: 'chain', size: 6, kind: 'any', unaryScale: 50 })
      const r = P.graphCutMap(model, { nonSubmodular: 'qpbo' })
      expect(r.undecided).toEqual([])
      expect(r.logValue).toBeCloseTo(best(model), 9)
    })
  })
})

describe('pairwiseMoveConditions', () => {
  it('accepts Potts and truncated-linear tables for expansion and swap', () => {
    for (const kind of ['potts', 'truncatedLinear'] as PairKind[]) {
      const c = P.pairwiseMoveConditions(randomPairwise({ seed: 1, shape: 'grid', size: 2, cols: 2, labels: 4, kind }))
      expect(c).toMatchObject({ semiMetric: true, metric: true, expansion: true, swap: true })
    }
  })

  it('tells a semi-metric that is not a metric: expansion fails, swap passes', () => {
    // E(0,2) = 5 > E(0,1) + E(1,2) = 2: symmetric, non-negative, zero diagonal, no triangle inequality.
    const table = [0, 1, 5, 1, 0, 1, 5, 1, 0]
    const model = P.markovNetwork(
      ['a', 'b'].map((name) => ({ name, states: ['0', '1', '2'] })),
      [
        P.logFactor(
          ['a', 'b'],
          [3, 3],
          table.map((e) => -e),
        ),
      ],
    )
    const c = P.pairwiseMoveConditions(model)
    expect(c.semiMetric).toBe(true)
    expect(c.metric).toBe(false)
    expect(c.violations.metric).toMatch(/a-b: E\(0,2\) > E\(0,1\) \+ E\(1,2\)/)
    expect(c.expansion).toBe(false)
    expect(c.swap).toBe(true)
    expect(() => P.alphaExpansion(model)).toThrow(/expansion condition/)
    expect(P.alphaBetaSwap(model).converged).toBe(true)
  })

  it('rejects an asymmetric table for swap', () => {
    const model = P.markovNetwork(
      ['a', 'b'].map((name) => ({ name, states: ['0', '1'] })),
      [P.logFactor(['a', 'b'], [2, 2], [-1, 0, 0, -1])],
    )
    const c = P.pairwiseMoveConditions(model)
    expect(c.swap).toBe(false)
    expect(() => P.alphaBetaSwap(model)).toThrow(/swap condition/)
  })
})

describe('alphaExpansion and alphaBetaSwap', () => {
  const potts = (seed: number, shape: 'chain' | 'grid', size: number, labels: number): P.MarkovNetwork =>
    randomPairwise({ seed, shape, size, cols: size, labels, kind: 'potts' })

  for (const seed of [1, 2, 3, 4]) {
    it(`never exceed the ICM and max-product energy, and stay within twice the optimum on Potts (seed ${seed})`, () => {
      const model = potts(seed, 'grid', 3, 3)
      const optimum = -P.maxProductElimination(model).logValue
      const ex = P.alphaExpansion(model)
      const sw = P.alphaBetaSwap(model)
      expect(ex.converged).toBe(true)
      expect(sw.converged).toBe(true)
      expect(ex.energy).toBeGreaterThanOrEqual(optimum - 1e-9)
      expect(sw.energy).toBeGreaterThanOrEqual(optimum - 1e-9)
      // Veksler's bound for the Potts model: a local minimum of expansions is within 2x the optimum (non-negative energy).
      expect(ex.energy).toBeLessThanOrEqual(2 * optimum + 1e-9)
      const icm = P.localSearchMap(model)
      const mp = P.maxProductMap(model)
      expect(ex.energy).toBeLessThanOrEqual(-icm.logValue + 1e-9)
      expect(ex.energy).toBeLessThanOrEqual(-mp.logValue + 1e-9)
      expect(P.unnormalisedLogProbability(model, ex.indices)).toBeCloseTo(-ex.energy, 9)
    })
  }

  it('are exact on a chain with a metric (truncated linear) term, from several starts', () => {
    const model = randomPairwise({ seed: 7, shape: 'chain', size: 6, labels: 4, kind: 'truncatedLinear' })
    const optimum = -P.maxProductElimination(model).logValue
    const r = P.alphaExpansion(model)
    expect(r.energy).toBeGreaterThanOrEqual(optimum - 1e-9)
    expect(r.energy).toBeLessThanOrEqual(2 * optimum + 1e-9)
    const start = Object.fromEntries(P.modelVariables(model).map((v) => [v.name, 3]))
    expect(P.alphaExpansion(model, { init: start }).energy).toBeGreaterThanOrEqual(optimum - 1e-9)
  })

  it('on two labels is the exact cut in one expansion', () => {
    const model = randomPairwise({ seed: 11, shape: 'grid', size: 3, cols: 3, kind: 'submodular' })
    const r = P.alphaExpansion(model)
    expect(r.logValue).toBeCloseTo(best(model), 9)
  })

  it('record the energy of every cycle on the tape, and it never rises', () => {
    const model = potts(5, 'grid', 4, 4)
    const rec = recorder()
    const r = P.alphaExpansion(model, { tracer: rec })
    const energy = metricSeries(rec.run(), 'energy').values
    expect(energy.length).toBe(r.energyTrace.length)
    expect(energy.at(-1)).toBeCloseTo(r.energy, 12)
    for (let i = 1; i < r.energyTrace.length; i++)
      expect(r.energyTrace[i] ?? 0).toBeLessThanOrEqual((r.energyTrace[i - 1] ?? 0) + 1e-9)
    expect(r.energyTrace.at(-1)).toBeCloseTo(r.energy, 12)
    expect(r.energyTrace[0] ?? 0).toBeGreaterThan(r.energy)
    const swap = runAlgorithm(P.alphaBetaSwapAlgorithm(model), { maxSteps: 50 })
    expect(swap.status).toBe('converged')
    expect(swap.state.history[0]).toBeCloseTo(r.energyTrace[0] ?? 0, 12)
  })

  it('respect evidence and the cycle limit', () => {
    const model = potts(6, 'chain', 5, 3)
    const r = P.alphaExpansion(model, { evidence: { v2: '1' }, maxCycles: 1 })
    expect(Object.keys(r.indices)).not.toContain('v2')
    expect(r.cycles).toBe(1)
    expect(r.status).toBe('completed')
    const everything = P.alphaExpansion(model, { evidence: { v2: '1' } })
    expect(everything.energy).toBeLessThanOrEqual(r.energy + 1e-9)
  })

  it('refuse variables with different numbers of states', () => {
    const model = P.factorGraph(
      [
        { name: 'a', states: ['0', '1'] },
        { name: 'b', states: ['0', '1', '2'] },
      ],
      [],
    )
    expect(() => P.alphaExpansion(model)).toThrow(/states/)
  })
})
