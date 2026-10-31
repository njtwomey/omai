import { describe, expect, it } from 'vitest'
import { key, uniformStream } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  groundMln,
  groundQuery,
  type MarkovLogicNetwork,
  mlnCounts,
  mlnExact,
  mlnGibbs,
  mlnGibbsAlgorithm,
  mlnLogWeight,
  mlnMap,
  mlnMcSat,
  mlnMcSatAlgorithm,
  sampleSat,
  withWeights,
  worldVector,
} from './index.ts'

/** The classic smokers-and-friends network. */
const smokers = (domain: string[]): MarkovLogicNetwork => ({
  domain,
  formulas: [
    { weight: 1.1, formula: 'friends(X,Y) & smokes(X) -> smokes(Y)' },
    { weight: 1.5, formula: 'smokes(X) -> cancer(X)' },
    { weight: -0.4, formula: 'friends(X,Y)' },
    { weight: -0.2, formula: 'smokes(X)' },
  ],
})

describe('grounding', () => {
  it('has an atom per predicate and tuple, and a ground formula per assignment of the free variables', () => {
    const g = groundMln(smokers(['anna', 'bob', 'carl']))
    expect(g.atoms.length).toBe(9 + 3 + 3)
    expect(g.atoms).toContain('friends(anna,bob)')
    expect(g.formulas.length).toBe(9 + 3 + 9 + 3)
    expect(g.weights).toEqual([1.1, 1.5, -0.4, -0.2])
  })

  it('expands quantifiers and decides equality', () => {
    const g = groundMln({
      domain: ['a', 'b'],
      formulas: [{ weight: 1, formula: 'all x. (p(x) -> exists y. (q(x,y) & x != y))' }],
    })
    expect(g.formulas.length).toBe(1)
    const w = (atoms: Record<string, boolean>): boolean => g.formulas[0]?.holds(worldVector(g, atoms)) ?? false
    expect(w({})).toBe(true)
    expect(w({ 'p(a)': true })).toBe(false)
    expect(w({ 'p(a)': true, 'q(a,b)': true })).toBe(true)
    expect(w({ 'p(a)': true, 'q(a,a)': true })).toBe(false)
  })

  it('rejects bad domains and inconsistent arities', () => {
    expect(() => groundMln({ domain: ['A'], formulas: [] })).toThrow()
    expect(() => groundMln({ domain: ['a', 'a'], formulas: [] })).toThrow()
    expect(() => groundMln({ domain: ['a'], formulas: [{ weight: 1, formula: 'p(X) & p(X,X)' }] })).toThrow()
  })

  it('counts true groundings and sums weights', () => {
    const g = groundMln(smokers(['anna', 'bob']))
    const w = worldVector(g, { 'smokes(anna)': true, 'cancer(anna)': true, 'friends(anna,bob)': true })
    // Implication groundings: friends(x,y)&smokes(x)->smokes(y) fails only for (anna,bob).
    expect(mlnCounts(g, w)).toEqual([3, 2, 1, 1])
    expect(mlnLogWeight(g, w)).toBeCloseTo(1.1 * 3 + 1.5 * 2 - 0.4 - 0.2, 12)
  })
})

describe('exact inference', () => {
  it('a single weighted atom has marginal sigmoid(w)', () => {
    const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
    const e = mlnExact(g)
    expect(e.marginals['p(a)']).toBeCloseTo(0.75, 12)
    expect(e.logPartition).toBeCloseTo(Math.log(1 + 3), 12)
  })

  it('matches a hand computation for two atoms and an implication', () => {
    const g = groundMln({ domain: ['a'], formulas: [{ weight: 2, formula: 'p(a) -> q(a)' }] })
    // Worlds (p,q): the implication fails only at (1,0).
    const z = 3 * Math.exp(2) + 1
    const e = mlnExact(g, { queries: ['p(a) & q(a)', 'p(a)'] })
    expect(e.logPartition).toBeCloseTo(Math.log(z), 12)
    expect(e.queries[0]).toBeCloseTo(Math.exp(2) / z, 12)
    expect(e.queries[1]).toBeCloseTo((Math.exp(2) + 1) / z, 12)
    expect(e.expectedCounts[0]).toBeCloseTo((3 * Math.exp(2)) / z, 12)
  })

  it('conditions on evidence, and respects hard formulas', () => {
    const g = groundMln({
      domain: ['a'],
      formulas: [
        { weight: Number.POSITIVE_INFINITY, formula: 'p(a) -> q(a)' },
        { weight: 1, formula: 'p(a)' },
      ],
    })
    expect(mlnExact(g).marginals['p(a)']).toBeCloseTo(Math.E / (Math.E + 1 + 1), 12)
    expect(mlnExact(g, { evidence: { 'p(a)': true } }).marginals['q(a)']).toBe(1)
    expect(() => mlnExact(g, { evidence: { 'p(a)': true, 'q(a)': false } })).toThrow()
  })

  it('a ground query needs ground atoms', () => {
    const g = groundMln(smokers(['anna']))
    expect(() => groundQuery(g, 'smokes(X)')).toThrow()
  })
})

describe('Gibbs sampling', () => {
  const g = groundMln(smokers(['anna', 'bob', 'carl']))
  const queries = ['smokes(anna) & cancer(anna)', 'friends(anna,bob) & smokes(anna) & smokes(bob)']
  const evidence = { 'friends(anna,bob)': true, 'smokes(anna)': true }

  it('matches the exact marginals within Monte Carlo error', () => {
    const exact = mlnExact(g, { queries })
    const est = mlnGibbs(g, { samples: 12000, seed: 5, queries })
    for (const a of g.atoms)
      expect(Math.abs((est.marginals[a] ?? 0) - (exact.marginals[a] as number))).toBeLessThan(0.03)
    expect(Math.abs((est.queries[0] as number) - (exact.queries[0] as number))).toBeLessThan(0.03)
  }, 60_000)

  it('conditions on evidence', () => {
    const exact = mlnExact(g, { evidence, queries })
    const est = mlnGibbs(g, { samples: 8000, seed: 2, evidence, queries, scan: 'random' })
    expect(est.marginals['smokes(anna)']).toBe(1)
    for (const a of g.atoms)
      expect(Math.abs((est.marginals[a] ?? 0) - (exact.marginals[a] as number))).toBeLessThan(0.04)
    expect(Math.abs((est.queries[1] as number) - (exact.queries[1] as number))).toBeLessThan(0.04)
  }, 60_000)

  it('puts the marginals, queries and energy on the tape, and its state round-trips', () => {
    const alg = mlnGibbsAlgorithm(g, { queries, burnIn: 5 })
    const rec = recorder()
    const run = runAlgorithm(alg, { seed: 1, maxSteps: 20, tracer: rec })
    expect(run.steps).toBe(20)
    const state = run.state
    expect(state.samples).toBe(15)
    const back = alg.deserialise?.(alg.serialise?.(state) as never)
    expect(Array.from(back?.world ?? [])).toEqual(Array.from(state.world))
    expect(Array.from(back?.atomCounts ?? [])).toEqual(Array.from(state.atomCounts))
    const names = JSON.stringify(rec.run())
    for (const p of ['mln/marginals', 'mln/queries', 'mln/energy', 'mln/world']) expect(names).toContain(p)
  })

  it('a resumed run continues like an uninterrupted one', () => {
    const alg = mlnGibbsAlgorithm(g, { burnIn: 2 })
    const whole = runAlgorithm(alg, { seed: 3, maxSteps: 12 }).state
    const half = runAlgorithm(alg, { seed: 3, maxSteps: 6 }).state
    const rest = runAlgorithm(alg, {
      seed: 3,
      maxSteps: 6,
      start: { state: alg.serialise?.(half), step: 6, serialised: true },
    }).state
    expect(Array.from(rest.atomCounts)).toEqual(Array.from(whole.atomCounts))
  })
})

describe('MC-SAT', () => {
  const small = groundMln(smokers(['anna', 'bob']))
  const exact = mlnExact(small, { queries: ['smokes(bob) & cancer(bob)'] })

  it('with an exactly sampled slice matches the exact marginals', () => {
    const est = mlnMcSat(small, { sampler: 'exact', samples: 6000, seed: 4, queries: ['smokes(bob) & cancer(bob)'] })
    for (const a of small.atoms)
      expect(Math.abs((est.marginals[a] ?? 0) - (exact.marginals[a] as number))).toBeLessThan(0.03)
    expect(Math.abs((est.queries[0] as number) - (exact.queries[0] as number))).toBeLessThan(0.03)
  }, 60_000)

  it('with SampleSAT matches the exact marginals within a looser tolerance', () => {
    const est = mlnMcSat(small, { samples: 6000, seed: 4, queries: ['smokes(bob) & cancer(bob)'] })
    for (const a of small.atoms)
      expect(Math.abs((est.marginals[a] ?? 0) - (exact.marginals[a] as number))).toBeLessThan(0.05)
  }, 60_000)

  it('conditions on evidence and handles hard formulas', () => {
    const hard = groundMln({
      domain: ['anna', 'bob'],
      formulas: [
        { weight: Number.POSITIVE_INFINITY, formula: 'smokes(X) -> cancer(X)' },
        { weight: 0.8, formula: 'friends(X,Y) & smokes(X) -> smokes(Y)' },
        { weight: 0.5, formula: 'smokes(X)' },
        { weight: -0.5, formula: 'friends(X,Y)' },
      ],
    })
    const evidence = { 'friends(anna,bob)': true }
    const truth = mlnExact(hard, { evidence })
    const est = mlnMcSat(hard, { sampler: 'exact', samples: 5000, seed: 8, evidence })
    for (const a of hard.atoms)
      expect(Math.abs((est.marginals[a] ?? 0) - (truth.marginals[a] as number))).toBeLessThan(0.04)
    // The hard formula is never violated by a sampled world.
    const run = runAlgorithm(mlnMcSatAlgorithm(hard, { burnIn: 0, evidence }), { seed: 1, maxSteps: 30 })
    expect(run.state.samples).toBe(30)
    expect(mlnLogWeight(hard, run.state.world)).toBeGreaterThan(Number.NEGATIVE_INFINITY)
  }, 60_000)

  it('SampleSAT returns models of satisfiable clause sets and nothing for unsatisfiable ones', () => {
    const u = uniformStream(key(0))
    const clauses = [
      [1, 2, 3],
      [-1, -2],
      [-2, -3],
      [-1, -3],
      [2, 3],
    ]
    for (let i = 0; i < 20; i++) {
      const m = sampleSat(clauses, 3, u)
      expect(m).not.toBeNull()
      expect(clauses.every((c) => c.some((l) => (m?.[Math.abs(l) - 1] === 1) === l > 0))).toBe(true)
    }
    expect(sampleSat([[1], [-1]], 1, u, { maxFlips: 200 })).toBeNull()
  })

  it('SampleSAT spreads over the models: exactly-one-of-four is close to uniform', () => {
    const u = uniformStream(key(9))
    const clauses = [
      [1, 2, 3, 4],
      [-1, -2],
      [-1, -3],
      [-1, -4],
      [-2, -3],
      [-2, -4],
      [-3, -4],
    ]
    const hits = [0, 0, 0, 0]
    const n = 2000
    for (let i = 0; i < n; i++) {
      const m = sampleSat(clauses, 4, u) as Uint8Array
      hits[m.indexOf(1)] = (hits[m.indexOf(1)] as number) + 1
    }
    for (const h of hits) expect(Math.abs(h / n - 0.25)).toBeLessThan(0.05)
  })
})

describe('MAP inference', () => {
  it('equals the enumerated maximum of the log weight', () => {
    const g = groundMln(smokers(['anna', 'bob']))
    const evidence = { 'friends(anna,bob)': true, 'smokes(anna)': true }
    let best = Number.NEGATIVE_INFINITY
    const ev = worldVector(g, evidence)
    const free = g.atoms.map((_a, i) => i).filter((i) => !((g.atoms[i] as string) in evidence))
    for (let m = 0; m < 2 ** free.length; m++) {
      const w = Uint8Array.from(ev)
      free.forEach((i, j) => {
        w[i] = (m >> j) & 1
      })
      best = Math.max(best, mlnLogWeight(g, w))
    }
    const r = mlnMap(g, { evidence })
    expect(r.status).toBe('optimal')
    expect(r.logWeight).toBeCloseTo(best, 10)
    expect(r.world?.['smokes(anna)']).toBe(true)
    expect(r.world?.['cancer(anna)']).toBe(true)
  })

  it('handles negative weights, hard formulas and compound formulas', () => {
    const g = groundMln({
      domain: ['a', 'b'],
      formulas: [
        { weight: Number.POSITIVE_INFINITY, formula: 'p(X) -> q(X)' },
        { weight: 2, formula: '(p(X) & -q(X)) | r(X)' },
        { weight: -1.5, formula: 'r(X)' },
        { weight: 1, formula: 'p(a) <-> p(b)' },
      ],
    })
    let best = Number.NEGATIVE_INFINITY
    for (let m = 0; m < 2 ** g.atoms.length; m++) {
      const w = Uint8Array.from({ length: g.atoms.length }, (_, i) => (m >> i) & 1)
      best = Math.max(best, mlnLogWeight(g, w))
    }
    expect(mlnMap(g).logWeight).toBeCloseTo(best, 10)
  })

  it('reweighting keeps the grounding', () => {
    const g = groundMln(smokers(['anna']))
    const h = withWeights(g, [0, 0, 0, 3])
    expect(mlnExact(h).marginals['smokes(anna)']).toBeCloseTo(1 / (1 + Math.exp(-3)), 10)
    expect(() => withWeights(g, [1])).toThrow()
  })
})
