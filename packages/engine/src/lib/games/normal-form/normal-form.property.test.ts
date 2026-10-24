import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import {
  bimatrixGame,
  correlatedEquilibrium,
  distinctEquilibria,
  expectedPayoffs,
  isCorrelatedEquilibrium,
  isNash,
  iteratedElimination,
  jointDistribution,
  lemkeHowson,
  type NormalFormGame,
  payoffMatrices,
  pureNashEquilibria,
  randomGame,
  solveZeroSum,
  supportEnumeration,
  vertexEnumeration,
} from '../index.ts'

const RUNS = 40

/** Generic games: independent uniform payoffs (non-degenerate with probability one). */
const shape = fc.tuple(fc.integer({ min: 1, max: 4 }), fc.integer({ min: 1, max: 4 }))
const generic: fc.Arbitrary<NormalFormGame> = fc
  .tuple(shape, fc.integer({ min: 0, max: 2 ** 31 - 1 }))
  .map(([[m, n], seed]) => randomGame(key(seed), [m, n], { low: -1, high: 1 }))
const matrix = (m: number, n: number): fc.Arbitrary<number[][]> =>
  fc
    .integer({ min: 0, max: 2 ** 31 - 1 })
    .map((seed) => payoffMatrices(randomGame(key(seed), [m, n], { low: -1, high: 1 }))[0])
/** Small integer payoffs: degenerate games with ties and continua of equilibria are common. */
const degenerate: fc.Arbitrary<NormalFormGame> = shape.chain(([m, n]) => {
  const small = fc.array(fc.array(fc.integer({ min: 0, max: 2 }), { minLength: n, maxLength: n }), {
    minLength: m,
    maxLength: m,
  })
  return fc.tuple(small, small).map(([a, b]) => bimatrixGame(a, b))
})

const same = (a: readonly (readonly number[])[], b: readonly (readonly number[])[], tol: number): boolean =>
  a.every((s, p) => s.every((v, k) => Math.abs(v - (b[p]?.[k] ?? Number.NaN)) <= tol))

describe('two-player equilibria', () => {
  it('support enumeration returns best responses, an odd number of them, equal to the vertex enumeration set', () => {
    fc.assert(
      fc.property(generic, (game) => {
        const se = supportEnumeration(game)
        expect(se.length % 2).toBe(1)
        for (const e of se) expect(isNash(game, e.strategies, 1e-9)).toBe(true)
        const ve = vertexEnumeration(game)
        expect(ve.length).toBe(se.length)
        for (const e of ve) expect(se.some((f) => same(e.strategies, f.strategies, 1e-7))).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('Lemke-Howson reaches one of the enumerated equilibria from every label', () => {
    fc.assert(
      fc.property(generic, (game) => {
        const se = supportEnumeration(game)
        const labels = (game.shape[0] ?? 0) + (game.shape[1] ?? 0)
        for (let k = 0; k < labels; k++) {
          const r = lemkeHowson(game, { initialLabel: k })
          expect(r.status).toBe('equilibrium')
          expect(se.some((f) => same(r.strategies, f.strategies, 1e-7))).toBe(true)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('on degenerate games every method terminates with best responses', () => {
    fc.assert(
      fc.property(degenerate, (game) => {
        const ve = vertexEnumeration(game)
        expect(ve.length).toBeGreaterThan(0)
        for (const coarse of [false, true]) {
          const ce = correlatedEquilibrium(game, { coarse })
          expect(isCorrelatedEquilibrium(game, ce.distribution, { coarse, tol: 1e-9 })).toBe(true)
        }
        for (const e of [...ve, ...supportEnumeration(game)]) expect(isNash(game, e.strategies, 1e-9)).toBe(true)
        for (let k = 0; k < (game.shape[0] ?? 0) + (game.shape[1] ?? 0); k++) {
          const r = lemkeHowson(game, { initialLabel: k, maxPivots: 200 })
          expect(r.status).toBe('equilibrium')
          expect(isNash(game, r.strategies, 1e-9)).toBe(true)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('every pure equilibrium is among the enumerated ones', () => {
    fc.assert(
      fc.property(generic, (game) => {
        const se = supportEnumeration(game)
        for (const p of pureNashEquilibria(game))
          expect(se.some((f) => same(p.strategies, f.strategies, 1e-9))).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('distinctEquilibria removes the repeats from all Lemke-Howson labels', () => {
    fc.assert(
      fc.property(generic, (game) => {
        const all = Array.from({ length: (game.shape[0] ?? 0) + (game.shape[1] ?? 0) }, (_, k) =>
          lemkeHowson(game, { initialLabel: k }),
        )
        const d = distinctEquilibria(all)
        for (const [i, e] of d.entries())
          for (const f of d.slice(i + 1)) expect(same(e.strategies, f.strategies, 1e-8)).toBe(false)
      }),
      { numRuns: RUNS },
    )
  })
})

describe('zero-sum games', () => {
  it('the LP value equals the payoff of every Nash equilibrium, with a zero duality gap', () => {
    fc.assert(
      fc.property(
        shape.chain(([m, n]) => matrix(m, n)),
        (a) => {
          const r = solveZeroSum(a)
          expect(Math.abs(r.gap)).toBeLessThan(1e-9)
          const game = bimatrixGame(a)
          expect(isNash(game, [r.row, r.column], 1e-9)).toBe(true)
          for (const e of supportEnumeration(game)) expect(e.payoffs[0]).toBeCloseTo(r.value, 8)
        },
      ),
      { numRuns: RUNS },
    )
  })
})

describe('correlated equilibria', () => {
  it('the welfare optimum is a correlated equilibrium at least as good as every Nash equilibrium', () => {
    fc.assert(
      fc.property(generic, (game) => {
        const ce = correlatedEquilibrium(game)
        expect(isCorrelatedEquilibrium(game, ce.distribution, { tol: 1e-9 })).toBe(true)
        const cce = correlatedEquilibrium(game, { coarse: true })
        expect(cce.value).toBeGreaterThanOrEqual(ce.value - 1e-9)
        for (const e of supportEnumeration(game)) {
          const p = jointDistribution(game, e.strategies)
          expect(isCorrelatedEquilibrium(game, p, { tol: 1e-9 })).toBe(true)
          expect(ce.value).toBeGreaterThanOrEqual((e.payoffs[0] ?? 0) + (e.payoffs[1] ?? 0) - 1e-9)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('works for three players', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 2 ** 31 - 1 }), (seed) => {
        const game = randomGame(key(seed), [2, 2, 2])
        const ce = correlatedEquilibrium(game, { objective: 'egalitarian' })
        expect(isCorrelatedEquilibrium(game, ce.distribution, { tol: 1e-9 })).toBe(true)
        for (const p of pureNashEquilibria(game)) expect(isNash(game, p.strategies)).toBe(true)
      }),
      { numRuns: 15 },
    )
  })
})

describe('near ties', () => {
  // fast-check's integers favour the ends of their range, so these payoffs often tie to within 1e-6: the linear
  // programs must still solve.
  const close = fc.integer({ min: 0, max: 1_000_000 }).map((v) => v / 1_000_000 - 0.5 + v * 1e-13)
  const nearTies: fc.Arbitrary<number[][][]> = shape.chain(([m, n]) => {
    const mat = fc.array(fc.array(close, { minLength: n, maxLength: n }), { minLength: m, maxLength: m })
    return fc.tuple(mat, mat)
  })
  it('equilibrium programs solve on nearly tied payoffs', () => {
    fc.assert(
      fc.property(nearTies, ([a, b]) => {
        const game = bimatrixGame(a ?? [], b)
        for (const coarse of [false, true]) {
          const ce = correlatedEquilibrium(game, { coarse })
          expect(isCorrelatedEquilibrium(game, ce.distribution, { coarse, tol: 1e-7 })).toBe(true)
        }
        expect(Math.abs(solveZeroSum(a ?? []).gap)).toBeLessThan(1e-7)
        iteratedElimination(game, { mixed: true })
      }),
      { numRuns: RUNS },
    )
  })
})

describe('dominance', () => {
  it('iterated strict elimination keeps every Nash equilibrium', () => {
    fc.assert(
      fc.property(generic, fc.boolean(), (game, mixed) => {
        const r = iteratedElimination(game, { mixed })
        for (const e of supportEnumeration(game))
          for (const [p, s] of e.strategies.entries())
            for (const [a, v] of s.entries()) if (v > 1e-9) expect(r.actions[p]).toContain(a)
        // The reduced game's equilibria are the original's.
        for (const e of supportEnumeration(r.game)) {
          const lifted = e.strategies.map((s, p) => {
            const full = new Array<number>(game.shape[p] ?? 0).fill(0)
            for (const [k, v] of s.entries()) full[r.actions[p]?.[k] ?? 0] = v
            return full
          })
          expect(isNash(game, lifted, 1e-9)).toBe(true)
          expect(expectedPayoffs(game, lifted)[0]).toBeCloseTo(e.payoffs[0] ?? 0, 9)
        }
      }),
      { numRuns: RUNS },
    )
  })
})
