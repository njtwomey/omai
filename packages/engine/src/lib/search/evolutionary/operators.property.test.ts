import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import {
  arithmeticCrossover,
  bitFlipMutation,
  blendCrossover,
  cycleCrossover,
  gaussianMutation,
  inversionMutation,
  onePointCrossover,
  orderCrossover,
  pmxCrossover,
  polynomialMutation,
  rankSelection,
  rouletteSelection,
  scrambleMutation,
  simulatedBinaryCrossover,
  stochasticUniversalSampling,
  swapMutation,
  tournamentSelection,
  truncationSelection,
  twoPointCrossover,
  uniformCrossover,
} from '../index.ts'

const perm = fc
  .integer({ min: 1, max: 12 })
  .chain((n) =>
    fc.tuple(
      fc.shuffledSubarray([...Array(n).keys()], { minLength: n, maxLength: n }),
      fc.shuffledSubarray([...Array(n).keys()], { minLength: n, maxLength: n }),
    ),
  )
const sorted = (p: readonly number[]): number[] => [...p].sort((a, b) => a - b)
const isPerm = (p: readonly number[], n: number): boolean => sorted(p).join() === [...Array(n).keys()].join()

describe('variation operators', () => {
  it('permutation crossovers and mutations return permutations', () => {
    fc.assert(
      fc.property(perm, fc.nat(1000), ([a, b], seed) => {
        const k = key(seed)
        for (const [c, d] of [orderCrossover(k, a, b), pmxCrossover(k, a, b), cycleCrossover(a, b)]) {
          expect(isPerm(c, a.length)).toBe(true)
          expect(isPerm(d, a.length)).toBe(true)
        }
        for (const m of [swapMutation(k, a), inversionMutation(k, a), scrambleMutation(k, a)])
          expect(isPerm(m, a.length)).toBe(true)
        // Cycle crossover keeps every element at a position it holds in a parent.
        const [c] = cycleCrossover(a, b)
        c.forEach((x, i) => {
          expect(x === a[i] || x === b[i]).toBe(true)
        })
      }),
    )
  })

  it('string crossovers exchange material position by position', () => {
    const bits = fc
      .integer({ min: 1, max: 20 })
      .chain((n) =>
        fc.tuple(
          fc.array(fc.integer({ min: 0, max: 1 }), { minLength: n, maxLength: n }),
          fc.array(fc.integer({ min: 0, max: 1 }), { minLength: n, maxLength: n }),
        ),
      )
    fc.assert(
      fc.property(bits, fc.nat(1000), ([a, b], seed) => {
        const k = key(seed)
        for (const [c, d] of [onePointCrossover(k, a, b), twoPointCrossover(k, a, b), uniformCrossover(k, a, b)]) {
          c.forEach((x, i) => {
            expect([x, d[i]].sort()).toEqual([a[i], b[i]].sort())
          })
        }
        expect(bitFlipMutation(k, a, 0)).toEqual(a)
        expect(bitFlipMutation(k, a, 1)).toEqual(a.map((x) => 1 - x))
      }),
    )
  })

  it('real crossovers and mutations respect their definitions and bounds', () => {
    const vec = fc
      .integer({ min: 1, max: 6 })
      .chain((n) =>
        fc.tuple(
          fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: n, maxLength: n }),
          fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: n, maxLength: n }),
        ),
      )
    fc.assert(
      fc.property(vec, fc.nat(1000), ([a, b], seed) => {
        const k = key(seed)
        const [s1, s2] = simulatedBinaryCrossover(k, a, b)
        s1.forEach((x, i) => {
          expect(x + (s2[i] ?? 0)).toBeCloseTo((a[i] ?? 0) + (b[i] ?? 0), 10)
        })
        const [r1, r2] = arithmeticCrossover(k, a, b)
        r1.forEach((x, i) => {
          expect(x + (r2[i] ?? 0)).toBeCloseTo((a[i] ?? 0) + (b[i] ?? 0), 10)
          expect(x).toBeGreaterThanOrEqual(Math.min(a[i] ?? 0, b[i] ?? 0) - 1e-12)
        })
        const [bl] = blendCrossover(k, a, b, 0)
        bl.forEach((x, i) => {
          expect(x).toBeGreaterThanOrEqual(Math.min(a[i] ?? 0, b[i] ?? 0) - 1e-12)
          expect(x).toBeLessThanOrEqual(Math.max(a[i] ?? 0, b[i] ?? 0) + 1e-12)
        })
        const lower = a.map(() => 0)
        const upper = a.map(() => 1)
        for (const m of [
          polynomialMutation(k, a, lower, upper, { rate: 1 }),
          gaussianMutation(k, a, { sigma: 3, lower, upper }),
        ]) {
          for (const x of m) {
            expect(x).toBeGreaterThanOrEqual(0)
            expect(x).toBeLessThanOrEqual(1)
          }
        }
      }),
    )
  })
})

describe('selection', () => {
  it('returns valid indices, deterministic by key, favouring the fit', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 10, noNaN: true }), { minLength: 1, maxLength: 20 }),
        fc.nat(1000),
        (f, seed) => {
          const k = key(seed)
          for (const pick of [
            tournamentSelection(k, f, 30),
            rouletteSelection(k, f, 30),
            stochasticUniversalSampling(k, f, 30),
            rankSelection(k, f, 30),
          ]) {
            expect(pick.length).toBe(30)
            for (const i of pick) expect(i >= 0 && i < f.length).toBe(true)
          }
          expect(tournamentSelection(k, f, 10)).toEqual(tournamentSelection(key(seed), f, 10))
          const top = truncationSelection(f, 1)[0] ?? 0
          for (const x of f) expect(f[top] ?? 0).toBeGreaterThanOrEqual(x)
        },
      ),
    )
  })

  it('samples roulette and SUS in proportion to fitness', () => {
    const f = [1, 2, 7]
    for (const pick of [rouletteSelection(key(5), f, 10_000), stochasticUniversalSampling(key(5), f, 10_000)]) {
      const share = pick.filter((i) => i === 2).length / pick.length
      expect(share).toBeCloseTo(0.7, 1)
    }
    const sus = stochasticUniversalSampling(key(1), f, 10)
    expect(sus.filter((i) => i === 2).length).toBe(7)
  })

  it('rejects invalid input', () => {
    expect(() => rouletteSelection(key(0), [-1, 2], 1)).toThrow()
    expect(() => tournamentSelection(key(0), [], 1)).toThrow()
    expect(() => rankSelection(key(0), [1], 1, 3)).toThrow()
    expect(() => truncationSelection([1], 2)).toThrow()
    expect(() => orderCrossover(key(0), [0, 0], [0, 1])).toThrow()
    expect(() => onePointCrossover(key(0), [0], [0, 1])).toThrow()
    expect(() => polynomialMutation(key(0), [0.5], [], [])).toThrow()
  })
})
