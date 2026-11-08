import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray, toFlat } from '../tensor/index.ts'
import {
  convolutionalAnalysis,
  convolutionalDictionaryLearning,
  convolutionalSparseCode,
  convolutionalSynthesis,
} from './convolutional-sparse.ts'

const value = fc.double({ min: -3, max: 3, noNaN: true, noDefaultInfinity: true })
const rows = (k: number, n: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(value, { minLength: n, maxLength: n }), { minLength: k, maxLength: k })
const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0)
  return s
}

// A shape (filters, length, map length) and data to go with it.
const problem = fc
  .record({ K: fc.integer({ min: 1, max: 3 }), L: fc.integer({ min: 1, max: 5 }), M: fc.integer({ min: 1, max: 9 }) })
  .chain(({ K, L, M }) =>
    fc.record({
      K: fc.constant(K),
      L: fc.constant(L),
      M: fc.constant(M),
      D: rows(K, L),
      Z: rows(K, M),
      r: fc.array(value, { minLength: M + L - 1, maxLength: M + L - 1 }),
    }),
  )

describe('convolutional synthesis and analysis', () => {
  it('make a signal of M + L - 1 samples, and analysis is its adjoint: <A z, r> = <z, A^T r>', () => {
    fc.assert(
      fc.property(problem, ({ K, L, M, D, Z, r }) => {
        const signal = toFlat(convolutionalSynthesis(D, Z))
        expect(signal.length).toBe(M + L - 1)
        const adjoint = toArray(convolutionalAnalysis(D, r)) as number[][]
        expect(adjoint.length).toBe(K)
        expect(adjoint[0]?.length).toBe(M)
        const lhs = dot(signal, r)
        const rhs = dot(Z.flat(), adjoint.flat())
        expect(lhs).toBeCloseTo(rhs, 8)
      }),
      { numRuns: 60 },
    )
  })

  it('are linear in the maps', () => {
    fc.assert(
      fc.property(problem, fc.double({ min: -2, max: 2, noNaN: true }), ({ D, Z }, c) => {
        const a = toFlat(convolutionalSynthesis(D, Z))
        const b = toFlat(
          convolutionalSynthesis(
            D,
            Z.map((row) => row.map((v) => c * v)),
          ),
        )
        for (const [i, v] of a.entries()) expect(b[i]).toBeCloseTo(c * v, 9)
      }),
      { numRuns: 40 },
    )
  })
})

describe('convolutional sparse coding', () => {
  it('has the code zero once lambda exceeds the largest correlation', () => {
    fc.assert(
      fc.property(problem, ({ D, r }) => {
        const top = Math.max(0, ...toFlat(convolutionalAnalysis(D, r)).map(Math.abs))
        fc.pre(top > 1e-3)
        expect(convolutionalSparseCode(D, r, { lambda: top * 1.01 }).nonzeros).toBe(0)
        // Below it the code is not zero (the first step of the proximal gradient already moves where the correlation is
        // above lambda). The number of nonzeros is not monotone in lambda in general, so nothing is claimed about it.
        expect(convolutionalSparseCode(D, r, { lambda: top * 0.5, maxSteps: 50 }).nonzeros).toBeGreaterThan(0)
      }),
      { numRuns: 25 },
    )
  })

  it('does not increase the objective from step to step (FISTA with the 1 / 1.1 L step, ISTA exactly)', () => {
    fc.assert(
      fc.property(problem, ({ D, r }) => {
        fc.pre(D.some((row) => row.some((v) => Math.abs(v) > 0.1)))
        const objective = (steps: number): number =>
          convolutionalSparseCode(D, r, { lambda: 0.2, accelerate: false, maxSteps: steps }).objective
        let previous = objective(0)
        for (const t of [1, 3, 8]) {
          const next = objective(t)
          expect(next).toBeLessThanOrEqual(previous + 1e-9)
          previous = next
        }
      }),
      { numRuns: 25 },
    )
  })
})

describe('convolutional dictionary learning', () => {
  it('keeps every filter in the unit ball and never increases the objective', () => {
    fc.assert(
      fc.property(
        fc.array(value, { minLength: 24, maxLength: 40 }),
        fc.integer({ min: 1, max: 2 }),
        fc.integer({ min: 2, max: 5 }),
        (y, K, L) => {
          fc.pre(y.some((v) => Math.abs(v) > 0.2))
          const r = convolutionalDictionaryLearning(y, {
            filters: K,
            length: L,
            lambda: 0.3,
            maxSteps: 4,
            codeSteps: 10,
            filterSteps: 15,
          })
          for (let i = 1; i < r.history.length; i++)
            expect(r.history[i] as number).toBeLessThanOrEqual((r.history[i - 1] as number) * (1 + 1e-6) + 1e-9)
          for (const f of toArray(r.filters) as number[][]) expect(Math.hypot(...f)).toBeLessThanOrEqual(1 + 1e-9)
          expect(toFlat(r.codes).length).toBe(K * (y.length - L + 1))
        },
      ),
      { numRuns: 15 },
    )
  })
})
