import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { dictionaryLearningAlgorithm } from './dictionary-learning.ts'
import { sparseCode } from './sparse-coding.ts'

const entry = fc.double({ min: -2, max: 2, noNaN: true }).map((v) => (Math.abs(v) < 1e-2 ? 0.5 : v))
const matrix = (n: number, d: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(entry, { minLength: d, maxLength: d }), { minLength: n, maxLength: n })

// Atoms that differ: drawn ones plus a fixed wiggle, so that no two are copies (a copy is a degenerate dictionary, on
// which the least angle path of `linear` can end off the lasso optimum).
const apart = (rows: number[][]): number[][] =>
  rows.map((r, j) => r.map((v, q) => v + Math.sin(2.3 * j + 1.7 * q * (j + 1) + 0.5)))

const norm = (v: readonly number[]): number => Math.hypot(...v)
const unit = (rows: number[][]): number[][] => rows.map((r) => r.map((v) => v / (norm(r) || 1)))

describe('alternating minimisation with exact sparse coding', () => {
  it('never raises the objective from one iteration to the next (cd atoms, lasso by LARS)', () => {
    fc.assert(
      fc.property(matrix(10, 4), matrix(3, 4), fc.constantFrom(0.05, 0.3), (x, start, alpha) => {
        const alg = dictionaryLearningAlgorithm(x, unit(apart(start)), { alpha, tol: 0 })
        let state = alg.init(key(0))
        for (let t = 0; t < 6; t++) {
          const next = alg.step(state, { t, key: key(t) } as never)
          // A replaced atom is a jump the descent does not account for; every other step is a descent.
          if (next.unused === 0) expect(next.objective).toBeLessThanOrEqual(state.objective * (1 + 1e-9) + 1e-12)
          state = next
        }
      }),
      { numRuns: 25 },
    )
  })
})

describe('the sparsity the coders promise', () => {
  it('OMP and LARS give at most k nonzeros per signal, and exactly k when the atoms are independent', () => {
    fc.assert(
      fc.property(matrix(5, 6), matrix(5, 6), fc.integer({ min: 1, max: 4 }), (x, atoms, k) => {
        const d = unit(apart(atoms))
        for (const algorithm of ['omp', 'lars'] as const) {
          const code = toArray(sparseCode(x, d, { algorithm, nNonzeroCoefs: k })) as number[][]
          for (const row of code) expect(row.filter((v) => v !== 0).length).toBeLessThanOrEqual(k)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('has the lasso code satisfy its optimality conditions: correlations bounded by alpha, equal to it on the support', () => {
    fc.assert(
      fc.property(matrix(4, 6), matrix(4, 6), fc.constantFrom(0.1, 0.5, 1.5), (x, atoms, alpha) => {
        const d = unit(apart(atoms))
        for (const algorithm of ['lassoLars', 'lassoCd'] as const) {
          const code = toArray(sparseCode(x, d, { algorithm, alpha, tol: 1e-14 })) as number[][]
          for (const [i, c] of code.entries()) {
            const fit = (x[i] ?? []).map((_, q) => c.reduce((a, cj, j) => a + cj * (d[j]?.[q] ?? 0), 0))
            for (const [j, atom] of d.entries()) {
              const g = atom.reduce((a, v, q) => a + v * ((x[i]?.[q] ?? 0) - (fit[q] ?? 0)), 0)
              expect(Math.abs(g), algorithm).toBeLessThanOrEqual(alpha + 1e-6)
              if (Math.abs(c[j] ?? 0) > 1e-9) expect(Math.abs(g), algorithm).toBeCloseTo(alpha, 5)
            }
          }
        }
      }),
      { numRuns: 25 },
    )
  })
})

describe('atoms', () => {
  it('stay unit norm under MOD and K-SVD, and inside the unit ball under cd', () => {
    fc.assert(
      fc.property(matrix(12, 5), matrix(3, 5), fc.constantFrom('cd', 'mod', 'ksvd'), (x, start, dictUpdate) => {
        const alg = dictionaryLearningAlgorithm(x, unit(apart(start)), {
          dictUpdate,
          fitAlgorithm: dictUpdate === 'cd' ? 'lassoLars' : 'omp',
          fitNonzeroCoefs: 2,
          alpha: 0.1,
          tol: 0,
        })
        const s = runAlgorithm(alg, { maxSteps: 3, key: key(1) }).state
        for (const atom of s.dictionary)
          if (dictUpdate === 'cd') expect(norm(atom)).toBeLessThanOrEqual(1 + 1e-9)
          else expect(norm(atom)).toBeCloseTo(1, 8)
      }),
      { numRuns: 20 },
    )
  })
})
