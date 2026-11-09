import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { sparseCodedSignal } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { type SparseCodeAlgorithm, sparseCode, sparseCoder, splitSign, synthesise } from './sparse-coding.ts'

const planted = generate(sparseCodedSignal, {
  key: key(4),
  knobs: { n: 30, components: 8, dim: 40, nonzeros: 2, noise: 0 },
})
const truth = planted.truth.extra?.['dictionary'] as number[][]
const codes = toArray(planted.data.y) as number[][]
const x = toArray(planted.data.x) as number[][]

describe('sparseCode', () => {
  it('recovers the planted codes of easy signals: exactly by OMP and LARS, to the penalty by the lassos', () => {
    for (const algorithm of ['omp', 'lars'] as const) {
      const c = toArray(sparseCode(x, truth, { algorithm, nNonzeroCoefs: 2 })) as number[][]
      for (const [i, row] of c.entries())
        for (const [j, v] of row.entries()) expect(v).toBeCloseTo(codes[i]?.[j] ?? 0, 8)
    }
    for (const algorithm of ['lassoLars', 'lassoCd'] as const) {
      const c = toArray(sparseCode(x, truth, { algorithm, alpha: 0.01, tol: 1e-14 })) as number[][]
      for (const [i, row] of c.entries())
        for (const [j, v] of row.entries()) expect(v).toBeCloseTo(codes[i]?.[j] ?? 0, 1)
    }
  })

  it('has the two lasso solvers agree on one objective', () => {
    const a = toFlat(sparseCode(x, truth, { algorithm: 'lassoLars', alpha: 0.5 }))
    const b = toFlat(sparseCode(x, truth, { algorithm: 'lassoCd', alpha: 0.5, tol: 1e-14 }))
    for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? 0, 7)
  })

  it('thresholds the correlations, positively when asked', () => {
    const d = [
      [1, 0],
      [0, 1],
    ]
    expect(toArray(sparseCode([[3, -0.5]], d, { algorithm: 'threshold', alpha: 1 }))).toEqual([[2, 0]])
    expect(toArray(sparseCode([[3, -4]], d, { algorithm: 'threshold', alpha: 1, positive: true }))).toEqual([[2, 0]])
  })

  it('solves the non-negative lasso by coordinate descent (the soft threshold on an orthonormal dictionary)', () => {
    const d = [
      [1, 0],
      [0, 1],
    ]
    const c = toArray(
      sparseCode([[3, -4]], d, { algorithm: 'lassoCd', alpha: 1, positive: true, tol: 1e-14 }),
    ) as number[][]
    expect(c[0]?.[0]).toBeCloseTo(2, 12)
    expect(c[0]?.[1]).toBe(0)
  })

  it('refuses a positive code from the coders that cannot make one, and bad shapes', () => {
    for (const algorithm of ['omp', 'lars', 'lassoLars'] as const)
      expect(() => sparseCode(x, truth, { algorithm, positive: true })).toThrow(/positive/)
    expect(() =>
      sparseCode(
        [[1, 2, 3]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow()
  })

  it("uses scikit-learn's default number of atoms, a tenth of the features (at least one)", () => {
    for (const algorithm of ['omp', 'lars'] as SparseCodeAlgorithm[]) {
      const c = toArray(sparseCode(x, truth, { algorithm })) as number[][]
      for (const row of c) expect(row.filter((v) => v !== 0).length).toBeLessThanOrEqual(4)
    }
  })

  it('splits signs and writes the signals back', () => {
    expect(toArray(splitSign([[2, -3]]))).toEqual([[2, 0, 0, 3]])
    expect(
      toArray(
        synthesise(
          [[2, 0, 0, 3]],
          [
            [1, 0],
            [0, 1],
          ],
          true,
        ),
      ),
    ).toEqual([[2, -3]])
  })
})

describe('sparseCoder', () => {
  it('is a transformer over a fixed dictionary: transform, split sign, inverse transform', () => {
    const coder = sparseCoder({ dictionary: truth, transformNonzeroCoefs: 2 }).fit({ x })
    const z = coder.transform(x)
    expect(z.shape).toEqual([30, 8])
    const back = toFlat(coder.inverseTransform(z))
    const flat = x.flat()
    for (const [i, v] of back.entries()) expect(v).toBeCloseTo(flat[i] ?? 0, 8)
    const split = sparseCoder({ dictionary: truth, transformNonzeroCoefs: 2, splitSign: true }).fit({ x })
    expect(split.transform(x).shape).toEqual([30, 16])
    expect(toFlat(split.transform(x)).every((v) => v >= 0)).toBe(true)
    for (const [i, v] of toFlat(split.inverseTransform(split.transform(x))).entries())
      expect(v).toBeCloseTo(flat[i] ?? 0, 8)
    expect(coder.nFeaturesIn).toBe(40)
  })

  it('revives from its parameters', () => {
    const est = sparseCoder({
      dictionary: [
        [1, 0],
        [0, 1],
      ],
      transformAlgorithm: 'threshold',
      transformAlpha: 1,
    })
    const fitted = est.fit({ x: [[3, 0.5]] })
    expect(toArray(est.revive(fitted.params).transform([[3, 0.5]]))).toEqual([[2, 0]])
  })
})
