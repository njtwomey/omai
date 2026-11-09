import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { betaDivergence, miniBatchNmf, miniBatchNmfAlgorithm, solveWeights } from './minibatch-nmf.ts'
import { nmf } from './nmf.ts'

// Two parts with disjoint supports: rows are mixtures of (1, 2, 0, 0) and (0, 0, 3, 1).
const weights = Array.from({ length: 24 }, (_, i) => [1 + (i % 5), 1 + ((i * 3) % 4)])
const x = weights.map(([a = 0, b = 0]) => [a, 2 * a, 3 * b, b])

describe('mini-batch NMF', () => {
  it('reconstructs the data and keeps every factor non-negative', () => {
    const f = miniBatchNmf.with({ nComponents: 2, batchSize: 6, maxIter: 300, tol: 1e-6 }).fit({ x })
    expect(f.params.reconstructionErr / Math.sqrt(x.flat().reduce((a, v) => a + v * v, 0))).toBeLessThan(0.05)
    for (const row of [...f.params.components, ...f.params.weights])
      for (const v of row) expect(v).toBeGreaterThanOrEqual(0)
    const back = toArray(f.inverseTransform(f.transform(x))) as number[][]
    for (const [i, r] of back.entries()) for (const [j, v] of r.entries()) expect(v).toBeCloseTo(x[i]?.[j] ?? 0, 0)
  })

  it('a batch as large as the data is a pass of the multiplicative updates (forgetting aside, the first step is the same)', () => {
    const start = {
      w: weights.map(() => [1, 1]),
      h: [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
      ],
    }
    const one = runAlgorithm(
      miniBatchNmfAlgorithm(x, start.w, start.h, { batchSize: 24, tol: 0, maxNoImprovement: null }),
      {
        maxSteps: 1,
      },
    ).state
    const reference = runAlgorithm(
      // The same W update as NMF's multiplicative rule; H then averages with its own start, so only W is compared.
      miniBatchNmfAlgorithm(x, start.w, start.h, { batchSize: 24, tol: 0, maxNoImprovement: null, forgetFactor: 0 }),
      { maxSteps: 1 },
    ).state
    expect(one.w).toEqual(reference.w)
    expect(one.step).toBe(1)
  })

  it('observes the batch cost and stops by the rules it names', () => {
    const f = miniBatchNmf.with({ nComponents: 2, batchSize: 6, tol: 1e-3, maxIter: 500 }).fit({ x }, { logs: 'all' })
    expect(f.params.nSteps).toBeLessThan(500 * 4)
    const s = miniBatchNmf.with({ nComponents: 2, batchSize: 6 }).session({ x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect(s.fittedAt(3).params.nSteps).toBe(3)
    expect((s.probe(3, 'model/embedding') as number[][]).length).toBe(24)
  })

  it('the state serialises and a run resumes from a checkpoint', () => {
    const start = {
      w: weights.map(() => [1, 1]),
      h: [
        [1, 2, 1, 1],
        [1, 1, 2, 1],
      ],
    }
    const alg = miniBatchNmfAlgorithm(x, start.w, start.h, { batchSize: 8, tol: 0 })
    const mid = runAlgorithm(alg, { maxSteps: 5 }).state
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(mid))))).toEqual(mid)
  })

  it('the beta divergence is zero at a perfect factorisation and the transform solves the weights', () => {
    expect(betaDivergence([[2, 4]], [[2]], [[1, 2]], 'frobenius')).toBeCloseTo(0, 12)
    expect(betaDivergence([[2, 4]], [[2]], [[1, 2]], 'kullback-leibler')).toBeCloseTo(0, 10)
    expect(betaDivergence([[2, 4]], [[2]], [[1, 2]], 'itakura-saito')).toBeCloseTo(0, 10)
    expect(
      solveWeights(
        [
          [2, 4],
          [3, 6],
        ],
        [[1, 2]],
        { maxIter: 300, tol: 1e-12 },
      ).map((r) => r[0] ?? 0)[1],
    ).toBeCloseTo(3, 4)
  })

  it('refuses negative data, and zeros with a beta of at most zero', () => {
    expect(() =>
      miniBatchNmf.fit({
        x: [
          [1, -1],
          [2, 1],
        ],
      }),
    ).toThrow()
    expect(() => miniBatchNmfAlgorithm([[0, 1]], [[1]], [[1, 1]], { betaLoss: 'itakura-saito' })).toThrow()
  })

  it('agrees with plain NMF in kind: both reach a similar error on the same data', () => {
    const batch = miniBatchNmf
      .with({ nComponents: 2, batchSize: 24, maxIter: 400, tol: 0, maxNoImprovement: null })
      .fit({ x })
    const plain = nmf.with({ nComponents: 2, maxIter: 400, tol: 0 }).fit({ x })
    expect(batch.params.reconstructionErr).toBeLessThan(plain.params.reconstructionErr * 3 + 0.5)
  })
})
