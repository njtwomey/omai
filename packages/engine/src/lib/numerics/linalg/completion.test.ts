import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as L from './index.ts'

/** A rank-r matrix with a share of its entries observed (noise-free), from a seed. */
function problem(seed: number, m = 14, n = 12, r = 2, share = 0.6): { obs: L.ObservedEntries; truth: number[] } {
  const a = toFlat(normal(key(seed), [m, r]))
  const b = toFlat(normal(key(seed + 1), [n, r]))
  const draw = toFlat(uniform(key(seed + 2), [m, n]))
  const truth = new Array<number>(m * n).fill(0)
  const rows: number[] = []
  const cols: number[] = []
  const values: number[] = []
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let v = 0
      for (let c = 0; c < r; c++) v += (a[i * r + c] ?? 0) * (b[j * r + c] ?? 0)
      truth[i * n + j] = v
      if ((draw[i * n + j] ?? 1) < share) {
        rows.push(i)
        cols.push(j)
        values.push(v)
      }
    }
  return { obs: { shape: [m, n], rows, cols, values }, truth }
}

const rmse = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Math.sqrt(
    Array.from({ length: a.length }, (_, i) => ((a[i] ?? 0) - (b[i] ?? 0)) ** 2).reduce((x, y) => x + y, 0) / a.length,
  )

describe('alternating least squares', () => {
  it('recovers a planted low-rank matrix from a share of its entries', () => {
    const { obs, truth } = problem(1)
    const alg = L.alsCompletionAlgorithm(obs, { rank: 2, lambda: 1e-4 })
    const s = runAlgorithm(alg, { maxSteps: 60, key: key(7) }).state
    expect(s.rmse).toBeLessThan(0.02)
    expect(rmse(toFlat(L.factorsToMatrix(s, 2)), truth)).toBeLessThan(0.1)
  })

  it('never increases the objective, whatever the data and the start', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 50 }), fc.double({ min: 0.01, max: 2, noNaN: true }), (seed, lambda) => {
        const { obs } = problem(seed, 8, 6, 2, 0.5)
        const alg = L.alsCompletionAlgorithm(obs, { rank: 3, lambda })
        let state = alg.init(key(seed))
        let previous = state.objective
        for (let t = 0; t < 8; t++) {
          state = alg.step(state, undefined as never)
          expect(state.objective).toBeLessThanOrEqual(previous + 1e-9)
          previous = state.objective
        }
      }),
      { numRuns: 15 },
    )
  })

  it('serialises: pausing and resuming gives the same factors', () => {
    const { obs } = problem(3)
    const alg = L.alsCompletionAlgorithm(obs, { rank: 2 })
    const straight = runAlgorithm(alg, { maxSteps: 6, key: key(1) }).state
    const first = runAlgorithm(alg, { maxSteps: 2, key: key(1) })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
    const resumed = runAlgorithm(alg, {
      maxSteps: 4,
      key: key(1),
      start: { state: stored, step: 2, serialised: true },
    }).state
    expect(resumed.u).toEqual(straight.u)
    expect(resumed.v).toEqual(straight.v)
  })

  it('rejects entries outside the matrix and a rank below one', () => {
    expect(() => L.alsCompletionAlgorithm({ shape: [2, 2], rows: [2], cols: [0], values: [1] }, { rank: 1 })).toThrow(
      /outside/,
    )
    expect(() => L.alsCompletionAlgorithm({ shape: [2, 2], rows: [0], cols: [0], values: [1] }, { rank: 0 })).toThrow(
      /rank/,
    )
  })
})

describe('biased SGD factorisation', () => {
  it('fits the observed entries and generalises to held-out ones on a planted low-rank matrix with biases', () => {
    const { obs, truth } = problem(5, 20, 16, 2, 0.7)
    const shifted = { ...obs, values: Array.from(obs.values, (v, k) => v + 3 + 0.5 * ((obs.rows[k] ?? 0) % 3)) }
    const alg = L.sgdCompletionAlgorithm(shifted, { rank: 2, lambda: 0.01, learningRate: 0.03, decay: 0.98 })
    const s = runAlgorithm(alg, { maxSteps: 150, key: key(2) }).state
    expect(s.rmse).toBeLessThan(0.15)
    const first = alg.init(key(2))
    expect(s.objective).toBeLessThan(first.objective / 5)
    // The mean is the data's mean and the fit reproduces the shift in the unobserved cells.
    const prediction = toFlat(L.factorsToMatrix(s, 2))
    const shift = (i: number): number => 3 + 0.5 * (i % 3)
    const target = truth.map((v, idx) => v + shift(Math.floor(idx / 16)))
    expect(rmse(prediction, target)).toBeLessThan(0.6)
  })

  it('is deterministic in the key (the epoch order is drawn from it) and the order matters', () => {
    const { obs } = problem(6)
    const alg = L.sgdCompletionAlgorithm(obs, { rank: 2 })
    const a = runAlgorithm(alg, { maxSteps: 3, key: key(4) }).state
    const b = runAlgorithm(alg, { maxSteps: 3, key: key(4) }).state
    const c = runAlgorithm(alg, { maxSteps: 3, key: key(5) }).state
    expect(a.u).toEqual(b.u)
    expect(a.u).not.toEqual(c.u)
  })

  it('without biases the mean and biases stay zero', () => {
    const { obs } = problem(8)
    const s = runAlgorithm(L.sgdCompletionAlgorithm(obs, { rank: 2, biases: false }), {
      maxSteps: 3,
      key: key(1),
    }).state
    expect(s.mean).toBe(0)
    expect(s.rowBias.every((b) => b === 0)).toBe(true)
  })
})

describe('implicit-feedback ALS', () => {
  it('ranks the observed cells above unobserved ones and lowers its objective', () => {
    const { obs } = problem(9, 12, 10, 2, 0.3)
    const counts: L.ObservedEntries = { ...obs, values: Array.from(obs.values, (v) => 1 + Math.abs(Math.round(2 * v))) }
    const alg = L.implicitAlsAlgorithm(counts, { rank: 3, lambda: 0.1, alpha: 10 })
    const start = alg.init(key(1))
    const s = runAlgorithm(alg, { maxSteps: 12, key: key(1) }).state
    expect(s.objective).toBeLessThan(start.objective)
    const m = toFlat(L.factorsToMatrix(s, 3))
    const seen = new Set(Array.from(counts.rows, (r, k) => r * 10 + (counts.cols[k] ?? 0)))
    const observedMean = Array.from(seen, (idx) => m[idx] ?? 0).reduce((a, b) => a + b, 0) / seen.size
    const others = m.filter((_, idx) => !seen.has(idx))
    expect(observedMean).toBeGreaterThan(others.reduce((a, b) => a + b, 0) / others.length + 0.3)
  })

  it('rejects non-positive counts', () => {
    expect(() => L.implicitAlsAlgorithm({ shape: [1, 1], rows: [0], cols: [0], values: [0] }, { rank: 1 })).toThrow(
      /positive/,
    )
  })
})

describe('soft-impute', () => {
  it('returns the zero matrix at the largest penalty, and more rank as the penalty falls', () => {
    const { obs } = problem(11)
    const lambdaMax = L.softImputeLambdaMax(obs)
    const atMax = L.softImpute(obs, { lambda: lambdaMax * 1.0001 })
    expect(toFlat(atMax.completed).every((v) => Math.abs(v) < 1e-12)).toBe(true)
    const rank = (lambda: number): number => {
      const s = runAlgorithm(L.softImputeAlgorithm(obs, { lambda }), { maxSteps: 300, seed: 0 }).state
      return s.singularValues.length
    }
    expect(rank(lambdaMax * 0.5)).toBeLessThanOrEqual(rank(lambdaMax * 0.05))
  })

  it('lowers its objective at every step and recovers a planted low-rank matrix at a small penalty', () => {
    const { obs, truth } = problem(12, 16, 14, 2, 0.7)
    const alg = L.softImputeAlgorithm(obs, { lambda: 0.05, tol: 1e-12 })
    let state = alg.init(key(0))
    let previous = state.objective
    for (let t = 0; t < 40; t++) {
      state = alg.step(state, undefined as never)
      expect(state.objective).toBeLessThanOrEqual(previous + 1e-9)
      previous = state.objective
    }
    const result = L.softImpute(obs, { lambda: 0.05, maxIterations: 2000, tol: 1e-12 })
    expect(rmse(toFlat(result.completed), truth)).toBeLessThan(0.3)
    expect(result.history[result.history.length - 1] ?? 0).toBeLessThan(result.history[0] ?? 0)
  })

  it('serialises, including the infinite initial change', () => {
    const { obs } = problem(13)
    const alg = L.softImputeAlgorithm(obs, { lambda: 0.2 })
    const first = runAlgorithm(alg, { maxSteps: 2, seed: 0 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
    const straight = runAlgorithm(alg, { maxSteps: 5, seed: 0 }).state
    const resumed = runAlgorithm(alg, {
      maxSteps: 3,
      seed: 0,
      start: { state: stored, step: 2, serialised: true },
    }).state
    expect(resumed.z).toEqual(straight.z)
    const initial = JSON.parse(JSON.stringify(alg.serialise?.(alg.init(key(0)))))
    expect(alg.deserialise?.(initial)?.relativeChange).toBe(Number.POSITIVE_INFINITY)
  })
})
