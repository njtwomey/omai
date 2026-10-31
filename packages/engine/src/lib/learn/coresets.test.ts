import { describe, expect, it } from 'vitest'
import { key, normal, permutation } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  bayesianCoreset,
  bayesianCoresetAlgorithm,
  herdingAlgorithm,
  kCenterGreedy,
  kCenterGreedyAlgorithm,
  kernelHerding,
  leverageScoreCoreset,
  sensitivitySampling,
} from './coresets.ts'

describe('coresets', () => {
  it('kCenterGreedy traverses farthest points first with monotonically non-increasing covering radius', () => {
    const rec = recorder()
    const pts = [[0], [1], [5], [10], [9]]
    const r = kCenterGreedy(pts, 3, { seedIndex: 0, tracer: rec })
    expect(r.indices).toEqual([0, 3, 2])
    expect(r.radii[0]).toBe(10)
    expect(r.radii[1]).toBe(5)
    expect(r.radii[2]).toBe(1)
    expect(r.radius).toBe(1)

    const alg = kCenterGreedyAlgorithm(pts, 3, { seedIndex: 0 })
    const res = runAlgorithm(alg, { maxSteps: 2 })
    const restored = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(res.state))))
    expect(restored?.selected).toEqual([0, 3])
  })

  it('kernelHerding reports the RKHS MMD of its picks, checked against a direct computation, and beats random subsets', () => {
    const z = toFlat(normal(key(7), [40, 2]))
    const pts = Array.from({ length: 40 }, (_, i) => [z[2 * i] ?? 0, (z[2 * i + 1] ?? 0) * 0.5 + (i % 2) * 3])
    const gamma = 0.4
    const at = (p: readonly number[] | undefined, j: number): number => p?.[j] ?? 0
    const kern = (a: number[] | undefined, b: number[] | undefined): number =>
      Math.exp(-gamma * ((at(a, 0) - at(b, 0)) ** 2 + (at(a, 1) - at(b, 1)) ** 2))
    const mmdOf = (idx: readonly number[]): number => {
      let kk = 0
      let cross = 0
      let within = 0
      for (const i of pts) for (const j of pts) kk += kern(i, j) / pts.length ** 2
      for (const a of idx) for (const j of pts) cross += kern(pts[a], j) / (idx.length * pts.length)
      for (const a of idx) for (const b of idx) within += kern(pts[a], pts[b]) / idx.length ** 2
      return Math.sqrt(Math.max(0, kk - 2 * cross + within))
    }
    const r = kernelHerding(pts, 8, { gamma })
    expect(r.indices.length).toBe(8)
    expect(new Set(r.indices).size).toBe(8)
    expect(r.mmd).toBeCloseTo(mmdOf(r.indices), 10)
    for (const t of [0, 3, 7]) expect(r.mmds[t]).toBeCloseTo(mmdOf(r.indices.slice(0, t + 1)), 10)
    // Herding is a greedy minimiser of the MMD: it is far below the average random subset of the same size.
    let randomMean = 0
    for (let rep = 0; rep < 30; rep++) {
      randomMean += mmdOf(toFlat(permutation(key(100 + rep), 40)).slice(0, 8)) / 30
    }
    expect(r.mmd).toBeLessThan(0.5 * randomMean)

    const alg = herdingAlgorithm(pts, 3, { gamma })
    const res = runAlgorithm(alg, { maxSteps: 2 })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(res.state)))
    expect(alg.deserialise?.(json)?.selected).toEqual(res.state.selected)
  })

  it('kCenterGreedy is a 2-approximation of the best covering radius, and covers around fixed centres', () => {
    for (let seed = 0; seed < 6; seed++) {
      const z = toFlat(normal(key(seed), [9, 2]))
      const pts = Array.from({ length: 9 }, (_, i) => [z[2 * i] ?? 0, z[2 * i + 1] ?? 0])
      const k = 3
      const radiusOf = (centres: readonly number[]): number =>
        Math.max(
          ...pts.map((p) =>
            Math.min(
              ...centres.map((c) => Math.hypot((p[0] ?? 0) - (pts[c]?.[0] ?? 0), (p[1] ?? 0) - (pts[c]?.[1] ?? 0))),
            ),
          ),
        )
      let best = Number.POSITIVE_INFINITY
      for (let a = 0; a < 9; a++)
        for (let b = a + 1; b < 9; b++) for (let c = b + 1; c < 9; c++) best = Math.min(best, radiusOf([a, b, c]))
      const r = kCenterGreedy(pts, k)
      expect(r.radius).toBeCloseTo(radiusOf(r.indices), 12)
      expect(r.radius).toBeLessThanOrEqual(2 * best + 1e-12)
      expect(r.radii).toEqual([...r.radii].sort((a, b) => b - a))
    }
    // Fixed centres outside x are covered first and never returned.
    const covered = kCenterGreedy([[0], [1], [5], [10]], 2, { centres: [[0]] })
    expect(covered.indices).toEqual([3, 2])
    expect(covered.radius).toBe(1)
  })

  it('leverageScoreCoreset and sensitivitySampling assign higher probabilities to high-leverage and outlier points', () => {
    const design = [
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 10],
    ]
    const lev = leverageScoreCoreset(design, 2, { deterministic: true })
    expect(lev.indices[0]).toBe(4)
    expect(lev.scores[4]).toBeCloseTo(1, 10)
    expect(lev.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    // Leverage scores of a full-rank design sum to its rank.
    expect(lev.scores.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 10)
    expect(lev.scores[0]).toBeCloseTo(0.25, 10)

    // The importance weights make the coreset's weighted sum unbiased for the full sum of any function of the rows.
    const f = [3, 1, 4, 1, 5]
    let mean = 0
    const reps = 400
    for (let r = 0; r < reps; r++) {
      const c = leverageScoreCoreset(design, 3, { key: key(1000 + r) })
      mean += c.indices.reduce((a, idx, t) => a + (c.weights[t] ?? 0) * (f[idx] ?? 0), 0) / reps
    }
    expect(mean).toBeGreaterThan(0.9 * 14)
    expect(mean).toBeLessThan(1.1 * 14)

    const sampledLev = leverageScoreCoreset(design, 3, { lambda: 0.1, key: key(2) })
    expect(sampledLev.indices.length).toBe(3)
    expect(sampledLev.weights.length).toBe(3)

    const clusterWithOutlier = [[0], [0.1], [-0.1], [0.05], [50]]
    const sens = sensitivitySampling(clusterWithOutlier, 2, { k: 1, deterministic: true })
    expect(sens.indices[0]).toBe(4)
    expect(sensitivitySampling(clusterWithOutlier, 1, { k: 1, deterministic: true }).indices).toEqual([4])
  })

  it('bayesianCoreset shrinks the log-likelihood error monotonically with a sparse nonnegative support', () => {
    const z = toFlat(normal(key(3), [60, 12]))
    const L = Array.from({ length: 60 }, (_, i) => Array.from({ length: 12 }, (_, j) => 1 + (z[i * 12 + j] ?? 0)))
    const total = Array.from({ length: 12 }, (_, j) => L.reduce((a, row) => a + (row[j] ?? 0), 0))
    const errors = [1, 2, 4, 8, 16].map((size) => bayesianCoreset(L, size).relativeError)
    for (let i = 1; i < errors.length; i++) expect(errors[i]).toBeLessThanOrEqual((errors[i - 1] ?? 0) + 1e-12)
    expect(errors[4]).toBeLessThan(0.5 * (errors[0] ?? 0))
    const bc = bayesianCoreset(L, 8)
    expect(bc.support.length).toBeLessThanOrEqual(8)
    expect(bc.weights.every((w) => w >= 0)).toBe(true)
    // The reported error is the actual error of the weighted sum.
    const approx = total.map((_, j) => bc.weights.reduce((a, w, i) => a + w * (L[i]?.[j] ?? 0), 0))
    const resid = Math.hypot(...approx.map((a, j) => a - (total[j] ?? 0)))
    expect(resid).toBeCloseTo(bc.residualNorm, 10)
    expect(resid / Math.hypot(...total)).toBeCloseTo(bc.relativeError, 10)

    // Two distinct atoms: the first step gets close, and the rest is the Frank-Wolfe rate.
    const exact = bayesianCoreset(
      [
        [2, 0],
        [2, 0],
        [0, 3],
        [0, 3],
      ],
      12,
    )
    expect(exact.relativeError).toBeLessThan(5e-3)

    const alg = bayesianCoresetAlgorithm(L, 2, { centre: true })
    const res = runAlgorithm(alg, { maxSteps: 1 })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(res.state)))
    expect(alg.deserialise?.(json)?.step).toBe(1)
  })
})
