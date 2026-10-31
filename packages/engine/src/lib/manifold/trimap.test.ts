import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { swissRoll } from '../datasets/index.ts'
import { smoothSolvers } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { spearman } from '../stats/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { nestedCircles, sphereClusters, twoScaleClusters } from './_planted.ts'
import { pcaStart } from './_support.ts'
import { trustworthiness } from './quality.ts'
import { temperedLog, trimap, trimapAlgorithm, trimapLoss, trimapProblem, trimapTriplets } from './trimap.ts'
import { umap } from './umap.ts'

// Ten unit-spaced points on a line: the scales and scaled distances are easy to write down independently.
const line = Array.from({ length: 10 }, (_, i) => [i])
const gap = (i: number, j: number): number => Math.abs(i - j)
/** sigma_i: the mean distance to the 4th to 6th nearest other row, by brute force. */
const sigma = (i: number): number => {
  const d = line
    .map((_, j) => gap(i, j))
    .filter((_, j) => j !== i)
    .sort((a, b) => a - b)
  return (d[3] ?? 0) / 3 + (d[4] ?? 0) / 3 + (d[5] ?? 0) / 3
}
const scaled = (i: number, j: number): number => gap(i, j) ** 2 / (sigma(i) * sigma(j))

describe('TriMap triplets', () => {
  it('the scales on the line are what the definition gives (row 0: 4, 5, 6; row 5: 2, 3, 3)', () => {
    expect(sigma(0)).toBeCloseTo(5, 14)
    expect(sigma(5)).toBeCloseTo(8 / 3, 14)
  })

  it('has n nInliers nOutliers + n nRandom triplets; j is a scaled-nearest neighbour; k is not i or an inlier; random ones are ordered', () => {
    const { triplets } = trimapTriplets(line, key(0), { nInliers: 2, nOutliers: 1, nRandom: 2 })
    expect(triplets.length).toBe(10 * 2 * 1 + 10 * 2)
    for (let i = 0; i < 10; i++) {
      const mine = triplets.filter((t) => t[0] === i)
      const inliers = [...Array(10).keys()]
        .filter((j) => j !== i)
        .sort((a, b) => scaled(i, a) - scaled(i, b) || a - b)
        .slice(0, 2)
      const outlier = mine.slice(0, 2)
      expect(outlier.map((t) => t[1])).toEqual(inliers)
      for (const [, , k] of outlier) {
        expect(k).not.toBe(i)
        expect(inliers).not.toContain(k)
      }
      for (const [, j, k] of mine.slice(2)) {
        expect(j).not.toBe(i)
        expect(k).not.toBe(i)
        expect(j).not.toBe(k)
        expect(scaled(i, j)).toBeLessThanOrEqual(scaled(i, k))
      }
    }
  })

  it('weights are the tempered log of 1 + 500 times the similarity ratio over its largest, computed independently here', () => {
    const adj = 500
    for (const temp of [1, 0.5]) {
      const { triplets, weights } = trimapTriplets(line, key(2), {
        nInliers: 2,
        nOutliers: 1,
        nRandom: 1,
        weightAdj: adj,
        weightTemp: temp,
      })
      const logRatio = triplets.map(([i, j, k]) => scaled(i, k) - scaled(i, j))
      const top = Math.max(...logRatio)
      const expected = logRatio.map((l) => {
        const x = 1 + adj * Math.exp(l - top)
        return temp === 1 ? Math.log(x) : (Math.sqrt(x) - 1) / 0.5
      })
      for (const [t, w] of weights.entries()) expect(w).toBeCloseTo(expected[t] ?? Number.NaN, 12)
      // the triplet with the largest ratio has the largest weight, exactly the tempered log of 501
      expect(Math.max(...weights)).toBeCloseTo(temperedLog(1 + adj, temp), 12)
      const order = [...logRatio.keys()].sort((a, b) => (logRatio[a] ?? 0) - (logRatio[b] ?? 0))
      for (let t = 1; t < order.length; t++)
        expect(weights[order[t] ?? 0] ?? 0).toBeGreaterThanOrEqual(weights[order[t - 1] ?? 0] ?? 0)
    }
    expect(temperedLog(Math.E, 1)).toBeCloseTo(1, 14)
    expect(temperedLog(9, 0.5)).toBeCloseTo(4, 14)
  })

  it('is deterministic from a key; the key changes outliers and random triplets, not the inliers', () => {
    const a = trimapTriplets(line, key(0), { nInliers: 2, nOutliers: 1, nRandom: 2 })
    expect(trimapTriplets(line, key(0), { nInliers: 2, nOutliers: 1, nRandom: 2 })).toEqual(a)
    const b = trimapTriplets(line, key(1), { nInliers: 2, nOutliers: 1, nRandom: 2 })
    expect(b.triplets.map((t) => [t[0], t[1]])).not.toEqual([])
    expect(b.triplets).not.toEqual(a.triplets)
    expect(() => trimapTriplets(line.slice(0, 5), key(0))).toThrow()
    expect(() => trimapTriplets(line, key(0), { nInliers: 9 })).toThrow(/nInliers/)
  })
})

describe('TriMap loss', () => {
  it('equals the hand-computed triplet loss, counts violations, and the gradient matches finite differences', () => {
    // y_i = (0,0), y_j = (1,0), y_k = (0,2), y_l = (0.5, 0): d_ij = 2, d_ik = 5, d_il = 1.25
    const y = [0, 0, 1, 0, 0, 2, 0.5, 0]
    const triplets = [
      [0, 1, 2],
      [0, 2, 3],
    ] as [number, number, number][]
    const w = [2, 0.5]
    const r = trimapLoss(y, 2, triplets, w)
    // second triplet: d_ij = 5 (to y2), d_ik = 1.25 (to y3): a violation
    expect(r.loss).toBeCloseTo((2 * 2) / 7 + (0.5 * 5) / 6.25, 14)
    expect(r.violations).toBe(1)
    const h = 1e-6
    for (let q = 0; q < y.length; q++) {
      const up = y.map((v, i) => (i === q ? v + h : v))
      const dn = y.map((v, i) => (i === q ? v - h : v))
      const fd = (trimapLoss(up, 2, triplets, w).loss - trimapLoss(dn, 2, triplets, w).loss) / (2 * h)
      expect(r.grad[q]).toBeCloseTo(fd, 8)
    }
    // an embedding that has j closer than k costs less than half the weight
    expect(trimapLoss([0, 1, 3], 1, [[0, 1, 2]], [1]).loss).toBeLessThan(0.5)
    expect(trimapLoss([0, 3, 1], 1, [[0, 1, 2]], [1]).loss).toBeGreaterThan(0.5)
  })
})

describe('TriMap algorithm', () => {
  const p = sphereClusters(key(2), 12)
  const { triplets, weights } = trimapTriplets(p.x, key(0), { nInliers: 5, nOutliers: 3, nRandom: 2 })
  const start = pcaStart(p.x, 2, 0.01)

  it('lowers the loss and the violations, keeps the embedding on the tape, and completes after nIters', () => {
    const f = trimap.with({ nInliers: 5, nOutliers: 3, nIters: 120 }).fit(p, { key: key(0), logs: 'all' })
    const loss = metricSeries(f.run, 'solver/loss').values
    expect(loss.length).toBe(121)
    expect(loss.at(-1) ?? 0).toBeLessThan((loss[0] ?? 0) / 10)
    expect(f.params.objective).toBeCloseTo(loss.at(-1) ?? 0, 12)
    expect(f.params.embedding.length).toBe(48)
    expect(metricSeries(f.run, 'solver/lr').values.length).toBe(121)
  })

  it('is deterministic from a key and the key matters', () => {
    const run = (k: number) => trimap.with({ nInliers: 5, nIters: 20 }).fit(p, { key: key(k) }).params.embedding
    expect(run(0)).toEqual(run(0))
    expect(run(1)).not.toEqual(run(0))
  })

  it('serialises its state, and a stored state resumes to the same run', () => {
    const alg = trimapAlgorithm(triplets, weights, start, { nIters: 12 })
    const whole = runAlgorithm(alg, { maxSteps: 12, key: key(0) })
    const half = runAlgorithm(alg, { maxSteps: 5, key: key(0) })
    expect(stateFromJson(stateToJson(half.state))).toEqual(half.state)
    const resumed = runAlgorithm(alg, {
      maxSteps: 12,
      key: key(0),
      start: { state: stateToJson(half.state), step: 5, serialised: true },
    })
    expect(resumed.state.y).toEqual(whole.state.y)
  })

  it('the gains follow delta-bar-delta: unit at the start, +0.2 when the descent continues, x0.8 when it flips, floored at 0.01', () => {
    const alg = trimapAlgorithm(triplets, weights, start, { nIters: 30 })
    const s0 = runAlgorithm(alg, { maxSteps: 0, key: key(0) }).state
    expect(s0.gain.flat().every((g) => g === 1)).toBe(true)
    // one step from a state with velocity: +0.2 where the velocity and the gradient point opposite ways, x0.8 where they
    // agree (the descent overshot), floored at 0.01
    const a = runAlgorithm(alg, { maxSteps: 4, key: key(0) }).state
    const b = runAlgorithm(alg, { maxSteps: 5, key: key(0) }).state
    let grew = 0
    let shrank = 0
    for (const [i, row] of b.gain.entries())
      for (const [q, g] of row.entries()) {
        const before = a.gain[i]?.[q] ?? 1
        const opposite = (a.vel[i]?.[q] ?? 0) > 0 !== (a.grad[i * 2 + q] ?? 0) > 0
        expect(g).toBeCloseTo(Math.max(opposite ? before + 0.2 : before * 0.8, 0.01), 12)
        if (opposite) grew++
        else shrank++
      }
    expect(grew).toBeGreaterThan(0)
    expect(shrank).toBeGreaterThan(0)
    const later = runAlgorithm(alg, { maxSteps: 30, key: key(0) }).state
    expect(Math.min(...later.gain.flat())).toBeGreaterThanOrEqual(0.01)
  })

  it('the optimiser is a slot: delta-bar-delta is the reference, an optim optimiser and the smooth solvers also lower the loss', () => {
    const base = trimapLoss(start.flat(), 2, triplets, weights).loss
    const adam = trimap
      .with({ nInliers: 5, nOutliers: 3, nRandom: 2, optimiser: 'adam', learningRate: 0.1, nIters: 100 })
      .fit(p, { key: key(0) })
    expect(adam.params.objective).toBeLessThan(base / 5)
    const inner = smoothSolvers.make(trimapProblem(triplets, weights, start), { kind: 'lbfgs' }, key(0))
    const out = runAlgorithm(inner, { maxSteps: 200, key: key(0) })
    const x = (out.state as { x: Float64Array }).x
    expect(trimapLoss(x, 2, triplets, weights).loss).toBeLessThan(base / 5)
  })
})

describe('TriMap on planted manifolds', () => {
  it('lays out a swiss roll with trustworthiness above 0.9', () => {
    const roll = generate(swissRoll, { key: key(1), knobs: { n: 300 } }).data
    const f = trimap.with({}).fit({ x: roll.x }, { key: key(0) })
    // measured 0.907 with delta-bar-delta, 0.916 to 0.922 with other optimisers and settings: the triplets pair each row's
    // neighbours with random outliers only, so the sampled roll's local order is kept less tightly than PaCMAP keeps it
    const t = trustworthiness(roll.x, f.params.embedding, 10)
    expect(t).toBeGreaterThan(0.9)
    expect(t).toBeGreaterThan(
      trustworthiness(
        roll.x,
        Array.from({ length: 300 }, (_, i) => [
          (Math.sin(i * 12.9898) * 43758.5453) % 1,
          (Math.sin(i * 78.233) * 12345.6789) % 1,
        ]),
        10,
      ) + 0.3,
    )
  })

  it('separates four clusters on a sphere and keeps two nested circles as rings', () => {
    const s = sphereClusters(key(2), 30)
    const fs = trimap.with({}).fit({ x: s.x }, { key: key(0) })
    expect(trustworthiness(s.x, fs.params.embedding, 10)).toBeGreaterThan(0.93)
    const c = nestedCircles(key(3), 80)
    const fc = trimap.with({}).fit({ x: c.x }, { key: key(0) })
    expect(trustworthiness(c.x, fc.params.embedding, 10)).toBeGreaterThan(0.97)
  })

  it('preserves the layout of cluster centres better than UMAP (the global-structure claim of the paper)', () => {
    const p = twoScaleClusters(key(4), 20)
    const centreCorrelation = (emb: number[][]): number => {
      const centres = p.centres.map((_, c) => {
        const rows = emb.filter((_, i) => p.labels[i] === c)
        return [0, 1].map((q) => rows.reduce((a, r) => a + (r[q] ?? 0), 0) / rows.length)
      })
      const d = (a: number[], b: number[]): number => Math.hypot(...a.map((v, q) => v - (b[q] ?? 0)))
      const truth: number[] = []
      const got: number[] = []
      for (let i = 0; i < centres.length; i++)
        for (let j = i + 1; j < centres.length; j++) {
          truth.push(d(p.centres[i] as number[], p.centres[j] as number[]))
          got.push(d(centres[i] as number[], centres[j] as number[]))
        }
      return spearman(truth, got).statistic
    }
    const seeds = [0, 1, 2]
    const mean = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length
    const tri = mean(
      seeds.map((s) => centreCorrelation(trimap.with({}).fit({ x: p.x }, { key: key(s) }).params.embedding)),
    )
    const uma = mean(
      seeds.map((s) =>
        centreCorrelation(
          umap.with({ nNeighbors: 10, nEpochs: 200 }).fit({ x: p.x }, { key: key(s) }).params.embedding,
        ),
      ),
    )
    // measured over these seeds: TriMap 0.43, UMAP -0.09 (no graph edge between the two groups, so their placement is arbitrary)
    expect(tri).toBeGreaterThan(0.3)
    expect(tri).toBeGreaterThan(uma + 0.3)
  })
})
