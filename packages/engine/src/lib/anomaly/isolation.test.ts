import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { recordsOf } from '../exposition/index.ts'
import { key, normal } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  averagePathLength,
  ExtendedIsolationForest,
  IsolationForest,
  isolationForest,
  isolationForestAlgorithm,
  isolationPath,
  isolationTree,
  pathLength,
} from './isolation.ts'

const gaussian = (seed: number, n: number, d: number): number[][] => toArray(normal(key(seed), [n, d])) as number[][]

const std = (v: readonly number[]): number => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

describe('isolation trees', () => {
  it('has the average path length of a binary search tree', () => {
    expect([averagePathLength(0), averagePathLength(1), averagePathLength(2)]).toEqual([0, 1 * 0, 1])
    expect(averagePathLength(256)).toBeCloseTo(10.2448, 4)
    // c(3) = 2 H(2) - 2 * 2 / 3 = 2 * 1.5 - 4/3
    expect(averagePathLength(3)).toBeCloseTo(2 * (Math.log(2) + 0.5772156649015329) - 4 / 3, 12)
  })

  it('separates a far point in few splits and the path ends at the leaf the length counts', () => {
    const x = [...gaussian(1, 63, 2), [30, 30]]
    const rows = x.map((_, i) => i)
    const lengths = [0, 1, 2, 3, 4, 5, 6, 7].map((s) => {
      const tree = isolationTree(x, rows, key(s), { maxDepth: 6 })
      return { far: pathLength(tree, [30, 30]), near: pathLength(tree, [0, 0]), tree }
    })
    const mean = (f: (r: (typeof lengths)[number]) => number): number =>
      lengths.reduce((a, r) => a + f(r), 0) / lengths.length
    expect(mean((r) => r.far)).toBeLessThan(mean((r) => r.near) - 1)
    for (const { tree } of lengths) {
      const path = isolationPath(tree, [0, 0])
      const leaf = path.at(-1)
      expect(leaf?.leaf).toBe(true)
      expect(pathLength(tree, [0, 0])).toBeCloseTo((leaf?.depth ?? 0) + averagePathLength(leaf?.size ?? 0), 12)
      // Every split on the path sends the point the way its value says.
      for (const step of path.slice(0, -1)) expect(step.direction).toBe(step.value < step.threshold ? 'left' : 'right')
    }
  })

  it('grows a tree that holds every row once at the leaves', () => {
    const x = gaussian(2, 40, 3)
    const tree = isolationTree(
      x,
      x.map((_, i) => i),
      key(4),
      { maxDepth: 40 },
    )
    const leafSizes = tree.filter((n) => n.leaf).map((n) => (n.leaf ? n.size : 0))
    expect(leafSizes.reduce((a, b) => a + b, 0)).toBe(40)
    expect(tree.filter((n) => !n.leaf).length).toBe(tree.filter((n) => n.leaf).length - 1)
  })
})

describe('the isolation forest', () => {
  const inliers = gaussian(7, 200, 2)
  const x = [...inliers, [7, 7]]

  it('scores an outlier above the inliers, and inliers near the middle below 0.5', () => {
    const f = new IsolationForest({ nEstimators: 100, contamination: 0.01 }).fit(x, undefined, { key: key(0) }).model
    const scores = f.params.trees.length === 100 ? toFlat(f.scoreSamples(x)) : []
    expect(scores.length).toBe(201)
    expect(scores[200]).toBeGreaterThan(Math.max(...scores.slice(0, 200)))
    expect(scores[200]).toBeGreaterThan(0.6)
    expect(toFlat(f.scoreSamples([[0, 0]]))[0]).toBeLessThan(0.5)
    expect(toFlat(f.predict(x)).filter((v) => v === -1).length).toBeGreaterThanOrEqual(1)
    expect(scores.every((s) => s > 0 && s <= 1)).toBe(true)
  })

  it('is reproducible under a key, and the trees differ under another', () => {
    const fit = (k: number) => isolationForest({ nEstimators: 20 }).fit({ x }, { key: key(k) }).params
    expect(fit(1).trees).toEqual(fit(1).trees)
    expect(fit(1).trees).not.toEqual(fit(2).trees)
    const trees = fit(1).trees
    expect(new Set(trees.map((t) => JSON.stringify(t))).size).toBe(20)
  })

  it('has scores that settle as trees are added', () => {
    const outlier = (trees: number, seed: number): number =>
      toFlat(
        isolationForest({ nEstimators: trees })
          .fit({ x }, { key: key(seed) })
          .scoreSamples([[7, 7]]),
      )[0] ?? 0
    const spread = (trees: number): number => std([0, 1, 2, 3, 4, 5, 6, 7].map((s) => outlier(trees, s)))
    expect(spread(80)).toBeLessThan(spread(4))
  })

  it('grows one tree per step, serialises its state and shows the score change fall', () => {
    const alg = isolationForestAlgorithm(x, { nEstimators: 30, maxSamples: 64 })
    const out = runAlgorithm(alg, { maxSteps: 40, key: key(3) })
    expect(out.steps).toBe(30)
    expect(out.state.trees.length).toBe(30)
    expect(alg.deserialise?.(alg.serialise?.(out.state) ?? null)).toEqual(out.state)
    const early = runAlgorithm(alg, { maxSteps: 3, key: key(3) }).state.scoreChange
    expect(out.state.scoreChange).toBeLessThan(early)
  })

  it('flags about the contamination fraction of the training rows', () => {
    const f = isolationForest({ nEstimators: 60, contamination: 0.1 }).fit({ x }, { key: key(5) })
    const flagged = toFlat(f.predict(x)).filter((v) => v === -1).length
    expect(flagged).toBeGreaterThanOrEqual(Math.floor(0.1 * 201))
    expect(flagged).toBeLessThanOrEqual(Math.ceil(0.1 * 201) + 1)
    const auto = isolationForest({ nEstimators: 20 }).fit({ x }, { key: key(5) })
    expect(auto.params.threshold).toBe(0.5)
    const [rec] = recordsOf(f.exposition, 'anomalyScores')
    expect(rec?.scores.length).toBe(201)
    expect(rec?.evidence?.name).toBe('meanPathLength')
  })

  it('runs a session tree by tree and answers for the model at any step', () => {
    const est = isolationForest({ nEstimators: 12 })
    const s = est.session({ x }, { mode: 'eager', key: key(1) })
    expect(s.t).toBe(12)
    expect(s.fittedAt(5).params.trees.length).toBe(5)
  })

  it('property: path lengths are non-negative and scores lie in (0, 1]', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.double({ min: -5, max: 5, noNaN: true }), fc.double({ min: -5, max: 5, noNaN: true })), {
          minLength: 4,
          maxLength: 30,
        }),
        fc.integer({ min: 0, max: 50 }),
        (pts, seed) => {
          const data = pts.map(([a, b]) => [a, b])
          const f = isolationForest({ nEstimators: 5 }).fit({ x: data }, { key: key(seed) })
          for (const s of toFlat(f.scoreSamples(data))) {
            expect(s).toBeGreaterThan(0)
            expect(s).toBeLessThanOrEqual(1)
          }
          for (const t of f.params.trees) expect(pathLength(t, data[0] ?? [0, 0])).toBeGreaterThanOrEqual(0)
        },
      ),
      { numRuns: 25 },
    )
  })
})

describe('the extended isolation forest', () => {
  // Points on the diagonal band, and a point off it that lies inside the bounding box of the band.
  const band = gaussian(11, 150, 1).map(([t]) => [
    (t ?? 0) * 2 + 0.05 * Math.sin((t ?? 0) * 50),
    (t ?? 0) * 2 - 0.05 * Math.cos((t ?? 0) * 50),
  ])
  const x = [...band, [2, -2]]

  it('separates the off-diagonal point that axis-parallel splits cannot', () => {
    const rankOf = (extended: boolean): number => {
      let total = 0
      for (let seed = 0; seed < 4; seed++) {
        const f = isolationForest({ nEstimators: 100, extended }).fit({ x }, { key: key(seed) })
        const s = toFlat(f.scoreSamples(x))
        total += s.filter((v) => v > (s[150] ?? 0)).length
      }
      return total / 4
    }
    // The mean rank over four keys (0 is the top score): the hyperplanes rank the point nearer the top.
    expect(rankOf(true)).toBeLessThanOrEqual(4)
    expect(rankOf(true)).toBeLessThan(rankOf(false))
  })

  it('is a class and an estimator with hyperplane nodes', () => {
    const f = new ExtendedIsolationForest({ nEstimators: 10 }).fit(x, undefined, { key: key(0) }).model
    expect(f.estimator).toBe('extended-isolation-forest')
    const hyperplanes = f.params.trees.flat().filter((n) => !n.leaf && n.normal !== null)
    expect(hyperplanes.length).toBeGreaterThan(0)
    expect(hyperplanes.every((n) => !n.leaf && n.normal?.length === 2)).toBe(true)
  })
})
