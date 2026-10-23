import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  aggregateCurves,
  aggregateScores,
  improvementEstimate,
  interquartileMean,
  performanceProfile,
  probabilityOfImprovement,
  stratifiedBootstrap,
} from './evaluation.ts'

const scoreMatrix = (runs: number, tasks: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: tasks, maxLength: tasks }), {
    minLength: runs,
    maxLength: runs,
  })

describe('stratified bootstrap intervals', () => {
  it('cover the point estimate and are reproducible from the key', () => {
    fc.assert(
      fc.property(scoreMatrix(6, 3), fc.nat(1000), (s, seed) => {
        const a = stratifiedBootstrap(s, interquartileMean, { key: key(seed), reps: 200 })
        const b = stratifiedBootstrap(s, interquartileMean, { key: key(seed), reps: 200 })
        expect(a).toEqual(b)
        expect(a.value).toBe(interquartileMean(s))
        // The interval spans the replicates' middle 95%, which lies within the range of the resampled values.
        expect(a.interval[0]).toBeLessThanOrEqual(a.interval[1])
        expect(a.samples).toHaveLength(200)
      }),
      { numRuns: 20 },
    )
  })

  it('covers the IQM of well-behaved data, and a different key gives different replicates', () => {
    const s = Array.from({ length: 10 }, (_, i) => [i / 10, 1 + i / 20, 2 - i / 30])
    const a = stratifiedBootstrap(s, interquartileMean, { seed: 0, reps: 500 })
    expect(a.interval[0]).toBeLessThanOrEqual(a.value)
    expect(a.value).toBeLessThanOrEqual(a.interval[1])
    expect(stratifiedBootstrap(s, interquartileMean, { seed: 1, reps: 500 }).samples).not.toEqual(a.samples)
  })

  it('keeps tasks apart: resampling never mixes columns', () => {
    // Task 0 is always 0 and task 1 always 1, so the mean of task means is 0.5 in every resample.
    const s = [
      [0, 1],
      [0, 1],
      [0, 1],
    ]
    const a = aggregateScores(s, { seed: 3, reps: 100, metrics: ['mean'] })
    expect(new Set(a.mean?.samples)).toEqual(new Set([0.5]))
    // With task resampling the replicates can draw one task twice.
    const t = aggregateScores(s, { seed: 3, reps: 100, metrics: ['mean'], taskBootstrap: true })
    expect(new Set(t.mean?.samples).size).toBeGreaterThan(1)
  })

  it('resamples two algorithms independently for the probability of improvement', () => {
    fc.assert(
      fc.property(scoreMatrix(5, 2), scoreMatrix(4, 2), (x, y) => {
        const e = improvementEstimate(x, y, { seed: 0, reps: 100 })
        expect(e.value).toBe(probabilityOfImprovement(x, y))
        expect(e.interval[0]).toBeGreaterThanOrEqual(0)
        expect(e.interval[1]).toBeLessThanOrEqual(1)
        // P(X > Y) + P(Y > X) = 1 with ties split evenly.
        expect(probabilityOfImprovement(x, y) + probabilityOfImprovement(y, x)).toBeCloseTo(1, 12)
      }),
      { numRuns: 20 },
    )
  })
})

describe('performance profiles and learning curves', () => {
  it('profiles decrease in the threshold and their intervals bracket them', () => {
    fc.assert(
      fc.property(scoreMatrix(5, 3), (s) => {
        const taus = [-6, -2, 0, 2, 6]
        const p = performanceProfile(s, taus, { seed: 0, reps: 100 })
        for (let i = 1; i < taus.length; i++) expect(p.value[i]).toBeLessThanOrEqual(p.value[i - 1] ?? 1)
        expect(p.value[0]).toBe(1)
        expect(p.value[4]).toBe(0)
        for (const [i, v] of p.value.entries()) {
          expect(p.lower[i]).toBeLessThanOrEqual(v + 1e-12)
          expect(p.upper[i]).toBeGreaterThanOrEqual(v - 1e-12)
        }
      }),
      { numRuns: 20 },
    )
  })

  it('aggregates curves of different lengths and step grids by interpolation, per task', () => {
    const curves = [
      { steps: [0, 10, 20], values: [0, 1, 2] },
      { steps: [0, 5, 15, 20], values: [0, 0.5, 1.5, 2] },
      { steps: [0, 20], values: [10, 12] },
      { steps: [0, 10, 20, 30], values: [10, 11, 12, 13] },
    ]
    const r = aggregateCurves(curves, { aggregate: 'mean', tasks: [0, 0, 1, 1], points: 5, seed: 0, reps: 200 })
    expect(r.x).toEqual([0, 5, 10, 15, 20])
    // Task 0 curves both follow x / 10, task 1 curves 10 + x / 10; the mean of task means is 5 + x / 10.
    for (const [i, x] of r.x.entries()) {
      expect(r.value[i]).toBeCloseTo(5 + x / 10, 12)
      expect(r.lower[i]).toBeCloseTo(5 + x / 10, 12)
    }
    expect(() => aggregateCurves(curves, { tasks: [0, 0, 0, 1] })).toThrow(/same number of runs/)
    const iqm = aggregateCurves(
      [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 100],
      ],
      { reps: 50, seed: 1 },
    )
    expect(iqm.value).toEqual([0, 2.5])
    expect(iqm.lower[1]).toBeLessThanOrEqual(2.5)
  })
})
