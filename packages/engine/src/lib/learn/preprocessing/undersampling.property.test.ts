import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { stratifiedKFold } from '../model-selection/splitters.ts'
import { naiveBayes } from '../naive-bayes.ts'
import { chainResamplers } from './combined.ts'
import { type Resampled, type Resampler, randomUnderSampler } from './resampling.ts'
import {
  allKnn,
  clusterCentroids,
  condensedNearestNeighbour,
  editedNearestNeighbours,
  instanceHardnessThreshold,
  neighbourhoodCleaningRule,
  oneSidedSelection,
  repeatedEditedNearestNeighbours,
} from './undersampling.ts'

// Two overlapping clouds (20 majority rows, 10 minority rows) on a grid of values without exact duplicates.
const clouds = fc
  .tuple(
    fc.array(fc.integer({ min: -300, max: 300 }), { minLength: 60, maxLength: 60 }),
    fc.integer({ min: 0, max: 99 }),
  )
  .map(([v, seed]) => {
    const x = Float64Array.from(v, (t, i) => t / 100 + 0.001 * i + (i >= 40 ? 1 : 0))
    return { x: fromStorage(x, [30, 2], 'float64'), y: Array.from({ length: 30 }, (_, i) => (i >= 20 ? 1 : 0)), seed }
  })

const cleaners: readonly [string, Resampler][] = [
  ['enn', editedNearestNeighbours()],
  ['enn mode', editedNearestNeighbours({ kindSel: 'mode', nNeighbors: 5 })],
  ['renn', repeatedEditedNearestNeighbours()],
  ['allknn', allKnn()],
  ['ncr', neighbourhoodCleaningRule()],
  ['cnn', condensedNearestNeighbour()],
  ['oss', oneSidedSelection({ nNeighbors: 3 })],
  ['iht', instanceHardnessThreshold({ estimator: naiveBayes(), cv: stratifiedKFold(3) })],
]

/** A resampled set is a subset of the original rows: distinct valid indices, rows and labels copied from them. */
function expectSubset(r: Resampled, x: ReturnType<typeof fromStorage>, y: readonly number[]): void {
  const original = rows(x)
  const kept = Array.from(r.indices)
  expect(new Set(kept).size).toBe(kept.length)
  if (kept.length === 0) return
  expect(rows(r.x)).toEqual(kept.map((i) => original[i]))
  expect(Array.from(toFlat(r.y))).toEqual(kept.map((i) => y[i]))
}

describe('cleaning under-samplers', () => {
  it('return subsets of the input that keep every minority sample', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        for (const [name, sampler] of cleaners) {
          const r = sampler.fitResample({ x, y }, { key: key(seed) })
          expectSubset(r, x, y)
          // With ties, a repeated pass may edit the class that began as the minority, as imbalanced-learn does.
          const minority = Array.from(r.indices).filter((i) => y[i] === 1)
          if (name !== 'renn' && name !== 'allknn') expect(minority.length, name).toBe(10)
          expect(Array.from(r.indices).filter((i) => y[i] === 0).length, name).toBeLessThanOrEqual(20)
        }
      }),
      { numRuns: 15 },
    )
  })

  it('are deterministic for a key', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        for (const sampler of [condensedNearestNeighbour(), oneSidedSelection(), clusterCentroids()]) {
          const a = sampler.fitResample({ x, y }, { key: key(seed) })
          const b = sampler.fitResample({ x, y }, { key: key(seed) })
          expect(rows(a.x)).toEqual(rows(b.x))
        }
      }),
      { numRuns: 10 },
    )
  })

  it('edited nearest neighbours: requiring all neighbours removes at least what the majority vote removes', () => {
    fc.assert(
      fc.property(clouds, ({ x, y }) => {
        const all = new Set(editedNearestNeighbours().fitResample({ x, y }).indices)
        const mode = new Set(editedNearestNeighbours({ kindSel: 'mode' }).fitResample({ x, y }).indices)
        for (const i of all) expect(mode.has(i)).toBe(true)
      }),
      { numRuns: 20 },
    )
  })

  it('AllKNN with one neighbour is edited nearest neighbours with one neighbour; repeating never keeps more than one pass', () => {
    fc.assert(
      fc.property(clouds, ({ x, y }) => {
        const one = editedNearestNeighbours({ nNeighbors: 1 }).fitResample({ x, y }).indices
        expect(Array.from(allKnn({ nNeighbors: 1 }).fitResample({ x, y }).indices)).toEqual(Array.from(one))
        const once = new Set(editedNearestNeighbours().fitResample({ x, y }).indices)
        for (const i of repeatedEditedNearestNeighbours().fitResample({ x, y }).indices) expect(once.has(i)).toBe(true)
      }),
      { numRuns: 20 },
    )
  })

  it('a strategy naming every class edits the minority too', () => {
    fc.assert(
      fc.property(clouds, ({ x, y }) => {
        const r = editedNearestNeighbours({ samplingStrategy: 'all', nNeighbors: 5 }).fitResample({ x, y })
        expect(Array.from(r.indices).filter((i) => y[i] === 1).length).toBeLessThanOrEqual(10)
        expectSubset(r, x, y)
      }),
      { numRuns: 20 },
    )
  })

  it('instance hardness keeps the easiest majority samples, as many as the minority has', () => {
    fc.assert(
      fc.property(clouds, ({ x, y }) => {
        const r = instanceHardnessThreshold({ estimator: naiveBayes(), cv: stratifiedKFold(3) }).fitResample({ x, y })
        const majority = Array.from(r.indices).filter((i) => y[i] === 0).length
        expect(majority).toBeGreaterThanOrEqual(10)
        expect(majority).toBeLessThanOrEqual(11)
      }),
      { numRuns: 15 },
    )
  })

  it('condensed nearest neighbour keeps the starting sample it was given', () => {
    fc.assert(
      fc.property(clouds, fc.integer({ min: 0, max: 19 }), ({ x, y }, seed) => {
        const r = condensedNearestNeighbour({ seeds: { 0: [seed] } }).fitResample({ x, y })
        expect(Array.from(r.indices)).toContain(seed)
      }),
      { numRuns: 20 },
    )
  })
})

describe('cluster centroids', () => {
  it('replace the majority class by as many centroids as the target, inside its bounding box', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const r = clusterCentroids().fitResample({ x, y }, { key: key(seed) })
        const labels = toFlat(r.y)
        expect(labels.filter((v) => v === 0).length).toBe(10)
        expect(labels.filter((v) => v === 1).length).toBe(10)
        const original = rows(x)
        const major = original.filter((_, i) => y[i] === 0)
        const synthetic = rows(r.x).filter((_, i) => r.indices[i] === -1)
        expect(synthetic).toHaveLength(10)
        for (const c of synthetic) {
          for (let j = 0; j < 2; j++) {
            expect(c[j] ?? 0).toBeGreaterThanOrEqual(Math.min(...major.map((p) => p[j] ?? 0)) - 1e-9)
            expect(c[j] ?? 0).toBeLessThanOrEqual(Math.max(...major.map((p) => p[j] ?? 0)) + 1e-9)
          }
        }
      }),
      { numRuns: 15 },
    )
  })

  it('hard voting returns real rows', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const r = clusterCentroids({ voting: 'hard' }).fitResample({ x, y }, { key: key(seed) })
        const original = rows(x)
        expect(Array.from(r.indices).every((i) => i >= 0)).toBe(true)
        expect(rows(r.x)).toEqual(Array.from(r.indices).map((i) => original[i]))
        expect(toFlat(r.y).filter((v) => v === 0).length).toBe(10)
      }),
      { numRuns: 15 },
    )
  })
})

describe('chains and the balanced bag', () => {
  it('compose indices back to the original rows', () => {
    fc.assert(
      fc.property(clouds, ({ x, y, seed }) => {
        const r = chainResamplers('chain', [editedNearestNeighbours(), randomUnderSampler()]).fitResample(
          { x, y },
          { key: key(seed) },
        )
        expectSubset(r, x, y)
      }),
      { numRuns: 15 },
    )
  })
})
