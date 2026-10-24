import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import {
  blockedTimeSeriesSplit,
  clusterBlockSplit,
  expandingWindowSplit,
  purgedKFold,
  slidingWindowSplit,
  spatialBlockSplit,
  timeSeriesSplit,
} from './blocked-splitters.ts'
import {
  foldsAreDisjoint,
  groupKFold,
  groupShuffleSplit,
  holdout,
  kFold,
  leaveOneGroupOut,
  repeatedKFold,
  type Splitter,
  shuffleSplit,
  splitIndices,
  stratifiedGroupKFold,
  stratifiedKFold,
  stratifiedShuffleSplit,
  trainValidationTestSplit,
} from './splitters.ts'

const sizes = fc.integer({ min: 30, max: 80 })
const seeds = fc.integer({ min: 0, max: 1000 })

function expectPartition(s: Splitter, input: Parameters<Splitter['split']>[0], seed: number): void {
  const folds = splitIndices(s, input, key(seed))
  expect(foldsAreDisjoint(folds, input.n)).toBe(true)
  const tested = new Uint8Array(input.n)
  for (const f of folds) {
    expect(f.train.length + f.test.length).toBe(input.n)
    for (const i of f.test) tested[i] = (tested[i] ?? 0) + 1
  }
  expect(Array.from(tested).every((c) => c === 1)).toBe(true)
}

describe('splitters partition the samples with no leakage', () => {
  it('k-fold variants test every sample exactly once', () => {
    fc.assert(
      fc.property(sizes, seeds, fc.integer({ min: 2, max: 5 }), (n, seed, k) => {
        const y = Array.from({ length: n }, (_, i) => (i * 7) % 3)
        const groups = Array.from({ length: n }, (_, i) => Math.floor(i / 2))
        expectPartition(kFold(k), { n }, seed)
        expectPartition(kFold(k, { shuffle: true }), { n }, seed)
        expectPartition(stratifiedKFold(k, { shuffle: seed % 2 === 0 }), { n, y }, seed)
        expectPartition(groupKFold(k, { shuffle: seed % 2 === 0 }), { n, groups }, seed)
        expectPartition(stratifiedGroupKFold(k, { shuffle: seed % 2 === 0 }), { n, y, groups }, seed)
        expectPartition(leaveOneGroupOut(), { n, groups }, seed)
        expectPartition(purgedKFold(k), { n }, seed)
      }),
      { numRuns: 40 },
    )
  })

  it('group splitters never split a group', () => {
    fc.assert(
      fc.property(sizes, seeds, (n, seed) => {
        const groups = Array.from({ length: n }, (_, i) => (i * 5) % 7)
        for (const s of [groupKFold(3), groupShuffleSplit(3, { testSize: 0.3 }), stratifiedGroupKFold(3)]) {
          const y = Array.from({ length: n }, (_, i) => i % 2)
          for (const f of splitIndices(s, { n, groups, y }, key(seed))) {
            const testGroups = new Set(Array.from(f.test, (i) => groups[i]))
            expect(Array.from(f.train).some((i) => testGroups.has(groups[i]))).toBe(false)
          }
        }
      }),
      { numRuns: 30 },
    )
  })

  it('stratified splitters keep class proportions within one sample per class', () => {
    fc.assert(
      fc.property(fc.integer({ min: 20, max: 80 }), seeds, (n, seed) => {
        const y = Array.from({ length: n }, (_, i) => (i % 5 === 0 ? 1 : 0))
        const ones = y.filter((v) => v === 1).length
        for (const f of splitIndices(stratifiedKFold(4, { shuffle: true }), { n, y }, key(seed))) {
          const share = Array.from(f.test).filter((i) => y[i] === 1).length
          expect(Math.abs(share - (ones * f.test.length) / n)).toBeLessThanOrEqual(1)
        }
        for (const f of splitIndices(stratifiedShuffleSplit(3, { testSize: 0.25 }), { n, y }, key(seed))) {
          expect(foldsAreDisjoint([f], n)).toBe(true)
          const share = Array.from(f.test).filter((i) => y[i] === 1).length
          expect(Math.abs(share - (ones * f.test.length) / n)).toBeLessThanOrEqual(1)
        }
      }),
      { numRuns: 30 },
    )
  })

  it('shuffle splits are disjoint, sized and reproducible from the key', () => {
    fc.assert(
      fc.property(sizes, seeds, (n, seed) => {
        for (const s of [shuffleSplit(4, { testSize: 0.2 }), holdout({ testSize: 0.3 }), repeatedKFold(3, 2)]) {
          const a = splitIndices(s, { n }, key(seed))
          expect(foldsAreDisjoint(a, n)).toBe(true)
          expect(splitIndices(s, { n }, key(seed))).toEqual(a)
        }
        const [f] = splitIndices(trainValidationTestSplit(), { n }, key(seed))
        expect(f !== undefined && foldsAreDisjoint([f], n)).toBe(true)
        expect((f?.train.length ?? 0) + (f?.validation?.length ?? 0) + (f?.test.length ?? 0)).toBe(n)
      }),
      { numRuns: 30 },
    )
  })

  it('time-series splitters train only on the past', () => {
    fc.assert(
      fc.property(fc.integer({ min: 30, max: 60 }), fc.integer({ min: 0, max: 3 }), (n, gap) => {
        for (const s of [
          timeSeriesSplit(3, { gap }),
          slidingWindowSplit(5, { testSize: 3, gap }),
          expandingWindowSplit(5, { testSize: 2, gap }),
          blockedTimeSeriesSplit(3, { gap, trainFraction: 0.5 }),
        ]) {
          for (const f of splitIndices(s, { n })) {
            const lastTrain = f.train[f.train.length - 1] ?? -1
            expect(lastTrain + gap).toBeLessThan(f.test[0] ?? 0)
          }
        }
      }),
      { numRuns: 30 },
    )
  })

  it('purged k-fold keeps training labels clear of the test span and its embargo', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 20, max: 60 }),
        fc.integer({ min: 0, max: 4 }),
        fc.integer({ min: 0, max: 3 }),
        (n, h, e) => {
          for (const f of splitIndices(purgedKFold(4, { horizon: h, embargo: e }), { n })) {
            const a = f.test[0] ?? 0
            const b = Math.min(n - 1, (f.test[f.test.length - 1] ?? 0) + h)
            for (const i of f.train) {
              expect(i + h < a || i > b + e).toBe(true)
            }
          }
        },
      ),
      { numRuns: 30 },
    )
  })
})

describe('spatial and cluster blocks', () => {
  it('keep each block in one fold', () => {
    const data: number[] = []
    for (let i = 0; i < 40; i++) data.push((i * 37) % 10, (i * 53) % 10)
    const input = { n: 40, coordinates: { data, dims: 2 } }
    expectPartition(spatialBlockSplit(3, { blockSize: 2.5 }), input, 0)
    const folds = splitIndices(spatialBlockSplit(3, { blockSize: 2.5 }), input)
    const cell = (i: number): string =>
      `${Math.floor((data[2 * i] ?? 0) / 2.5)},${Math.floor((data[2 * i + 1] ?? 0) / 2.5)}`
    for (const f of folds) {
      const cells = new Set(Array.from(f.test, cell))
      expect(Array.from(f.train).some((i) => cells.has(cell(i)))).toBe(false)
    }
    expectPartition(clusterBlockSplit(3, { nClusters: 6 }), input, 1)
  })

  it('refuse bad configurations', () => {
    expect(() => splitIndices(kFold(5), { n: 3 })).toThrow()
    expect(() => splitIndices(groupKFold(4), { n: 4, groups: [0, 0, 1, 1] })).toThrow()
    expect(() => splitIndices(stratifiedKFold(3), { n: 4 })).toThrow(/labels/)
    expect(() => splitIndices(timeSeriesSplit(5), { n: 5 })).toThrow()
    expect(() => spatialBlockSplit(2, { blockSize: 0 })).toThrow()
  })
})
