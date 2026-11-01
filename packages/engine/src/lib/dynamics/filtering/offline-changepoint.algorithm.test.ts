import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import {
  binarySegmentation,
  binarySegmentationAlgorithm,
  binsegContract,
  pelt,
  peltAlgorithm,
  peltContract,
  peltSegments,
} from './changepoint.ts'
import { costOf, type SegmentCost, segmentCosts } from './offline-costs.ts'

const y = [0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 4.1, 3.8, 4.4, 3.9, 4.2, 4.0, 3.7, 0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.2]
const counts = [1, 0, 2, 1, 1, 0, 7, 9, 8, 6, 7, 8, 1, 0, 2, 1, 0, 1]

const runPelt = (series: readonly number[], penalty: number, options = {}): number[] =>
  peltSegments(
    runAlgorithm(peltAlgorithm(series, penalty, options), { maxSteps: series.length + 1 }).state,
    series.length,
  )

describe('PELT as an algorithm', () => {
  it('is pelt: run to the end it gives the segmentations the function gave before', () => {
    // Before the algorithm was exported (copied from the old function's output on these series).
    expect(runPelt(y, 3)).toEqual([6, 13, 20])
    expect(runPelt(y, 3, { minSize: 4 })).toEqual([6, 13, 20])
    expect(runPelt(y, 5, { cost: 'normal' })).toEqual([6, 13, 20])
    expect(runPelt(y, 3, { cost: 'l1', jump: 2 })).toEqual([6, 12, 20])
    expect(runPelt(counts, 8, { cost: 'poisson' })).toEqual([6, 12, 18])
    expect(runPelt(y, 1000)).toEqual([20])
    expect(pelt(y, 3, { cost: 'l1', jump: 2 })).toEqual([6, 12, 20])
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const alg = peltAlgorithm(y, 3, { cost: 'l1', jump: 2 })
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const half = runAlgorithm(alg, { maxSteps: 4 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 100, start: { state: stored, step: 4, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('logs F, the last start, the admissible set and what it pruned, by path', () => {
    const rec = recorder()
    runAlgorithm(peltAlgorithm(y, 3), {
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    const ends = metric('pelt/end')
    expect(ends[0]).toBe(2)
    expect(ends.at(-1)).toBe(20)
    expect(metric('pelt/F')).toHaveLength(ends.length)
    expect(metric('pelt/pruned').some((v) => v > 0)).toBe(true)
    expect(metric('pelt/F').at(-1)).toBeCloseTo(
      costOf(y, 'l2').error(0, 6) + costOf(y, 'l2').error(6, 13) + costOf(y, 'l2').error(13, 20) + 3 * 3,
      10,
    )
    expect(Object.keys(peltContract)).toHaveLength(5)
  })

  it('is exact: the same optimum as the full dynamic program over every start (property)', () => {
    const brute = (series: readonly number[], penalty: number, cost: SegmentCost, minSize: number): number => {
      const c = costOf(series, cost)
      const F = [0, ...new Array<number>(series.length).fill(Number.POSITIVE_INFINITY)]
      for (let b = minSize; b <= series.length; b++)
        for (let t = 0; t <= b - minSize; t++)
          if (Number.isFinite(F[t] ?? Number.NaN)) F[b] = Math.min(F[b] ?? 0, (F[t] ?? 0) + c.error(t, b) + penalty)
      return F[series.length] ?? 0
    }
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -3, max: 6, noNaN: true }), { minLength: 6, maxLength: 24 }),
        fc.double({ min: 0.1, max: 20, noNaN: true }),
        fc.constantFrom<SegmentCost>('l2', 'normal'),
        fc.integer({ min: 2, max: 4 }),
        (series, penalty, cost, minSize) => {
          if (series.length < minSize) return true
          const state = runAlgorithm(peltAlgorithm(series, penalty, { cost, minSize }), { maxSteps: 100 }).state
          return Math.abs((state.F[series.length] ?? Number.NaN) - brute(series, penalty, cost, minSize)) < 1e-8
        },
      ),
      { numRuns: 60 },
    )
  })
})

describe('binary segmentation as an algorithm', () => {
  const runBs = (
    series: readonly number[],
    stop: { changepoints?: number; penalty?: number },
    options = {},
  ): number[] => [
    ...runAlgorithm(binarySegmentationAlgorithm(series, stop, options), { maxSteps: series.length }).state.bkps,
  ]

  it('is binarySegmentation: run to the end it gives the segmentations the function gave before', () => {
    expect(runBs(y, { changepoints: 2 })).toEqual([6, 13, 20])
    expect(runBs(y, { penalty: 3 })).toEqual([6, 13, 20])
    expect(runBs(y, { changepoints: 3 }, { cost: 'l1', minSize: 3 })).toEqual([6, 13, 16, 20])
    expect(runBs(y, { changepoints: 0 })).toEqual([20])
    expect(binarySegmentation(y, { changepoints: 3 }, { cost: 'l1', minSize: 3 })).toEqual([6, 13, 16, 20])
    expect(() => binarySegmentationAlgorithm(y, {})).toThrow()
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const alg = binarySegmentationAlgorithm(y, { changepoints: 3 })
    const whole = runAlgorithm(alg, { maxSteps: 50 })
    const half = runAlgorithm(alg, { maxSteps: 1 })
    expect(half.state.bkps).toHaveLength(2)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 50, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('logs the split, its gain and the gain on offer; the gains add up to the cost saved', () => {
    const rec = recorder()
    const alg = binarySegmentationAlgorithm(y, { changepoints: 3 })
    const final = runAlgorithm(alg, {
      maxSteps: 50,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, roles: ['output'], detail: 'full' },
    }).state
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('binseg/split')).toHaveLength(3)
    expect(metric('binseg/segments')).toEqual([2, 3, 4])
    const c = costOf(y, 'l2')
    const total = (bkps: readonly number[]): number =>
      bkps.reduce((acc, b, i) => acc + c.error(i === 0 ? 0 : (bkps[i - 1] ?? 0), b), 0)
    const saved = metric('binseg/gain').reduce((a, b) => a + b, 0)
    expect(total([y.length]) - total(final.bkps)).toBeCloseTo(saved, 10)
    expect(Object.keys(binsegContract)).toHaveLength(4)
  })
})

describe('segment cost table', () => {
  it('runs every cost in both searches', () => {
    for (const name of segmentCosts.names) {
      const series = name === 'poisson' ? counts : y
      const cost = name as SegmentCost
      expect(runPelt(series, 6, { cost }).at(-1)).toBe(series.length)
      expect(
        runAlgorithm(binarySegmentationAlgorithm(series, { changepoints: 1 }, { cost }), { maxSteps: 5 }).state.bkps
          .length,
      ).toBe(2)
    }
    expect(segmentCosts.where((p) => p.counts)).toEqual(['poisson'])
    expect(segmentCosts.make({ kind: 'l1' })([0, 1, 9]).error(0, 3)).toBe(9)
  })
})
