import { describe, expect, it } from 'vitest'
import type { MetricResult } from '../base/index.ts'
import * as M from './index.ts'

// The rules of the metrics folder, one table per rule: weights count repetitions or are rejected, a NaN score or weight
// and an empty input are errors, and an invalid sampling rate cannot hang the audio metrics.
const w = [1, 2, 1, 3, 1, 2]
const rep = <T>(a: readonly T[]): T[] => a.flatMap((v, i) => Array<T>(w[i] ?? 0).fill(v))
const values: readonly [number[], number[]] = [
  [1, 2, 3, 5, 4, 7],
  [1.5, 2.5, 2, 5.5, 3, 8],
]
const labels: readonly [number[], number[]] = [
  [0, 1, 1, 0, 2, 1],
  [0, 1, 0, 0, 2, 2],
]
const scores: readonly [number[], number[]] = [
  [0, 1, 1, 0, 1, 0],
  [0.2, 0.8, 0.6, 0.3, 0.9, 0.4],
]

type Call = (a: never, b: never, o?: never) => MetricResult
const asCall = (f: unknown): Call => f as Call

describe('sample weights', () => {
  // Integer weights give the metric of the rows repeated that many times.
  const honoured: readonly (readonly [string, Call, readonly [number[], number[]]])[] = [
    ['concordanceCorrelation', asCall(M.concordanceCorrelation), values],
    ['cramersV', asCall(M.cramersV), labels],
    ['theilsU', asCall(M.theilsU), labels],
    ['pearsonCorrelation', asCall(M.pearsonCorrelation), values],
    ['meanSquaredError', asCall(M.meanSquaredError), values],
    ['medianAbsoluteError', asCall(M.medianAbsoluteError), values],
    ['accuracyScore', asCall(M.accuracyScore), labels],
    ['averagePrecisionScore', asCall(M.averagePrecisionScore), scores],
  ]
  it('equal repeating the rows', () => {
    for (const [name, call, [a, b]] of honoured) {
      const weighted = call(a as never, b as never, { sampleWeight: w } as never).estimate.value
      const repeated = call(rep(a) as never, rep(b) as never).estimate.value
      expect(weighted, name).toBeCloseTo(repeated, 12)
    }
  })

  it('of zero remove a sample from maxError', () => {
    const o = { sampleWeight: [1, 1, 1, 1, 1, 0] }
    expect(M.maxError([1, 2, 3, 5, 4, 7], [1, 2, 3, 5, 4, 70], o).estimate.value).toBe(0)
    expect(M.maxError([1, 2, 3, 5, 4, 7], [1, 2, 3, 5, 4, 70]).estimate.value).toBe(63)
  })

  it('are rejected by the metrics that have no weighted definition, and when negative or not finite', () => {
    const o = { sampleWeight: w }
    const rejected: readonly (readonly [string, () => unknown])[] = [
      ['spearmanCorrelation', () => M.spearmanCorrelation(values[0], values[1], o)],
      ['kendallTau', () => M.kendallTau(values[0], values[1], o)],
      ['theilsU2', () => M.theilsU2(values[0], values[1], o)],
      ['brierDecomposition', () => M.brierDecomposition(scores[0], scores[1], o)],
      ['a negative weight', () => M.accuracyScore([1, 0], [1, 0], { sampleWeight: [1, -1] })],
      ['a NaN weight', () => M.meanSquaredError([1, 0], [1, 0], { sampleWeight: [1, Number.NaN] })],
      ['an infinite weight', () => M.meanSquaredError([1, 0], [1, 0], { sampleWeight: [1, Number.POSITIVE_INFINITY] })],
    ]
    for (const [name, call] of rejected) expect(call, name).toThrow(expect.objectContaining({ code: 'value' }))
  })
})

describe('non-finite and empty input', () => {
  it('is rejected with value: a NaN score or label, and no samples', () => {
    const nan = [0.1, Number.NaN, 0.3, 0.9]
    const calls: readonly (readonly [string, () => unknown])[] = [
      ['rocAucScore with a NaN score', () => M.rocAucScore([0, 1, 0, 1], nan)],
      ['averagePrecisionScore with a NaN score', () => M.averagePrecisionScore([0, 1, 0, 1], nan)],
      ['fprAtTpr with a NaN score', () => M.fprAtTpr([0, 1, 0, 1], nan, { min: 0.5 })],
      ['accuracyScore with a NaN label', () => M.accuracyScore([0, 1, Number.NaN, 1], [0, 1, Number.NaN, 0])],
      ['accuracyScore of nothing', () => M.accuracyScore([], [])],
      ['informedness of nothing', () => M.informedness([], [])],
      ['adjustedRandScore of nothing', () => M.adjustedRandScore([], [])],
      ['meanSquaredError of nothing', () => M.meanSquaredError([], [])],
      ['maxError of nothing', () => M.maxError([], [])],
      ['r2Score of nothing', () => M.r2Score([], [])],
    ]
    for (const [name, call] of calls) expect(call, name).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('leaves average precision at 0 when no label is positive (sklearn)', () => {
    expect(M.averagePrecisionScore([0, 0, 0], [0.1, 0.5, 0.9]).estimate.value).toBe(0)
    expect(M.averagePrecisionScore([1, 1, 1], [0.1, 0.5, 0.9]).estimate.value).toBe(1)
  })
})

describe('the sampling rate of the audio metrics', () => {
  it('must be a positive integer (positive and finite for the segmental SNR), not hang', () => {
    const x = Array.from({ length: 2000 }, (_, i) => Math.sin(i / 5))
    const bad = [Number.NaN, undefined, 0, -16000, 16000.5, Number.POSITIVE_INFINITY]
    for (const fs of bad) {
      const o = { fs } as unknown as { fs: number }
      expect(() => M.shortTimeObjectiveIntelligibility(x, x, o), `stoi fs ${fs}`).toThrow(
        expect.objectContaining({ code: 'value' }),
      )
    }
    for (const fs of [Number.NaN, 0, -1, Number.POSITIVE_INFINITY, 1e-9]) {
      expect(() => M.segmentalSnr(x, x, { fs }), `segmentalSnr fs ${fs}`).toThrow(
        expect.objectContaining({ code: 'value' }),
      )
    }
  })
})
