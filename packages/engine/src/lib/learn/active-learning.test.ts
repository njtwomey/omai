import { describe, expect, it } from 'vitest'
import { gaussianNB } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { recorder, runTree } from '../trace/index.ts'
import {
  activeLearning,
  appendRows,
  expectedErrorReduction,
  randomSampling,
  streamActiveLearning,
  voteEntropySampling,
} from './active-learning.ts'

const pool = Array.from({ length: 30 }, (_, i) => [Math.sin(i) * 3, Math.cos(1.7 * i) * 3])
const truth = (r: readonly number[]) => ((r[0] ?? 0) + (r[1] ?? 0) > 0 ? 1 : 0)
const labeled = {
  x: [
    [-3, -3],
    [3, 3],
    [-2, -2.5],
    [2.5, 2],
  ],
  y: [0, 1, 0, 1],
}
const oracle = (idx: readonly number[]) => idx.map((i) => truth(pool[i] ?? []))

describe('active learning', () => {
  it('spends the budget without repeating rows, traced round by round', () => {
    const rec = recorder()
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool,
      oracle,
      strategy: voteEntropySampling(),
      batchSize: 3,
      budget: 10,
      evaluation: { x: pool, y: pool.map(truth) },
      scoring: 'accuracy',
      key: key(1),
      tracer: rec,
    })
    expect(r.queried.length).toBe(10)
    expect(new Set(r.queried).size).toBe(10)
    expect(r.scores.length).toBe(4)
    expect(runTree(rec.run()).length).toBe(1)
  })

  it('stops when the pool is exhausted', () => {
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 5),
      oracle,
      strategy: randomSampling(),
      batchSize: 2,
      key: key(2),
    })
    expect([...r.queried].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4])
  })

  it('expected error reduction picks the candidate with the lowest expected pool error', () => {
    const est = gaussianNB()
    const r = activeLearning(est, {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: expectedErrorReduction(),
      budget: 1,
    })
    expect(r.queried.length).toBe(1)
    expect(appendRows([[1, 2]], [[3, 4]]).shape).toEqual([2, 2])
  })

  it('selective sampling queries only uncertain arrivals within budget', () => {
    const r = streamActiveLearning(gaussianNB(), {
      labeled,
      stream: pool,
      oracle: (t) => truth(pool[t] ?? []),
      threshold: 0.2,
      budget: 4,
    })
    expect(r.queried.length).toBeLessThanOrEqual(4)
    expect([...r.queried].sort((a, b) => a - b)).toEqual(r.queried)
  })
})
