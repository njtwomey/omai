import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { sparseCodedSignal } from '../datasets/index.ts'
import { dictionaryRecovery } from '../explain/index.ts'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  dictLearningOnline,
  miniBatchDictionaryLearning,
  miniBatchDictionaryLearningAlgorithm,
} from './dictionary-learning-online.ts'
import { sparseCode } from './sparse-coding.ts'

const planted = generate(sparseCodedSignal, {
  key: key(2),
  knobs: { n: 240, components: 6, dim: 20, nonzeros: 2, noise: 0.01 },
})
const truth = planted.truth.extra?.['dictionary'] as number[][]
const x = toArray(planted.data.x) as number[][]
const base = { nComponents: 6, alpha: 0.05, batchSize: 24, tol: 0, maxNoImprovement: null } as const

describe('online dictionary learning', () => {
  it('lowers the cost over all the signals as batches arrive, and finds the planted atoms', () => {
    const f = miniBatchDictionaryLearning
      .with({ ...base, maxIter: 8, init: 'random', monitor: 'full' })
      .fit({ x }, { key: key(3), logs: 'all' })
    const cost = metricSeries(f.run, 'solver/objective').values
    expect(cost.at(-1) as number).toBeLessThan((cost[1] as number) * 0.7)
    expect(f.params.nSteps).toBe(80)
    expect(f.params.nIter).toBe(8)
    const long = miniBatchDictionaryLearning.with({ ...base, maxIter: 40, init: 'samples' }).fit({ x }, { key: key(3) })
    expect(dictionaryRecovery(truth, long.params.components).mean).toBeGreaterThan(0.9)
  }, 30_000)

  it("keeps the running statistics of Mairal: after one batch A = C'C / m and B = X'C / m", () => {
    const algorithm = miniBatchDictionaryLearningAlgorithm(x, truth, { alpha: 0.1, batchSize: 24, shuffle: false })
    const { state } = runAlgorithm(algorithm, { maxSteps: 1 })
    const batch = x.slice(0, 24)
    const code = toArray(sparseCode(batch, truth, { algorithm: 'lassoLars', alpha: 0.1 })) as number[][]
    for (let p = 0; p < 6; p++)
      for (let q = 0; q < 6; q++) {
        let a = 0
        for (const row of code) a += (row[p] ?? 0) * (row[q] ?? 0)
        expect(state.a[p]?.[q]).toBeCloseTo(a / 24, 10)
      }
    for (let f = 0; f < 20; f++)
      for (let q = 0; q < 6; q++) {
        let b = 0
        for (const [i, row] of code.entries()) b += (batch[i]?.[f] ?? 0) * (row[q] ?? 0)
        expect(state.b[f]?.[q]).toBeCloseTo(b / 24, 10)
      }
  })

  it('visits the signals in the order of a keyed shuffle, cycling through batches', () => {
    const run = (k: number) =>
      runAlgorithm(miniBatchDictionaryLearningAlgorithm(x, truth, { batchSize: 24 }), { maxSteps: 1, key: key(k) })
        .state.order
    expect(run(1)).toEqual(run(1))
    expect(run(1)).not.toEqual(run(2))
    expect([...run(1)].sort((a, b) => a - b)).toEqual(Array.from({ length: 240 }, (_, i) => i))
  })

  it('stops when the atoms barely move or the smoothed cost stops improving, and not before the first epoch', () => {
    const f = miniBatchDictionaryLearning
      .with({ ...base, maxIter: 400, tol: 1e-3, maxNoImprovement: 10, init: 'samples' })
      .fit({ x }, { key: key(3) })
    expect(f.params.nSteps).toBeLessThan(4000)
    expect(f.params.nSteps).toBeGreaterThan(10)
    const noStop = miniBatchDictionaryLearning.with({ ...base, maxIter: 3 }).fit({ x }, { key: key(3) })
    expect(noStop.params.nSteps).toBe(30)
  })

  it('replaces atoms the statistics show unused, counting them on the tape', () => {
    const flat = [
      [2, 0, 0],
      [0, 3, 0],
      [0, 0, 1],
      [1, 1, 0],
    ]
    const f = miniBatchDictionaryLearning
      .with({ nComponents: 5, alpha: 0.1, batchSize: 2, maxIter: 2, tol: 0, maxNoImprovement: null })
      .fit({ x: flat }, { key: key(0), logs: 'all' })
    expect(metricSeries(f.run, 'solver/unused').values.some((v) => v > 0)).toBe(true)
    expect(f.params.components.every((a) => Math.hypot(...a) > 0)).toBe(true)
  })

  it('transforms, inverts and plays through a session, atoms and usage probed at a batch', () => {
    const est = miniBatchDictionaryLearning.with({ ...base, maxIter: 4 })
    const f = est.fit({ x }, { key: key(0) })
    expect(f.transform(x).shape).toEqual([240, 6])
    expect(f.inverseTransform(f.transform(x)).shape).toEqual([240, 20])
    const s = est.session({ x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 5 })
    expect((s.probe(3, 'dict/atoms') as { series: unknown[] }).series.length).toBe(6)
    expect(s.fittedAt(4).params.nSteps).toBe(4)
  })

  it('has custom starts checked, and states round-trip', () => {
    expect(() => miniBatchDictionaryLearning.with({ ...base, init: 'custom' }).fit({ x })).toThrow(/dictInit/)
    expect(() => miniBatchDictionaryLearning.with({ ...base, init: 'custom', dictInit: [[1]] }).fit({ x })).toThrow(
      /dictInit/,
    )
    const { state } = runAlgorithm(miniBatchDictionaryLearningAlgorithm(x, truth, { batchSize: 24 }), { maxSteps: 3 })
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

describe('dictLearningOnline', () => {
  it('returns the atoms and the lasso codes over them', () => {
    const r = dictLearningOnline(x, 6, {
      alpha: 0.05,
      batchSize: 24,
      maxIter: 5,
      dictInit: truth,
      shuffle: false,
      tol: 0,
      maxNoImprovement: null,
    })
    expect(r.dictionary.length).toBe(6)
    expect(r.code.length).toBe(240)
    expect(r.nSteps).toBe(50)
    expect(dictLearningOnline(x, 6, { method: 'cd', maxIter: 1, batchSize: 24 }).code.length).toBe(240)
  })
})
