import { describe, expect, it } from 'vitest'
import { key, randint } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { recorder, runAlgorithm } from '../../trace/index.ts'
import {
  adaptiveOverfittingDemo,
  type HoldoutQuery,
  ladderAlgorithm,
  leaderboardAttack,
  thresholdout,
  thresholdoutAlgorithm,
} from './adaptive.ts'

const signs = (seed: number, n: number): number[] => Array.from(toFlat(randint(key(seed), [n], 0, 2)), (v) => 2 * v - 1)
const sample = (seed: number, n: number): { x: number[][]; y: number[] } => ({
  x: signs(seed, n * 3).reduce<number[][]>((rows, v, i) => {
    if (i % 3 === 0) rows.push([])
    rows[rows.length - 1]?.push(v)
    return rows
  }, []),
  y: signs(seed + 1, n),
})

describe('thresholdout', () => {
  it('answers a non-adaptive query from the training set when the two samples agree', () => {
    const train = sample(1, 2000)
    const holdout = sample(2, 2000)
    const q: HoldoutQuery = (x, y) => ((x[0] ?? 0) === y ? 1 : 0)
    const r = thresholdout(train, holdout, [q, q, q], { threshold: 0.3, sigma: 0.005 })
    expect(r.released).toEqual([false, false, false])
    expect(r.answers[0]).toBe(r.trainValues[0])
    expect(r.budget).toBe(100)
  })

  it('releases a noised holdout value when the training value is far off, and spends budget', () => {
    // A query whose training average is biased by construction: all training labels agree with the feature.
    const train = { x: [[1], [1], [1], [1]], y: [1, 1, 1, 1] }
    const holdout = { x: [[1], [1], [1], [1]], y: [1, -1, -1, -1] }
    const q: HoldoutQuery = (x, y) => ((x[0] ?? 0) === y ? 1 : 0)
    const r = thresholdout(train, holdout, [q], { threshold: 0.1, sigma: 0.01, budget: 3 })
    expect(r.trainValues[0]).toBe(1)
    expect(r.holdoutValues[0]).toBe(0.25)
    expect(r.released[0]).toBe(true)
    expect(Math.abs((r.answers[0] ?? 0) - 0.25)).toBeLessThan(0.2)
    expect(r.budget).toBe(2)
  })

  it('stops answering once the budget is spent', () => {
    const train = { x: [[1], [1]], y: [1, 1] }
    const holdout = { x: [[1], [1]], y: [-1, -1] }
    const q: HoldoutQuery = (x, y) => ((x[0] ?? 0) === y ? 1 : 0)
    const r = thresholdout(train, holdout, [q, q, q], { threshold: 0.05, sigma: 0.005, budget: 1 })
    expect(r.released.filter(Boolean).length).toBe(1)
    expect(Number.isNaN(r.answers[2])).toBe(true)
  })

  it('is deterministic given the key, traces answers and serialises its state', () => {
    const a = sample(3, 100)
    const b = sample(4, 100)
    const qs: HoldoutQuery[] = [(x) => x[0] ?? 0, (x) => x[1] ?? 0]
    const alg = thresholdoutAlgorithm(a, b, qs, { threshold: 0.05 })
    const rec = recorder()
    const r1 = runAlgorithm(alg, { key: key(5), maxSteps: 2, tracer: rec })
    const r2 = runAlgorithm(alg, { key: key(5), maxSteps: 2 })
    expect(r1.state.answers).toEqual(r2.state.answers)
    expect(r1.steps).toBe(2)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(r1.state))))
    expect(back?.answers).toEqual(r1.state.answers)
    expect(rec.run()).toBeDefined()
  })

  it('takes an adaptive analyst that sees the previous answers', () => {
    const d = { x: [[1], [2], [3]], y: [0, 0, 0] }
    const analyst = (answers: readonly number[]): HoldoutQuery | undefined =>
      answers.length === 0
        ? (x) => x[0] ?? 0
        : answers.length === 1
          ? (x) => (x[0] ?? 0) * (answers[0] ?? 0)
          : undefined
    const r = thresholdout(d, d, analyst, { threshold: 0.5 })
    expect(r.answers).toEqual([2, 4])
  })
})

describe('the planted adaptive overfitting problem', () => {
  const opts = { n: 1000, features: 1000 }
  it('inflates the reported accuracy under naive reuse of the holdout while the truth stays at one half', () => {
    const rs = [0, 1, 2].map((s) => adaptiveOverfittingDemo('naive', { ...opts, key: key(s) }))
    for (const r of rs) {
      expect(r.reported).toBeGreaterThan(0.65)
      expect(Math.abs(r.freshAccuracy - 0.5)).toBeLessThan(0.08)
    }
  })

  it('reports honest accuracies through thresholdout', () => {
    const rs = [0, 1, 2].map((s) => adaptiveOverfittingDemo('thresholdout', { ...opts, key: key(s) }))
    for (const r of rs) {
      // Thresholdout answers from the training set while it agrees with the holdout; the final claim is checked against
      // the holdout, where the classifier has no skill.
      expect(Math.abs(r.reported - 0.5)).toBeLessThan(0.08)
      expect(r.trainAccuracy).toBeGreaterThan(0.65)
    }
    const naive = adaptiveOverfittingDemo('naive', { ...opts, key: key(0) })
    const thr = adaptiveOverfittingDemo('thresholdout', { ...opts, key: key(0) })
    expect(naive.reported - thr.reported).toBeGreaterThan(0.12)
  })
})

describe('the ladder', () => {
  it('moves only for improvements larger than the step, in multiples of the step', () => {
    const alg = ladderAlgorithm([0.5, 0.495, 0.42, 0.43, 0.3], { step: 0.05 })
    const r = runAlgorithm(alg, { maxSteps: 5 })
    const rep = r.state.reported
    expect(rep.slice(0, 4).map((v) => Math.round(v * 100) / 100)).toEqual([0.5, 0.5, 0.4, 0.4])
    expect(Math.round((rep[4] ?? 0) * 100) / 100).toBe(0.3)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(r.state))))
    expect(back?.reported).toEqual(rep)
  })

  it('shows the plain leaderboard falling to the boosting attack while the ladder stays near one half', () => {
    const naive = [0, 1, 2].map((s) => leaderboardAttack('naive', { n: 1000, features: 300, key: key(s) }))
    const ladder = [0, 1, 2].map((s) => leaderboardAttack('ladder', { n: 1000, features: 300, key: key(s) }))
    for (const r of naive) {
      expect(r.reportedError).toBeLessThan(0.38)
      expect(Math.abs(r.freshError - 0.5)).toBeLessThan(0.08)
    }
    for (const r of ladder) expect(r.reportedError).toBeGreaterThan(0.43)
  })

  it('rejects a non-positive step', () => {
    expect(() => ladderAlgorithm([0.1], { step: 0 })).toThrow()
  })
})
