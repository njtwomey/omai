import { describe, expect, it } from 'vitest'
import { bernoulliBandit, gaussianBandit } from '../environments/index.ts'
import { key, uniform } from '../random/index.ts'
import { recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import {
  characteristicTime,
  hardness,
  lucbAlgorithm,
  medianEliminationAlgorithm,
  runBestArm,
  successiveEliminationAlgorithm,
  trackAndStopAlgorithm,
} from './best-arm.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('the quantities of best-arm identification', () => {
  it('has the weights and rate of the Gaussian lower bound', () => {
    // Two arms: equal weights, T* = Delta^2 / 8.
    const two = characteristicTime([1, 0])
    expect(two.weights[0]).toBeCloseTo(0.5, 8)
    expect(two.time).toBeCloseTo(8, 6)
    expect(characteristicTime([0.3, 0.1], 0.5).time).toBeCloseTo((8 * 0.25) / 0.04, 6)
    // The optimum equalises the terms and has sum_a (w_a / w_best)^2 = 1.
    const means = [1, 0.4, 0.3, 0]
    const { weights, time } = characteristicTime(means)
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    const w1 = weights[0] ?? 0
    expect(weights.slice(1).reduce((a, w) => a + (w / w1) ** 2, 0)).toBeCloseTo(1, 8)
    const objective = (w: readonly number[]): number =>
      Math.min(
        ...means
          .slice(1)
          .map((m, j) => ((((w[0] ?? 0) * (w[j + 1] ?? 0)) / ((w[0] ?? 0) + (w[j + 1] ?? 0))) * (1 - m) ** 2) / 2),
      )
    expect(objective(weights)).toBeCloseTo(1 / time, 10)
    // No other allocation does better (random points of the simplex and perturbations of the optimum).
    const draws = uniform(key(7), [3000, 2 * means.length]).data
    const rand = (i: number, j: number): number => draws[i * 2 * means.length + j] ?? 0
    for (let i = 0; i < 3000; i++) {
      const raw = means.map((_, j) => 0.01 + rand(i, j))
      const total = raw.reduce((a, b) => a + b, 0)
      expect(objective(raw.map((x) => x / total))).toBeLessThanOrEqual(1 / time + 1e-12)
      const perturbed = weights.map((w, j) => Math.max(1e-6, w + 0.02 * (rand(i, means.length + j) - 0.5)))
      const t2 = perturbed.reduce((a, b) => a + b, 0)
      expect(objective(perturbed.map((x) => x / t2))).toBeLessThanOrEqual(1 / time + 1e-12)
    }
    expect(() => characteristicTime([1, 1, 0])).toThrow()
  })

  it('measures the hardness by the inverse squared gaps', () => {
    expect(hardness([0.9, 0.5, 0.5])).toBeCloseTo(18.75, 10)
    expect(hardness([0.9, 0.5])).toBeCloseTo(2 / 0.16, 10)
  })
})

describe('best-arm identification', () => {
  const probs = [0.3, 0.4, 0.5, 0.6, 0.8]
  const env = bernoulliBandit.create({ probs })

  it('names the best arm with probability at least 1 - delta (successive elimination, LUCB)', () => {
    const delta = 0.1
    for (const [name, make] of [
      ['successive elimination', () => successiveEliminationAlgorithm(env, { delta })],
      ['lucb', () => lucbAlgorithm(env, { delta })],
    ] as const) {
      let wrong = 0
      for (let seed = 0; seed < 100; seed++) {
        const r = runBestArm(env, make(), { seed })
        expect(r.stopped, name).toBe(true)
        if (!r.correct) wrong += 1
      }
      expect(wrong, name).toBeLessThanOrEqual(10)
    }
  })

  it('returns an epsilon-optimal arm from median elimination, whatever the gaps', () => {
    const arms = [0.2, 0.35, 0.5, 0.55, 0.6]
    const epsilon = 0.3
    const e = bernoulliBandit.create({ probs: arms })
    let bad = 0
    const times: number[] = []
    for (let seed = 0; seed < 40; seed++) {
      const r = runBestArm(e, medianEliminationAlgorithm(e, { delta: 0.1, epsilon }), { seed })
      expect(r.stopped).toBe(true)
      times.push(r.stoppingTime)
      if ((arms[r.recommendation] ?? 0) < 0.6 - epsilon) bad += 1
    }
    expect(bad).toBeLessThanOrEqual(4)
    // The sample complexity does not depend on the gaps: a fixed schedule, so every run takes the same pulls.
    expect(new Set(times).size).toBe(1)
  })

  it('stops later as the gap shrinks, by about the factor 4 that the 1/gap^2 scaling predicts', () => {
    const stoppingTimes = (gap: number): number => {
      const e = bernoulliBandit.create({ probs: [0.5 + gap / 2, 0.5 - gap / 2] })
      return mean(
        Array.from(
          { length: 40 },
          (_, seed) => runBestArm(e, successiveEliminationAlgorithm(e, { delta: 0.1 }), { seed }).stoppingTime,
        ),
      )
    }
    const wide = stoppingTimes(0.4)
    const narrow = stoppingTimes(0.2)
    expect(narrow / wide).toBeGreaterThan(3.3)
    expect(narrow / wide).toBeLessThan(5)
  })

  it('records the pulls, the eliminations in order and the decision on the tape', () => {
    const rec = recorder()
    const result = runAlgorithm(successiveEliminationAlgorithm(env, { delta: 0.1 }), {
      maxSteps: 1_000_000,
      seed: 3,
      tracer: rec,
      policy: tracePolicy({ metrics: ['pull/arm', 'arms/active'], messages: true }),
    })
    const events = rec.run().events
    const messages = events.filter((e) => e.type === 'message') as unknown as {
      kind: string
      payload: { arm: number }
    }[]
    const dropped = messages.filter((m) => m.kind === 'arm/eliminated').map((m) => m.payload.arm)
    expect(dropped).toHaveLength(probs.length - 1)
    expect(dropped[0]).toBe(0) // the worst arm goes first
    expect(messages.filter((m) => m.kind === 'stop/decision')).toHaveLength(1)
    expect(dropped).not.toContain(result.state.recommendation)
    const pulls = events.filter((e) => e.type === 'metric' && (e as { name: string }).name === 'pull/arm')
    expect(pulls).toHaveLength(result.state.pulls)
    expect(runBestArm(env, successiveEliminationAlgorithm(env, { delta: 0.1 }), { seed: 3 }).eliminated).toEqual(
      dropped,
    )
  })

  it('has Gaussian track-and-stop concentrate its pulls near the optimal allocation and respect the lower bound', () => {
    const means = [1, 0.5, 0, -0.5]
    const g = gaussianBandit.create({ means, sigma: 1 })
    const delta = 0.01
    const { weights, time } = characteristicTime(means, 1)
    const runs = Array.from({ length: 30 }, (_, seed) =>
      runBestArm(g, trackAndStopAlgorithm(g, { delta, sigma: 1 }), { seed }),
    )
    expect(runs.every((r) => r.stopped)).toBe(true)
    expect(runs.filter((r) => !r.correct).length).toBeLessThanOrEqual(1)
    const tau = mean(runs.map((r) => r.stoppingTime))
    // No delta-correct algorithm stops sooner on average than T*^-1 log(1 / (2.4 delta)); the algorithm is within a small factor.
    const bound = time * Math.log(1 / (2.4 * delta))
    expect(tau).toBeGreaterThan(bound)
    expect(tau).toBeLessThan(4 * bound)
    const share = means.map((_, a) => mean(runs.map((r) => (r.counts[a] ?? 0) / r.stoppingTime)))
    expect(share[0]).toBeGreaterThan(0.25)
    expect(share[0] ?? 0).toBeCloseTo(weights[0] ?? 0, 0)
  })

  it('stops after the budget when asked to, naming the empirical leader', () => {
    const r = runBestArm(env, lucbAlgorithm(env, { delta: 0.0001, maxPulls: 50 }), { seed: 1 })
    expect(r.stopped).toBe(false)
    expect(r.status).toBe('stopped')
    expect(r.stoppingTime).toBe(50)
    expect(() => successiveEliminationAlgorithm(env, { delta: 0 })).toThrow()
    expect(() => lucbAlgorithm(bernoulliBandit.create({ probs: [0.5] }))).toThrow()
  })
})
