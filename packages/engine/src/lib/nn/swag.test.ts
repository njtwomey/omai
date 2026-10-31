import { describe, expect, it } from 'vitest'
import { expectClose } from '../../testing/close.ts'
import { key, multivariateNormal, normal, subsample } from '../random/index.ts'
import { add, matmul, mul, sub, sum, take, tensor, toArray, toFlat, transpose, type Value } from '../tensor/index.ts'
import { metricSeries, policies, recorder, runAlgorithm, tensorSeries } from '../trace/index.ts'
import {
  type SwagState,
  swagAlgorithm,
  swagCollect,
  swagCovariance,
  swagPredictive,
  swagSample,
  swagState,
  swagUnravel,
  swagVariance,
} from './swag.ts'

const collectAll = (rows: readonly (readonly number[])[], rank = 100): SwagState =>
  rows.reduce<SwagState>((s, r) => swagCollect(s, r), swagState(rows[0]?.length ?? 0, rank))

describe('SWAG moments', () => {
  it('keep the running mean, the diagonal variance and the last deviations from the running mean', () => {
    const rows = [
      [1, 4],
      [3, 0],
      [5, 2],
      [7, 6],
    ]
    const s = collectAll(rows, 3)
    expect(s.count).toBe(4)
    expect(s.mean).toEqual([4, 3])
    // Population variance of each coordinate.
    expectClose(swagVariance(s), [5, 5], 1e-12)
    // Three columns kept: iterates 2, 3, 4 against the mean after they were added.
    expect(s.deviations).toHaveLength(3)
    expect(s.deviations[2]).toEqual([3, 3])
    expect(s.deviations[0]).toEqual([1, -2])
    expect(s.deviations[1]).toEqual([2, 0])
    expect(() => swagCollect(s, [1])).toThrow(/expected 2/)
  })

  it('recover a Gaussian from its iterates: the mean, the variances and the low-rank covariance', () => {
    const mu = [1, -2, 0.5]
    const chol = tensor([
      [1, 0, 0],
      [0.8, 0.6, 0],
      [-0.3, 0.4, 0.5],
    ])
    const draws = toArray(multivariateNormal(key(1), tensor(mu), chol, 600)) as number[][]
    const s = collectAll(draws, 600)
    expectClose(s.mean, mu, 0.12)
    const truth = toArray(matmul(chol, transpose(chol))) as number[][]
    expectClose(
      swagVariance(s),
      [0, 1, 2].map((i) => (truth[i] as number[])[i] as number),
      0.15,
    )
    // The paper's covariance averages the diagonal and the low-rank (sample covariance) estimates.
    const cov = toArray(swagCovariance(s)) as number[][]
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const t = ((truth[i] as number[])[j] as number) * (i === j ? 1 : 0.5)
        expect(Math.abs(((cov[i] as number[])[j] as number) - t)).toBeLessThan(0.2)
      }
  })
})

describe('SWAG sampling', () => {
  const draws = toArray(normal(key(2), [200, 3])) as number[][]
  const rows = draws.map((r, i) => [
    r[0] as number,
    (r[0] as number) * 0.9 + 0.3 * (r[1] as number),
    (r[2] as number) + i * 0,
  ])
  const state = collectAll(rows, 50)

  it('samples have the SWA mean and the stated covariance', () => {
    const n = 20000
    const samples = Array.from({ length: n }, (_, i) => swagSample(state, key(i)))
    const m = [0, 1, 2].map((j) => samples.reduce((a, r) => a + (r[j] as number), 0) / n)
    const cov = swagCovariance(state)
    const target = toArray(cov) as number[][]
    for (let j = 0; j < 3; j++) expect(m[j]).toBeCloseTo(state.mean[j] as number, 1)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const c =
          samples.reduce(
            (a, r) => a + ((r[i] as number) - (m[i] as number)) * ((r[j] as number) - (m[j] as number)),
            0,
          ) /
          (n - 1)
        expect(c).toBeCloseTo((target[i] as number[])[j] as number, 1)
      }
  })

  it('the diagonal mode drops the low-rank part, the scale multiplies the covariance, scale 0 is the SWA point', () => {
    expect(
      toFlat(swagCovariance(state, { mode: 'diagonal' }))
        .filter((_, i) => i % 4 !== 0)
        .every((v) => v === 0),
    ).toBe(true)
    expectClose(swagCovariance(state, { scale: 3 }), mul(swagCovariance(state), 3), 1e-12)
    expect(swagSample(state, key(5), { scale: 0 })).toEqual(state.mean.map((m) => m + 0))
    expect(swagSample(state, key(5))).toEqual(swagSample(state, key(5)))
    expect(swagSample(state, key(5))).not.toEqual(swagSample(state, key(6)))
    const diag = Array.from({ length: 4000 }, (_, i) => swagSample(state, key(i), { mode: 'diagonal' }))
    const v0 = diag.reduce((a, r) => a + ((r[0] as number) - (state.mean[0] as number)) ** 2, 0) / 4000
    expect(v0).toBeCloseTo(swagVariance(state)[0] as number, 1)
  })
})

describe('SWAG along an SGD trajectory', () => {
  // Linear regression with noisy minibatches: constant-step SGD bounces around the least-squares solution.
  const n = 200
  const X = normal(key(10), [n, 3])
  const w0 = tensor([1, -2, 0.5])
  const y = add(matmul(X, w0), mul(normal(key(11), [n]), 0.5))
  const lr = 0.15
  const trajectory = {
    init: (): Record<string, Value> => ({ w: tensor([0, 0, 0]) }),
    step: (p: Record<string, unknown>, ctx: { key: Parameters<typeof subsample>[0] }): Record<string, Value> => {
      const idx = subsample(ctx.key, n, 4)
      const xb = take(X, idx, 0)
      const yb = take(y, idx, 0)
      const w = p['w'] as Value
      const grad = mul(matmul(transpose(xb), sub(matmul(xb, w), yb)), 1 / 4)
      return { w: sub(w, mul(grad, lr)) }
    },
  }
  const algo = swagAlgorithm(trajectory, { burnIn: 100, collectEvery: 2, rank: 20 })

  it('the average of the iterates is closer to the solution than the iterates themselves', () => {
    const rec = recorder()
    const result = runAlgorithm(algo, { maxSteps: 500, seed: 3, tracer: rec, policy: policies.teaching })
    const solution = toFlat(
      matmul(
        matmul(
          // (X^T X)^-1 X^T y by the normal equations.
          tensor(invert3(toArray(matmul(transpose(X), X)) as number[][])),
          transpose(X),
        ),
        y,
      ),
    )
    const dist = (w: readonly number[]): number => Math.hypot(...w.map((v, i) => v - (solution[i] as number)))
    // The distance of the iterates over the collection window, from the trace's mean tensors.
    const last = toFlat(result.state.params['w'] as Value)
    const swa = result.state.swag.mean
    expect(result.state.swag.count).toBe(200)
    expect(result.state.swag.deviations).toHaveLength(20)
    expect(dist(swa)).toBeLessThan(0.5 * dist(last) + 0.02)
    expect(dist(swa)).toBeLessThan(0.1)
    const run = rec.run()
    expect(metricSeries(run, 'swag/collected').values.at(-1)).toBe(200)
    expect(metricSeries(run, 'swag/rank').values.at(-1)).toBe(20)
    expect(tensorSeries(run, 'swag/mean').length).toBeGreaterThan(0)
    expect(tensorSeries(run, 'swag/deviations').length).toBeGreaterThan(0)
  })

  it('serialises and resumes to the same state', () => {
    const full = runAlgorithm(algo, { maxSteps: 150, seed: 4 })
    const half = runAlgorithm(algo, { maxSteps: 90, seed: 4 })
    const json = JSON.parse(JSON.stringify(algo.serialise?.(half.state)))
    const resumed = runAlgorithm(algo, { maxSteps: 60, seed: 4, start: { state: json, step: 90, serialised: true } })
    expect(resumed.state.swag.mean).toEqual(full.state.swag.mean)
    expectClose(resumed.state.params['w'] as Value, full.state.params['w'] as Value, 0)
  })

  it('the predictive averages the network over sampled weights', () => {
    const result = runAlgorithm(algo, { maxSteps: 300, seed: 5 })
    const unravel = swagUnravel(result.state.params)
    const x = tensor([
      [1, 0, 0],
      [0.5, 0.5, 1],
    ])
    const predict = (p: Readonly<Record<string, unknown>>): Value => matmul(x, p['w'] as Value)
    const exact = swagPredictive(result.state.swag, unravel, predict, key(6), 5, { scale: 0 })
    expectClose(exact.mean, matmul(x, tensor(result.state.swag.mean)), 1e-12)
    const wide = swagPredictive(result.state.swag, unravel, predict, key(7), 2000)
    // A linear network's mean prediction at sampled weights is the prediction at the mean weights.
    for (const [i, v] of toFlat(wide.mean).entries()) expect(v).toBeCloseTo(toFlat(exact.mean)[i] as number, 1)
    const spread = (wide.samples.map((s) => toFlat(s)[0]) as number[]).reduce(
      (a, v) => a + (v - (toFlat(wide.mean)[0] as number)) ** 2,
      0,
    )
    expect(spread).toBeGreaterThan(0)
    expect(sum(wide.mean).shape).toEqual([])
  })
})

/** The inverse of a 3x3 matrix by cofactors. */
function invert3(m: number[][]): number[][] {
  const [a, b, c] = m[0] as [number, number, number]
  const [d, e, f] = m[1] as [number, number, number]
  const [g, h, i] = m[2] as [number, number, number]
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ]
}
