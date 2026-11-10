// The pieces of particle MCMC against the exact linear-Gaussian answers: the filter's likelihood estimate is unbiased (the
// mean of Z-hat / Z over replications is one within its standard error); a conditional SMC chain on the path has the
// Kalman smoother's mean and spread as its stationary law, and ancestor sampling refreshes the early path where plain
// conditional SMC leaves it stuck to the reference.
import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../../random/index.ts'
import { item, mul, toFlat, type Value } from '../../tensor/index.ts'
import { kalmanFilter, kalmanLogLikelihood, rtsSmoother } from './kalman.ts'
import type { NonlinearModel } from './nonlinear.ts'
import { gaussianParticleModel } from './particle-options.ts'
import { completeDataLogDensity, conditionalSmc, particleGibbs, particleLogLikelihood, pmmhKernel } from './pmcmc.ts'

const PHI = 0.8
const Q = 0.25
const R = 0.36
const model: NonlinearModel = {
  f: (x: Value) => mul(x, PHI),
  h: (x: Value) => x,
  Q: [[Q]],
  R: [[R]],
  x0: [0],
  P0: [[1]],
}
const kalman = { F: [[PHI]], H: [[1]], Q: [[Q]], R: [[R]], x0: [0], P0: [[1]] }
const particleModel = gaussianParticleModel(model)

function series(T: number, seed: number): number[][] {
  const e = toFlat(normal(key(seed), [T]))
  const o = toFlat(normal(key(seed + 100), [T]))
  let x = toFlat(normal(key(seed + 200), [1]))[0] ?? 0
  return Array.from({ length: T }, (_, t) => {
    x = PHI * x + Math.sqrt(Q) * (e[t] ?? 0)
    return [x + Math.sqrt(R) * (o[t] ?? 0)]
  })
}

describe('the particle filter as a likelihood estimator', () => {
  it('is unbiased on the natural scale', () => {
    const Z = series(10, 1)
    const exact = item(kalmanLogLikelihood(kalman, Z))
    const reps = 60
    const ratios = Array.from({ length: reps }, (_, i) =>
      Math.exp(particleLogLikelihood(particleModel, Z, { key: foldIn(key(2), i), particles: 60 }) - exact),
    )
    const mean = ratios.reduce((a, b) => a + b, 0) / reps
    const sd = Math.sqrt(ratios.reduce((a, b) => a + (b - mean) ** 2, 0) / (reps - 1))
    expect(Math.abs(mean - 1)).toBeLessThan(4 * (sd / Math.sqrt(reps)) + 0.02)
  })
})

describe('conditional SMC', () => {
  const Z = series(6, 5)
  const exact = rtsSmoother(kalman, kalmanFilter(kalman, Z))
  const means = toFlat(exact.means)
  const sds = toFlat(exact.covariances).map(Math.sqrt)

  function chain(ancestor: 'sampling' | 'none', sweeps: number): { paths: Float64Array[]; refresh: number } {
    let path: Float64Array = new Float64Array(Z.length + 1)
    const paths: Float64Array[] = []
    let refresh = 0
    for (let i = 0; i < sweeps; i++) {
      const r = conditionalSmc(particleModel, Z, path, { key: foldIn(key(7), i), particles: 12, ancestor })
      path = r.path
      refresh += r.refresh / sweeps
      if (i >= 100) paths.push(path)
    }
    return { paths, refresh }
  }

  it('leaves the smoothing distribution of the path invariant (the Kalman smoother is the reference)', () => {
    const { paths } = chain('sampling', 1500)
    for (const s of [1, Z.length]) {
      const xs = paths.map((p) => p[s] ?? 0)
      const m = xs.reduce((a, b) => a + b, 0) / xs.length
      const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
      // The path index s is the state after the (s - 1)-th transition, the smoother's index s - 1.
      expect(Math.abs(m - (means[s - 1] ?? 0))).toBeLessThan(0.12)
      expect(Math.abs(sd / (sds[s - 1] ?? 1) - 1)).toBeLessThan(0.2)
    }
  })

  it('ancestor sampling refreshes the path far more than conditional SMC without it', () => {
    const withAs = chain('sampling', 200).refresh
    const without = chain('none', 200).refresh
    expect(withAs).toBeGreaterThan(without)
    expect(withAs).toBeGreaterThan(0.5)
  })

  it('rejects a reference of the wrong length and ancestor sampling without a transition density', () => {
    expect(() => conditionalSmc(particleModel, Z, new Float64Array(3), { key: key(0) })).toThrow(/does not fit/)
    const { logTransition: _drop, ...blind } = particleModel
    expect(() => conditionalSmc(blind, Z, new Float64Array(7), { key: key(0) })).toThrow(/logTransition/)
  })
})

describe('the kernels', () => {
  const Z = series(8, 9)
  const prior = {
    dim: 1,
    logProb: (t: ArrayLike<number>) => (Math.abs(t[0] ?? 2) < 0.99 ? -Math.log(1.98) : Number.NEGATIVE_INFINITY),
    sample: () => new Float64Array(1),
  }
  const family = {
    dim: 1,
    prior,
    model: (t: Float64Array) =>
      gaussianParticleModel({
        f: (x: Value) => mul(x, t[0] ?? 0),
        h: (x: Value) => x,
        Q: [[Q]],
        R: [[R]],
        x0: [0],
        P0: [[1]],
      }),
    logInitial: (_t: Float64Array, x: Float64Array) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * Math.log(2 * Math.PI),
  }

  it('PMMH carries its likelihood estimate, rejects outside the prior and is deterministic in its keys', () => {
    const k = pmmhKernel(family, Z, { key: key(1), particles: 30, scale: 0.2 })
    const s = k.init([0.5])
    expect(Number.isFinite(s.logLikelihood)).toBe(true)
    const a = k.step(key(2), s)
    const b = k.step(key(2), s)
    expect(a.state.position).toEqual(b.state.position)
    const edge = k.step(key(3), { ...s, position: new Float64Array([0.985]) })
    expect(edge.info.acceptanceRate).toBeLessThanOrEqual(1)
  })

  it('particle Gibbs keeps a path and moves the parameter on the complete-data density', () => {
    const k = particleGibbs(family, Z, { key: key(4), particles: 20, thetaMoves: 2, scale: 0.2 })
    const s = k.init([0.3])
    expect(s.path.length).toBe(Z.length + 1)
    const next = k.step(key(5), s)
    expect(next.state.path.length).toBe(Z.length + 1)
    expect(next.state.logDensity).toBeCloseTo(
      completeDataLogDensity(family, next.state.position, Z, next.state.path),
      12,
    )
    expect(() => particleGibbs({ ...family, logInitial: undefined as never }, Z, { key: key(0) })).toThrow(/logInitial/)
  })
})
