// Guided proposals of the particle filters against the Kalman filter's exact likelihood: the evidence estimate of the
// bootstrap, auxiliary and guided filters is unbiased (the mean of Z-hat / Z over replications is one within its standard
// error), the locally optimal proposal has the smallest variance on an informative linear-Gaussian model, and on a
// nonlinear model every filter lands on the same evidence.
import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../../random/index.ts'
import { add, div, item, mul, square, toFlat, type Value } from '../../tensor/index.ts'
import { gaussianGuidedModel } from './guided.ts'
import { kalmanLogLikelihood, type LinearGaussianModel } from './kalman.ts'
import type { NonlinearModel } from './nonlinear.ts'
import { auxiliaryParticleFilter, bootstrapParticleFilter } from './particle.ts'
import { gaussianParticleModel } from './particle-options.ts'

const PHI = 0.9
const Q = 0.09
const R = 0.04
const linear: NonlinearModel = {
  f: (x: Value) => mul(x, PHI),
  h: (x: Value) => x,
  Q: [[Q]],
  R: [[R]],
  x0: [0],
  P0: [[1]],
}
const kalman: LinearGaussianModel = { F: [[PHI]], H: [[1]], Q: [[Q]], R: [[R]], x0: [0], P0: [[1]] }

// A series from the model itself (x_{-1} ~ N(0, 1), then the transition and the observation at every step).
function simulate(seed: number, T: number): number[][] {
  const e = toFlat(normal(key(seed), [T]))
  const o = toFlat(normal(key(seed + 1000), [T]))
  let x = toFlat(normal(key(seed + 2000), [1]))[0] ?? 0
  const z: number[][] = []
  for (let t = 0; t < T; t++) {
    x = PHI * x + Math.sqrt(Q) * (e[t] ?? 0)
    z.push([x + Math.sqrt(R) * (o[t] ?? 0)])
  }
  return z
}

const Z = simulate(3, 12)
const exact = item(kalmanLogLikelihood(kalman, Z))

function replicate(run: (k: ReturnType<typeof key>) => number, reps: number): { ratios: number[]; logs: number[] } {
  const logs = Array.from({ length: reps }, (_, i) => run(foldIn(key(11), i)))
  return { ratios: logs.map((l) => Math.exp(l - exact)), logs }
}

const meanOf = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length
const sdOf = (v: number[]): number => {
  const m = meanOf(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
}

describe('guided particle filters on a linear-Gaussian model', () => {
  const bootstrapModel = gaussianParticleModel(linear)
  const guidedModel = gaussianGuidedModel(linear, { observationMatrix: [[1]] })
  const reps = 30

  it('every filter estimates the evidence without bias (the mean of Z-hat / Z is one within its standard error)', () => {
    const runs = {
      bootstrap: replicate(
        (k) => bootstrapParticleFilter(bootstrapModel, Z, { particles: 80, key: k }).logEvidence,
        reps,
      ),
      auxiliary: replicate(
        (k) => auxiliaryParticleFilter(bootstrapModel, Z, { particles: 80, key: k }).logEvidence,
        reps,
      ),
      guided: replicate((k) => bootstrapParticleFilter(guidedModel, Z, { particles: 80, key: k }).logEvidence, reps),
    }
    for (const [name, r] of Object.entries(runs)) {
      const se = sdOf(r.ratios) / Math.sqrt(reps)
      expect(Math.abs(meanOf(r.ratios) - 1), name).toBeLessThan(4 * se + 0.02)
    }
  })

  it('the locally optimal proposal has a far smaller variance of log Z-hat than the bootstrap filter', () => {
    const boot = replicate(
      (k) => bootstrapParticleFilter(bootstrapModel, Z, { particles: 80, key: k }).logEvidence,
      reps,
    )
    const guided = replicate(
      (k) => bootstrapParticleFilter(guidedModel, Z, { particles: 80, key: k }).logEvidence,
      reps,
    )
    expect(sdOf(guided.logs)).toBeLessThan(0.5 * sdOf(boot.logs))
    expect(Math.abs(meanOf(guided.logs) - exact)).toBeLessThan(0.1)
  })

  it('the guided filter composes with the auxiliary first stage and every resampling scheme', () => {
    for (const resampling of ['multinomial', 'systematic', 'stratified', 'residual'] as const) {
      const r = bootstrapParticleFilter(guidedModel, Z, { particles: 200, key: key(5), resampling, essThreshold: 1 })
      expect(Math.abs(r.logEvidence - exact)).toBeLessThan(0.5)
    }
    const aux = auxiliaryParticleFilter(guidedModel, Z, { particles: 200, key: key(6) })
    expect(Math.abs(aux.logEvidence - exact)).toBeLessThan(0.5)
  })

  it('the Jacobian of the observation can come from autodiff (same proposal as the constant matrix)', () => {
    const viaAutodiff = gaussianGuidedModel(linear)
    const a = bootstrapParticleFilter(viaAutodiff, Z, { particles: 60, key: key(9) })
    const b = bootstrapParticleFilter(guidedModel, Z, { particles: 60, key: key(9) })
    expect(Math.abs(a.logEvidence - b.logEvidence)).toBeLessThan(1e-8)
  })
})

describe('guided particle filters on a nonlinear model', () => {
  // The classic nonlinear benchmark without its forcing term: x_t = x/2 + 25 x / (1 + x^2) + w, z = x^2 / 20 + v.
  const nonlinear: NonlinearModel = {
    f: (x: Value) => add(mul(x, 0.5), mul(div(x, add(square(x), 1)), 25)),
    h: (x: Value) => div(square(x), 20),
    Q: [[4]],
    R: [[0.5]],
    x0: [0],
    P0: [[1]],
  }
  const series = (() => {
    const e = toFlat(normal(key(21), [10]))
    const o = toFlat(normal(key(22), [10]))
    let x = 0.1
    return Array.from({ length: 10 }, (_, t) => {
      x = 0.5 * x + (25 * x) / (1 + x * x) + 2 * (e[t] ?? 0)
      return [(x * x) / 20 + Math.sqrt(0.5) * (o[t] ?? 0)]
    })
  })()

  it('the bootstrap, auxiliary and guided filters land on the same evidence (within the spread of the estimates)', () => {
    const reference = bootstrapParticleFilter(gaussianParticleModel(nonlinear), series, {
      particles: 4000,
      key: key(31),
    })
    const guided = gaussianGuidedModel(nonlinear)
    const g = Array.from(
      { length: 6 },
      (_, i) => bootstrapParticleFilter(guided, series, { particles: 100, key: foldIn(key(32), i) }).logEvidence,
    )
    const b = Array.from(
      { length: 6 },
      (_, i) =>
        bootstrapParticleFilter(gaussianParticleModel(nonlinear), series, { particles: 400, key: foldIn(key(33), i) })
          .logEvidence,
    )
    const a = auxiliaryParticleFilter(gaussianParticleModel(nonlinear), series, { particles: 400, key: key(34) })
    // Z-hat is unbiased, log Z-hat is biased low by about var / 2: compare the means with the reference loosely.
    expect(Math.abs(meanOf(g) - reference.logEvidence)).toBeLessThan(1)
    expect(Math.abs(meanOf(b) - reference.logEvidence)).toBeLessThan(1)
    expect(Math.abs(a.logEvidence - reference.logEvidence)).toBeLessThan(1.5)
  })
})
