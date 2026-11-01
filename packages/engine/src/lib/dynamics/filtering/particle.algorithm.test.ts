import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { add, mul, square, sub, sum, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { kalmanFilter } from './kalman.ts'
import {
  auxiliaryParticleAlgorithm,
  auxiliaryParticleFilter,
  bootstrapParticleAlgorithm,
  bootstrapParticleFilter,
  type ParticleModel,
  pfContract,
  type Resampling,
} from './particle.ts'
import { gaussianParticleModel, particleProposals, resamplers } from './particle-options.ts'

const q = 0.2
const r = 0.5
const model = {
  initial: (k: never, n: number): Value => normal(k, [n, 1]),
  transition: (k: never, x: Value): Value => add(x, mul(normal(k, x.shape), Math.sqrt(q))),
  logLikelihood: (z: Value, x: Value): Value =>
    sub(mul(sum(square(sub(x, z)), -1), -0.5 / r), 0.5 * Math.log(2 * Math.PI * r)),
  predict: (x: Value): Value => x,
} as unknown as ParticleModel
const zs = [[0.3], [0.9], [0.5], [1.4], [1.1], [1.8]]

// What the functions returned before they became runs of the algorithms (20 particles, 13 significant digits).
const OLD: Record<string, { means: number[]; logEvidence: number; ess: number[]; resampled: boolean[] }> = {
  multinomial: {
    means: [0.1530116710346, 0.5580007875811, 0.4944481574307, 1.038383244274, 1.086141441923, 1.636893642921],
    logEvidence: -6.686652153037,
    ess: [20, 13.23980124407, 14.67085657711, 11.7998406578, 13.52204806913, 17.51977500132],
    resampled: [false, true, false, true, true, false],
  },
  systematic: {
    means: [0.1530116710346, 0.651391282326, 0.5708582458872, 1.015447493945, 1.041083049429, 1.420064836476],
    logEvidence: -6.549725028908,
    ess: [20, 13.23980124407, 13.98375725696, 16.30167124685, 11.37231584154, 19.33731288642],
    resampled: [false, true, true, false, true, false],
  },
  stratified: {
    means: [0.1530116710346, 0.5692551147119, 0.5638859515891, 0.951949505556, 1.129644098744, 1.550194045076],
    logEvidence: -6.727535189837,
    ess: [20, 13.23980124407, 13.82073654674, 16.68368176887, 10.89558923965, 16.27748618931],
    resampled: [false, true, true, false, true, false],
  },
  residual: {
    means: [0.1530116710346, 0.6026303146558, 0.500642508035, 0.7180572936244, 0.9943684253932, 1.297788280265],
    logEvidence: -7.318437200023,
    ess: [20, 13.23980124407, 14.07981948781, 11.20354157266, 15.17996623332, 12.89993878178],
    resampled: [false, true, false, true, false, true],
  },
}
const OLD_APF = {
  means: [0.07864041265183, 0.6050397061491, 0.5574855687913, 1.162489826567, 1.083672858062, 1.325600281476],
  logEvidence: -7.044495786179,
  ess: [18.41517095258, 11.77303127798, 18.60735250074, 17.07623649845, 18.63353350567, 18.01884325002],
}

const near = (a: ArrayLike<number>, b: readonly number[]): void => {
  expect(a.length).toBe(b.length)
  b.forEach((v, i) => {
    expect(Math.abs((a[i] ?? 0) - v)).toBeLessThan(1e-11 * Math.max(1, Math.abs(v)))
  })
}

describe('particle filter algorithms', () => {
  it('are the filters: the algorithm run to the end gives the numbers the functions gave before', () => {
    for (const [scheme, old] of Object.entries(OLD)) {
      const options = { particles: 20, key: key(1), resampling: scheme as Resampling, essThreshold: 0.7 }
      const s = runAlgorithm(bootstrapParticleAlgorithm(model, zs, options), { maxSteps: zs.length }).state
      near(Float64Array.from(s.means.flatMap((m) => Array.from(m))), old.means)
      near(s.ess, old.ess)
      expect(s.resampled).toEqual(old.resampled)
      expect(s.logEvidence).toBeCloseTo(old.logEvidence, 10)
      const fn = bootstrapParticleFilter(model, zs, options)
      expect(fn.logEvidence).toBe(s.logEvidence)
      expect(fn.ess).toEqual(s.ess)
    }
    const a = runAlgorithm(auxiliaryParticleAlgorithm(model, zs, { particles: 20, key: key(2) }), {
      maxSteps: zs.length,
    }).state
    near(Float64Array.from(a.means.flatMap((m) => Array.from(m))), OLD_APF.means)
    near(a.ess, OLD_APF.ess)
    expect(a.logEvidence).toBeCloseTo(OLD_APF.logEvidence, 10)
    expect(auxiliaryParticleFilter(model, zs, { particles: 20, key: key(2) }).logEvidence).toBe(a.logEvidence)
  })

  it('have a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    for (const alg of [
      bootstrapParticleAlgorithm(model, zs, { particles: 30, key: key(4), essThreshold: 0.9 }),
      auxiliaryParticleAlgorithm(model, zs, { particles: 30, key: key(5) }),
    ]) {
      const whole = runAlgorithm(alg, { maxSteps: zs.length }).state
      const half = runAlgorithm(alg, { maxSteps: 3 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 3, start: { state: stored, step: 3, serialised: true } })
      expect(rest.steps).toBe(zs.length)
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    }
  })

  it('logs its contract by path with the right shapes', () => {
    const rec = recorder()
    runAlgorithm(bootstrapParticleAlgorithm(model, zs, { particles: 16, key: key(1) }), {
      maxSteps: 4,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const shapes = (name: string): number[][] =>
      rec.events.flatMap((e) => (e.type === 'tensor' && e.name === name ? [[...e.shape]] : []))
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(shapes('pf/particles')).toEqual(Array(4).fill([16, 1]))
    expect(shapes('pf/weights')).toEqual(Array(4).fill([16]))
    expect(shapes('pf/ancestors')).toEqual(Array(4).fill([16]))
    expect(shapes('pf/mean')).toEqual(Array(4).fill([1]))
    expect(metric('pf/ess')).toHaveLength(4)
    expect(metric('pf/resampled').every((v) => v === 0 || v === 1)).toBe(true)
    expect(Object.keys(pfContract)).toHaveLength(7)
  })

  it('keeps its weights a distribution, its ESS within (0, N] and its ancestors indices of the cloud', () => {
    const N = 25
    const s = runAlgorithm(bootstrapParticleAlgorithm(model, zs, { particles: N, key: key(9), essThreshold: 1 }), {
      maxSteps: zs.length,
    }).state
    for (const w of s.weights) expect(Math.abs(Array.from(w).reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-12)
    expect(s.ess.every((e) => e > 0 && e <= N + 1e-9)).toBe(true)
    expect(s.ancestors.every((a) => Number.isInteger(a) && a >= 0 && a < N)).toBe(true)
    // With a threshold of one it resamples at every step but the first (equal weights give ESS = N, not below).
    expect(s.resampled.slice(1).every(Boolean)).toBe(true)
  })

  it('with a Gaussian model the evidence tracks the Kalman filter, through the bridge from the nonlinear form', () => {
    const lin = { f: (x: Value) => x, h: (x: Value) => x, Q: [[q]], R: [[r]], x0: [0], P0: [[1]] }
    const exact = kalmanFilter({ F: [[1]], H: [[1]], Q: [[q]], R: [[r]], x0: [0], P0: [[1]] }, zs)
    const s = runAlgorithm(
      bootstrapParticleAlgorithm(gaussianParticleModel(lin), zs, { particles: 3000, key: key(1) }),
      {
        maxSteps: zs.length,
      },
    ).state
    const truth = toFlat(exact.means)
    s.means.forEach((m, i) => {
      expect(Math.abs((m[0] ?? 0) - (truth[i] ?? 0))).toBeLessThan(0.06)
    })
  })

  it('run every option of the resampling and proposal tables', () => {
    for (const name of resamplers.names) {
      const s = runAlgorithm(
        bootstrapParticleAlgorithm(model, zs, {
          particles: 20,
          key: key(1),
          resampling: resamplers.make({ kind: name }),
        }),
        { maxSteps: zs.length },
      ).state
      expect(s.t).toBe(zs.length)
    }
    expect(resamplers.where((p) => p.keepsIntegerCopies)).toEqual(['residual'])
    expect(particleProposals.where((p) => p.needsPredict)).toEqual(['auxiliary'])
    expect(particleProposals.make({ kind: 'auxiliary' }).auxiliary).toBe(true)
  })
})
