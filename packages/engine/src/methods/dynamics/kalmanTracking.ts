/**
 * Kalman tracking: a point moving in the plane with a wandering velocity, seen through noisy position readings, and a
 * constant-velocity Kalman filter following it. The world is simulated from the run's key (the truth and the
 * observations), the filter is the library's, and each step puts the truth, the observation, the filtered position with
 * its covariance, the normalised innovation and the position error on the tape. The noise the filter assumes is a
 * knob, so the page shows what trusting the model too much or too little does to the estimate.
 */

import {
  type KalmanState,
  kalmanFilterAlgorithm,
  type LinearGaussianModel,
} from '../../lib/dynamics/filtering/kalman.ts'
import { ldsSample, linearDynamicalSystem } from '../../lib/pgm/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { dict, real } from '../../lib/space/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  defineMethod,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../lib/trace/index.ts'

/** The length of the simulated run, in steps. */
const STEPS = 300
/** The noise of the simulated world: acceleration spectral density and position reading standard deviation. */
const WORLD = { processNoise: 0.03, measurementNoise: 5 } as const

type Matrix = number[][]

/** The constant-velocity model of state `[x, y, vx, vy]` for a sampling interval and the noise levels assumed. */
function trackingMatrices(
  dt: number,
  processNoise: number,
  measurementNoise: number,
): { F: Matrix; H: Matrix; Q: Matrix; R: Matrix } {
  const q = processNoise * processNoise
  const [a, b, c] = [(q * dt ** 3) / 3, (q * dt ** 2) / 2, q * dt]
  return {
    F: [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    H: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    // Continuous white-noise acceleration, discretised: positive definite, unlike the rank-one piecewise-constant form.
    Q: [
      [a, 0, b, 0],
      [0, a, 0, b],
      [b, 0, c, 0],
      [0, b, 0, c],
    ],
    R: [
      [measurementNoise ** 2, 0],
      [0, measurementNoise ** 2],
    ],
  }
}

/** The simulated world: the true positions and the noisy readings of them, one pair per step. */
function simulate(dt: number, key: Key): { truth: number[][]; observations: number[][] } {
  const w = trackingMatrices(dt, WORLD.processNoise, WORLD.measurementNoise)
  const lds = linearDynamicalSystem({
    transition: w.F,
    observation: w.H,
    transitionCovariance: w.Q,
    observationCovariance: w.R,
    initialMean: [0, 0, 0.5, 0.25],
    initialCovariance: [
      [0.01, 0, 0, 0],
      [0, 0.01, 0, 0],
      [0, 0, 0.04, 0],
      [0, 0, 0, 0.04],
    ],
  })
  // The sampled initial state is the filter's prior; the first reading is of the step after it.
  const sample = ldsSample(lds, key, STEPS + 1)
  return {
    truth: sample.states.slice(1).map((s) => [s[0] ?? 0, s[1] ?? 0]),
    observations: sample.observations.slice(1),
  }
}

/** A filter state without its past: the tape keeps that, and a checkpoint stays small. */
function present(s: KalmanState): KalmanState {
  return { ...s, means: [], covs: [], pmeans: [], pcovs: [], observed: [], nis: s.nis.slice(-1) }
}

/**
 * Track a moving point with a Kalman filter: a constant-velocity model in the plane, 300 steps of a simulated truth
 * and noisy position readings, and the library's Kalman filter run over them one reading per step. The world has fixed
 * noise (acceleration 0.03, reading noise 5); `processNoise` and `measurementNoise` are what the filter assumes, so
 * the defaults are matched and moving either shows over- or under-trusting the model (the normalised innovation
 * drifts away from 2). On the tape: `truth`, `observation`, `estimate` (the filtered position) and `covariance` (its
 * 2-by-2 block) at every step, with the scalars `nis` and `positionError`.
 *
 * @example
 * ```ts
 * import { kalmanTracking } from 'mdml-engine/methods/dynamics'
 * import { session } from 'mdml-engine/trace'
 * session(kalmanTracking, { mode: 'eager', steps: 10 }).t // => 10
 * ```
 */
export const kalmanTracking: Method<KalmanState, Config> = defineMethod<KalmanState, Config>({
  name: 'dynamics/kalman-tracking',
  config: dict({
    processNoise: real(0.001, 10, {
      log: true,
      default: WORLD.processNoise,
      label: 'process noise',
      description: `The acceleration noise the filter assumes (the world’s is ${WORLD.processNoise}).`,
    }),
    measurementNoise: real(0.1, 20, {
      log: true,
      default: WORLD.measurementNoise,
      label: 'measurement noise',
      description: `The standard deviation of the position readings the filter assumes (the world’s is ${WORLD.measurementNoise}).`,
    }),
    dt: real(0.1, 2, { default: 1, label: 'time step', description: 'The time between readings.' }),
  }),
  make: (c, key): Algorithm<KalmanState> => {
    const dt = c['dt'] as number
    const { truth, observations } = simulate(dt, key)
    const m = trackingMatrices(dt, c['processNoise'] as number, c['measurementNoise'] as number)
    const model: LinearGaussianModel = {
      F: m.F,
      H: m.H,
      Q: m.Q,
      R: m.R,
      x0: [0, 0, 0, 0],
      P0: [
        [25, 0, 0, 0],
        [0, 25, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
    }
    const inner = kalmanFilterAlgorithm(model, observations).algorithm
    return {
      name: 'kalman-tracking',
      config: { dt, processNoise: c['processNoise'] as number, measurementNoise: c['measurementNoise'] as number },
      init: inner.init,
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        const x = toFlat(next.x)
        const P = toFlat(next.P)
        const [tx, ty] = truth[ctx.t] ?? [0, 0]
        const estimate = [x[0] ?? 0, x[1] ?? 0]
        ctx.log('truth', [tx ?? 0, ty ?? 0])
        ctx.log('observation', observations[ctx.t] ?? [0, 0])
        ctx.log('estimate', estimate)
        ctx.log('covariance', [
          [P[0] ?? 0, P[1] ?? 0],
          [P[4] ?? 0, P[5] ?? 0],
        ])
        ctx.log('nis', next.nis.at(-1) ?? 0)
        ctx.log('positionError', Math.hypot((estimate[0] ?? 0) - (tx ?? 0), (estimate[1] ?? 0) - (ty ?? 0)))
        return present(next)
      },
      done: (_, t) => (t >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(present(s)),
      deserialise: (j) => stateFromJson(j) as KalmanState,
    }
  },
  contract: {
    truth: tensor({ role: 'state', shape: [2], description: 'the true position' }),
    observation: tensor({ role: 'state', shape: [2], description: 'the noisy position reading' }),
    estimate: tensor({ role: 'state', shape: [2], description: 'the filtered position' }),
    covariance: tensor({ role: 'state', shape: [2, 2], description: 'the filtered position’s covariance' }),
    nis: scalar({ description: 'the normalised innovation squared of this step (about 2 for an honest filter)' }),
    positionError: scalar({ direction: 'lower', description: 'the distance from the estimate to the true position' }),
  },
  // Tensors are drawn element by element, so they are kept in full (`all` keeps summaries).
  defaults: {
    truth: Log({ detail: 'full' }),
    observation: Log({ detail: 'full' }),
    estimate: Log({ detail: 'full' }),
    covariance: Log({ detail: 'full' }),
    nis: Log(),
    positionError: Log(),
  },
})
