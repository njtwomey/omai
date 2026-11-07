/**
 * The slots of nonlinear state estimation as option tables: the filter (the extended Kalman filter, the unscented
 * Kalman filter with Van der Merwe or Julier sigma points and its knobs, the ensemble Kalman filter with its size and
 * inflation), each building a steppable algorithm over a model and observations, and the state-space model (a random
 * walk, a pendulum observed through its sine, a body tracked by its range) with the noise as knobs. A model also
 * simulates itself, so a filter has a truth to be compared with. Each table is one `union` space for controls, grids
 * and tuners.
 */

import { MdmlError } from '../../base/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  bool,
  type DictSpace,
  dict,
  fillKnobs,
  int,
  type Knobs,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../../space/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import {
  add,
  asTensor,
  matmul,
  mul,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  square,
  stack,
  sum,
  type Tensor,
  type Value,
} from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import {
  ensembleKalmanAlgorithm,
  extendedKalmanAlgorithm,
  julierSigmaPoints,
  merweSigmaPoints,
  type NonlinearFilterPath,
  type NonlinearFilterState,
  type NonlinearModel,
  unscentedKalmanAlgorithm,
} from './nonlinear.ts'

/** The paths any filter of the {@link nonlinearFilters} table may log. */
export type AnyNonlinearFilterPath = NonlinearFilterPath | 'filter/sigma-points' | 'filter/ensemble'

/** A filter of the table: the algorithm over a model and its observations (the key feeds the ensemble's draws). */
export type NonlinearFilterScheme = (
  model: NonlinearModel,
  observations: TensorLike,
  key: Key,
) => Algorithm<NonlinearFilterState, AnyNonlinearFilterPath>

/** What a filter does, for validity rules and selectors. */
export interface NonlinearFilterProperties {
  /** Whether it linearises the dynamics (a Jacobian) rather than sampling them. */
  readonly linearises: boolean
  /** Whether it is random (the ensemble draws noise; the others are deterministic). */
  readonly stochastic: boolean
  /** What the state holds beyond the moments: nothing, the sigma points, or the ensemble. */
  readonly carries: 'moments' | 'sigma points' | 'ensemble'
}

function scheme(
  description: string,
  properties: NonlinearFilterProperties,
  space: DictSpace,
  build: (k: Knobs, config: Readonly<Record<string, SpaceValue>>) => NonlinearFilterScheme,
): Option<NonlinearFilterScheme, NonlinearFilterProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return build((name) => Number(full[name]), full)
    },
  }
}

// The algorithms carry a richer state than the common one (sigma points, an ensemble); the table hands them out as the
// common algorithm, whose steps read only the common fields and pass the rest through.
const asCommon = (a: unknown): Algorithm<NonlinearFilterState, AnyNonlinearFilterPath> =>
  a as Algorithm<NonlinearFilterState, AnyNonlinearFilterPath>

/**
 * The nonlinear Kalman filters as one slot: `extended` (a Jacobian by autodiff), `unscented` (Van der Merwe's scaled
 * sigma points, with alpha, beta and kappa), `unscented-julier` (Julier's points with kappa) and `ensemble` (perturbed
 * observations with a size and a multiplicative inflation). Each makes a steppable algorithm over a model and its
 * observations.
 *
 * @concept unscented-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.nonlinearFilters.names // => ['extended', 'unscented', 'unscented-julier', 'ensemble']
 * ```
 */
export const nonlinearFilters: OptionTable<NonlinearFilterScheme, NonlinearFilterProperties> = optionTable(
  {
    extended: scheme(
      'linearise the dynamics and the observation at the estimate with their Jacobians',
      { linearises: true, stochastic: false, carries: 'moments' },
      dict({}),
      () => (model, observations) => asCommon(extendedKalmanAlgorithm(model, observations)),
    ),
    unscented: scheme(
      'push Van der Merwe’s scaled sigma points through the dynamics and the observation',
      { linearises: false, stochastic: false, carries: 'sigma points' },
      dict({
        alpha: real(0.001, 1, { log: true, default: 0.1, label: 'spread α' }),
        beta: real(0, 5, { default: 2, label: 'prior knowledge β (2 for Gaussians)' }),
        kappa: real(-10, 10, { default: 0, label: 'secondary scaling κ' }),
        redraw: bool({ default: false, label: 'redraw the sigma points of the prediction' }),
      }),
      (k, c) => (model, observations) =>
        asCommon(
          unscentedKalmanAlgorithm(model, observations, {
            points: merweSigmaPoints(vecSize(model), k('alpha'), k('beta'), k('kappa')),
            redraw: c['redraw'] === true,
          }),
        ),
    ),
    'unscented-julier': scheme(
      'push Julier’s symmetric sigma points, with one spread parameter, through the dynamics and the observation',
      { linearises: false, stochastic: false, carries: 'sigma points' },
      dict({
        kappa: real(-10, 10, { default: 1, label: 'spread κ' }),
        redraw: bool({ default: false, label: 'redraw the sigma points of the prediction' }),
      }),
      (k, c) => (model, observations) =>
        asCommon(
          unscentedKalmanAlgorithm(model, observations, {
            points: julierSigmaPoints(vecSize(model), k('kappa')),
            redraw: c['redraw'] === true,
          }),
        ),
    ),
    ensemble: scheme(
      'propagate an ensemble of members and update each with its own perturbed observation',
      { linearises: false, stochastic: true, carries: 'ensemble' },
      dict({
        size: int(2, 5000, { log: true, default: 100, label: 'ensemble members' }),
        inflation: real(1, 2, { default: 1, label: 'covariance inflation' }),
      }),
      (k) => (model, observations, key) =>
        asCommon(ensembleKalmanAlgorithm(model, observations, { size: k('size'), inflation: k('inflation'), key })),
    ),
  },
  {
    default: 'extended',
    label: 'filter',
    description: 'How the filter carries the state estimate through a nonlinear model.',
  },
)

function vecSize(model: NonlinearModel): number {
  return asTensor(model.x0).shape.reduce((a, b) => a * b, 1)
}

/** A model that simulates: the filtering model, and the truth of a draw from it. */
export interface StateSpaceModel extends NonlinearModel {
  /** The names of the state's components. */
  readonly stateNames: readonly string[]
}

/** What a state-space model is about, for validity rules and selectors. */
export interface StateSpaceProperties {
  /** The state's dimension. */
  readonly dimension: number
  /** Whether the observation is a nonlinear function of the state. */
  readonly nonlinearObservation: boolean
  /** Whether the dynamics are nonlinear. */
  readonly nonlinearDynamics: boolean
}

const at = (x: Value, i: number): Value => sum(sliceAxis(x, 0, i, i + 1))
const diag = (...v: number[]): number[][] => v.map((a, i) => v.map((_, j) => (i === j ? a : 0)))

function ssModel(
  description: string,
  properties: StateSpaceProperties,
  space: DictSpace,
  build: (k: Knobs) => StateSpaceModel,
): Option<StateSpaceModel, StateSpaceProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return build((name) => Number(full[name]))
    },
  }
}

const noise = (label: string, def: number): ReturnType<typeof real> =>
  real(0.001, 10, { log: true, default: def, label })

/**
 * The state-space models as one slot: `random-walk` (a level observed with noise), `pendulum` (an angle and its rate,
 * observed through the sine of the angle) and `range` (a position and velocity, observed through the range to a
 * sensor at a fixed offset). The noise standard deviations are knobs.
 *
 * @concept extended-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.stateSpaceModels.make({ kind: 'pendulum', dt: 0.1 }).stateNames // => ['angle', 'rate']
 * ```
 */
export const stateSpaceModels: OptionTable<StateSpaceModel, StateSpaceProperties> = optionTable(
  {
    'random-walk': ssModel(
      'a level that drifts as a random walk, observed with noise',
      { dimension: 1, nonlinearObservation: false, nonlinearDynamics: false },
      dict({
        processStd: noise('process noise standard deviation', 0.3),
        observationStd: noise('observation noise standard deviation', 0.5),
      }),
      (k) => ({
        stateNames: ['level'],
        f: (x) => x,
        h: (x) => x,
        Q: diag(k('processStd') ** 2),
        R: diag(k('observationStd') ** 2),
        x0: [0],
        P0: diag(1),
      }),
    ),
    pendulum: ssModel(
      'a pendulum (angle and rate, Euler steps) of which only the sine of the angle is observed',
      { dimension: 2, nonlinearObservation: true, nonlinearDynamics: true },
      dict({
        dt: real(0.001, 1, { log: true, default: 0.1, label: 'time step' }),
        gravity: real(0.1, 50, { log: true, default: 9.81, label: 'g / length' }),
        processStd: noise('process noise standard deviation', 0.03),
        observationStd: noise('observation noise standard deviation', 0.2),
      }),
      (k) => ({
        stateNames: ['angle', 'rate'],
        f: (x) =>
          stack([add(at(x, 0), mul(at(x, 1), k('dt'))), add(at(x, 1), mul(sin(at(x, 0)), -k('dt') * k('gravity')))]),
        h: (x) => stack([sin(at(x, 0))]),
        Q: diag(k('processStd') ** 2, k('processStd') ** 2),
        R: diag(k('observationStd') ** 2),
        x0: [0.5, 0],
        P0: diag(0.1, 0.1),
      }),
    ),
    range: ssModel(
      'a body moving at near-constant velocity, observed only through its distance to a sensor off its path',
      { dimension: 2, nonlinearObservation: true, nonlinearDynamics: false },
      dict({
        dt: real(0.01, 5, { log: true, default: 0.5, label: 'time step' }),
        offset: real(0.1, 100, { log: true, default: 2, label: 'distance of the sensor from the path' }),
        processStd: noise('process noise standard deviation', 0.05),
        observationStd: noise('observation noise standard deviation', 0.2),
      }),
      (k) => ({
        stateNames: ['position', 'velocity'],
        f: (x) => stack([add(at(x, 0), mul(at(x, 1), k('dt'))), at(x, 1)]),
        h: (x) => stack([sqrt(add(square(at(x, 0)), k('offset') ** 2))]),
        Q: diag(k('processStd') ** 2, k('processStd') ** 2),
        R: diag(k('observationStd') ** 2),
        x0: [-3, 1],
        P0: diag(1, 1),
      }),
    ),
  },
  {
    default: 'random-walk',
    label: 'state-space model',
    description: 'The hidden process and what is observed of it, with the noise of each.',
  },
)

/**
 * A draw from a state-space model: the hidden states and the observations they produced.
 *
 * @param model - The model.
 * @param steps - Number of time steps.
 * @param key - The randomness.
 * @returns The true states `[steps, n]` and the observations `[steps, p]`.
 * @throws MdmlError with code `value` for fewer than one step.
 * @concept extended-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { key } from 'mdml-engine/random'
 * const { states, observations } = filtering.simulateStateSpace(filtering.stateSpaceModels.make({ kind: 'range' }), 5, key(0))
 * [states.shape, observations.shape] // => [[5, 2], [5, 1]]
 * ```
 */
export function simulateStateSpace(
  model: NonlinearModel,
  steps: number,
  key: Key,
): { readonly states: Tensor; readonly observations: Tensor } {
  if (!(steps >= 1)) throw new MdmlError('value', 'simulateStateSpace: need at least one step')
  const x0 = reshape(asTensor(model.x0), [-1])
  const n = x0.shape[0] ?? 0
  const LQ = cholesky(asTensor(model.Q))
  const LR = cholesky(asTensor(model.R))
  const p = asTensor(model.R).shape[0] ?? 0
  const LP = cholesky(asTensor(model.P0))
  let x: Value = add(x0, reshape(matmul(LP, reshape(normal(foldIn(key, 0), [n]), [n, 1])), [n]))
  const states: Value[] = []
  const obs: Value[] = []
  for (let t = 0; t < steps; t++) {
    const k = foldIn(key, t + 1)
    x = add(model.f(x), reshape(matmul(LQ, reshape(normal(foldIn(k, 0), [n]), [n, 1])), [n]))
    states.push(x)
    obs.push(add(model.h(x), reshape(matmul(LR, reshape(normal(foldIn(k, 1), [p]), [p, 1])), [p])))
  }
  return {
    states: asTensor(stack(states)) as Tensor,
    observations: asTensor(stack(obs)) as Tensor,
  }
}
