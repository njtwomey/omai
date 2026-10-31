/**
 * Linear function approximation for reinforcement learning (Sutton and Barto, Chapters 9 to 12): tile coding in
 * Sutton's form (overlapping grids with asymmetric offsets and optional hashing), sparse feature maps over
 * observations, and the learners that run on them as agents with serialisable state: semi-gradient TD(lambda) with
 * accumulating, replacing and dutch (true online) traces, semi-gradient Sarsa(lambda) for control, the gradient-TD
 * methods GTD2 and TDC, and emphatic TD(lambda), the last three being the answers to the divergence of off-policy
 * semi-gradient TD that Baird's counterexample shows. The learners take an optional gradient transformation from
 * `optim` as their update rule, so the direction `delta z` can be fed to momentum or Adam instead of a plain step.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { applyUpdates, type GradientTransformation } from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import type { Agent, Transition } from '../rl/index.ts'
import type { Space } from '../space/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { type Rate, rateAt, unit } from './_tabular.ts'

// ------------------------------------------------------------------------------------------------ feature maps

/** Sparse real features: the positions of the non-zero entries and their values (all ones when `values` is absent). */
export interface SparseFeatures {
  readonly indices: readonly number[]
  readonly values?: readonly number[]
}

/** A map from observations to sparse features of a fixed dimension. */
export interface FeatureMap<O> {
  /** The number of features. */
  readonly size: number
  features(observation: O): SparseFeatures
}

/** A map from observations and actions to sparse features, for action values. */
export interface ActionFeatureMap<O> {
  /** The number of features (all actions). */
  readonly size: number
  features(observation: O, action: number): SparseFeatures
}

/** The inner product of a weight vector and sparse features. */
function dotSparse(w: ArrayLike<number>, f: SparseFeatures): number {
  let total = 0
  for (const [k, i] of f.indices.entries()) total += (w[i] ?? 0) * (f.values?.[k] ?? 1)
  return total
}

/**
 * The value of sparse features under a weight vector.
 *
 * @param w - The weights.
 * @param f - The features.
 * @returns `w . f`.
 * @example
 * ```ts
 * import { featureValue } from 'mdml-engine/agents'
 * featureValue([1, 2, 3], { indices: [0, 2] }) // => 4
 * ```
 */
export function featureValue(w: ArrayLike<number>, f: SparseFeatures): number {
  return dotSparse(w, f)
}

/**
 * One-hot features of a state index: the tabular case of linear function approximation.
 *
 * @param n - The number of states.
 * @returns The feature map over observations that are state indices.
 * @example
 * ```ts
 * import { oneHotFeatures } from 'mdml-engine/agents'
 * oneHotFeatures(4).features(2).indices // => [2]
 * ```
 */
export function oneHotFeatures(n: number): FeatureMap<number> {
  return { size: n, features: (s) => ({ indices: [s] }) }
}

/**
 * Dense features from a matrix with one row per state: row `s` is the feature vector of state index `s`; zero entries are
 * left out of the sparse form.
 *
 * @param matrix - The rows.
 * @returns The feature map over state indices.
 * @example
 * ```ts
 * import { matrixFeatures } from 'mdml-engine/agents'
 * matrixFeatures([[2, 0, 1], [0, 1, 1]]).features(0) // => { indices: [0, 2], values: [2, 1] }
 * ```
 */
export function matrixFeatures(matrix: readonly (readonly number[])[]): FeatureMap<number> {
  const size = matrix[0]?.length ?? 0
  return {
    size,
    features: (s) => {
      const indices: number[] = []
      const values: number[] = []
      for (const [j, x] of (matrix[s] ?? []).entries()) {
        if (x !== 0) {
          indices.push(j)
          values.push(x)
        }
      }
      return { indices, values }
    },
  }
}

// ----------------------------------------------------------------------------------------------- tile coding

/** How the tilings are displaced from one another. */
export type TileOffsets = 'asymmetric' | 'symmetric' | 'none'

/** Options of {@link tileCoder}. */
export interface TileCoderOptions {
  /** The lower bound of each input dimension. */
  readonly low: readonly number[]
  /** The upper bound of each input dimension. */
  readonly high: readonly number[]
  /** The number of overlapping tilings (default 8). */
  readonly tilings?: number
  /** The number of tiles per dimension within one tiling: one number or one per dimension (default 8). */
  readonly tiles?: number | readonly number[]
  /**
   * The displacement of tiling `t`: `asymmetric` (default) moves dimension `d` by `t (2 d + 1) / tilings` of a tile,
   * Albus' first odd numbers, which avoids the diagonal stripes of `symmetric` (every dimension by `t / tilings`);
   * `none` stacks the tilings (one grid, `tilings` times).
   */
  readonly offsets?: TileOffsets
  /** Hash the tile coordinates into this many weights (default: no hashing, one weight per tile). */
  readonly memory?: number
}

/** A tile coder: the active tile of each tiling for an input point. */
export interface TileCoder {
  /** The input dimension. */
  readonly dims: number
  /** The number of tilings, which is the number of active features. */
  readonly tilings: number
  /** The number of features: tiles over all tilings, or the hash memory. */
  readonly size: number
  /** The index of the active tile of every tiling. */
  indices(x: readonly number[]): number[]
}

const hashCoordinates = (tiling: number, q: readonly number[], memory: number): number => {
  let h = Math.imul(tiling + 1, 0x9e3779b1) >>> 0
  for (const c of q) {
    h = Math.imul(h ^ (c + 0x7f4a7c15), 0x85ebca6b) >>> 0
    h ^= h >>> 13
    h = Math.imul(h, 0xc2b2ae35) >>> 0
    h ^= h >>> 16
  }
  return (h >>> 0) % memory
}

/**
 * A tile coder in the manner of Sutton's tiles: `tilings` overlapping grids, each displaced by a fraction of a tile, so a
 * point activates exactly one tile per tiling and generalises to the points that share tiles with it. A grid has
 * `tiles + 1` cells per dimension, the last making room for the displacement, and the index of the active cell is its
 * mixed-radix code offset by the tiling, or a hash of the tiling and the cell coordinates modulo `memory`.
 *
 * @param options - The bounds of the input, the tilings, tiles, offsets and optional hash memory.
 * @returns The coder.
 * @throws MdmlError with code `value` for bounds that do not match or are empty.
 * @example
 * ```ts
 * import { tileCoder } from 'mdml-engine/agents'
 * const coder = tileCoder({ low: [0], high: [1], tilings: 4, tiles: 4 })
 * [coder.size, coder.indices([0.1])] // => [20, [0, 5, 10, 16]]
 * ```
 */
export function tileCoder(options: TileCoderOptions): TileCoder {
  const d = options.low.length
  if (d === 0 || options.high.length !== d) throw new MdmlError('shape', 'tileCoder: low and high must match')
  const tilings = options.tilings ?? 8
  const tiles = Array.from({ length: d }, (_, i) =>
    typeof options.tiles === 'number' ? options.tiles : (options.tiles?.[i] ?? 8),
  )
  const offsets = options.offsets ?? 'asymmetric'
  const cells = tiles.map((n) => n + 1)
  const perTiling = cells.reduce((a, b) => a * b, 1)
  const memory = options.memory
  for (let i = 0; i < d; i++) {
    if (!((options.high[i] ?? 0) > (options.low[i] ?? 0))) throw new MdmlError('value', 'tileCoder: need low < high')
  }
  const displacement = (t: number, i: number): number =>
    offsets === 'none' ? 0 : offsets === 'symmetric' ? t / tilings : ((t * (2 * i + 1)) % tilings) / tilings
  return {
    dims: d,
    tilings,
    size: memory ?? tilings * perTiling,
    indices: (x) => {
      const scaled = Array.from({ length: d }, (_, i) => {
        const lo = options.low[i] ?? 0
        const u = (((x[i] ?? lo) - lo) / ((options.high[i] ?? 1) - lo)) * (tiles[i] ?? 1)
        return Math.min(Math.max(u, 0), (tiles[i] ?? 1) - 1e-9)
      })
      return Array.from({ length: tilings }, (_, t) => {
        const q = scaled.map((u, i) => Math.floor(u + displacement(t, i)))
        if (memory !== undefined) return hashCoordinates(t, q, memory)
        let code = 0
        for (let i = 0; i < d; i++) code = code * (cells[i] ?? 1) + (q[i] ?? 0)
        return t * perTiling + code
      })
    },
  }
}

/**
 * Tile-coded features of an observation: the coder's active tiles of `project(observation)` (the observation itself
 * by default).
 *
 * @param coder - The tile coder.
 * @param project - How an observation becomes the coder's input (default: the observation, or a number as one value).
 * @returns The feature map.
 * @example
 * ```ts
 * import { tileCoder, tileFeatures } from 'mdml-engine/agents'
 * tileFeatures(tileCoder({ low: [0], high: [1], tilings: 2, tiles: 2 })).features([0.3]).indices // => [0, 4]
 * ```
 */
export function tileFeatures<O = readonly number[]>(
  coder: TileCoder,
  project?: (observation: O) => readonly number[],
): FeatureMap<O> {
  const toInput = project ?? ((o: O): readonly number[] => (typeof o === 'number' ? [o] : (o as readonly number[])))
  return { size: coder.size, features: (o) => ({ indices: coder.indices(toInput(o)) }) }
}

/**
 * Tile-coded features of an observation and an action for action values: the active tiles of the observation, shifted
 * by the action's block of weights.
 *
 * @param coder - The tile coder.
 * @param nActions - The number of actions.
 * @param project - How an observation becomes the coder's input.
 * @returns The feature map, of `nActions * coder.size` features.
 * @example
 * ```ts
 * import { actionTileFeatures, tileCoder } from 'mdml-engine/agents'
 * actionTileFeatures(tileCoder({ low: [0], high: [1], tilings: 2, tiles: 2 }), 3).features([0.3], 1).indices // => [6, 10]
 * ```
 */
export function actionTileFeatures<O = readonly number[]>(
  coder: TileCoder,
  nActions: number,
  project?: (observation: O) => readonly number[],
): ActionFeatureMap<O> {
  const base = tileFeatures<O>(coder, project)
  return {
    size: nActions * coder.size,
    features: (o, a) => ({ indices: base.features(o).indices.map((i) => a * coder.size + i) }),
  }
}

// -------------------------------------------------------------------------------------------- shared machinery

/** The trace kinds of the linear learners. */
export type LinearTraceKind = 'accumulating' | 'replacing' | 'dutch'

/** The state of a linear prediction agent: weights, traces and what the method needs to carry between steps. */
export interface LinearTdState {
  /** The weights of the value function. */
  readonly w: Float64Array
  /** The eligibility trace (all but TD(0)). */
  readonly z: Float64Array
  /** The secondary weights of the gradient-TD methods. */
  readonly h?: Float64Array
  /** The value of the next state before the last update (true online TD). */
  readonly vOld: number
  /** The importance-weighted followon trace of emphatic TD, carried to the next step. */
  readonly followon: number
  /** The summed updates of the episode so far (offline updating only). */
  readonly pending?: Float64Array
  /** The state of the update rule (when an optimiser is given). */
  readonly rule?: unknown
  readonly steps: number
  readonly episodes: number
  readonly tdError: number
  /** The Euclidean norm of the weights. */
  readonly weightNorm: number
}

const norm = (w: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < w.length; i++) s += (w[i] ?? 0) ** 2
  return Math.sqrt(s)
}

/** Options shared by the linear prediction agents. */
export interface LinearPredictionOptions<O> {
  /** The features of an observation. */
  readonly features: FeatureMap<O>
  /** The number of actions (default from a discrete action space). */
  readonly nActions?: number
  /** The probabilities with which the agent acts, per observation (default uniform): the behaviour policy. */
  readonly behaviour?: (observation: O) => readonly number[]
  /** The policy being evaluated, per observation; with it the updates are importance-weighted (off-policy). */
  readonly target?: (observation: O) => readonly number[]
  /** The step size (default 0.01). */
  readonly alpha?: number
  /** The discount (default 0.99). */
  readonly gamma?: number
  /** The initial weights (default zeros). */
  readonly initialWeights?: readonly number[]
}

const actionCount = (env: { readonly actionSpace: Space }, given: number | undefined, name: string): number => {
  const n = given ?? (env.actionSpace.kind === 'discrete' ? env.actionSpace.n : undefined)
  if (n === undefined) throw new MdmlError('value', `${name}: needs a discrete action space or nActions`)
  return n
}

const sample = (probs: readonly number[], k: Key): number => {
  const u = unit(k)
  let acc = 0
  for (const [a, p] of probs.entries()) {
    acc += p
    if (u < acc) return a
  }
  return probs.length - 1
}

const configOf = (config: Readonly<Record<string, Json>>): Json => config

// -------------------------------------------------------------------------------------- semi-gradient TD(lambda)

/** Options of {@link semiGradientTd}. */
export interface SemiGradientTdOptions<O> extends LinearPredictionOptions<O> {
  /** The trace decay (default 0). */
  readonly lambda?: number
  /**
   * `accumulating` (default) adds the features to the trace, `replacing` sets the active entries to their feature
   * values, `dutch` is true online TD(lambda) (van Seijen and Sutton 2014; on-policy only, no optimiser).
   */
  readonly trace?: LinearTraceKind
  /**
   * Apply the updates of an episode together at its end, with the weights held fixed during it: the offline form,
   * whose summed updates equal those of the forward view with lambda returns.
   */
  readonly offline?: boolean
  /**
   * The update rule: a gradient transformation fed the negated direction `-rho delta z`, replacing the plain step
   * `alpha rho delta z` (then `alpha` is not used; the transformation carries its own learning rate).
   */
  readonly optimiser?: GradientTransformation
}

function applyDirection(
  state: { readonly w: Float64Array; readonly rule?: unknown },
  direction: Float64Array,
  alpha: number,
  optimiser: GradientTransformation | undefined,
): { readonly w: Float64Array; readonly rule: unknown } {
  const w = state.w.slice()
  if (optimiser === undefined) {
    for (let i = 0; i < w.length; i++) w[i] = (w[i] ?? 0) + alpha * (direction[i] ?? 0)
    return { w, rule: undefined }
  }
  const params = { w: tensor(Array.from(state.w)) }
  const rule = state.rule ?? optimiser.init(params)
  const grads = { w: tensor(Array.from(direction, (x) => -x)) }
  const out = optimiser.update(grads, rule, params)
  const next = applyUpdates(params, out.updates) as { readonly w: Tensor }
  return { w: Float64Array.from(toFlat(next.w)), rule: out.state }
}

/**
 * Semi-gradient TD(lambda) with linear features (Sutton and Barto, Chapters 9, 12 and 11): the weights move along
 * `alpha delta z`, the TD error times the eligibility trace of the features, ignoring the dependence of the target on the
 * weights. With a `target` policy the trace carries the importance ratio `rho = pi(a|s) / b(a|s)` of every step (the
 * per-decision form `z = rho (gamma lambda z + x)`), which makes the method off-policy and, on Baird's counterexample,
 * divergent. Traces end with the episode; a terminal next state has value zero and a truncated one bootstraps.
 *
 * @param env - The environment (its action space).
 * @param options - The features, policies, step size, discount, lambda and trace kind, and the update rule.
 * @returns The agent; the value of an observation is `featureValue(state.w, options.features.features(observation))`.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { oneHotFeatures, semiGradientTd } from 'mdml-engine/agents'
 * import { randomWalk } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = randomWalk.create({ n: 5 })
 * const agent = semiGradientTd(env, { features: oneHotFeatures(5), alpha: 0.5, gamma: 1 })
 * const t = { observation: 2, action: 1, reward: 1, nextObservation: 3, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).w[2] // => 0.5
 * ```
 */
export function semiGradientTd<O>(
  env: { readonly actionSpace: Space },
  options: SemiGradientTdOptions<O>,
): Agent<LinearTdState, O, number> {
  const lambda = options.lambda ?? 0
  const kind = options.trace ?? 'accumulating'
  if (kind === 'dutch' && (options.target !== undefined || options.optimiser !== undefined || options.offline === true))
    throw new MdmlError('value', 'semiGradientTd: dutch traces are on-policy, online and take no optimiser')
  return linearPrediction(
    'semiGradientTd',
    env,
    options,
    configOf({ lambda, trace: kind, offline: options.offline === true, rule: options.optimiser?.name ?? 'sgd' }),
    (s, c) => {
      const { alpha, gamma, phi, v, v2, delta, rho, ended } = c
      const n = s.w.length
      if (kind === 'dutch') {
        const decay = gamma * lambda
        const z = s.z.slice()
        const zPhi = dotSparse(z, phi)
        for (let i = 0; i < n; i++) z[i] = decay * (z[i] ?? 0)
        const factor = 1 - alpha * decay * zPhi
        for (const [k, i] of phi.indices.entries()) z[i] = (z[i] ?? 0) + factor * (phi.values?.[k] ?? 1)
        const w = s.w.slice()
        const extra = delta + v - s.vOld
        for (let i = 0; i < n; i++) w[i] = (w[i] ?? 0) + alpha * extra * (z[i] ?? 0)
        for (const [k, i] of phi.indices.entries()) w[i] = (w[i] ?? 0) - alpha * (v - s.vOld) * (phi.values?.[k] ?? 1)
        return { w, z: ended ? new Float64Array(n) : z, vOld: ended ? 0 : v2 }
      }
      const z = s.z.slice()
      for (let i = 0; i < n; i++) z[i] = rho * gamma * lambda * (z[i] ?? 0)
      for (const [k, i] of phi.indices.entries()) {
        const x = phi.values?.[k] ?? 1
        z[i] = kind === 'replacing' ? rho * x : (z[i] ?? 0) + rho * x
      }
      const direction = new Float64Array(n)
      for (let i = 0; i < n; i++) direction[i] = delta * (z[i] ?? 0)
      const znext = ended ? new Float64Array(n) : z
      if (options.offline === true) {
        const pending = (s.pending ?? new Float64Array(n)).slice()
        for (let i = 0; i < n; i++) pending[i] = (pending[i] ?? 0) + (direction[i] ?? 0)
        if (!ended) return { w: s.w, z: znext, pending }
        const applied = applyDirection(s, pending, alpha, options.optimiser)
        return { w: applied.w, z: znext, rule: applied.rule, pending: new Float64Array(n) }
      }
      const applied = applyDirection(s, direction, alpha, options.optimiser)
      return { w: applied.w, z: znext, rule: applied.rule }
    },
  )
}

// ------------------------------------------------------------------------------------- gradient-TD and emphatic

/** Options of {@link gradientTd}. */
export interface GradientTdOptions<O> extends LinearPredictionOptions<O> {
  /** `tdc` (default, TD with gradient correction) or `gtd2`. */
  readonly variant?: 'tdc' | 'gtd2'
  /** The step size of the secondary weights (default `alpha`). */
  readonly beta?: number
}

/**
 * Gradient-TD learning with linear features (Sutton et al. 2009; Sutton and Barto, section 11.7): GTD2 and TDC follow the
 * true gradient of the projected Bellman error, learning a second weight vector `h` that estimates the expected TD
 * error given the features. With `rho` the importance ratio of the step, `delta` the TD error and `x`, `x'` the
 * features of the state and its successor: `h <- h + beta rho (delta - h . x) x`, and `w` moves by
 * `alpha rho (x - gamma x') (x . h)` for GTD2 or `alpha rho (delta x - gamma x' (x . h))` for TDC. They converge under
 * off-policy sampling where semi-gradient TD diverges.
 *
 * @param env - The environment (its action space).
 * @param options - The features, policies, step sizes, discount and variant.
 * @returns The agent.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { gradientTd, oneHotFeatures } from 'mdml-engine/agents'
 * import { randomWalk } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = gradientTd(randomWalk.create({ n: 5 }), { features: oneHotFeatures(5), alpha: 0.5, gamma: 1 })
 * agent.init(key(0)).h?.length // => 5
 * ```
 */
export function gradientTd<O>(
  env: { readonly actionSpace: Space },
  options: GradientTdOptions<O>,
): Agent<LinearTdState, O, number> {
  const variant = options.variant ?? 'tdc'
  const beta = options.beta ?? options.alpha ?? 0.01
  return linearPrediction(
    variant,
    env,
    options,
    configOf({ variant, beta }),
    (s, c) => {
      const { alpha, gamma, phi, phi2, delta, rho } = c
      const n = s.w.length
      const h = s.h ?? new Float64Array(n)
      const xh = dotSparse(h, phi)
      const w = s.w.slice()
      const grow = new Float64Array(n)
      if (variant === 'tdc') for (const [k, i] of phi.indices.entries()) grow[i] = delta * (phi.values?.[k] ?? 1)
      else for (const [k, i] of phi.indices.entries()) grow[i] = xh * (phi.values?.[k] ?? 1)
      for (const [k, i] of phi2.indices.entries()) grow[i] = (grow[i] ?? 0) - gamma * xh * (phi2.values?.[k] ?? 1)
      for (let i = 0; i < n; i++) w[i] = (w[i] ?? 0) + alpha * rho * (grow[i] ?? 0)
      const h2 = h.slice()
      for (const [k, i] of phi.indices.entries())
        h2[i] = (h2[i] ?? 0) + beta * rho * (delta - xh) * (phi.values?.[k] ?? 1)
      return { w, h: h2, z: s.z }
    },
    true,
  )
}

/** Options of {@link emphaticTd}. */
export interface EmphaticTdOptions<O> extends LinearPredictionOptions<O> {
  /** The trace decay (default 0). */
  readonly lambda?: number
  /** The interest in each state (default 1). */
  readonly interest?: (observation: O) => number
}

/**
 * Emphatic TD(lambda) (Sutton, Mahmood and White 2016; Sutton and Barto, section 11.8): off-policy TD that reweights
 * every update by an emphasis, the followon trace `F_t = gamma rho_{t-1} F_{t-1} + i_t` of the interest accumulated
 * along the trajectory, so that the expected update is stable under off-policy sampling. With `M_t = lambda i_t +
 * (1 - lambda) F_t` the trace is `z_t = rho_t (gamma lambda z_{t-1} + M_t x_t)` and the weights move by
 * `alpha delta_t z_t`. The emphasis has high variance when the ratios are large, so the step size must be small.
 *
 * @param env - The environment (its action space).
 * @param options - The features, policies, step size, discount, lambda and interest.
 * @returns The agent.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { emphaticTd, oneHotFeatures } from 'mdml-engine/agents'
 * import { randomWalk } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = emphaticTd(randomWalk.create({ n: 5 }), { features: oneHotFeatures(5), alpha: 0.5, gamma: 1 })
 * const t = { observation: 2, action: 1, reward: 1, nextObservation: 3, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).w[2] // => 0.5
 * ```
 */
export function emphaticTd<O>(
  env: { readonly actionSpace: Space },
  options: EmphaticTdOptions<O>,
): Agent<LinearTdState, O, number> {
  const lambda = options.lambda ?? 0
  return linearPrediction('emphaticTd', env, options, configOf({ lambda }), (s, c) => {
    const { alpha, gamma, phi, delta, rho, ended, observation } = c
    const n = s.w.length
    const interest = options.interest?.(observation as O) ?? 1
    const F = gamma * s.followon + interest
    const M = lambda * interest + (1 - lambda) * F
    const z = s.z.slice()
    for (let i = 0; i < n; i++) z[i] = rho * gamma * lambda * (z[i] ?? 0)
    for (const [k, i] of phi.indices.entries()) z[i] = (z[i] ?? 0) + rho * M * (phi.values?.[k] ?? 1)
    const w = s.w.slice()
    for (let i = 0; i < n; i++) w[i] = (w[i] ?? 0) + alpha * delta * (z[i] ?? 0)
    return { w, z: ended ? new Float64Array(n) : z, followon: ended ? 0 : rho * F }
  })
}

// -------------------------------------------------------------------------------------------- the common agent

interface StepContext {
  readonly observation: unknown
  readonly alpha: number
  readonly gamma: number
  readonly phi: SparseFeatures
  readonly phi2: SparseFeatures
  readonly v: number
  readonly v2: number
  readonly delta: number
  readonly rho: number
  /** The episode ends with this transition (termination or truncation). */
  readonly ended: boolean
}

type Update = Partial<LinearTdState> & { readonly w: Float64Array }

function linearPrediction<O>(
  name: string,
  env: { readonly actionSpace: Space },
  options: LinearPredictionOptions<O>,
  config: Json,
  update: (state: LinearTdState, context: StepContext) => Update,
  secondary = false,
): Agent<LinearTdState, O, number> {
  const nActions = actionCount(env, options.nActions, name)
  const alpha = options.alpha ?? 0.01
  const gamma = options.gamma ?? 0.99
  const size = options.features.size
  const uniformProbs = Array.from({ length: nActions }, () => 1 / nActions)
  const behaviour = options.behaviour ?? (() => uniformProbs)
  const w0 = Float64Array.from({ length: size }, (_, i) => options.initialWeights?.[i] ?? 0)
  return {
    name,
    config: { alpha, gamma, features: size, nActions, offPolicy: options.target !== undefined, ...(config as object) },
    metrics: {
      tdError: { description: 'TD error of the last update' },
      weightNorm: { description: 'Euclidean norm of the weights' },
    },
    init: () => ({
      w: w0,
      z: new Float64Array(size),
      ...(secondary ? { h: new Float64Array(size) } : {}),
      vOld: 0,
      followon: 0,
      steps: 0,
      episodes: 0,
      tdError: 0,
      weightNorm: norm(w0),
    }),
    act: (_, o, k) => sample(behaviour(o), k),
    greedy: (_, o) => argmaxOf((options.target ?? behaviour)(o)),
    learn: (s, t: Transition<O, number>) => {
      const phi = options.features.features(t.observation)
      const phi2 = options.features.features(t.nextObservation)
      const v = dotSparse(s.w, phi)
      const v2 = t.terminated ? 0 : dotSparse(s.w, phi2)
      const delta = t.reward + gamma * v2 - v
      const b = behaviour(t.observation)[t.action] ?? 1
      const rho = options.target === undefined ? 1 : (options.target(t.observation)[t.action] ?? 0) / b
      const ended = t.terminated || t.truncated
      const next = update(s, {
        observation: t.observation,
        alpha,
        gamma,
        phi,
        phi2: t.terminated ? { indices: [] } : phi2,
        v,
        v2,
        delta,
        rho,
        ended,
      })
      return {
        ...s,
        ...next,
        steps: s.steps + 1,
        episodes: s.episodes + (ended ? 1 : 0),
        tdError: delta,
        weightNorm: norm(next.w),
      }
    },
    stats: (s) => ({ tdError: s.tdError, weightNorm: s.weightNorm }),
  }
}

// ----------------------------------------------------------------------------------------- Sarsa(lambda)

/** Options of {@link semiGradientSarsa}. */
export interface SemiGradientSarsaOptions<O> {
  /** The features of an observation and an action. */
  readonly features: ActionFeatureMap<O>
  /** The number of actions (default from a discrete action space). */
  readonly nActions?: number
  /** The step size (default 0.1; with tile coding divide by the number of tilings). */
  readonly alpha?: number
  /** The discount (default 1). */
  readonly gamma?: number
  /** The exploration rate: a number or a schedule over steps (default 0). */
  readonly epsilon?: Rate
  /** The trace decay (default 0.9). */
  readonly lambda?: number
  /** `accumulating` (default) or `replacing`. */
  readonly trace?: 'accumulating' | 'replacing'
  /** The initial weights (default zeros; optimistic values explore). */
  readonly initialWeights?: number
  /** The update rule, as for {@link semiGradientTd}. */
  readonly optimiser?: GradientTransformation
}

/** The state of {@link semiGradientSarsa}. */
export interface LinearSarsaState {
  readonly w: Float64Array
  readonly z: Float64Array
  /** The previous step awaiting its successor action: the features it took, its reward and the next observation. */
  readonly previous?: {
    readonly indices: readonly number[]
    readonly values?: readonly number[]
    readonly reward: number
  }
  readonly rule?: unknown
  readonly steps: number
  readonly episodes: number
  readonly tdError: number
  readonly weightNorm: number
}

/**
 * Episodic semi-gradient Sarsa(lambda) with linear features (Sutton and Barto, sections 10.1 and 12.7): action values
 * `q(s, a) = w . x(s, a)`, an epsilon-greedy policy, and the weights moving by `alpha delta z` with the TD error of the
 * step once its successor action is known (the update of a step waits for the next one, or for the end of the episode),
 * and traces that accumulate or are replaced and decay by `gamma lambda`. With tile coding and `alpha` divided by the
 * number of tilings this learns Mountain Car in a few episodes.
 *
 * @param env - The environment (its action space).
 * @param options - The features, step size, discount, exploration, lambda and trace kind.
 * @returns The agent.
 * @concept sarsa
 * @example
 * ```ts
 * import { actionTileFeatures, semiGradientSarsa, tileCoder } from 'mdml-engine/agents'
 * import { mountainCar } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = mountainCar.create()
 * const coder = tileCoder({ low: [-1.2, -0.07], high: [0.6, 0.07], tilings: 8, tiles: 8 })
 * const agent = semiGradientSarsa(env, { features: actionTileFeatures(coder, 3), alpha: 0.5 / 8 })
 * agent.init(key(0)).w.length // => 1944
 * ```
 */
export function semiGradientSarsa<O>(
  env: { readonly actionSpace: Space },
  options: SemiGradientSarsaOptions<O>,
): Agent<LinearSarsaState, O, number> {
  const nActions = actionCount(env, options.nActions, 'semiGradientSarsa')
  const alpha = options.alpha ?? 0.1
  const gamma = options.gamma ?? 1
  const lambda = options.lambda ?? 0.9
  const kind = options.trace ?? 'accumulating'
  const epsilon = options.epsilon ?? 0
  const size = options.features.size
  const qs = (w: ArrayLike<number>, o: O): number[] =>
    Array.from({ length: nActions }, (_, a) => dotSparse(w, options.features.features(o, a)))
  const bestAction = (w: ArrayLike<number>, o: O, k?: Key): number => {
    const q = qs(w, o)
    if (k === undefined) return argmaxOf(q)
    const top = Math.max(...q)
    const ties = q.flatMap((x, a) => (x === top ? [a] : []))
    return ties[Math.min(ties.length - 1, Math.floor(unit(k) * ties.length))] ?? 0
  }
  const process = (
    s: LinearSarsaState,
    features: SparseFeatures,
    reward: number,
    bootstrap: number,
  ): Pick<LinearSarsaState, 'w' | 'z' | 'rule' | 'tdError'> => {
    const z = s.z.slice()
    for (const [k, i] of features.indices.entries()) {
      const x = features.values?.[k] ?? 1
      z[i] = kind === 'replacing' ? x : (z[i] ?? 0) + x
    }
    const delta = reward + gamma * bootstrap - dotSparse(s.w, features)
    const direction = new Float64Array(size)
    for (let i = 0; i < size; i++) direction[i] = delta * (z[i] ?? 0)
    const applied = applyDirection(s, direction, alpha, options.optimiser)
    for (let i = 0; i < size; i++) z[i] = gamma * lambda * (z[i] ?? 0)
    return { w: applied.w, z, rule: applied.rule, tdError: delta }
  }
  return {
    name: 'semiGradientSarsa',
    config: {
      alpha,
      gamma,
      lambda,
      trace: kind,
      epsilon: typeof epsilon === 'number' ? epsilon : 'schedule',
      features: size,
      nActions,
      rule: options.optimiser?.name ?? 'sgd',
    },
    metrics: {
      tdError: { description: 'TD error of the last update' },
      weightNorm: { description: 'Euclidean norm of the weights' },
    },
    init: () => ({
      w: new Float64Array(size).fill(options.initialWeights ?? 0),
      z: new Float64Array(size),
      steps: 0,
      episodes: 0,
      tdError: 0,
      weightNorm: 0,
    }),
    act: (s, o, k) => {
      const eps = rateAt(epsilon, s.steps)
      if (eps > 0 && unit(k) < eps) return Math.min(nActions - 1, Math.floor(unit(foldIn(k, 1)) * nActions))
      return bestAction(s.w, o, k)
    },
    greedy: (s, o) => bestAction(s.w, o),
    learn: (s, t) => {
      let state = s
      let delta = s.tdError
      const p = s.previous
      if (p !== undefined) {
        // The step before this one learns now that its successor action is this one.
        const q = dotSparse(s.w, options.features.features(t.observation, t.action))
        const u = process(s, p, p.reward, q)
        state = { ...s, ...u }
        delta = u.tdError
      }
      const here = options.features.features(t.observation, t.action)
      const ended = t.terminated || t.truncated
      const { previous: _drop, ...rest } = state
      if (!ended) {
        return {
          ...rest,
          previous: { ...here, reward: t.reward },
          steps: s.steps + 1,
          tdError: delta,
          weightNorm: norm(state.w),
        }
      }
      const boot = t.terminated ? 0 : Math.max(...qs(state.w, t.nextObservation))
      const u = process(state, here, t.reward, boot)
      return {
        ...rest,
        w: u.w,
        z: new Float64Array(size),
        rule: u.rule,
        steps: s.steps + 1,
        episodes: s.episodes + 1,
        tdError: u.tdError,
        weightNorm: norm(u.w),
      }
    },
    stats: (s) => ({ tdError: s.tdError, weightNorm: s.weightNorm }),
  }
}

/**
 * The cost-to-go surface of a linear action-value function over a grid of two-dimensional observations: the negated
 * value of the best action at each grid point (for Mountain Car's position and velocity).
 *
 * @param state - The Sarsa agent state.
 * @param features - The feature map the agent was built with.
 * @param nActions - The number of actions.
 * @param grid - The two axes' values, with the first axis varying slowest.
 * @returns One cost-to-go per grid point, row by row.
 * @example
 * ```ts
 * import { costToGo } from 'mdml-engine/agents'
 * const features = { size: 2, features: (o: readonly number[], a: number) => ({ indices: [a] }) }
 * costToGo({ w: Float64Array.of(1, 3) } as never, features, 2, [[0], [0]]) // => [-3]
 * ```
 */
export function costToGo(
  state: Pick<LinearSarsaState, 'w'>,
  features: ActionFeatureMap<readonly number[]>,
  nActions: number,
  grid: readonly [readonly number[], readonly number[]],
): number[] {
  const out: number[] = []
  for (const x of grid[0]) {
    for (const y of grid[1]) {
      let best = Number.NEGATIVE_INFINITY
      for (let a = 0; a < nActions; a++) best = Math.max(best, dotSparse(state.w, features.features([x, y], a)))
      out.push(-best)
    }
  }
  return out
}
