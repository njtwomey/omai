import { argmaxOf, WhiteboxError } from '../base/index.ts'
import { categorical, normal } from '../dists/index.ts'
import { foldIn, type Key, normal as normalNoise, uniform } from '../random/index.ts'
import {
  add,
  arctanh,
  asTensor,
  clip,
  exp,
  item,
  mul,
  softplus,
  sub,
  sum,
  type TensorLike,
  tanh,
  type Value,
} from '../tensor/index.ts'

const u1 = (k: Key): number => uniform(k, []).data[0] ?? 0

/**
 * The action probabilities of an epsilon-greedy policy: `1 - epsilon + epsilon / n` on the first maximising action and
 * `epsilon / n` on the others.
 *
 * @param q - Action values.
 * @param epsilon - The exploration rate.
 * @returns The probabilities.
 * @concept exploration
 * @example
 * ```ts
 * import { epsilonGreedyProbabilities } from 'whitebox-engine/rl'
 * epsilonGreedyProbabilities([1, 3, 2], 0.75) // => [0.25, 0.5, 0.25]
 * ```
 */
export function epsilonGreedyProbabilities(q: ArrayLike<number>, epsilon: number): number[] {
  const n = q.length
  const best = argmaxOf(q)
  return Array.from({ length: n }, (_, a) => (a === best ? 1 - epsilon + epsilon / n : epsilon / n))
}

/**
 * Sample an epsilon-greedy action: uniformly at random with probability `epsilon`, else the first greedy action.
 *
 * @param q - Action values.
 * @param epsilon - The exploration rate.
 * @param key - The key.
 * @returns The action.
 * @concept exploration
 * @example
 * ```ts
 * import { epsilonGreedy } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * epsilonGreedy([0, 5, 1], 0, key(0)) // => 1
 * ```
 */
export function epsilonGreedy(q: ArrayLike<number>, epsilon: number, key: Key): number {
  if (q.length === 0) throw new WhiteboxError('value', 'epsilonGreedy: no actions')
  if (u1(foldIn(key, 0)) < epsilon) return Math.min(q.length - 1, Math.floor(u1(foldIn(key, 1)) * q.length))
  return argmaxOf(q)
}

/**
 * The Boltzmann (softmax) action probabilities `exp(q / T) / sum exp(q / T)`, computed stably.
 *
 * @param q - Action values.
 * @param temperature - The temperature `T > 0`.
 * @returns The probabilities.
 * @concept exploration
 * @example
 * ```ts
 * import { boltzmannProbabilities } from 'whitebox-engine/rl'
 * boltzmannProbabilities([0, Math.log(3)], 1).map((p) => Number(p.toFixed(12))) // => [0.25, 0.75]
 * ```
 */
export function boltzmannProbabilities(q: ArrayLike<number>, temperature: number): number[] {
  if (!(temperature > 0)) throw new WhiteboxError('value', 'boltzmann: the temperature must be positive')
  let m = -Infinity
  for (let i = 0; i < q.length; i++) m = Math.max(m, (q[i] ?? 0) / temperature)
  const w = Array.from({ length: q.length }, (_, i) => Math.exp((q[i] ?? 0) / temperature - m))
  let z = 0
  for (const x of w) z += x
  return w.map((x) => x / z)
}

/**
 * Sample a Boltzmann (softmax) action.
 *
 * @param q - Action values.
 * @param temperature - The temperature.
 * @param key - The key.
 * @returns The action.
 * @concept exploration
 * @example
 * ```ts
 * import { boltzmann } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * boltzmann([0, 100], 1, key(0)) // => 1
 * ```
 */
export function boltzmann(q: ArrayLike<number>, temperature: number, key: Key): number {
  const p = boltzmannProbabilities(q, temperature)
  const u = u1(key)
  let c = 0
  for (const [i, x] of p.entries()) {
    c += x
    if (u < c) return i
  }
  return p.length - 1
}

/**
 * A distribution over actions produced by a network head, with what policy-gradient and actor-critic methods need:
 * samples (and reparameterised samples for continuous actions), log-probabilities summed over action dimensions,
 * entropy and the deterministic action.
 *
 * @concept policy-gradient
 */
export interface PolicyHead {
  sample(key: Key): Value
  /** A sample that is a differentiable function of the head's parameters (continuous heads only). */
  rsample(key: Key): Value
  logProb(action: TensorLike): Value
  /** A sample and its log-probability together (cheaper and more stable for squashed Gaussians). */
  sampleWithLogProb(key: Key, reparameterise?: boolean): { readonly action: Value; readonly logProb: Value }
  entropy(): Value
  /** The deterministic (greedy) action. */
  mode(): Value
}

/**
 * A categorical policy over discrete actions from logits (categories on the last axis; leading axes are a batch).
 *
 * @param logits - Unnormalised log-probabilities `[..., nActions]`.
 * @returns The head.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { categoricalPolicy } from 'whitebox-engine/rl'
 * import { item } from 'whitebox-engine/tensor'
 * item(categoricalPolicy([0, 0]).logProb(1)) === Math.log(0.5) // => true
 * ```
 */
export function categoricalPolicy(logits: TensorLike): PolicyHead {
  const d = categorical({ logits })
  return {
    sample: (k) => d.sample(k),
    rsample: () => {
      throw new WhiteboxError('not-implemented', 'categoricalPolicy: discrete actions have no reparameterisation')
    },
    logProb: (a) => d.logProb(a),
    sampleWithLogProb: (k) => {
      const action = d.sample(k)
      return { action, logProb: d.logProb(action) }
    },
    entropy: () => d.entropy(),
    mode: () => d.mode(),
  }
}

/** Options of {@link gaussianPolicy}. */
export interface GaussianPolicyOptions {
  /** Squash samples through tanh into (-1, 1) (SAC), correcting the log-probability. */
  readonly squash?: boolean
  /** Rescale squashed actions to `[low, high]` (per dimension or one bound). */
  readonly low?: number | readonly number[]
  readonly high?: number | readonly number[]
}

// log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)), stable for large |u| (as in SAC implementations).
const logTanhJacobian = (u: Value): Value => mul(2, sub(sub(Math.LN2, u), softplus(mul(-2, u))))

/**
 * A diagonal Gaussian policy over continuous actions, optionally tanh-squashed and rescaled to bounds (the SAC
 * actor): log-probabilities sum over the last axis and include the change-of-variables correction
 * `- sum log(1 - tanh(u)^2)` and the rescaling.
 *
 * @param mean - Means `[..., actionDim]`.
 * @param logStd - Log standard deviations (broadcast to the mean).
 * @param options - Squashing and bounds.
 * @returns The head. With squashing, `entropy` is that of the unsquashed Gaussian (the squashed one has no closed form).
 * @concept policy-gradient
 * @example
 * ```ts
 * import { gaussianPolicy } from 'whitebox-engine/rl'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(gaussianPolicy([0], [0]).logProb([0])) + 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function gaussianPolicy(mean: TensorLike, logStd: TensorLike, options: GaussianPolicyOptions = {}): PolicyHead {
  const mu = asTensor(mean) as Value
  const ls = asTensor(logStd) as Value
  const base = normal(mu, exp(ls))
  const squash = options.squash ?? false
  const bounds = options.low !== undefined && options.high !== undefined
  const n = mu.shape[mu.shape.length - 1] ?? 1
  const at = (b: number | readonly number[] | undefined, i: number, fallback: number): number =>
    typeof b === 'number' ? b : (b?.[i] ?? fallback)
  const lowArr = Array.from({ length: n }, (_, i) => at(options.low, i, -1))
  const highArr = Array.from({ length: n }, (_, i) => at(options.high, i, 1))
  const scale = lowArr.map((l, i) => ((highArr[i] ?? 1) - l) / 2)
  const centre = lowArr.map((l, i) => ((highArr[i] ?? 1) + l) / 2)
  let logScale = 0
  for (const s of scale) logScale += Math.log(s)
  const toAction = (u: Value): Value => {
    if (!squash) return u
    const a = tanh(u)
    return bounds ? add(mul(a, scale), centre) : a
  }
  const squashedLogProb = (u: Value): Value => {
    const lp = sum(base.logProb(u), -1)
    if (!squash) return lp
    const corrected = sub(lp, sum(logTanhJacobian(u), -1))
    return bounds ? sub(corrected, logScale) : corrected
  }
  const fromAction = (a: Value): Value => {
    if (!squash) return a
    const unit = bounds
      ? mul(
          sub(a, centre),
          scale.map((s) => 1 / s),
        )
      : a
    return arctanh(clip(unit, -1 + 1e-7, 1 - 1e-7))
  }
  const draw = (k: Key, reparameterise: boolean): Value => (reparameterise ? base.rsample(k) : base.sample(k))
  return {
    sample: (k) => toAction(draw(k, false)),
    rsample: (k) => toAction(draw(k, true)),
    logProb: (a) => squashedLogProb(fromAction(asTensor(a) as Value)),
    sampleWithLogProb: (k, reparameterise = true) => {
      const u = draw(k, reparameterise)
      return { action: toAction(u), logProb: squashedLogProb(u) }
    },
    entropy: () => sum(base.entropy(), -1),
    mode: () => toAction(mu),
  }
}

/**
 * A deterministic policy head (DDPG and TD3 actors): sampling returns the action itself.
 *
 * @param action - The action.
 * @returns The head; `logProb` and `entropy` are not defined and throw.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { deterministicPolicy } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(deterministicPolicy([0.5]).sample(key(0))) // => [0.5]
 * ```
 */
export function deterministicPolicy(action: TensorLike): PolicyHead {
  const a = asTensor(action) as Value
  const none = (): never => {
    throw new WhiteboxError('not-implemented', 'deterministicPolicy: a deterministic policy has no density')
  }
  return {
    sample: () => a,
    rsample: () => a,
    logProb: none,
    sampleWithLogProb: none,
    entropy: none,
    mode: () => a,
  }
}

/** A noise process for exploration in continuous action spaces (stable-baselines3's ActionNoise). */
export interface NoiseProcess {
  readonly size: number
  init(): number[]
  /** The next noise value from the previous one. */
  step(previous: readonly number[], key: Key): number[]
}

/**
 * Independent Gaussian action noise `N(mean, sigma^2)` per step (stable-baselines3's NormalActionNoise).
 *
 * @param size - The action dimension.
 * @param options - `sigma` (default 0.1) and `mean` (default 0).
 * @returns The process.
 * @concept exploration
 * @example
 * ```ts
 * import { gaussianNoise } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * gaussianNoise(2, { sigma: 0 }).step([0, 0], key(0)) // => [0, 0]
 * ```
 */
export function gaussianNoise(
  size: number,
  options: { readonly sigma?: number; readonly mean?: number } = {},
): NoiseProcess {
  const sigma = options.sigma ?? 0.1
  const mean = options.mean ?? 0
  return {
    size,
    init: () => new Array<number>(size).fill(mean),
    step: (_, k) => Array.from(normalNoise(k, [size]).data, (z) => mean + sigma * z),
  }
}

/**
 * Ornstein-Uhlenbeck action noise (Uhlenbeck and Ornstein 1930; DDPG's exploration; stable-baselines3's
 * OrnsteinUhlenbeckActionNoise): `x' = x + theta (mean - x) dt + sigma sqrt(dt) N(0, 1)`, temporally correlated.
 *
 * @param size - The action dimension.
 * @param options - `theta` (0.15), `sigma` (0.2), `dt` (0.01), `mean` (0) and the start value.
 * @returns The process.
 * @concept exploration
 * @example
 * ```ts
 * import { ornsteinUhlenbeckNoise } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * ornsteinUhlenbeckNoise(1, { sigma: 0, theta: 1, dt: 0.5 }).step([2], key(0)) // => [1]
 * ```
 */
export function ornsteinUhlenbeckNoise(
  size: number,
  options: {
    readonly theta?: number
    readonly sigma?: number
    readonly dt?: number
    readonly mean?: number
    readonly initial?: number
  } = {},
): NoiseProcess {
  const theta = options.theta ?? 0.15
  const sigma = options.sigma ?? 0.2
  const dt = options.dt ?? 1e-2
  const mean = options.mean ?? 0
  return {
    size,
    init: () => new Array<number>(size).fill(options.initial ?? 0),
    step: (x, k) => {
      const z = normalNoise(k, [size]).data
      return Array.from({ length: size }, (_, i) => {
        const xi = x[i] ?? 0
        return xi + theta * (mean - xi) * dt + sigma * Math.sqrt(dt) * (z[i] ?? 0)
      })
    },
  }
}

/**
 * The log-probability of an action under a head as a number (for storing in rollout buffers).
 *
 * @param head - The policy head.
 * @param action - The action.
 * @returns The log-probability.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { categoricalPolicy, logProbOf } from 'whitebox-engine/rl'
 * logProbOf(categoricalPolicy([0, 0, 0, 0]), 2) === Math.log(0.25) // => true
 * ```
 */
export function logProbOf(head: PolicyHead, action: TensorLike): number {
  return item(head.logProb(action))
}
