/**
 * Return and advantage estimators, the targets that value-based and policy-gradient methods learn from: discounted
 * returns, one-step TD targets and errors, n-step and lambda returns, generalised advantage estimation (GAE), and the
 * off-policy corrections V-trace (IMPALA) and Retrace.
 */

import { MdmlError } from '../base/index.ts'

/** A per-step flag as a number (0 or 1) or a boolean. */
type Flags = ArrayLike<number | boolean>

const flag = (xs: Flags | undefined, t: number): number => (xs === undefined ? 0 : Number(xs[t] ?? 0))

function sameLength(name: string, n: number, ...arrays: (ArrayLike<unknown> | undefined)[]): void {
  for (const a of arrays) {
    if (a !== undefined && a.length !== n) {
      throw new MdmlError('shape', `${name}: inputs must have the same length (${n}), got ${a.length}`)
    }
  }
}

/**
 * Discounted returns `G_t = r_t + gamma (1 - d_t) G_{t+1}`, from the end backwards, where `d_t` marks the last step of
 * an episode; after the final step the return is `bootstrap` (the value of the state the data stops in, default 0).
 *
 * @param rewards - Rewards `r_0 .. r_{T-1}`.
 * @param gamma - The discount.
 * @param options - Episode ends `dones` (default none) and the bootstrap value.
 * @returns The returns.
 * @concept returns
 * @example
 * ```ts
 * import { discountedReturns } from 'mdml-engine/rl'
 * discountedReturns([1, 1, 1], 0.5) // => [1.75, 1.5, 1]
 * ```
 */
export function discountedReturns(
  rewards: ArrayLike<number>,
  gamma: number,
  options: { readonly dones?: Flags; readonly bootstrap?: number } = {},
): number[] {
  const n = rewards.length
  sameLength('discountedReturns', n, options.dones)
  const out = new Array<number>(n)
  let g = options.bootstrap ?? 0
  for (let t = n - 1; t >= 0; t--) {
    g = (rewards[t] ?? 0) + gamma * (1 - flag(options.dones, t)) * g
    out[t] = g
  }
  return out
}

/**
 * One-step TD targets `r_t + gamma (1 - terminated_t) V(s_{t+1})`.
 *
 * @param rewards - Rewards.
 * @param nextValues - Values of the next states.
 * @param terminated - Whether each next state is terminal (no bootstrap).
 * @param gamma - The discount.
 * @returns The targets.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { tdTargets } from 'mdml-engine/rl'
 * tdTargets([1, 1], [2, 2], [0, 1], 0.5) // => [2, 1]
 * ```
 */
export function tdTargets(
  rewards: ArrayLike<number>,
  nextValues: ArrayLike<number>,
  terminated: Flags,
  gamma: number,
): number[] {
  sameLength('tdTargets', rewards.length, nextValues, terminated)
  return Array.from(
    { length: rewards.length },
    (_, t) => (rewards[t] ?? 0) + gamma * (1 - flag(terminated, t)) * (nextValues[t] ?? 0),
  )
}

/**
 * One-step TD errors `delta_t = r_t + gamma (1 - terminated_t) V(s_{t+1}) - V(s_t)`.
 *
 * @param rewards - Rewards.
 * @param values - Values of the states.
 * @param nextValues - Values of the next states.
 * @param terminated - Whether each next state is terminal.
 * @param gamma - The discount.
 * @returns The errors.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { tdErrors } from 'mdml-engine/rl'
 * tdErrors([1], [3], [2], [0], 0.5) // => [-1]
 * ```
 */
export function tdErrors(
  rewards: ArrayLike<number>,
  values: ArrayLike<number>,
  nextValues: ArrayLike<number>,
  terminated: Flags,
  gamma: number,
): number[] {
  sameLength('tdErrors', rewards.length, values, nextValues, terminated)
  return tdTargets(rewards, nextValues, terminated, gamma).map((y, t) => y - (values[t] ?? 0))
}

/**
 * n-step returns `G_t^(n) = sum_{k<n} gamma^k r_{t+k} + gamma^n V(s_{t+n})`, truncated at episode ends (no bootstrap
 * past a done) and at the end of the data (bootstrapping from the last value).
 *
 * @param rewards - Rewards `r_0 .. r_{T-1}`.
 * @param values - State values `V(s_0) .. V(s_T)` (one more than the rewards; the last bootstraps the data's end).
 * @param dones - Whether step `t` ended its episode in a terminal state.
 * @param gamma - The discount.
 * @param n - The number of steps.
 * @returns The returns.
 * @concept returns
 * @example
 * ```ts
 * import { nStepReturns } from 'mdml-engine/rl'
 * nStepReturns([1, 1, 1], [0, 0, 0, 10], [0, 0, 0], 0.5, 2) // => [1.5, 4, 6]
 * ```
 */
export function nStepReturns(
  rewards: ArrayLike<number>,
  values: ArrayLike<number>,
  dones: Flags,
  gamma: number,
  n: number,
): number[] {
  const T = rewards.length
  if (values.length !== T + 1) throw new MdmlError('shape', `nStepReturns: need ${T + 1} values, got ${values.length}`)
  sameLength('nStepReturns', T, dones)
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'nStepReturns: n must be a positive integer')
  return Array.from({ length: T }, (_, t) => {
    let g = 0
    let discount = 1
    for (let k = 0; k < n; k++) {
      const i = t + k
      if (i >= T) return g + discount * (values[T] ?? 0)
      g += discount * (rewards[i] ?? 0)
      discount *= gamma
      if (flag(dones, i) === 1) return g
    }
    return g + discount * (values[t + n] ?? 0)
  })
}

/**
 * Lambda returns (the forward view of TD(lambda)):
 * `G_t = r_t + gamma (1 - d_t) [(1 - lambda) V(s_{t+1}) + lambda G_{t+1}]`, with `G_{T-1}` bootstrapping from
 * `V(s_T)`. Lambda 0 gives TD targets and lambda 1 Monte Carlo returns.
 *
 * @param rewards - Rewards.
 * @param nextValues - Values of the next states `V(s_1) .. V(s_T)`.
 * @param dones - Episode ends (terminal next states).
 * @param gamma - The discount.
 * @param lambda - The trace parameter.
 * @returns The returns.
 * @concept eligibility-traces
 * @example
 * ```ts
 * import { lambdaReturns } from 'mdml-engine/rl'
 * lambdaReturns([0, 0], [1, 2], [0, 0], 1, 0.5) // => [1.5, 2]
 * ```
 */
export function lambdaReturns(
  rewards: ArrayLike<number>,
  nextValues: ArrayLike<number>,
  dones: Flags,
  gamma: number,
  lambda: number,
): number[] {
  const T = rewards.length
  sameLength('lambdaReturns', T, nextValues, dones)
  const out = new Array<number>(T)
  let g = nextValues[T - 1] ?? 0
  for (let t = T - 1; t >= 0; t--) {
    const v = nextValues[t] ?? 0
    const blend = t === T - 1 ? v : (1 - lambda) * v + lambda * g
    g = (rewards[t] ?? 0) + gamma * (1 - flag(dones, t)) * blend
    out[t] = g
  }
  return out
}

/** Advantages and the matching returns. */
export interface Advantages {
  readonly advantages: number[]
  readonly returns: number[]
}

/**
 * Generalised advantage estimation (Schulman et al. 2016), as stable-baselines3 and CleanRL compute it:
 * `delta_t = r_t + gamma (1 - d_t) V(s_{t+1}) - V(s_t)` and `A_t = delta_t + gamma lambda (1 - d_t) A_{t+1}`, where
 * `d_t` marks steps after which the episode ended and `V(s_T)` is `lastValue`. Returns are `A_t + V(s_t)` (the
 * lambda-returns, the value targets).
 *
 * @param rewards - Rewards `r_0 .. r_{T-1}`.
 * @param values - Values `V(s_0) .. V(s_{T-1})`.
 * @param dones - Episode ends.
 * @param lastValue - The value of the state after the last step.
 * @param gamma - The discount (default 0.99).
 * @param lambda - The GAE parameter (default 0.95).
 * @returns Advantages and returns.
 * @concept advantage-estimation
 * @index function Generalised advantage estimation.
 * @example
 * ```ts
 * import { gae } from 'mdml-engine/rl'
 * gae([1, 1], [0, 0], [0, 1], 5, 1, 1).advantages // => [2, 1]
 * ```
 */
export function gae(
  rewards: ArrayLike<number>,
  values: ArrayLike<number>,
  dones: Flags,
  lastValue: number,
  gamma = 0.99,
  lambda = 0.95,
): Advantages {
  const T = rewards.length
  sameLength('gae', T, values, dones)
  const advantages = new Array<number>(T)
  let last = 0
  for (let t = T - 1; t >= 0; t--) {
    const notDone = 1 - flag(dones, t)
    const next = t === T - 1 ? lastValue : (values[t + 1] ?? 0)
    const delta = (rewards[t] ?? 0) + gamma * next * notDone - (values[t] ?? 0)
    last = delta + gamma * lambda * notDone * last
    advantages[t] = last
  }
  return { advantages, returns: advantages.map((a, t) => a + (values[t] ?? 0)) }
}

/** The result of {@link vtrace}. */
export interface VTraceResult {
  /** The V-trace value targets `v_s`. */
  readonly vs: number[]
  /** The policy-gradient advantages `rho_t (r_t + gamma v_{t+1} - V(s_t))`. */
  readonly pgAdvantages: number[]
}

/**
 * V-trace off-policy value targets and policy-gradient advantages (Espeholt et al. 2018, IMPALA; rlax's
 * `vtrace_td_error_and_advantage`): with importance ratios `rho_t = pi(a_t | s_t) / mu(a_t | s_t)`, clipped at
 * `clipRho` for the TD errors and `lambda min(1, rho)` for the traces.
 *
 * @param rewards - Rewards.
 * @param discounts - Per-step discounts (`gamma`, or 0 at episode ends).
 * @param values - `V(s_0) .. V(s_{T-1})` under the current estimate.
 * @param bootstrapValue - `V(s_T)`.
 * @param logRhos - Log importance ratios `log pi - log mu` of the actions taken.
 * @param options - `clipRho` (default 1), `clipPgRho` (default 1) and `lambda` (default 1).
 * @returns The value targets and the advantages.
 * @concept off-policy-learning
 * @example
 * ```ts
 * import { vtrace } from 'mdml-engine/rl'
 * vtrace([1], [0.9], [0], 2, [0]).vs // => [2.8]
 * ```
 */
export function vtrace(
  rewards: ArrayLike<number>,
  discounts: ArrayLike<number>,
  values: ArrayLike<number>,
  bootstrapValue: number,
  logRhos: ArrayLike<number>,
  options: { readonly clipRho?: number; readonly clipPgRho?: number; readonly lambda?: number } = {},
): VTraceResult {
  const T = rewards.length
  sameLength('vtrace', T, discounts, values, logRhos)
  const clipRho = options.clipRho ?? 1
  const clipPg = options.clipPgRho ?? 1
  const lambda = options.lambda ?? 1
  const rhos = Array.from({ length: T }, (_, t) => Math.exp(logRhos[t] ?? 0))
  const next = (t: number): number => (t + 1 < T ? (values[t + 1] ?? 0) : bootstrapValue)
  const vs = new Array<number>(T)
  let acc = 0
  for (let t = T - 1; t >= 0; t--) {
    const rho = rhos[t] ?? 1
    const delta = Math.min(clipRho, rho) * ((rewards[t] ?? 0) + (discounts[t] ?? 0) * next(t) - (values[t] ?? 0))
    acc = delta + (discounts[t] ?? 0) * lambda * Math.min(1, rho) * acc
    vs[t] = acc + (values[t] ?? 0)
  }
  const vsNext = (t: number): number => (t + 1 < T ? (vs[t + 1] ?? 0) : bootstrapValue)
  const pgAdvantages = Array.from(
    { length: T },
    (_, t) => Math.min(clipPg, rhos[t] ?? 1) * ((rewards[t] ?? 0) + (discounts[t] ?? 0) * vsNext(t) - (values[t] ?? 0)),
  )
  return { vs, pgAdvantages }
}

/**
 * Retrace targets for action values (Munos et al. 2016; rlax's `general_off_policy_returns_from_q_and_v`):
 * `G_t = r_t + gamma_t [E_pi Q(s_{t+1}) + c_{t+1} (G_{t+1} - Q(s_{t+1}, a_{t+1}))]`, ending with
 * `G_{T-1} = r_{T-1} + gamma_{T-1} E_pi Q(s_T)`, with traces `c = lambda min(1, pi / mu)`.
 *
 * @param rewards - Rewards `r_0 .. r_{T-1}`.
 * @param discounts - Discounts (0 at episode ends).
 * @param nextExpectedQ - `E_pi Q(s_{t+1}, .)` for `t = 0 .. T-1`.
 * @param nextTakenQ - `Q(s_{t+1}, a_{t+1})` for `t = 0 .. T-2`.
 * @param traces - `c_{t+1} = lambda min(1, pi(a_{t+1}) / mu(a_{t+1}))` for `t = 0 .. T-2`.
 * @returns The targets for `Q(s_t, a_t)`.
 * @concept off-policy-learning
 * @example
 * ```ts
 * import { retrace } from 'mdml-engine/rl'
 * retrace([1, 1], [1, 1], [0, 0], [0], [1]) // => [2, 1]
 * ```
 */
export function retrace(
  rewards: ArrayLike<number>,
  discounts: ArrayLike<number>,
  nextExpectedQ: ArrayLike<number>,
  nextTakenQ: ArrayLike<number>,
  traces: ArrayLike<number>,
): number[] {
  const T = rewards.length
  sameLength('retrace', T, discounts, nextExpectedQ)
  if (nextTakenQ.length !== T - 1 || traces.length !== T - 1) {
    throw new MdmlError('shape', `retrace: nextTakenQ and traces need ${T - 1} entries`)
  }
  const out = new Array<number>(T)
  let g = (rewards[T - 1] ?? 0) + (discounts[T - 1] ?? 0) * (nextExpectedQ[T - 1] ?? 0)
  out[T - 1] = g
  for (let t = T - 2; t >= 0; t--) {
    const c = traces[t] ?? 0
    g = (rewards[t] ?? 0) + (discounts[t] ?? 0) * ((nextExpectedQ[t] ?? 0) - c * (nextTakenQ[t] ?? 0) + c * g)
    out[t] = g
  }
  return out
}

/**
 * Retrace traces from target and behaviour probabilities: `c = lambda min(1, pi / mu)`.
 *
 * @param targetProbs - `pi(a | s)` of the actions taken.
 * @param behaviourProbs - `mu(a | s)` of the actions taken.
 * @param lambda - The trace parameter (default 1).
 * @returns The traces.
 * @concept off-policy-learning
 * @example
 * ```ts
 * import { retraceTraces } from 'mdml-engine/rl'
 * retraceTraces([0.5, 0.9], [0.25, 1]) // => [1, 0.9]
 * ```
 */
export function retraceTraces(targetProbs: ArrayLike<number>, behaviourProbs: ArrayLike<number>, lambda = 1): number[] {
  sameLength('retraceTraces', targetProbs.length, behaviourProbs)
  return Array.from(
    { length: targetProbs.length },
    (_, t) => lambda * Math.min(1, (targetProbs[t] ?? 0) / Math.max(behaviourProbs[t] ?? 0, 1e-300)),
  )
}
