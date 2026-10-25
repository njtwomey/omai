/**
 * Tabular reinforcement learning after Sutton and Barto, for environments small enough to keep a value per state or
 * state-action pair. TD control (Q-learning, SARSA, expected SARSA, n-step SARSA, SARSA(lambda) and double Q-learning),
 * Monte Carlo control, the planning learners Dyna-Q and prioritised sweeping, and TD(0) and TD(lambda) prediction of a
 * fixed policy's values; helpers read the learnt tables back as value, action-value and greedy-policy grids for
 * figures.
 */

import { argmaxOf, heap, type Json, maxOf, WhiteboxError } from '../../../lib/base/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../../lib/random/index.ts'
import type { Agent, Policy, Transition } from '../../../lib/rl/index.ts'
import {
  epsilonGreedyRow,
  epsilonGreedyValue,
  filled,
  nested,
  type Rate,
  randomIndex,
  rateAt,
  row,
  type StepSize,
  stepSizeAt,
  type TabularCoding,
  type TabularCodingOptions,
  type TabularEnvLike,
  tabularCoding,
  unit,
  withEntry,
} from './_tabular.ts'

// ------------------------------------------------------------------------------------------------ shared machinery

/** Options shared by the tabular control agents. */
export interface TabularOptions<O> extends TabularCodingOptions<O> {
  /** Step size: a constant (default 0.1), `'sample-average'` (`1 / n(s, a)`) or a function of the visit count. */
  readonly alpha?: StepSize
  /** Discount (default 0.99). */
  readonly gamma?: number
  /** Exploration rate of the epsilon-greedy behaviour: a constant (default 0.1) or a schedule. */
  readonly epsilon?: Rate
  /** What the epsilon schedule counts: learning steps (default) or finished episodes. */
  readonly epsilonUnit?: 'steps' | 'episodes'
  /** Initial action values (default 0; large values give optimistic exploration). */
  readonly initialValue?: number
}

/** The state of a tabular control agent: the action-value table `Q[s * nActions + a]`, visit counts and counters. */
export interface TabularState {
  readonly nStates: number
  readonly nActions: number
  readonly Q: Float64Array
  /** Updates of each state-action pair from real experience. */
  readonly visits: Float64Array
  /** Transitions learned from. */
  readonly steps: number
  /** Episodes finished. */
  readonly episodes: number
  /** The last TD error (or Monte Carlo error). */
  readonly tdError: number
}

/** A transition with its observations replaced by state indices. */
interface Coded {
  readonly s: number
  readonly a: number
  readonly r: number
  readonly s2: number
  readonly terminated: boolean
  readonly truncated: boolean
}

/** What the factories pass to an agent's update. */
interface Context {
  readonly nA: number
  readonly gamma: number
  readonly alpha: StepSize
  /** The behaviour epsilon at the state's current count. */
  readonly epsilon: number
}

const GREEDY_KEY: Key = makeKey(0x67726565)

const stepSizeConfig = (alpha: StepSize): Json => (typeof alpha === 'function' ? 'schedule' : alpha)
const rateConfig = (rate: Rate): Json => (typeof rate === 'number' ? rate : 'schedule')

/** One step-size update of `Q[i]` towards `target` (counting a visit when `real`). */
function towards(
  s: { readonly Q: Float64Array; readonly visits: Float64Array },
  i: number,
  target: number,
  alpha: StepSize,
  real = true,
): { readonly Q: Float64Array; readonly visits: Float64Array; readonly delta: number } {
  const n = (s.visits[i] ?? 0) + (real ? 1 : 0)
  const q = s.Q[i] ?? 0
  const delta = target - q
  return {
    Q: withEntry(s.Q, i, q + stepSizeAt(alpha, Math.max(n, 1)) * delta),
    visits: real ? withEntry(s.visits, i, n) : s.visits,
    delta,
  }
}

/** The greedy value `max_a Q(s, a)` of a row. */
const best = (Q: Float64Array, s: number, nA: number): number => maxOf(row(Q, s, nA))

interface ControlSpec<O, G extends TabularState> {
  readonly name: string
  readonly env: TabularEnvLike<O>
  readonly options: TabularOptions<O>
  readonly config?: Readonly<Record<string, Json>>
  readonly metrics?: Agent<G, O, number>['metrics']
  /** Defaults that differ from the shared ones. */
  readonly defaults?: { readonly alpha?: StepSize; readonly epsilon?: Rate }
  extend(base: TabularState): G
  update(state: G, t: Coded, key: Key, ctx: Context): G
  /** The behaviour policy (default epsilon-greedy on `Q` with random tie-breaking). */
  act?(state: G, s: number, key: Key, ctx: Context): number
  stats?(state: G): Readonly<Record<string, number>>
}

function controlAgent<O, G extends TabularState>(spec: ControlSpec<O, G>): Agent<G, O, number> {
  const coding: TabularCoding<O> = tabularCoding(spec.env, spec.options, spec.name)
  const { nStates: nS, nActions: nA } = coding
  const o = spec.options
  const gamma = o.gamma ?? 0.99
  const alpha = o.alpha ?? spec.defaults?.alpha ?? 0.1
  const epsilon = o.epsilon ?? spec.defaults?.epsilon ?? 0.1
  const byEpisodes = o.epsilonUnit === 'episodes'
  const ctx = (s: TabularState): Context => ({
    nA,
    gamma,
    alpha,
    epsilon: rateAt(epsilon, byEpisodes ? s.episodes : s.steps),
  })
  const check = (code: number): number => {
    if (!(Number.isInteger(code) && code >= 0 && code < nS)) {
      throw new WhiteboxError('value', `${spec.name}: state index ${code} is outside [0, ${nS})`)
    }
    return code
  }
  return {
    name: spec.name,
    config: {
      nStates: nS,
      nActions: nA,
      gamma,
      alpha: stepSizeConfig(alpha),
      epsilon: rateConfig(epsilon),
      initialValue: o.initialValue ?? 0,
      ...(spec.config ?? {}),
    },
    metrics: {
      epsilon: { description: 'exploration rate' },
      tdError: { description: 'TD error of the last update' },
      ...(spec.metrics ?? {}),
    },
    init: () =>
      spec.extend({
        nStates: nS,
        nActions: nA,
        Q: filled(nS * nA, o.initialValue ?? 0),
        visits: new Float64Array(nS * nA),
        steps: 0,
        episodes: 0,
        tdError: 0,
      }),
    act: (s, obs, k) => {
      const code = check(coding.encode(obs))
      const c = ctx(s)
      return spec.act === undefined ? epsilonGreedyRow(s.Q, code, nA, c.epsilon, k) : spec.act(s, code, k, c)
    },
    greedy: (s, obs) => argmaxOf(row(s.Q, check(coding.encode(obs)), nA)),
    learn: (s, t: Transition<O, number>, k) => {
      const coded: Coded = {
        s: check(coding.encode(t.observation)),
        a: t.action,
        r: t.reward,
        s2: check(coding.encode(t.nextObservation)),
        terminated: t.terminated,
        truncated: t.truncated,
      }
      const next = spec.update(s, coded, k, ctx(s))
      return { ...next, steps: s.steps + 1, episodes: s.episodes + (t.terminated || t.truncated ? 1 : 0) }
    },
    stats: (s) => ({ epsilon: ctx(s).epsilon, tdError: s.tdError, ...(spec.stats === undefined ? {} : spec.stats(s)) }),
  }
}

// ------------------------------------------------------------------------------------- one-step TD control

/**
 * Q-learning (Watkins 1989; Sutton and Barto, section 6.5): off-policy TD control towards
 * `r + gamma max_a' Q(s', a')` (no bootstrap after termination; bootstrapped after truncation), with an epsilon-greedy
 * behaviour policy (random tie-breaking). The table is `state.Q` (see {@link qTable}).
 *
 * @param env - The environment (discrete actions; discrete or tuple-of-discrete observations, or give `encode`).
 * @param options - Step size, discount, exploration, initial values and the observation coding.
 * @returns The agent.
 * @concept q-learning
 * @example
 * ```ts
 * import { qLearning } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = qLearning(cliffWalking.create(), { alpha: 0.5, gamma: 1 })
 * const s = agent.learn(agent.init(key(0)), { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }, key(1))
 * s.Q[36 * 4] // => -0.5
 * ```
 */
export function qLearning<O>(env: TabularEnvLike<O>, options: TabularOptions<O> = {}): Agent<TabularState, O, number> {
  return controlAgent<O, TabularState>({
    name: 'qLearning',
    env,
    options,
    extend: (b) => b,
    update: (s, t, _k, c) => {
      const target = t.r + (t.terminated ? 0 : c.gamma * best(s.Q, t.s2, c.nA))
      const u = towards(s, t.s * c.nA + t.a, target, c.alpha)
      return { ...s, Q: u.Q, visits: u.visits, tdError: u.delta }
    },
  })
}

/** The state of {@link sarsa}: the table and the transition waiting for its next action. */
export interface SarsaState extends TabularState {
  readonly previous?: { readonly s: number; readonly a: number; readonly r: number; readonly s2: number }
}

/**
 * SARSA (Rummery and Niranjan 1994; Sutton and Barto, section 6.4): on-policy TD control towards
 * `r + gamma Q(s', a')` with the action `a'` the behaviour policy actually takes next. The update of a transition waits
 * for the next one (whose action is `a'`), which reproduces the book's order exactly: `a'` is chosen from `Q` before
 * `Q(s, a)` changes. At termination the target is `r`; after a truncation (or when the next transition does not start
 * where the last one ended) it bootstraps from the epsilon-greedy expectation at `s'`.
 *
 * @param env - The environment.
 * @param options - Step size, discount, exploration, initial values and the observation coding.
 * @returns The agent.
 * @concept sarsa
 * @example
 * ```ts
 * import { sarsa } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = sarsa(cliffWalking.create(), { alpha: 0.5 })
 * const t = { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).previous // => { s: 36, a: 0, r: -1, s2: 24 }
 * ```
 */
export function sarsa<O>(env: TabularEnvLike<O>, options: TabularOptions<O> = {}): Agent<SarsaState, O, number> {
  return controlAgent<O, SarsaState>({
    name: 'sarsa',
    env,
    options,
    extend: (b) => b,
    update: (s, t, _k, c) => {
      let tab: { readonly Q: Float64Array; readonly visits: Float64Array } = s
      let delta = s.tdError
      const p = s.previous
      if (p !== undefined) {
        const boot = p.s2 === t.s ? (tab.Q[t.s * c.nA + t.a] ?? 0) : epsilonGreedyValue(tab.Q, p.s2, c.nA, c.epsilon)
        const u = towards(tab, p.s * c.nA + p.a, p.r + c.gamma * boot, c.alpha)
        tab = u
        delta = u.delta
      }
      if (!(t.terminated || t.truncated)) {
        return { ...s, Q: tab.Q, visits: tab.visits, tdError: delta, previous: { s: t.s, a: t.a, r: t.r, s2: t.s2 } }
      }
      const boot = t.terminated ? 0 : c.gamma * epsilonGreedyValue(tab.Q, t.s2, c.nA, c.epsilon)
      const u = towards(tab, t.s * c.nA + t.a, t.r + boot, c.alpha)
      const { previous: _p, ...rest } = s
      return { ...rest, Q: u.Q, visits: u.visits, tdError: u.delta }
    },
  })
}

/** Options of {@link expectedSarsa}. */
export interface ExpectedSarsaOptions<O> extends TabularOptions<O> {
  /** The epsilon of the target policy (default the behaviour's; 0 gives Q-learning). */
  readonly targetEpsilon?: number
}

/**
 * Expected SARSA (van Seijen et al. 2009; Sutton and Barto, section 6.6): TD control towards the expectation of the
 * next action value under the epsilon-greedy target policy, `r + gamma sum_a' pi(a' | s') Q(s', a')`, which removes
 * the variance of sampling `a'`. With `targetEpsilon` the target policy differs from the behaviour (off-policy).
 *
 * @param env - The environment.
 * @param options - Step size, discount, exploration, the target policy's epsilon and the observation coding.
 * @returns The agent.
 * @concept sarsa
 * @example
 * ```ts
 * import { expectedSarsa } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = expectedSarsa(cliffWalking.create(), { alpha: 0.5, targetEpsilon: 0 })
 * const t = { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).Q[144] // => -0.5
 * ```
 */
export function expectedSarsa<O>(
  env: TabularEnvLike<O>,
  options: ExpectedSarsaOptions<O> = {},
): Agent<TabularState, O, number> {
  return controlAgent<O, TabularState>({
    name: 'expectedSarsa',
    env,
    options,
    config: { targetEpsilon: options.targetEpsilon ?? 'behaviour' },
    extend: (b) => b,
    update: (s, t, _k, c) => {
      const eps = options.targetEpsilon ?? c.epsilon
      const target = t.r + (t.terminated ? 0 : c.gamma * epsilonGreedyValue(s.Q, t.s2, c.nA, eps))
      const u = towards(s, t.s * c.nA + t.a, target, c.alpha)
      return { ...s, Q: u.Q, visits: u.visits, tdError: u.delta }
    },
  })
}

// ------------------------------------------------------------------------------------------------ n-step SARSA

/** One remembered step of an episode. */
export interface TabularStep {
  readonly s: number
  readonly a: number
  readonly r: number
  readonly s2: number
}

/** Options of {@link nStepSarsa}. */
export interface NStepSarsaOptions<O> extends TabularOptions<O> {
  /** The number of steps (default 4). */
  readonly n?: number
  /** Bootstrap from the expected value at `S_{t+n}` (n-step expected SARSA) instead of `Q(S_{t+n}, A_{t+n})`. */
  readonly expected?: boolean
}

/** The state of {@link nStepSarsa}: the table and the last `n` steps waiting for their returns. */
export interface NStepSarsaState extends TabularState {
  readonly window: readonly TabularStep[]
}

/**
 * n-step SARSA (Sutton and Barto, section 7.2): the update of `Q(S_tau, A_tau)` waits until `n` more steps are known
 * and uses `G = R_{tau+1} + ... + gamma^{n-1} R_{tau+n} + gamma^n Q(S_{tau+n}, A_{tau+n})`; at termination the
 * remaining steps are updated with their truncated returns, and after a truncation they bootstrap from the
 * epsilon-greedy expectation at the last state. With `expected` the bootstrap is that expectation throughout
 * (n-step expected SARSA). `n = 1` is SARSA.
 *
 * @param env - The environment.
 * @param options - `n`, the expected variant, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @throws WhiteboxError with code `value` when `n` is not a positive integer.
 * @concept sarsa
 * @example
 * ```ts
 * import { nStepSarsa } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = nStepSarsa(cliffWalking.create(), { n: 3 })
 * const t = { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).window // => [{ s: 36, a: 0, r: -1, s2: 24 }]
 * ```
 */
export function nStepSarsa<O>(
  env: TabularEnvLike<O>,
  options: NStepSarsaOptions<O> = {},
): Agent<NStepSarsaState, O, number> {
  const n = options.n ?? 4
  if (!(Number.isInteger(n) && n >= 1)) throw new WhiteboxError('value', 'nStepSarsa: n must be a positive integer')
  const expected = options.expected ?? false
  type Table = { readonly Q: Float64Array; readonly visits: Float64Array; readonly delta: number }
  // Update window[0] with its rewards and `tail` (gamma^k times the bootstrap value, already discounted).
  const updateFirst = (tab: Table, window: readonly TabularStep[], boot: number, c: Context): Table => {
    const first = window[0] as TabularStep
    let g = 0
    let discount = 1
    for (const w of window) {
      g += discount * w.r
      discount *= c.gamma
    }
    return towards(tab, first.s * c.nA + first.a, g + discount * boot, c.alpha)
  }
  const flush = (tab: Table, window: readonly TabularStep[], bootstrapAt: number | undefined, c: Context): Table => {
    let out = tab
    for (let i = 0; i < window.length; i++) {
      const boot = bootstrapAt === undefined ? 0 : epsilonGreedyValue(out.Q, bootstrapAt, c.nA, c.epsilon)
      out = updateFirst(out, window.slice(i), boot, c)
    }
    return out
  }
  return controlAgent<O, NStepSarsaState>({
    name: 'nStepSarsa',
    env,
    options,
    config: { n, expected },
    extend: (b) => ({ ...b, window: [] }),
    update: (s, t, _k, c) => {
      let tab: Table = { Q: s.Q, visits: s.visits, delta: s.tdError }
      let window = s.window
      const last = window[window.length - 1]
      if (last !== undefined && last.s2 !== t.s) {
        // The experience does not continue the remembered steps: close them as if truncated.
        tab = flush(tab, window, last.s2, c)
        window = []
      }
      if (window.length === n) {
        const boot = expected ? epsilonGreedyValue(tab.Q, t.s, c.nA, c.epsilon) : (tab.Q[t.s * c.nA + t.a] ?? 0)
        tab = updateFirst(tab, window, boot, c)
        window = window.slice(1)
      }
      window = [...window, { s: t.s, a: t.a, r: t.r, s2: t.s2 }]
      if (expected && window.length === n && !(t.terminated || t.truncated)) {
        tab = updateFirst(tab, window, epsilonGreedyValue(tab.Q, t.s2, c.nA, c.epsilon), c)
        window = window.slice(1)
      }
      if (t.terminated || t.truncated) {
        tab = flush(tab, window, t.terminated ? undefined : t.s2, c)
        window = []
      }
      return { ...s, Q: tab.Q, visits: tab.visits, tdError: tab.delta, window }
    },
  })
}

// ------------------------------------------------------------------------------------------- Monte Carlo control

/** Options of {@link monteCarloControl}. */
export interface MonteCarloOptions<O> extends TabularOptions<O> {
  /** Update every visit of a pair in an episode instead of only the first (default false: first-visit). */
  readonly everyVisit?: boolean
  /**
   * Exploring starts (Sutton and Barto, section 5.3): the first action of every episode is uniformly random, and the
   * behaviour is greedy afterwards unless `epsilon` is given (it defaults to 0 with exploring starts).
   */
  readonly exploringStarts?: boolean
}

/** The state of {@link monteCarloControl}: the table and the steps of the current episode. */
export interface MonteCarloState extends TabularState {
  readonly episode: readonly TabularStep[]
}

/**
 * On-policy Monte Carlo control (Sutton and Barto, sections 5.3 and 5.4): the steps of an episode are buffered in the
 * state, and at its end every first visit (or every visit) of a state-action pair moves `Q` towards the return that
 * followed it, by sample averages (the default step size) or a constant step. The behaviour is epsilon-soft
 * (epsilon-greedy), or greedy with exploring starts. After a truncation the returns bootstrap from the epsilon-greedy
 * value of the last state.
 *
 * @param env - The environment (Blackjack's tuple observations are coded automatically).
 * @param options - First- or every-visit, exploring starts, step size, discount, exploration and the coding.
 * @returns The agent.
 * @concept monte-carlo-methods
 * @example
 * ```ts
 * import { monteCarloControl } from 'whitebox-engine/methods/rl/agents'
 * import { blackjack } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * monteCarloControl(blackjack.create(), { exploringStarts: true }).init(key(0)).nStates // => 704
 * ```
 */
export function monteCarloControl<O>(
  env: TabularEnvLike<O>,
  options: MonteCarloOptions<O> = {},
): Agent<MonteCarloState, O, number> {
  const everyVisit = options.everyVisit ?? false
  const es = options.exploringStarts ?? false
  return controlAgent<O, MonteCarloState>({
    name: 'monteCarloControl',
    env,
    options,
    defaults: { alpha: 'sample-average', epsilon: es ? 0 : 0.1 },
    config: { everyVisit, exploringStarts: es },
    extend: (b) => ({ ...b, episode: [] }),
    act: (s, code, k, c) =>
      es && s.episode.length === 0 ? randomIndex(foldIn(k, 3), c.nA) : epsilonGreedyRow(s.Q, code, c.nA, c.epsilon, k),
    update: (s, t, _k, c) => {
      const episode = [...s.episode, { s: t.s, a: t.a, r: t.r, s2: t.s2 }]
      if (!(t.terminated || t.truncated)) return { ...s, episode }
      const first = new Map<number, number>()
      for (const [j, e] of episode.entries()) {
        const i = e.s * c.nA + e.a
        if (!first.has(i)) first.set(i, j)
      }
      const Q = s.Q.slice()
      const visits = s.visits.slice()
      let g = t.terminated ? 0 : epsilonGreedyValue(s.Q, t.s2, c.nA, c.epsilon)
      let delta = 0
      for (let j = episode.length - 1; j >= 0; j--) {
        const e = episode[j] as TabularStep
        g = e.r + c.gamma * g
        const i = e.s * c.nA + e.a
        if (!everyVisit && first.get(i) !== j) continue
        const nv = (visits[i] ?? 0) + 1
        visits[i] = nv
        delta = g - (Q[i] ?? 0)
        Q[i] = (Q[i] ?? 0) + stepSizeAt(c.alpha, nv) * delta
      }
      return { ...s, Q, visits, tdError: delta, episode: [] }
    },
  })
}

// ------------------------------------------------------------------------------------------------ SARSA(lambda)

/** Eligibility traces: accumulating (`z += 1`), replacing (`z = 1`) or dutch (true online TD(lambda)). */
export type TraceKind = 'accumulating' | 'replacing' | 'dutch'

/** Options of {@link sarsaLambda}. */
export interface SarsaLambdaOptions<O> extends TabularOptions<O> {
  /** The trace decay (default 0.9). */
  readonly lambda?: number
  /** Accumulating (default) or replacing traces. */
  readonly trace?: 'accumulating' | 'replacing'
}

/** The state of {@link sarsaLambda}: the table, the eligibility traces and the transition waiting for `a'`. */
export interface SarsaLambdaState extends SarsaState {
  readonly z: Float64Array
}

/**
 * SARSA(lambda) (Sutton and Barto, section 12.7): on-policy control with eligibility traces over state-action pairs.
 * Each step marks `z(s, a)` (accumulating or replacing), moves every entry by `alpha delta z` with
 * `delta = r + gamma Q(s', a') - Q(s, a)`, and decays the traces by `gamma lambda`; traces are cleared at the end of an
 * episode. As in {@link sarsa}, a transition's update waits for the next action.
 *
 * @param env - The environment.
 * @param options - `lambda`, trace kind, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @concept eligibility-traces
 * @example
 * ```ts
 * import { sarsaLambda } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * sarsaLambda(cliffWalking.create(), { lambda: 0.8, trace: 'replacing' }).init(key(0)).z.length // => 192
 * ```
 */
export function sarsaLambda<O>(
  env: TabularEnvLike<O>,
  options: SarsaLambdaOptions<O> = {},
): Agent<SarsaLambdaState, O, number> {
  const lambda = options.lambda ?? 0.9
  const kind = options.trace ?? 'accumulating'
  const process = (
    Q: Float64Array,
    z: Float64Array,
    visits: Float64Array,
    i: number,
    target: number,
    c: Context,
  ): number => {
    const n = (visits[i] ?? 0) + 1
    visits[i] = n
    z[i] = kind === 'replacing' ? 1 : (z[i] ?? 0) + 1
    const delta = target - (Q[i] ?? 0)
    const step = stepSizeAt(c.alpha, n) * delta
    for (let j = 0; j < Q.length; j++) {
      const zj = z[j] ?? 0
      if (zj !== 0) {
        Q[j] = (Q[j] ?? 0) + step * zj
        z[j] = c.gamma * lambda * zj
      }
    }
    return delta
  }
  return controlAgent<O, SarsaLambdaState>({
    name: 'sarsaLambda',
    env,
    options,
    config: { lambda, trace: kind },
    extend: (b) => ({ ...b, z: new Float64Array(b.Q.length) }),
    update: (s, t, _k, c) => {
      const Q = s.Q.slice()
      let z = s.z.slice()
      const visits = s.visits.slice()
      let delta = s.tdError
      const p = s.previous
      if (p !== undefined) {
        const boot = p.s2 === t.s ? (Q[t.s * c.nA + t.a] ?? 0) : epsilonGreedyValue(Q, p.s2, c.nA, c.epsilon)
        delta = process(Q, z, visits, p.s * c.nA + p.a, p.r + c.gamma * boot, c)
        if (p.s2 !== t.s) z = new Float64Array(Q.length)
      }
      const { previous: _p, ...rest } = s
      if (!(t.terminated || t.truncated)) {
        return { ...rest, Q, z, visits, tdError: delta, previous: { s: t.s, a: t.a, r: t.r, s2: t.s2 } }
      }
      const boot = t.terminated ? 0 : c.gamma * epsilonGreedyValue(Q, t.s2, c.nA, c.epsilon)
      delta = process(Q, z, visits, t.s * c.nA + t.a, t.r + boot, c)
      return { ...rest, Q, z: new Float64Array(Q.length), visits, tdError: delta }
    },
  })
}

// ------------------------------------------------------------------------------------------ double Q-learning

/** The state of {@link doubleQLearning}: the two estimates and their average `Q` (which the agent acts on). */
export interface DoubleQState extends TabularState {
  readonly QA: Float64Array
  readonly QB: Float64Array
}

/**
 * Double Q-learning (van Hasselt 2010; Sutton and Barto, section 6.7): two independent estimates; each step a fair
 * coin (`uniform(foldIn(key, 0)) < 0.5`) picks one to update, with the action chosen by that estimate and evaluated by
 * the other, `QA(s, a) <- r + gamma QB(s', argmax_a' QA(s', a'))`, which removes the maximisation bias of Q-learning.
 * The agent acts epsilon-greedily on the average `Q = (QA + QB) / 2`.
 *
 * @param env - The environment.
 * @param options - Step size, discount, exploration, initial values and the observation coding.
 * @returns The agent.
 * @concept q-learning
 * @example
 * ```ts
 * import { doubleQLearning } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * doubleQLearning(cliffWalking.create()).init(key(0)).QA.length // => 192
 * ```
 */
export function doubleQLearning<O>(
  env: TabularEnvLike<O>,
  options: TabularOptions<O> = {},
): Agent<DoubleQState, O, number> {
  return controlAgent<O, DoubleQState>({
    name: 'doubleQLearning',
    env,
    options,
    extend: (b) => ({ ...b, QA: b.Q.slice(), QB: b.Q.slice() }),
    update: (s, t, k, c) => {
      const useA = unit(foldIn(k, 0)) < 0.5
      const own = useA ? s.QA : s.QB
      const other = useA ? s.QB : s.QA
      const i = t.s * c.nA + t.a
      const target = t.r + (t.terminated ? 0 : c.gamma * (other[t.s2 * c.nA + argmaxOf(row(own, t.s2, c.nA))] ?? 0))
      const u = towards({ Q: own, visits: s.visits }, i, target, c.alpha)
      const QA = useA ? u.Q : s.QA
      const QB = useA ? s.QB : u.Q
      return {
        ...s,
        QA,
        QB,
        Q: withEntry(s.Q, i, ((QA[i] ?? 0) + (QB[i] ?? 0)) / 2),
        visits: u.visits,
        tdError: u.delta,
      }
    },
  })
}

// --------------------------------------------------------------------------------------------------------- Dyna-Q

/** A deterministic learned model: the last reward and next state of every tried pair (`next = -1` when terminal). */
export interface TabularModel {
  readonly reward: Float64Array
  readonly next: Int32Array
  /** The step at which each pair was last tried (for Dyna-Q+). */
  readonly lastTried: Float64Array
  /** Pairs tried so far, in the order first tried. */
  readonly observed: readonly number[]
  /** States visited so far, in the order first visited. */
  readonly states: readonly number[]
}

const UNSEEN = -2

function emptyModel(size: number): TabularModel {
  return {
    reward: new Float64Array(size),
    next: new Int32Array(size).fill(UNSEEN),
    lastTried: new Float64Array(size),
    observed: [],
    states: [],
  }
}

function recordModel(m: TabularModel, t: Coded, nA: number, step: number): TabularModel {
  const i = t.s * nA + t.a
  const isNew = (m.next[i] ?? UNSEEN) === UNSEEN
  const next = m.next.slice()
  next[i] = t.terminated ? -1 : t.s2
  const lastTried = withEntry(m.lastTried, i, step)
  const newState = !m.states.includes(t.s)
  return {
    reward: withEntry(m.reward, i, t.r),
    next,
    lastTried,
    observed: isNew ? [...m.observed, i] : m.observed,
    states: newState ? [...m.states, t.s] : m.states,
  }
}

/** Options of {@link dynaQ}. */
export interface DynaQOptions<O> extends TabularOptions<O> {
  /** Planning updates per real step (default 10). */
  readonly planningSteps?: number
  /**
   * Dyna-Q+ (Sutton and Barto, section 8.3): planning rewards get the bonus `kappa sqrt(tau)` for pairs untried for
   * `tau` steps, and untried actions of visited states are planned as self-loops with reward 0 (default 0: Dyna-Q).
   */
  readonly kappa?: number
}

/** The state of {@link dynaQ}: the table and the learned model. */
export interface DynaQState extends TabularState {
  readonly model: TabularModel
}

/**
 * Dyna-Q (Sutton and Barto, section 8.2): Q-learning from real experience, a deterministic model of the last outcome of
 * each tried pair, and `planningSteps` Q-learning updates per real step on pairs sampled uniformly from the model.
 * With `kappa > 0` it is Dyna-Q+, whose exploration bonus finds changed environments.
 *
 * @param env - The environment.
 * @param options - Planning steps, the Dyna-Q+ bonus, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @concept model-based-reinforcement-learning
 * @example
 * ```ts
 * import { dynaQ } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = dynaQ(cliffWalking.create(), { planningSteps: 5 })
 * const t = { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).model.observed // => [144]
 * ```
 */
export function dynaQ<O>(env: TabularEnvLike<O>, options: DynaQOptions<O> = {}): Agent<DynaQState, O, number> {
  const planning = options.planningSteps ?? 10
  const kappa = options.kappa ?? 0
  return controlAgent<O, DynaQState>({
    name: kappa > 0 ? 'dynaQPlus' : 'dynaQ',
    env,
    options,
    config: { planningSteps: planning, kappa },
    metrics: { modelSize: { description: 'state-action pairs in the learned model' } },
    extend: (b) => ({ ...b, model: emptyModel(b.Q.length) }),
    update: (s, t, k, c) => {
      const real = t.r + (t.terminated ? 0 : c.gamma * best(s.Q, t.s2, c.nA))
      let tab: { readonly Q: Float64Array; readonly visits: Float64Array } = towards(s, t.s * c.nA + t.a, real, c.alpha)
      const delta = real - (s.Q[t.s * c.nA + t.a] ?? 0)
      const model = recordModel(s.model, t, c.nA, s.steps + 1)
      const u = uniform(foldIn(k, 1), [2 * planning]).data
      for (let p = 0; p < planning; p++) {
        let i: number
        if (kappa > 0) {
          const st = model.states[Math.min(model.states.length - 1, Math.floor((u[2 * p] ?? 0) * model.states.length))]
          i = (st ?? 0) * c.nA + Math.min(c.nA - 1, Math.floor((u[2 * p + 1] ?? 0) * c.nA))
        } else
          i =
            model.observed[Math.min(model.observed.length - 1, Math.floor((u[2 * p] ?? 0) * model.observed.length))] ??
            0
        const nextState = model.next[i] ?? UNSEEN
        const sa = Math.floor(i / c.nA)
        // Untried actions (Dyna-Q+ only) lead back to their state with reward 0.
        const reward = nextState === UNSEEN ? 0 : (model.reward[i] ?? 0)
        const bonus = kappa > 0 ? kappa * Math.sqrt(s.steps + 1 - (model.lastTried[i] ?? 0)) : 0
        const to = nextState === UNSEEN ? sa : nextState
        const target = reward + bonus + (to < 0 ? 0 : c.gamma * best(tab.Q, to, c.nA))
        tab = towards(tab, i, target, c.alpha, false)
      }
      return { ...s, Q: tab.Q, visits: tab.visits, tdError: delta, model }
    },
    stats: (s) => ({ modelSize: s.model.observed.length }),
  })
}

// ------------------------------------------------------------------------------------------- prioritised sweeping

/** Options of {@link prioritizedSweeping}. */
export interface PrioritizedSweepingOptions<O> extends TabularOptions<O> {
  /** Updates from the priority queue per real step (default 10). */
  readonly planningSteps?: number
  /** Pairs enter the queue only when their priority exceeds this (default 1e-4). */
  readonly theta?: number
}

/** An entry of the priority queue: a state-action pair and its priority `|target - Q|`. */
export interface SweepEntry {
  readonly priority: number
  readonly pair: number
}

/** The state of {@link prioritizedSweeping}: the table, the model, predecessors and the priority queue. */
export interface PrioritizedSweepingState extends TabularState {
  readonly model: TabularModel
  /** For every state, the pairs the model predicts lead to it. */
  readonly predecessors: readonly (readonly number[])[]
  /** The queue's entries in heap order; entries whose priority differs from `queued` are stale. */
  readonly queue: readonly SweepEntry[]
  /** The current priority of every queued pair (0 when not queued). */
  readonly queued: Float64Array
}

// Highest priority first; equal priorities by the lower pair index, so the order is a total order.
const sweepOrder = (x: SweepEntry, y: SweepEntry): number => y.priority - x.priority || x.pair - y.pair

/**
 * Prioritised sweeping (Moore and Atkeson 1993; Sutton and Barto, section 8.4): a deterministic model plus a priority
 * queue (the binary heap of `lib/base`) of pairs whose values would change by more than `theta`. Each real step
 * queues its pair by `|r + gamma max Q(s') - Q(s, a)|`, then up to `planningSteps` of the most urgent pairs are updated
 * from the model, and the predecessors of each updated state are queued by their own priorities, so value changes
 * propagate backwards from where they happen.
 *
 * @param env - The environment.
 * @param options - Planning steps, the threshold, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @concept model-based-reinforcement-learning
 * @example
 * ```ts
 * import { prioritizedSweeping } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = prioritizedSweeping(cliffWalking.create(), { alpha: 1, gamma: 1 })
 * const t = { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).Q[144] // => -1
 * ```
 */
export function prioritizedSweeping<O>(
  env: TabularEnvLike<O>,
  options: PrioritizedSweepingOptions<O> = {},
): Agent<PrioritizedSweepingState, O, number> {
  const planning = options.planningSteps ?? 10
  const theta = options.theta ?? 1e-4
  return controlAgent<O, PrioritizedSweepingState>({
    name: 'prioritizedSweeping',
    env,
    options,
    config: { planningSteps: planning, theta },
    metrics: { queueSize: { description: 'pairs waiting in the priority queue' } },
    extend: (b) => ({
      ...b,
      model: emptyModel(b.Q.length),
      predecessors: Array.from({ length: b.nStates }, () => []),
      queue: [],
      queued: new Float64Array(b.Q.length),
    }),
    update: (s, t, _k, c) => {
      const model = recordModel(s.model, t, c.nA, s.steps + 1)
      const i0 = t.s * c.nA + t.a
      let predecessors = s.predecessors
      if (!t.terminated && !(predecessors[t.s2] ?? []).includes(i0)) {
        const copy = predecessors.slice()
        copy[t.s2] = [...(s.predecessors[t.s2] ?? []), i0]
        predecessors = copy
      }
      const Q = s.Q.slice()
      const visits = withEntry(s.visits, i0, (s.visits[i0] ?? 0) + 1)
      const queued = s.queued.slice()
      const q = heap(sweepOrder, s.queue)
      const targetOf = (i: number): number => {
        const nx = model.next[i] ?? -1
        return (model.reward[i] ?? 0) + (nx < 0 ? 0 : c.gamma * maxOf(row(Q, nx, c.nA)))
      }
      const push = (i: number): void => {
        const p = Math.abs(targetOf(i) - (Q[i] ?? 0))
        if (p > theta && p > (queued[i] ?? 0)) {
          queued[i] = p
          q.push({ priority: p, pair: i })
        }
      }
      const delta = targetOf(i0) - (Q[i0] ?? 0)
      push(i0)
      let done = 0
      while (done < planning && q.size > 0) {
        const e = q.pop() as SweepEntry
        if (e.priority !== queued[e.pair]) continue
        queued[e.pair] = 0
        done += 1
        Q[e.pair] =
          (Q[e.pair] ?? 0) +
          stepSizeAt(c.alpha, Math.max(visits[e.pair] ?? 0, 1)) * (targetOf(e.pair) - (Q[e.pair] ?? 0))
        for (const j of predecessors[Math.floor(e.pair / c.nA)] ?? []) push(j)
      }
      return { ...s, Q, visits, tdError: delta, model, predecessors, queue: q.toArray(), queued }
    },
    stats: (s) => {
      let n = 0
      for (const x of s.queued) if (x > 0) n += 1
      return { queueSize: n }
    },
  })
}

// ------------------------------------------------------------------------------------------ TD prediction

/** A policy to evaluate: a function of observations, or action probabilities per state index. */
export type EvaluatedPolicy<O> = Policy<O, number> | readonly (readonly number[])[]

/** Options of {@link td0} and {@link tdLambda}. */
export interface TdOptions<O> extends TabularCodingOptions<O> {
  /** The policy to evaluate and follow (default uniformly random actions). */
  readonly policy?: EvaluatedPolicy<O>
  /** Step size (default 0.1; `'sample-average'` uses `1 / n(s)`). */
  readonly alpha?: StepSize
  /** Discount (default 0.99). */
  readonly gamma?: number
  /** Initial state values (default 0). */
  readonly initialValue?: number
}

/** The state of a TD prediction agent: state values, visit counts and, for TD(lambda), the traces. */
export interface TdState {
  readonly nStates: number
  readonly nActions: number
  readonly V: Float64Array
  readonly visits: Float64Array
  readonly steps: number
  readonly episodes: number
  readonly tdError: number
  /** Eligibility traces (TD(lambda) only). */
  readonly z?: Float64Array
  /** The previous next-state value of true online TD(lambda). */
  readonly vOld?: number
}

function predictionAgent<O>(
  name: string,
  env: TabularEnvLike<O>,
  options: TdOptions<O>,
  config: Readonly<Record<string, Json>>,
  update: (s: TdState, t: Coded, alpha: number, gamma: number) => TdState,
): Agent<TdState, O, number> {
  const coding = tabularCoding(env, options, name)
  const { nStates: nS, nActions: nA } = coding
  const gamma = options.gamma ?? 0.99
  const alpha = options.alpha ?? 0.1
  const policy = options.policy
  const pick = (obs: O, k: Key): number => {
    if (policy === undefined) return randomIndex(k, nA)
    if (typeof policy === 'function') return policy(obs, k)
    const probs = policy[coding.encode(obs)] ?? []
    const u = unit(k)
    let acc = 0
    for (const [a, p] of probs.entries()) {
      acc += p
      if (u < acc) return a
    }
    return probs.length - 1
  }
  return {
    name,
    config: {
      nStates: nS,
      nActions: nA,
      gamma,
      alpha: stepSizeConfig(alpha),
      policy: policy === undefined ? 'uniform' : typeof policy === 'function' ? 'function' : 'table',
      ...config,
    },
    metrics: { tdError: { description: 'TD error of the last update' } },
    init: () => ({
      nStates: nS,
      nActions: nA,
      V: filled(nS, options.initialValue ?? 0),
      visits: new Float64Array(nS),
      steps: 0,
      episodes: 0,
      tdError: 0,
    }),
    act: (_, obs, k) => pick(obs, k),
    greedy: (_, obs) =>
      policy !== undefined && typeof policy !== 'function'
        ? argmaxOf(policy[coding.encode(obs)] ?? [0])
        : pick(obs, GREEDY_KEY),
    learn: (s, t, _k) => {
      const coded: Coded = {
        s: coding.encode(t.observation),
        a: t.action,
        r: t.reward,
        s2: coding.encode(t.nextObservation),
        terminated: t.terminated,
        truncated: t.truncated,
      }
      const n = (s.visits[coded.s] ?? 0) + 1
      const next = update({ ...s, visits: withEntry(s.visits, coded.s, n) }, coded, stepSizeAt(alpha, n), gamma)
      return { ...next, steps: s.steps + 1, episodes: s.episodes + (t.terminated || t.truncated ? 1 : 0) }
    },
    stats: (s) => ({ tdError: s.tdError }),
  }
}

/**
 * TD(0) prediction (Sutton 1988; Sutton and Barto, section 6.1): learns the state values of a given policy, which it
 * also follows, by `V(s) <- V(s) + alpha (r + gamma V(s') - V(s))` (no bootstrap after termination).
 *
 * @param env - The environment.
 * @param options - The policy (a function or a probability table; default uniform), step size, discount, initial
 *   values and the observation coding.
 * @returns The agent; its values are `state.V`.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { td0 } from 'whitebox-engine/methods/rl/agents'
 * import { riverSwim } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = td0(riverSwim.create(), { alpha: 0.5, gamma: 0.9 })
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).V[0] // => 0.5
 * ```
 */
export function td0<O>(env: TabularEnvLike<O>, options: TdOptions<O> = {}): Agent<TdState, O, number> {
  return predictionAgent('td0', env, options, {}, (s, t, alpha, gamma) => {
    const delta = t.r + (t.terminated ? 0 : gamma * (s.V[t.s2] ?? 0)) - (s.V[t.s] ?? 0)
    return { ...s, V: withEntry(s.V, t.s, (s.V[t.s] ?? 0) + alpha * delta), tdError: delta }
  })
}

/** Options of {@link tdLambda}. */
export interface TdLambdaOptions<O> extends TdOptions<O> {
  /** The trace decay (default 0.9). */
  readonly lambda?: number
  /** Accumulating (default), replacing, or dutch traces (true online TD(lambda)). */
  readonly trace?: TraceKind
}

/**
 * TD(lambda) prediction with eligibility traces (Sutton and Barto, sections 12.2 and 12.5): traces decay by
 * `gamma lambda` and mark the visited state (accumulating `z(s) += 1` or replacing `z(s) = 1`), and every value moves
 * by `alpha delta z`. Dutch traces give true online TD(lambda) (van Seijen and Sutton 2014), which matches the online
 * lambda-return algorithm exactly. Traces reset at the end of each episode; lambda = 0 is TD(0), lambda = 1 with
 * accumulating traces is every-visit Monte Carlo for episodic tasks.
 *
 * @param env - The environment.
 * @param options - `lambda`, trace kind, the policy, step size, discount and the observation coding.
 * @returns The agent; its values are `state.V`.
 * @concept eligibility-traces
 * @example
 * ```ts
 * import { tdLambda } from 'whitebox-engine/methods/rl/agents'
 * import { riverSwim } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = tdLambda(riverSwim.create(), { alpha: 0.5, lambda: 0.5, trace: 'dutch' })
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).V[0] // => 0.5
 * ```
 */
export function tdLambda<O>(env: TabularEnvLike<O>, options: TdLambdaOptions<O> = {}): Agent<TdState, O, number> {
  const lambda = options.lambda ?? 0.9
  const kind = options.trace ?? 'accumulating'
  return predictionAgent('tdLambda', env, options, { lambda, trace: kind }, (s, t, alpha, gamma) => {
    const V = s.V.slice()
    const z = s.z === undefined ? new Float64Array(V.length) : s.z.slice()
    const v = V[t.s] ?? 0
    const v2 = t.terminated ? 0 : (V[t.s2] ?? 0)
    const delta = t.r + gamma * v2 - v
    const decay = gamma * lambda
    if (kind === 'dutch') {
      // True online TD(lambda) with one-hot features: z <- gamma lambda z + (1 - alpha gamma lambda z_s) e_s and
      // V <- V + alpha (delta + v - vOld) z - alpha (v - vOld) e_s.
      const zs = z[t.s] ?? 0
      for (let j = 0; j < z.length; j++) z[j] = decay * (z[j] ?? 0)
      z[t.s] = (z[t.s] ?? 0) + 1 - alpha * decay * zs
      const vOld = s.vOld ?? 0
      for (let j = 0; j < V.length; j++) V[j] = (V[j] ?? 0) + alpha * (delta + v - vOld) * (z[j] ?? 0)
      V[t.s] = (V[t.s] ?? 0) - alpha * (v - vOld)
    } else {
      for (let j = 0; j < z.length; j++) z[j] = decay * (z[j] ?? 0)
      z[t.s] = kind === 'replacing' ? 1 : (z[t.s] ?? 0) + 1
      for (let j = 0; j < V.length; j++) V[j] = (V[j] ?? 0) + alpha * delta * (z[j] ?? 0)
    }
    const ended = t.terminated || t.truncated
    return { ...s, V, z: ended ? new Float64Array(V.length) : z, vOld: ended ? 0 : v2, tdError: delta }
  })
}

// ------------------------------------------------------------------------------------------------- inspection

/**
 * The action-value table of a tabular agent state as rows `Q[s][a]` (for heat maps and arrows in figures).
 *
 * @param state - A tabular control agent's state.
 * @returns `Q[s][a]`.
 * @concept q-learning
 * @example
 * ```ts
 * import { qLearning, qTable } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * qTable(qLearning(cliffWalking.create()).init(key(0)))[0] // => [0, 0, 0, 0]
 * ```
 */
export function qTable(state: TabularState): number[][] {
  return nested(state.Q, state.nStates, state.nActions)
}

/**
 * The state values of a tabular agent state: `V` for prediction agents, `max_a Q(s, a)` for control agents.
 *
 * @param state - A tabular agent's state.
 * @returns One value per state.
 * @concept temporal-difference-learning
 * @example
 * ```ts
 * import { td0, valueTable } from 'whitebox-engine/methods/rl/agents'
 * import { riverSwim } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * valueTable(td0(riverSwim.create(), { initialValue: 1 }).init(key(0))) // => [1, 1, 1, 1, 1, 1]
 * ```
 */
export function valueTable(state: TabularState | TdState): number[] {
  if ('V' in state) return Array.from(state.V)
  return Array.from({ length: state.nStates }, (_, s) => maxOf(row(state.Q, s, state.nActions)))
}

/**
 * The greedy policy of a tabular control agent state: the first maximising action of every state (the policy
 * `greedy` follows, as a table for figures and for `policyEvaluation`).
 *
 * @param state - A tabular control agent's state.
 * @returns One action per state.
 * @concept q-learning
 * @example
 * ```ts
 * import { greedyTable, qLearning } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * greedyTable(qLearning(cliffWalking.create()).init(key(0))).length // => 48
 * ```
 */
export function greedyTable(state: TabularState): number[] {
  return Array.from({ length: state.nStates }, (_, s) => argmaxOf(row(state.Q, s, state.nActions)))
}
