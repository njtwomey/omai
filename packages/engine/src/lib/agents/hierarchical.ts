/**
 * Hierarchical RL with options (Sutton, Precup and Singh 1999): temporally extended actions, each with an initiation
 * set, an internal policy and a termination condition. Option builders (primitive actions, subgoal options solved from
 * a model, the hallway options of the four-rooms domain), SMDP Q-learning and intra-option Q-learning over a set of
 * options, and the tabular option-critic, which learns the options' policies and terminations from reward alone.
 */

import { argmaxOf, type Json, OmaiError } from '../base/index.ts'
import { expit } from '../numerics/special/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import {
  type Agent,
  boltzmannProbabilities,
  type Env,
  type EnvModel,
  mdpFromModel,
  type Outcome,
  valueIteration,
} from '../rl/index.ts'
import { item } from '../tensor/index.ts'
import { FOUR_ROOMS } from '../worlds/index.ts'
import {
  filled,
  nested,
  type Rate,
  rateAt,
  type StepSize,
  stepSizeAt,
  type TabularCodingOptions,
  type TabularEnvLike,
  tabularCoding,
  unit,
} from './_tabular.ts'

// ------------------------------------------------------------------------------------------------------- options

/**
 * An option (Sutton, Precup and Singh 1999): a temporally extended action over state indices, with an initiation set
 * (where it may start), an intra-option policy (action probabilities) and a termination function (the probability
 * `beta(s)` of stopping on entering `s`).
 *
 * @concept options-framework
 */
export interface HierarchicalOption {
  readonly name: string
  initiation(state: number): boolean
  policy(state: number): readonly number[]
  termination(state: number): number
}

/**
 * The primitive actions as one-step options: available everywhere, taking their action and terminating after one
 * step, so a policy over options that includes them can always act as finely as a flat agent.
 *
 * @param nActions - The number of actions.
 * @returns One option per action.
 * @concept options-framework
 * @example
 * ```ts
 * import { primitiveOptions } from 'omai-engine/agents'
 * const [up] = primitiveOptions(4)
 * [up?.policy(7), up?.termination(7)] // => [[1, 0, 0, 0], 1]
 * ```
 */
export function primitiveOptions(nActions: number): HierarchicalOption[] {
  return Array.from({ length: nActions }, (_, a) => {
    const probs = Array.from({ length: nActions }, (_, b) => (b === a ? 1 : 0))
    return { name: `action ${a}`, initiation: () => true, policy: () => probs, termination: () => 1 }
  })
}

/**
 * An option that drives to a subgoal state within a region, with its policy derived from an environment model by
 * solving a sub-MDP (value iteration): reaching the subgoal pays 1 and ends the sub-task, leaving the region (or an
 * episode end elsewhere) ends it with 0, and the discount makes shorter routes better. The option may start anywhere
 * in the region and terminates everywhere outside it (including at the subgoal).
 *
 * @param model - The environment model (its states must be the indices `0 .. n - 1`).
 * @param region - The states the option may start and run in.
 * @param subgoal - The target state.
 * @param options - The discount of the sub-MDP (default 0.9) and a name.
 * @returns The option.
 * @concept options-framework
 * @example
 * ```ts
 * import { subgoalOption } from 'omai-engine/agents'
 * import { makeCliffWalking } from 'omai-engine/environments'
 * const env = makeCliffWalking()
 * const right = env.model === undefined ? undefined : subgoalOption(env.model, [0, 1, 2], 3)
 * right?.policy(0) // => [0, 1, 0, 0]
 * ```
 */
export function subgoalOption(
  model: EnvModel<number, number>,
  region: readonly number[],
  subgoal: number,
  options: { readonly gamma?: number; readonly name?: string } = {},
): HierarchicalOption {
  const inside = new Set(region)
  inside.delete(subgoal)
  const sub: EnvModel<number, number> = {
    states: model.states,
    actions: model.actions,
    stateIndex: (s) => model.stateIndex(s),
    actionIndex: (a) => model.actionIndex(a),
    initial: region.map((s) => ({ probability: 1 / region.length, state: s })),
    transitions: (s, a): Outcome<number>[] => {
      if (!inside.has(s)) return [{ probability: 1, state: s, reward: 0, terminated: true }]
      return model.transitions(s, a).map((o) => {
        if (o.state === subgoal) return { probability: o.probability, state: o.state, reward: 1, terminated: true }
        if (o.terminated || !inside.has(o.state)) {
          return { probability: o.probability, state: o.state, reward: 0, terminated: true }
        }
        return { probability: o.probability, state: o.state, reward: 0, terminated: false }
      })
    },
  }
  const solution = valueIteration(mdpFromModel(sub, options.gamma ?? 0.9), { tol: 1e-10 })
  const nA = model.actions.length
  const policies = solution.policy.map((a) => Array.from({ length: nA }, (_, b) => (b === a ? 1 : 0)))
  return {
    name: options.name ?? `to ${subgoal}`,
    initiation: (s) => inside.has(s),
    policy: (s) => policies[s] ?? new Array<number>(nA).fill(1 / nA),
    termination: (s) => (inside.has(s) ? 0 : 1),
  }
}

/** The rooms and hallways of a grid layout: free cells with walls on two opposite sides are hallways. */
function roomsOf(layout: readonly string[]): {
  readonly cells: readonly (readonly [number, number])[]
  readonly rooms: readonly (readonly number[])[]
  readonly hallways: readonly number[]
} {
  const free = (r: number, c: number): boolean => (layout[r]?.[c] ?? 'w') !== 'w'
  const cells: [number, number][] = []
  for (const [r, row] of layout.entries()) for (let c = 0; c < row.length; c++) if (free(r, c)) cells.push([r, c])
  const index = new Map(cells.map(([r, c], i) => [`${r},${c}`, i]))
  const isHallway = ([r, c]: readonly [number, number]): boolean =>
    (!free(r - 1, c) && !free(r + 1, c)) || (!free(r, c - 1) && !free(r, c + 1))
  const hallways = cells.flatMap((cell, i) => (isHallway(cell) ? [i] : []))
  const hall = new Set(hallways)
  const room = new Array<number>(cells.length).fill(-1)
  const rooms: number[][] = []
  for (let i = 0; i < cells.length; i++) {
    if (hall.has(i) || (room[i] ?? -1) >= 0) continue
    const members: number[] = []
    const stack = [i]
    room[i] = rooms.length
    while (stack.length > 0) {
      const j = stack.pop() as number
      members.push(j)
      const [r, c] = cells[j] ?? [0, 0]
      for (const [dr, dc] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ] as const) {
        const k = index.get(`${r + dr},${c + dc}`)
        if (k !== undefined && !hall.has(k) && (room[k] ?? -1) < 0) {
          room[k] = rooms.length
          stack.push(k)
        }
      }
    }
    rooms.push(members.sort((a, b) => a - b))
  }
  return { cells, rooms, hallways }
}

/**
 * The hallway options of the four-rooms domain (Sutton, Precup and Singh 1999, section 3): for each room and each of
 * its two hallways, an option that may start in the room (or at the room's other hallway) and drives to that hallway,
 * with its policy solved from the environment's model ({@link subgoalOption}). Rooms and hallways are read from the
 * layout (hallways are free cells between two walls); state indices are the layout's free cells in row-major order,
 * as in `makeFourRooms`.
 *
 * @param env - A four-rooms environment with its model (default goal and success probability, or any).
 * @param options - The layout (default `FOUR_ROOMS`) and the sub-MDP discount (default 0.9).
 * @returns Eight options for the standard layout.
 * @throws OmaiError with code `value` when the environment has no model.
 * @concept options-framework
 * @example
 * ```ts
 * import { hallwayOptions } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * hallwayOptions(makeFourRooms()).length // => 8
 * ```
 */
export function hallwayOptions(
  env: Env<number, number, number>,
  options: { readonly layout?: readonly string[]; readonly gamma?: number } = {},
): HierarchicalOption[] {
  const model = env.model
  if (model === undefined) throw new OmaiError('value', 'hallwayOptions: the environment has no model')
  const { cells, rooms, hallways } = roomsOf(options.layout ?? FOUR_ROOMS)
  const adjacent = (i: number, j: number): boolean => {
    const [r1, c1] = cells[i] ?? [0, 0]
    const [r2, c2] = cells[j] ?? [0, 0]
    return Math.abs(r1 - r2) + Math.abs(c1 - c2) === 1
  }
  const out: HierarchicalOption[] = []
  for (const [ri, members] of rooms.entries()) {
    const doors = hallways.filter((h) => members.some((m) => adjacent(m, h)))
    for (const door of doors) {
      const region = [...members, ...doors.filter((d) => d !== door)]
      const [r, c] = cells[door] ?? [0, 0]
      out.push(
        subgoalOption(model, region, door, {
          ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
          name: `room ${ri} to hallway (${r}, ${c})`,
        }),
      )
    }
  }
  return out
}

// ------------------------------------------------------------------------------------------------ option agents

/** Options of {@link smdpQLearning} and {@link intraOptionQLearning}. */
export interface OptionAgentOptions<O> extends TabularCodingOptions<O> {
  /** The options to choose among (include {@link primitiveOptions} to keep every action available). */
  readonly options: readonly HierarchicalOption[]
  /** Step size (default 0.1). */
  readonly alpha?: StepSize
  /** Discount per primitive step (default 0.99). */
  readonly gamma?: number
  /** Exploration rate of the epsilon-greedy policy over options (default 0.1), a constant or a schedule over decisions. */
  readonly epsilon?: Rate
  /** Initial option values (default 0). */
  readonly initialValue?: number
}

/** The option being executed: where it started and the discounted reward accumulated so far. */
export interface RunningOption {
  readonly option: number
  readonly start: number
  readonly reward: number
  /** `gamma^k` after `k` steps. */
  readonly discount: number
  readonly length: number
}

/** The state of an option agent: option values `Q[s * nOptions + o]`, the running option and the choice stream. */
export interface OptionAgentState {
  readonly nStates: number
  readonly nOptions: number
  readonly Q: Float64Array
  readonly visits: Float64Array
  readonly steps: number
  readonly episodes: number
  readonly tdError: number
  /** The agent's own key for choosing options, so `act` and `learn` agree on a new choice. */
  readonly key: Key
  /** Option choices made so far (the index into the choice stream). */
  readonly decisions: number
  readonly running?: RunningOption
  /** The length of the last option that terminated. */
  readonly lastOptionLength: number
}

/** Epsilon-greedy choice among the given indices of a value row (ties at random). */
function chooseAmong(
  values: ArrayLike<number>,
  offset: number,
  allowed: readonly number[],
  eps: number,
  k: Key,
): number {
  const u = uniform(k, [2 + allowed.length]).data
  if ((u[0] ?? 0) < eps) return allowed[Math.min(allowed.length - 1, Math.floor((u[1] ?? 0) * allowed.length))] ?? 0
  let best = -Infinity
  let choice = allowed[0] ?? 0
  let ties = 0
  for (const [j, o] of allowed.entries()) {
    const v = values[offset + o] ?? 0
    if (v > best) {
      best = v
      choice = o
      ties = 1
    } else if (v === best) {
      ties += 1
      if ((u[2 + j] ?? 0) * ties < 1) choice = o
    }
  }
  return choice
}

function sampleAction(probs: readonly number[], k: Key): number {
  const u = unit(k)
  let c = 0
  for (const [a, p] of probs.entries()) {
    c += p
    if (u < c) return a
  }
  return probs.length - 1
}

function optionAgent<O>(
  name: string,
  env: TabularEnvLike<O>,
  o: OptionAgentOptions<O>,
  intra: boolean,
): Agent<OptionAgentState, O, number> {
  const coding = tabularCoding(env, o, name)
  const nS = coding.nStates
  const opts = o.options
  const nO = opts.length
  if (nO === 0) throw new OmaiError('value', `${name}: give at least one option`)
  const gamma = o.gamma ?? 0.99
  const alpha = o.alpha ?? 0.1
  const epsilon = o.epsilon ?? 0.1
  const availableCache = new Map<number, number[]>()
  const available = (s: number): number[] => {
    const hit = availableCache.get(s)
    if (hit !== undefined) return hit
    const list = opts.flatMap((op, i) => (op.initiation(s) ? [i] : []))
    const out = list.length > 0 ? list : opts.map((_, i) => i)
    availableCache.set(s, out)
    return out
  }
  const bestValue = (Q: Float64Array, s: number): number => {
    let m = -Infinity
    for (const i of available(s)) m = Math.max(m, Q[s * nO + i] ?? 0)
    return m
  }
  const choose = (st: OptionAgentState, s: number): number =>
    chooseAmong(st.Q, s * nO, available(s), rateAt(epsilon, st.decisions), foldIn(st.key, st.decisions))
  const current = (st: OptionAgentState, s: number): number => st.running?.option ?? choose(st, s)
  return {
    name,
    config: {
      nStates: nS,
      options: opts.map((op) => op.name),
      gamma,
      alpha: typeof alpha === 'function' ? 'schedule' : alpha,
      epsilon: typeof epsilon === 'number' ? epsilon : 'schedule',
    } as Json,
    metrics: {
      tdError: { description: 'TD error of the last update' },
      optionLength: { description: 'length in steps of the last option to terminate' },
      meanOptionLength: { description: 'steps per option choice so far' },
    },
    init: (k) => ({
      nStates: nS,
      nOptions: nO,
      Q: filled(nS * nO, o.initialValue ?? 0),
      visits: new Float64Array(nS * nO),
      steps: 0,
      episodes: 0,
      tdError: 0,
      key: foldIn(k, 0x6f707473),
      decisions: 0,
      lastOptionLength: 0,
    }),
    act: (st, obs, k) => {
      const s = coding.encode(obs)
      return sampleAction(opts[current(st, s)]?.policy(s) ?? [1], k)
    },
    greedy: (st, obs) => {
      // Evaluation polls the best option at every step (interrupting options, Sutton et al. section 5).
      const s = coding.encode(obs)
      let best = -Infinity
      let choice = 0
      for (const i of available(s)) {
        const v = st.Q[s * nO + i] ?? 0
        if (v > best) {
          best = v
          choice = i
        }
      }
      return argmaxOf(opts[choice]?.policy(s) ?? [1])
    },
    learn: (st, t, k) => {
      const s = coding.encode(t.observation)
      const s2 = coding.encode(t.nextObservation)
      const oi = current(st, s)
      const run: RunningOption = st.running ?? { option: oi, start: s, reward: 0, discount: 1, length: 0 }
      const acc: RunningOption = {
        ...run,
        reward: run.reward + run.discount * t.reward,
        discount: run.discount * gamma,
        length: run.length + 1,
      }
      const ended = t.terminated || t.truncated
      const stop = ended || unit(foldIn(k, 0)) < (opts[oi]?.termination(s2) ?? 1)
      const Q = st.Q.slice()
      const visits = st.visits.slice()
      let delta = st.tdError
      const update = (i: number, target: number, weight: number): void => {
        const n = (visits[i] ?? 0) + 1
        visits[i] = n
        delta = target - (Q[i] ?? 0)
        Q[i] = (Q[i] ?? 0) + weight * stepSizeAt(alpha, n) * delta
      }
      if (intra) {
        // Every option whose policy could have taken this action learns from the step (Sutton et al. section 7).
        const vNext = t.terminated ? 0 : bestValue(st.Q, s2)
        for (const [i, op] of opts.entries()) {
          const p = op.policy(s)[t.action] ?? 0
          if (p === 0) continue
          const beta = op.termination(s2)
          const u = t.terminated ? 0 : (1 - beta) * (st.Q[s2 * nO + i] ?? 0) + beta * vNext
          update(s * nO + i, t.reward + gamma * u, p)
        }
      } else if (stop) {
        // The SMDP backup of the whole option: r_1 + ... + gamma^{k-1} r_k + gamma^k max_o' Q(s', o').
        update(acc.start * nO + oi, acc.reward + (t.terminated ? 0 : acc.discount * bestValue(st.Q, s2)), 1)
      }
      const base = {
        ...st,
        Q,
        visits,
        tdError: delta,
        steps: st.steps + 1,
        episodes: st.episodes + (ended ? 1 : 0),
      }
      if (!stop) return { ...base, running: acc }
      const { running: _r, ...rest } = base
      return { ...rest, decisions: st.decisions + 1, lastOptionLength: acc.length }
    },
    stats: (st) => ({
      tdError: st.tdError,
      optionLength: st.lastOptionLength,
      meanOptionLength: st.steps / Math.max(st.decisions, 1),
    }),
  }
}

/**
 * SMDP Q-learning over options (Bradtke and Duff 1995; Sutton, Precup and Singh 1999, section 3): an epsilon-greedy
 * policy over the options available in a state picks one, which runs until it terminates; then its value moves towards
 * the discounted reward it collected plus `gamma^k max_o' Q(s', o')`. The agent acts in primitive actions; its state
 * tracks the running option, and new choices come from the agent's own key stream so `act` and `learn` agree.
 *
 * @param env - The environment (discrete observations, or give `encode`).
 * @param options - The options, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @throws OmaiError with code `value` when no options are given.
 * @concept options-framework
 * @example
 * ```ts
 * import { primitiveOptions, smdpQLearning } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const env = makeFourRooms()
 * smdpQLearning(env, { options: primitiveOptions(4) }).init(key(0)).Q.length // => 416
 * ```
 */
export function smdpQLearning<O>(
  env: TabularEnvLike<O>,
  options: OptionAgentOptions<O>,
): Agent<OptionAgentState, O, number> {
  return optionAgent('smdpQLearning', env, options, false)
}

/**
 * Intra-option Q-learning (Sutton, Precup and Singh 1999, section 7): behaves like {@link smdpQLearning}, but every
 * primitive step updates every option consistent with it (weighted by its probability of the action), towards
 * `r + gamma U(s', o)` with `U(s', o) = (1 - beta_o(s')) Q(s', o) + beta_o(s') max_o' Q(s', o')`, so options learn
 * from experience gathered while executing others.
 *
 * @param env - The environment.
 * @param options - The options, step size, discount, exploration and the observation coding.
 * @returns The agent.
 * @concept options-framework
 * @example
 * ```ts
 * import { hallwayOptions, intraOptionQLearning, primitiveOptions } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * const env = makeFourRooms()
 * intraOptionQLearning(env, { options: [...hallwayOptions(env), ...primitiveOptions(4)] }).name // => 'intraOptionQLearning'
 * ```
 */
export function intraOptionQLearning<O>(
  env: TabularEnvLike<O>,
  options: OptionAgentOptions<O>,
): Agent<OptionAgentState, O, number> {
  return optionAgent('intraOptionQLearning', env, options, true)
}

/**
 * The option values of an option agent state as rows `Q[s][o]`.
 *
 * @param state - The agent state.
 * @returns `Q[s][o]`.
 * @concept options-framework
 * @example
 * ```ts
 * import { optionValues, primitiveOptions, smdpQLearning } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * optionValues(smdpQLearning(makeFourRooms(), { options: primitiveOptions(4) }).init(key(0)))[0] // => [0, 0, 0, 0]
 * ```
 */
export function optionValues(state: OptionAgentState | OptionCriticState): number[][] {
  return nested('Q' in state ? state.Q : state.QOmega, state.nStates, state.nOptions)
}

// ------------------------------------------------------------------------------------------------- option-critic

/** Options of {@link optionCritic}. */
export interface OptionCriticOptions<O> extends TabularCodingOptions<O> {
  /** The number of options (default 4). */
  readonly nOptions?: number
  /** Step size of the critics `Q_Omega` and `Q_U` (default 0.5). */
  readonly alphaCritic?: number
  /** Step size of the intra-option policies (default 0.25). */
  readonly alphaTheta?: number
  /** Step size of the terminations (default 0.25). */
  readonly alphaBeta?: number
  /** Discount (default 0.99). */
  readonly gamma?: number
  /** Exploration of the epsilon-greedy policy over options (default 0.05). */
  readonly epsilon?: number
  /** Temperature of the softmax intra-option policies (default 0.01, as Bacon et al.'s four-rooms experiments). */
  readonly temperature?: number
  /** Subtract `Q_Omega(s, o)` as a baseline in the policy gradient (default true). */
  readonly baseline?: boolean
  /**
   * The deliberation cost `xi` (Harb et al. 2018): a cost charged when an option terminates, which enters the
   * critic's target and the termination gradient and so favours longer options (default 0).
   */
  readonly deliberationCost?: number
}

/** The state of {@link optionCritic}: the critics, the intra-option policy and termination parameters. */
export interface OptionCriticState {
  readonly nStates: number
  readonly nOptions: number
  readonly nActions: number
  /** Intra-option policy parameters `theta[(o * nStates + s) * nActions + a]`. */
  readonly theta: Float64Array
  /** Termination parameters `vartheta[o * nStates + s]` (beta = sigmoid). */
  readonly vartheta: Float64Array
  /** Option values `Q_Omega[s * nOptions + o]`. */
  readonly QOmega: Float64Array
  /** Values of acting in the context of an option `Q_U[(s * nOptions + o) * nActions + a]`. */
  readonly QU: Float64Array
  readonly steps: number
  readonly episodes: number
  readonly tdError: number
  readonly key: Key
  readonly decisions: number
  readonly running?: number
  readonly lastOptionLength: number
  readonly optionLength: number
}

/**
 * The option-critic architecture (Bacon, Harb and Precup 2017) in its tabular form: options with softmax intra-option
 * policies and sigmoid terminations, all learned from reward alone. Each step the critics move `Q_Omega(s, o)` and
 * `Q_U(s, o, a)` towards `r + gamma U(s', o)`, `U(s', o) = (1 - beta_o(s')) Q_Omega(s', o) + beta_o(s') (V(s') - xi)`;
 * the intra-option policy follows the gradient `grad log pi_o(a | s) (Q_U(s, o, a) - Q_Omega(s, o))`; and the
 * termination follows `-grad beta_o(s') (Q_Omega(s', o) - V(s') + xi)`, so options stop where switching pays (more
 * reluctantly with a deliberation cost `xi`). A new option is chosen epsilon-greedily on `Q_Omega` whenever one
 * terminates, from the agent's own key stream.
 *
 * @param env - The environment.
 * @param options - Number of options, step sizes, discount, exploration, temperature, baseline and deliberation cost.
 * @returns The agent.
 * @concept option-critic
 * @example
 * ```ts
 * import { optionCritic } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * optionCritic(makeFourRooms(), { nOptions: 4 }).init(key(0)).theta.length // => 1664
 * ```
 */
export function optionCritic<O>(
  env: TabularEnvLike<O>,
  options: OptionCriticOptions<O> = {},
): Agent<OptionCriticState, O, number> {
  const coding = tabularCoding(env, options, 'optionCritic')
  const { nStates: nS, nActions: nA } = coding
  const nO = options.nOptions ?? 4
  const aC = options.alphaCritic ?? 0.5
  const aT = options.alphaTheta ?? 0.25
  const aB = options.alphaBeta ?? 0.25
  const gamma = options.gamma ?? 0.99
  const eps = options.epsilon ?? 0.05
  const temp = options.temperature ?? 0.01
  const useBaseline = options.baseline ?? true
  const xi = options.deliberationCost ?? 0
  const all = Array.from({ length: nO }, (_, i) => i)
  const pi = (st: OptionCriticState, o: number, s: number): number[] =>
    boltzmannProbabilities(st.theta.subarray((o * nS + s) * nA, (o * nS + s + 1) * nA), temp)
  const beta = (vartheta: Float64Array, o: number, s: number): number => item(expit(vartheta[o * nS + s] ?? 0))
  const value = (Q: Float64Array, s: number): number => {
    let m = -Infinity
    for (let o = 0; o < nO; o++) m = Math.max(m, Q[s * nO + o] ?? 0)
    return m
  }
  const choose = (st: OptionCriticState, s: number): number =>
    chooseAmong(st.QOmega, s * nO, all, eps, foldIn(st.key, st.decisions))
  const current = (st: OptionCriticState, s: number): number => st.running ?? choose(st, s)
  return {
    name: 'optionCritic',
    config: {
      nStates: nS,
      nActions: nA,
      nOptions: nO,
      alphaCritic: aC,
      alphaTheta: aT,
      alphaBeta: aB,
      gamma,
      epsilon: eps,
      temperature: temp,
      baseline: useBaseline,
      deliberationCost: xi,
    },
    metrics: {
      tdError: { description: 'TD error of the critic' },
      optionLength: { description: 'length in steps of the last option to terminate' },
      meanOptionLength: { description: 'steps per option choice so far' },
    },
    init: (k) => ({
      nStates: nS,
      nOptions: nO,
      nActions: nA,
      theta: new Float64Array(nO * nS * nA),
      vartheta: new Float64Array(nO * nS),
      QOmega: new Float64Array(nS * nO),
      QU: new Float64Array(nS * nO * nA),
      steps: 0,
      episodes: 0,
      tdError: 0,
      key: foldIn(k, 0x6f63),
      decisions: 0,
      lastOptionLength: 0,
      optionLength: 0,
    }),
    act: (st, obs, k) => {
      const s = coding.encode(obs)
      return sampleAction(pi(st, current(st, s), s), k)
    },
    greedy: (st, obs) => {
      const s = coding.encode(obs)
      const o = argmaxOf(st.QOmega.subarray(s * nO, (s + 1) * nO))
      return argmaxOf(st.theta.subarray((o * nS + s) * nA, (o * nS + s + 1) * nA))
    },
    learn: (st, t, k) => {
      const s = coding.encode(t.observation)
      const s2 = coding.encode(t.nextObservation)
      const a = t.action
      const o = current(st, s)
      const ended = t.terminated || t.truncated
      const b2 = beta(st.vartheta, o, s2)
      const stop = ended || unit(foldIn(k, 0)) < b2
      // Critic: Q_Omega and Q_U towards r + gamma U(s', o).
      const v2 = value(st.QOmega, s2)
      const u = t.terminated ? 0 : (1 - b2) * (st.QOmega[s2 * nO + o] ?? 0) + b2 * (v2 - xi)
      const target = t.reward + gamma * u
      const QOmega = st.QOmega.slice()
      const QU = st.QU.slice()
      const iO = s * nO + o
      const iU = iO * nA + a
      const delta = target - (QOmega[iO] ?? 0)
      QOmega[iO] = (QOmega[iO] ?? 0) + aC * delta
      QU[iU] = (QU[iU] ?? 0) + aC * (target - (QU[iU] ?? 0))
      // Actor: the intra-option policy gradient at (s, o, a).
      const theta = st.theta.slice()
      const probs = pi(st, o, s)
      const advantage = (QU[iU] ?? 0) - (useBaseline ? (QOmega[iO] ?? 0) : 0)
      const base = (o * nS + s) * nA
      for (let b = 0; b < nA; b++) {
        theta[base + b] = (theta[base + b] ?? 0) + (aT * ((b === a ? 1 : 0) - (probs[b] ?? 0)) * advantage) / temp
      }
      // Termination gradient at s': lower beta where the option is better than the best (plus the cost).
      const vartheta = st.vartheta.slice()
      if (!t.terminated) {
        const j = o * nS + s2
        vartheta[j] = (vartheta[j] ?? 0) - aB * b2 * (1 - b2) * ((QOmega[s2 * nO + o] ?? 0) - value(QOmega, s2) + xi)
      }
      const length = st.optionLength + 1
      const next = {
        ...st,
        theta,
        vartheta,
        QOmega,
        QU,
        tdError: delta,
        steps: st.steps + 1,
        episodes: st.episodes + (ended ? 1 : 0),
      }
      if (!stop) return { ...next, running: o, optionLength: length }
      const { running: _r, ...rest } = next
      return { ...rest, decisions: st.decisions + 1, lastOptionLength: length, optionLength: 0 }
    },
    stats: (st) => ({
      tdError: st.tdError,
      optionLength: st.lastOptionLength,
      meanOptionLength: st.steps / Math.max(st.decisions, 1),
    }),
  }
}

/**
 * The termination probabilities of an option-critic state, `beta[o][s]` (for figures of where options stop).
 *
 * @param state - The agent state.
 * @returns `beta[o][s]`.
 * @concept option-critic
 * @example
 * ```ts
 * import { optionCritic, optionTerminations } from 'omai-engine/agents'
 * import { makeFourRooms } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * optionTerminations(optionCritic(makeFourRooms()).init(key(0)))[0]?.[0] // => 0.5
 * ```
 */
export function optionTerminations(state: OptionCriticState): number[][] {
  return nested(
    Array.from(state.vartheta, (x) => item(expit(x))),
    state.nOptions,
    state.nStates,
  )
}
