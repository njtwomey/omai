import { type Json, linspaceNumbers, meanOf, WhiteboxError } from '../../lib/base/index.ts'
import { type EpisodeRecord, type Explained, episodeRecord } from '../../lib/exposition/index.ts'
import { foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  type Agent,
  type Env,
  type Policy,
  type TimeLimitState,
  type TrainState,
  type Transition,
  timeLimit,
  trainingAlgorithm,
} from '../../lib/rl/index.ts'
import type { Scene } from '../../lib/scene/index.ts'
import {
  bool,
  choice,
  type DictSpace,
  defaults,
  dict,
  discrete,
  int,
  real,
  type SpaceValue,
} from '../../lib/space/index.ts'
import { type Algorithm, type RunOptions, type RunStatus, runAlgorithm } from '../../lib/trace/index.ts'
import {
  a2c,
  epsilonGreedyBandit,
  expectedSarsa,
  greedyTable,
  qLearning,
  reinforce,
  type SweepingPlanState,
  sarsa,
  sarsaLambda,
  sweepingPlannerAgent,
  type TabularState,
  thompsonBernoulli,
  ucb1,
  valueTable,
} from './agents/index.ts'
import { bernoulliBandit, cartPole, cliffWalking, frozenLake, mountainCar } from './environments/index.ts'

// ------------------------------------------------------------------------------------------------ the catalogue

/** Parameter values of a playground pairing (its `hyperSpace`'s values). */
export type PlaygroundParams = Readonly<Record<string, SpaceValue>>

/** What a pairing builds from its parameters: the environment (without a time limit), the agent and the limit. */
export interface PlaygroundSetup {
  readonly env: Env<unknown, unknown, unknown>
  readonly agent: Agent<unknown, unknown, unknown>
  /** Steps after which an episode is truncated. */
  readonly maxEpisodeSteps: number
  /** The values and greedy policy of the agent's state (tabular agents), for snapshots. */
  readonly snapshot?: (agentState: unknown) => {
    readonly values?: readonly number[]
    readonly policy?: readonly number[]
  }
}

/** The family of environments a pairing belongs to (for pickers). */
export type PlaygroundFamily = 'bandits' | 'gridworlds' | 'classic-control'

/**
 * An environment and an agent with demo parameters that learn visibly in a short budget: the unit of the
 * reinforcement-learning playground.
 */
export interface PlaygroundPairing {
  /** `<environment>/<agent>`, e.g. `cliff-walking/q-learning`. */
  readonly id: string
  readonly title: string
  readonly summary: string
  readonly family: PlaygroundFamily
  /** The environment's short name, e.g. `cliff-walking`. */
  readonly environment: string
  /** The agent's short name, e.g. `q-learning`. */
  readonly agent: string
  /** The parameters, their defaults the demo's: the agent's, the environment's and the budget (`episodes`, ...). */
  readonly hyperSpace: DictSpace
  /** What snapshot values and policies are indexed by: environment states, or a bandit's arms. */
  readonly snapshotAxis?: 'state' | 'arm'
  /** Build the environment and agent; `key` initialises networks. */
  setup(params: PlaygroundParams, key: Key): PlaygroundSetup
}

const num = (params: PlaygroundParams, name: string): number => {
  const v = params[name]
  if (typeof v !== 'number') throw new WhiteboxError('value', `playground: parameter '${name}' must be a number`)
  return v
}
const flag = (params: PlaygroundParams, name: string): boolean => params[name] === true
const text = (params: PlaygroundParams, name: string): string => String(params[name])

type AnyAgent = Agent<unknown, unknown, unknown>
type AnyEnv = Env<unknown, unknown, unknown>
const asAgent = <G, O, A>(agent: Agent<G, O, A>): AnyAgent => agent as unknown as AnyAgent
const asEnv = <S, O, A>(env: Env<S, O, A>): AnyEnv => env as unknown as AnyEnv

const tabularSnapshot = (g: unknown): { values: number[]; policy: number[] } => ({
  values: valueTable(g as TabularState),
  policy: greedyTable(g as TabularState),
})
const plannerSnapshot = (g: unknown): { values: number[]; policy: number[] } => ({
  values: [...(g as SweepingPlanState).V],
  policy: [...(g as SweepingPlanState).policy],
})

const episodesSpace = (low: number, high: number, value: number) =>
  int(low, high, { default: value, label: 'episodes' })

// Bandits: an episode is a block of pulls; the agent keeps learning across blocks, so the return of a block (its
// total reward) climbs towards `pulls` times the best arm's probability.
const banditSpaces = {
  arms: int(2, 10, { default: 5, label: 'arms' }),
  pulls: int(5, 200, { default: 20, label: 'pulls per episode' }),
  episodes: episodesSpace(5, 500, 50),
}

function bandit(params: PlaygroundParams): Env<unknown, number, number> {
  const arms = num(params, 'arms')
  const probs = linspaceNumbers(0.2, 0.8, arms)
  return bernoulliBandit.create({ probs, horizon: num(params, 'pulls') }) as unknown as Env<unknown, number, number>
}

function banditPairing(
  agent: string,
  title: string,
  summary: string,
  spaces: DictSpace['spaces'],
  make: (env: Env<unknown, number, number>, params: PlaygroundParams) => AnyAgent,
  estimates: (g: unknown) => ArrayLike<number>,
): PlaygroundPairing {
  return {
    id: `bandit/${agent}`,
    title,
    summary,
    family: 'bandits',
    environment: 'bandit',
    agent,
    hyperSpace: dict({ ...spaces, ...banditSpaces }),
    snapshotAxis: 'arm',
    setup: (params) => {
      const env = bandit(params)
      const values = (g: unknown): number[] => Array.from(estimates(g))
      return {
        env: asEnv(env),
        agent: make(env, params),
        maxEpisodeSteps: num(params, 'pulls'),
        snapshot: (g) => {
          const v = values(g)
          let best = 0
          for (const [i, x] of v.entries()) if (x > (v[best] ?? -Infinity)) best = i
          return { values: v, policy: [best] }
        },
      }
    },
  }
}

const cliffSpaces = {
  alpha: real(0.01, 1, { default: 0.5, label: 'step size' }),
  epsilon: real(0, 0.5, { default: 0.1, label: 'exploration rate' }),
  gamma: real(0.5, 1, { default: 1, label: 'discount' }),
  episodes: episodesSpace(10, 1000, 200),
  maxEpisodeSteps: int(20, 500, { default: 100, label: 'steps per episode' }),
}

function cliffPairing(
  agent: string,
  title: string,
  summary: string,
  make: (env: Env<number, number, number>, options: { alpha: number; epsilon: number; gamma: number }) => AnyAgent,
): PlaygroundPairing {
  return {
    id: `cliff-walking/${agent}`,
    title,
    summary,
    family: 'gridworlds',
    environment: 'cliff-walking',
    agent,
    hyperSpace: dict(cliffSpaces),
    setup: (params) => {
      const env = cliffWalking.create()
      return {
        env: asEnv(env),
        agent: make(env, { alpha: num(params, 'alpha'), epsilon: num(params, 'epsilon'), gamma: num(params, 'gamma') }),
        maxEpisodeSteps: num(params, 'maxEpisodeSteps'),
        snapshot: tabularSnapshot,
      }
    },
  }
}

function lakePlannerPairing(
  planner: 'valueIteration' | 'policyIteration',
  agent: string,
  title: string,
  summary: string,
  spaces: DictSpace['spaces'],
): PlaygroundPairing {
  return {
    id: `frozen-lake/${agent}`,
    title,
    summary,
    family: 'gridworlds',
    environment: 'frozen-lake',
    agent,
    hyperSpace: dict({
      isSlippery: bool({ default: true, label: 'slippery ice' }),
      gamma: real(0.5, 0.999, { default: 0.99, label: 'discount' }),
      ...spaces,
      episodes: episodesSpace(10, 1000, 200),
    }),
    setup: (params) => {
      const env = frozenLake.create({ isSlippery: flag(params, 'isSlippery') })
      return {
        env: asEnv(env),
        agent: asAgent(
          sweepingPlannerAgent(env, {
            planner,
            gamma: num(params, 'gamma'),
            sweeps: num(params, 'sweeps'),
            every: num(params, 'every'),
          }),
        ),
        maxEpisodeSteps: 100,
        snapshot: plannerSnapshot,
      }
    },
  }
}

/** The grid cell of a box observation, `bins` per dimension (row-major), for tabular agents on continuous states. */
function binner(low: readonly number[], high: readonly number[], bins: number): (o: readonly number[]) => number {
  return (o) => {
    let code = 0
    for (const [d, lo] of low.entries()) {
      const hi = high[d] ?? lo
      const b = Math.floor((((o[d] ?? lo) - lo) / (hi - lo)) * bins)
      code = code * bins + Math.min(bins - 1, Math.max(0, b))
    }
    return code
  }
}

/**
 * The playground's pairings: bandits (epsilon-greedy, UCB1, Thompson sampling), cliff walking (Q-learning, SARSA,
 * expected SARSA), frozen lake (Q-learning, and value and policy iteration planning a little after every episode),
 * CartPole (REINFORCE and A2C) and MountainCar (SARSA(lambda) on a grid of the state). Every default learns visibly
 * within a second or so in a browser. DQN is left out: on CartPole it needs far more steps than REINFORCE or A2C
 * before its returns move.
 */
export const playgroundPairings: readonly PlaygroundPairing[] = [
  banditPairing(
    'epsilon-greedy',
    'Epsilon-greedy bandit',
    'Pull the best-looking arm, and a random one with probability epsilon.',
    {
      epsilon: real(0, 0.5, { default: 0.1, label: 'exploration rate' }),
      initialValue: real(0, 2, { default: 0, label: 'initial estimate' }),
    },
    (env, p) => asAgent(epsilonGreedyBandit(env, { epsilon: num(p, 'epsilon'), initialValue: num(p, 'initialValue') })),
    (g) => (g as { estimates: Float64Array }).estimates,
  ),
  banditPairing(
    'ucb1',
    'UCB1 bandit',
    'Pull the arm with the largest upper confidence bound on its mean.',
    { c: real(0.1, 10, { log: true, default: 2, label: 'exploration constant' }) },
    (env, p) => asAgent(ucb1(env, { c: num(p, 'c') })),
    (g) => (g as { means: Float64Array }).means,
  ),
  banditPairing(
    'thompson',
    'Thompson sampling bandit',
    'Sample each arm’s success probability from its Beta posterior and pull the largest.',
    {
      priorAlpha: real(0.1, 10, { log: true, default: 1, label: 'prior successes' }),
      priorBeta: real(0.1, 10, { log: true, default: 1, label: 'prior failures' }),
    },
    (env, p) => asAgent(thompsonBernoulli(env, { prior: [num(p, 'priorAlpha'), num(p, 'priorBeta')] })),
    (g) => {
      const s = g as { alpha: Float64Array; beta: Float64Array }
      return Array.from(s.alpha, (a, i) => a / (a + (s.beta[i] ?? 0)))
    },
  ),
  cliffPairing(
    'q-learning',
    'Q-learning on the cliff',
    'Off-policy TD control learns the shortest path along the cliff edge (and falls off while exploring).',
    (env, o) => asAgent(qLearning(env, o)),
  ),
  cliffPairing(
    'sarsa',
    'SARSA on the cliff',
    'On-policy TD control learns a safer path away from the edge.',
    (env, o) => asAgent(sarsa(env, o)),
  ),
  cliffPairing(
    'expected-sarsa',
    'Expected SARSA on the cliff',
    'TD control towards the expected next value under the exploring policy.',
    (env, o) => asAgent(expectedSarsa(env, o)),
  ),
  {
    id: 'frozen-lake/q-learning',
    title: 'Q-learning on the frozen lake',
    summary: 'Find the path across the ice to the goal, avoiding the holes.',
    family: 'gridworlds',
    environment: 'frozen-lake',
    agent: 'q-learning',
    hyperSpace: dict({
      isSlippery: bool({ default: false, label: 'slippery ice' }),
      alpha: real(0.01, 1, { default: 0.5, label: 'step size' }),
      epsilon: real(0, 0.5, { default: 0.1, label: 'exploration rate' }),
      gamma: real(0.5, 1, { default: 0.95, label: 'discount' }),
      episodes: episodesSpace(10, 2000, 300),
    }),
    setup: (params) => {
      const env = frozenLake.create({ isSlippery: flag(params, 'isSlippery') })
      return {
        env: asEnv(env),
        agent: asAgent(
          qLearning(env, { alpha: num(params, 'alpha'), epsilon: num(params, 'epsilon'), gamma: num(params, 'gamma') }),
        ),
        maxEpisodeSteps: 100,
        snapshot: tabularSnapshot,
      }
    },
  },
  lakePlannerPairing(
    'valueIteration',
    'value-iteration',
    'Value iteration on the frozen lake',
    'One Bellman sweep of the known model after every episode: values spread back from the goal.',
    {
      sweeps: int(1, 20, { default: 1, label: 'sweeps per block' }),
      every: int(1, 50, { default: 1, label: 'episodes per block' }),
    },
  ),
  lakePlannerPairing(
    'policyIteration',
    'policy-iteration',
    'Policy iteration on the frozen lake',
    'Evaluate the current policy exactly and improve it greedily, every few episodes.',
    {
      sweeps: int(1, 5, { default: 1, label: 'iterations per block' }),
      every: int(1, 50, { default: 10, label: 'episodes per block' }),
    },
  ),
  {
    id: 'cart-pole/reinforce',
    title: 'REINFORCE on CartPole',
    summary: 'A policy network trained by Monte Carlo policy gradients with a learned baseline.',
    family: 'classic-control',
    environment: 'cart-pole',
    agent: 'reinforce',
    hyperSpace: dict({
      hidden: int(4, 128, { log: true, default: 32, label: 'hidden units' }),
      learningRate: real(1e-4, 1e-1, { log: true, default: 5e-3, label: 'learning rate' }),
      gamma: real(0.9, 0.999, { default: 0.99, label: 'discount' }),
      baseline: choice(['value', 'average', 'none'], { default: 'value', label: 'baseline' }),
      episodes: episodesSpace(10, 1000, 150),
      maxSteps: int(1000, 100_000, { log: true, default: 10_000, label: 'step budget' }),
    }),
    setup: (params, key) => {
      const env = cartPole.create()
      return {
        env: asEnv(env),
        agent: asAgent(
          reinforce(env, {
            hidden: [num(params, 'hidden')],
            learningRate: num(params, 'learningRate'),
            gamma: num(params, 'gamma'),
            baseline: text(params, 'baseline') as 'value' | 'average' | 'none',
            key,
          }),
        ),
        maxEpisodeSteps: 500,
      }
    },
  },
  {
    id: 'cart-pole/a2c',
    title: 'Advantage actor-critic on CartPole',
    summary: 'A policy and a value network updated every few steps from n-step advantages.',
    family: 'classic-control',
    environment: 'cart-pole',
    agent: 'a2c',
    hyperSpace: dict({
      hidden: int(4, 128, { log: true, default: 64, label: 'hidden units' }),
      learningRate: real(1e-4, 1e-1, { log: true, default: 2e-3, label: 'learning rate' }),
      nSteps: int(1, 64, { log: true, default: 16, label: 'steps per update' }),
      entCoef: real(0, 0.1, { default: 0, label: 'entropy bonus' }),
      episodes: episodesSpace(10, 2000, 500),
      maxSteps: int(1000, 100_000, { log: true, default: 10_000, label: 'step budget' }),
    }),
    setup: (params, key) => {
      const env = cartPole.create()
      return {
        env: asEnv(env),
        agent: asAgent(
          a2c(env, {
            hidden: [num(params, 'hidden')],
            learningRate: num(params, 'learningRate'),
            nSteps: num(params, 'nSteps'),
            entCoef: num(params, 'entCoef'),
            key,
          }),
        ),
        maxEpisodeSteps: 500,
      }
    },
  },
  {
    id: 'mountain-car/sarsa-lambda',
    title: 'SARSA(lambda) on MountainCar',
    summary: 'Tabular SARSA with eligibility traces on a grid of position and velocity learns to rock up the hill.',
    family: 'classic-control',
    environment: 'mountain-car',
    agent: 'sarsa-lambda',
    hyperSpace: dict({
      bins: int(4, 30, { default: 12, label: 'grid cells per dimension' }),
      alpha: real(0.01, 1, { default: 0.3, label: 'step size' }),
      lambda: real(0, 1, { default: 0.9, label: 'trace decay' }),
      epsilon: real(0, 0.2, { default: 0, label: 'exploration rate' }),
      episodes: episodesSpace(10, 2000, 300),
    }),
    setup: (params) => {
      const env = mountainCar.create()
      const space = env.observationSpace
      if (space.kind !== 'box') throw new WhiteboxError('value', 'playground: MountainCar has box observations')
      const low = Array.isArray(space.low) ? (space.low as number[]) : [-1.2, -0.07]
      const high = Array.isArray(space.high) ? (space.high as number[]) : [0.6, 0.07]
      const bins = num(params, 'bins')
      return {
        env: asEnv(env),
        agent: asAgent(
          sarsaLambda(env, {
            encode: binner(low, high, bins),
            nStates: bins ** low.length,
            alpha: num(params, 'alpha'),
            lambda: num(params, 'lambda'),
            epsilon: num(params, 'epsilon'),
            gamma: 1,
            trace: 'replacing',
          }),
        ),
        maxEpisodeSteps: 200,
        snapshot: tabularSnapshot,
      }
    },
  },
]

/**
 * A playground pairing by id (`cliff-walking/q-learning`), or by environment and agent names.
 *
 * @param id - The pairing's id, or its environment's short name when `agent` is given.
 * @param agent - The agent's short name.
 * @returns The pairing.
 * @throws WhiteboxError with code `value` for an unknown pairing.
 */
export function playgroundPairing(id: string, agent?: string): PlaygroundPairing {
  const full = agent === undefined ? id : `${id}/${agent}`
  const found = playgroundPairings.find((p) => p.id === full)
  if (found === undefined) {
    const known = playgroundPairings.map((p) => p.id).join(', ')
    throw new WhiteboxError('value', `playground: no pairing '${full}' (known: ${known})`)
  }
  return found
}

// ------------------------------------------------------------------------------------------------ the run

/** One step of an episode in progress, as a persistent list (newest first) so recording is pure and O(1) a step. */
interface StepCell {
  readonly state: Json
  readonly action: Json
  readonly reward: number
  readonly frame?: Scene
  readonly previous?: StepCell
}

/** The episode in progress: its first state and its steps so far. */
interface Recording {
  readonly episode: number
  readonly start: Json
  readonly startFrame?: Scene
  readonly steps?: StepCell
}

/** A finished, recorded episode: per step its state, action and reward (and frames when asked for). */
export interface RecordedEpisode {
  readonly episode: number
  readonly states: readonly Json[]
  readonly actions: readonly Json[]
  readonly rewards: readonly number[]
  readonly frames?: readonly Scene[]
}

/** A snapshot of a tabular agent's values and greedy policy after a number of episodes. */
export interface PlaygroundSnapshot {
  readonly episode: number
  readonly values?: readonly number[]
  readonly policy?: readonly number[]
}

/** The agent state with the transition it learned from last (to read the action and the raw next state). */
interface Remembering {
  readonly inner: unknown
  readonly last?: Transition<unknown, unknown>
}

/** The state of {@link playgroundSession}'s algorithm. */
export interface PlaygroundState {
  readonly train: TrainState<Remembering, TimeLimitState<unknown>, unknown>
  readonly recording: Recording
  /** The scheduled episodes recorded so far. */
  readonly recorded: readonly RecordedEpisode[]
  /** The last finished episode (recorded or not). */
  readonly latest?: RecordedEpisode
  readonly snapshots: readonly PlaygroundSnapshot[]
}

/** Options of {@link playground} and {@link playgroundSession}. */
export interface PlaygroundOptions
  extends Partial<Pick<RunOptions, 'key' | 'seed' | 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** The pairing's id (`cliff-walking/q-learning`); or give `environment` and `agent`. */
  readonly pairing?: string
  /** The environment's short name (`cliff-walking`), with `agent`. */
  readonly environment?: string
  /** The agent's short name (`q-learning`), with `environment`. */
  readonly agent?: string
  /** Parameter values; missing ones take the pairing's demo defaults. */
  readonly params?: PlaygroundParams
  /** Episodes to train for (default the `episodes` parameter). */
  readonly episodes?: number
  /** A cap on environment steps (default the `maxSteps` parameter, else ten million). */
  readonly maxSteps?: number
  /** Episodes to record in full (default the first and the middle one); the last finished episode always is. */
  readonly record?: readonly number[]
  /** Episode counts at which to snapshot a tabular agent (default five, evenly from 0); the end always is. */
  readonly snapshots?: readonly number[]
  /** Render a frame of every recorded step (default false; frames are scenes, kept beside the record). */
  readonly frames?: boolean
  /** Called after every finished episode (for progress bars and live views; it must not change anything). */
  readonly onEpisode?: (progress: PlaygroundProgress) => void
}

/** What {@link PlaygroundOptions.onEpisode} receives. */
export interface PlaygroundProgress {
  readonly state: PlaygroundState
  /** Episodes finished. */
  readonly episode: number
  /** The share of the budget spent: the larger of the episode and step fractions. */
  readonly fraction: number
  /** The episode record so far. */
  record(): EpisodeRecord
}

/** A playground run, ready to step: its pairing, parameters, budget and algorithm, and the record of any state. */
export interface PlaygroundSession {
  readonly pairing: PlaygroundPairing
  readonly params: PlaygroundParams
  readonly episodes: number
  readonly maxSteps: number
  /** The environment the agent trains in (time-limited). */
  readonly env: Env<TimeLimitState<unknown>, unknown, unknown>
  /** Training as an algorithm: one environment step per algorithm step. */
  readonly algorithm: Algorithm<PlaygroundState>
  /** The episode record of a state (any step of the algorithm, so a page can show learning live). */
  record(state: PlaygroundState): EpisodeRecord
  /** The recorded episodes of a state, with frames when the session renders them. */
  episodesOf(state: PlaygroundState): RecordedEpisode[]
}

const RAW_STATE = 'playgroundState'

/** The environment with its raw next state in every step's info, so the terminal state reaches the recorder. */
function exposeState(env: AnyEnv): AnyEnv {
  return {
    ...env,
    step: (s, a, k) => {
      const r = env.step(s, a, k)
      return { ...r, info: { ...r.info, [RAW_STATE]: r.state as Json } }
    },
    wrapped: env,
  }
}

function remembering(agent: AnyAgent): Agent<Remembering, unknown, unknown> {
  const stats = agent.stats
  return {
    name: agent.name,
    config: agent.config,
    ...(agent.metrics === undefined ? {} : { metrics: agent.metrics }),
    init: (k) => ({ inner: agent.init(k) }),
    act: (g, o, k) => agent.act(g.inner, o, k),
    learn: (g, t, k) => ({ inner: agent.learn(g.inner, t, k), last: t }),
    greedy: (g, o) => agent.greedy(g.inner, o),
    ...(stats === undefined ? {} : { stats: (g: Remembering) => stats(g.inner) }),
  }
}

function finish(r: Recording, end: StepCell | undefined): RecordedEpisode {
  const cells: StepCell[] = []
  for (let c = end; c !== undefined; c = c.previous) cells.push(c)
  cells.reverse()
  const frames = r.startFrame === undefined ? undefined : [r.startFrame, ...cells.map((c) => c.frame as Scene)]
  return {
    episode: r.episode,
    states: [r.start, ...cells.map((c) => c.state)],
    actions: cells.map((c) => c.action),
    rewards: cells.map((c) => c.reward),
    ...(frames === undefined ? {} : { frames }),
  }
}

function resolvePairing(options: PlaygroundOptions): PlaygroundPairing {
  if (options.pairing !== undefined) return playgroundPairing(options.pairing)
  if (options.environment !== undefined && options.agent !== undefined)
    return playgroundPairing(options.environment, options.agent)
  throw new WhiteboxError('value', 'playground: give a pairing, or an environment and an agent')
}

const evenly = (n: number, count: number): number[] => [
  ...new Set(linspaceNumbers(0, n, count).map((x) => Math.round(x))),
]

/**
 * Set up a playground run without running it: the pairing's environment and agent built from its parameters, and
 * training as an {@link Algorithm} (one environment step per algorithm step, through `trainingAlgorithm` of
 * `lib/rl`) that also records chosen episodes in full and snapshots tabular values and policies. A page steps the
 * algorithm with `runAlgorithm` (or step by step) and draws `record(state)` as it goes.
 *
 * @param options - The pairing (or environment and agent), parameters, budget, episodes to record, snapshot
 *   schedule and frames.
 * @returns The session.
 * @throws WhiteboxError with code `value` for an unknown pairing or a parameter of the wrong type.
 */
export function playgroundSession(options: PlaygroundOptions): PlaygroundSession {
  const pairing = resolvePairing(options)
  const params: PlaygroundParams = {
    ...(defaults(pairing.hyperSpace) as PlaygroundParams),
    ...(options.params ?? {}),
  }
  const episodes = options.episodes ?? num(params, 'episodes')
  const maxSteps = options.maxSteps ?? (typeof params['maxSteps'] === 'number' ? params['maxSteps'] : 10_000_000)
  const root = options.key ?? makeKey(options.seed ?? 0)
  const setup = pairing.setup(params, foldIn(root, 7))
  const base = setup.env
  const env = timeLimit(exposeState(base), setup.maxEpisodeSteps) as Env<TimeLimitState<unknown>, unknown, unknown>
  const inner = trainingAlgorithm(env, remembering(setup.agent), { episodes, checkpoints: false })
  const recordSet = new Set(options.record ?? [0, Math.floor(episodes / 2)])
  const snapshotSet = new Set(options.snapshots ?? evenly(episodes, 5))
  const frames = options.frames === true
  const snap = setup.snapshot
  const onEpisode = options.onEpisode
  const snapshotOf = (episode: number, g: Remembering): PlaygroundSnapshot =>
    snap === undefined ? { episode } : { episode, ...snap(g.inner) }
  const startOf = (episode: number, state: TimeLimitState<unknown>): Recording => ({
    episode,
    start: state.inner as Json,
    ...(frames ? { startFrame: base.render(state.inner) } : {}),
  })

  const algorithm: Algorithm<PlaygroundState> = {
    name: `playground:${pairing.id}`,
    config: { pairing: pairing.id, params: params as Json, episodes, maxSteps } as Json,
    ...(inner.schema === undefined ? {} : { schema: inner.schema }),
    init: (k) => {
      const train = inner.init(k)
      return {
        train,
        recording: startOf(0, train.envState),
        recorded: [],
        snapshots: snap !== undefined && snapshotSet.has(0) ? [snapshotOf(0, train.agent)] : [],
      }
    },
    step: (s, ctx) => {
      const train = inner.step(s.train, ctx)
      const t = train.agent.last
      if (t === undefined) throw new WhiteboxError('value', 'playground: the agent learned from no transition')
      const raw = t.info?.[RAW_STATE] as unknown
      const cell: StepCell = {
        state: raw as Json,
        action: t.action as Json,
        reward: t.reward,
        ...(frames ? { frame: base.render(raw) } : {}),
        ...(s.recording.steps === undefined ? {} : { previous: s.recording.steps }),
      }
      if (train.episode === s.train.episode) return { ...s, train, recording: { ...s.recording, steps: cell } }
      const done = finish(s.recording, cell)
      const next: PlaygroundState = {
        train,
        recording: startOf(train.episode, train.envState),
        recorded: recordSet.has(done.episode) ? [...s.recorded, done] : s.recorded,
        latest: done,
        snapshots:
          snap !== undefined && snapshotSet.has(train.episode)
            ? [...s.snapshots, snapshotOf(train.episode, train.agent)]
            : s.snapshots,
      }
      onEpisode?.({
        state: next,
        episode: train.episode,
        fraction: Math.min(1, Math.max(train.episode / episodes, train.totalSteps / maxSteps)),
        record: () => record(next),
      })
      return next
    },
    done: (s, t) => inner.done?.(s.train, t),
    ...(inner.observe === undefined
      ? {}
      : {
          observe: (s: PlaygroundState, t: number) => (inner.observe as NonNullable<typeof inner.observe>)(s.train, t),
        }),
  }

  const episodesOf = (state: PlaygroundState): RecordedEpisode[] => {
    const out = [...state.recorded]
    const latest = state.latest
    if (latest !== undefined && !out.some((r) => r.episode === latest.episode)) out.push(latest)
    return out
  }
  const record = (state: PlaygroundState): EpisodeRecord => {
    const train = state.train
    const snapshots = [...state.snapshots]
    if (snap !== undefined && snapshots.at(-1)?.episode !== train.episode)
      snapshots.push(snapshotOf(train.episode, train.agent))
    return episodeRecord({
      environment: pairing.environment,
      agent: pairing.agent,
      returns: train.returns,
      lengths: train.lengths,
      recorded: episodesOf(state).map(({ frames: _f, ...r }) => r),
      ...(snap === undefined ? {} : { snapshots }),
    })
  }
  return { pairing, params, episodes, maxSteps, env, algorithm, record, episodesOf }
}

/** The result of {@link playground}. */
export interface PlaygroundResult {
  readonly pairing: string
  /** The environment's id (`CliffWalking-v1`). */
  readonly environment: string
  readonly agent: string
  readonly params: PlaygroundParams
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  readonly episodes: number
  readonly steps: number
  readonly status: RunStatus
  /** Mean return of the first and of the last tenth of the episodes (at least one episode each). */
  readonly before: number
  readonly after: number
  /** What snapshot values are indexed by. */
  readonly snapshotAxis: 'state' | 'arm'
  /** Frames of the recorded episodes, in the record's order, when asked for. */
  readonly frames?: readonly (readonly Scene[])[]
  /** The trained agent's state. */
  readonly agentState: unknown
  readonly runId: string
}

/**
 * Train an agent in an environment of the playground and explain it: returns and lengths per episode, a few episodes
 * recorded step by step for playback (the first, the middle and the last by default), and value and policy snapshots
 * for tabular agents, as one `episodes` exposition record. Every pairing's defaults learn visibly in about a second.
 * The run is `trainingAlgorithm` of `lib/rl` through `runAlgorithm`, traced (rewards, the agent's statistics, an
 * `episode` event per episode); use {@link playgroundSession} to step it live.
 *
 * @param options - The pairing (or environment and agent), parameters, budget, recording and run options.
 * @returns The result with its `episodes` record.
 * @throws WhiteboxError with code `value` for an unknown pairing.
 * @example
 * ```ts
 * import { playground } from 'whitebox-engine/methods/rl'
 * const run = playground({ pairing: 'cliff-walking/q-learning', episodes: 100, seed: 0 })
 * run.after > run.before // => true
 * ```
 */
export function playground(options: PlaygroundOptions): Explained<PlaygroundResult> {
  const session = playgroundSession(options)
  const { key, seed, tracer, policy, parent, runId } = options
  const result = runAlgorithm(session.algorithm, {
    maxSteps: session.maxSteps,
    ...(key === undefined ? { seed: seed ?? 0 } : { key }),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  })
  const s = result.state
  const returns = s.train.returns
  const tenth = Math.max(1, Math.floor(returns.length / 10))
  const frames = options.frames === true ? session.episodesOf(s).map((r) => r.frames ?? []) : undefined
  return {
    pairing: session.pairing.id,
    environment: session.env.spec.id,
    agent: session.pairing.agent,
    params: session.params,
    returns,
    lengths: s.train.lengths,
    episodes: s.train.episode,
    steps: s.train.totalSteps,
    status: result.status,
    before: returns.length === 0 ? Number.NaN : meanOf(returns.slice(0, tenth)),
    after: returns.length === 0 ? Number.NaN : meanOf(returns.slice(-tenth)),
    snapshotAxis: session.pairing.snapshotAxis ?? 'state',
    ...(frames === undefined ? {} : { frames }),
    agentState: s.train.agent.inner,
    runId: result.runId,
    exposition: [session.record(s)],
  }
}

// ------------------------------------------------------------------------------------------------ playback

/** A recorded episode as an environment and a policy, for players that run an episode (render's `EpisodePlayer`). */
export interface EpisodeReplay {
  /** States are step indices; each step moves to the recorded next state and pays the recorded reward. */
  readonly env: Env<number, number, Json>
  /** The recorded action at each step. */
  readonly policy: Policy<number, Json>
}

/**
 * Play a recorded episode back exactly: an environment whose state is the step index, which moves to the recorded
 * next state and pays the recorded reward whatever the dynamics' randomness (slippery ice, bandit draws), drawn by
 * the pairing's own `render`, and the policy that takes the recorded actions. A page builds it from an `episodes`
 * record to hand to a player, since the trained agent stays where it was trained (a worker, say).
 *
 * @param pairing - The pairing's id (its environment draws the frames).
 * @param recorded - A recorded episode of its `episodes` record.
 * @param params - The parameters of the run (the environment's options among them; default the demo's).
 * @returns The environment and policy.
 * @throws WhiteboxError with code `value` for an unknown pairing.
 */
export function replayRecorded(
  pairing: string,
  recorded: {
    readonly states: readonly Json[]
    readonly actions: readonly Json[]
    readonly rewards: readonly number[]
  },
  params: PlaygroundParams = {},
): EpisodeReplay {
  const p = playgroundPairing(pairing)
  const base = p.setup({ ...(defaults(p.hyperSpace) as PlaygroundParams), ...params }, makeKey(0)).env
  const n = recorded.actions.length
  const env: Env<number, number, Json> = {
    spec: { id: `${base.spec.id} (replay)`, description: `A recorded episode of ${base.spec.id}`, deterministic: true },
    observationSpace: discrete(n + 1, { label: 'step' }),
    actionSpace: base.actionSpace,
    reset: () => ({ state: 0, observation: 0, info: {} }),
    step: (t) => {
      const next = Math.min(t + 1, n)
      return {
        state: next,
        observation: next,
        reward: recorded.rewards[t] ?? 0,
        terminated: next >= n,
        truncated: false,
        info: {},
      }
    },
    render: (t) => base.render(recorded.states[t] as unknown),
    observe: (t) => t,
  }
  return { env, policy: (t) => recorded.actions[t] ?? 0 }
}
