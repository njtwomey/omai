/**
 * The test beds of prediction and off-policy learning, each carrying the truth it is used to check: the random walk of
 * Sutton and Barto (the 5-, 19- and 1000-state chains, whose true values under the random policy are known), Baird's
 * counterexample (seven states, linear features and the weights that make semi-gradient off-policy TD diverge, whose
 * true values are all zero) and the one-state process of Example 5.5, where ordinary importance sampling has infinite
 * variance and weighted importance sampling does not.
 */

import { type Env, type EnvDefinition, mdpFromEnv, type Outcome, policyEvaluation, tabularEnv } from '../rl/index.ts'
import { circle, line, type Scene, type SceneNode, scene, text } from '../scene/index.ts'
import { dict, int, real } from '../space/index.ts'

// -------------------------------------------------------------------------------------------------- random walk

/** Options of {@link makeRandomWalk}. */
export interface RandomWalkOptions {
  /** The number of non-terminal states (default 19). */
  readonly n?: number
  /** The most states one move can jump (default 1; Sutton and Barto's 1000-state walk jumps up to 100). */
  readonly jump?: number
  /** The reward for leaving on the left (default 0; the 19-state walk of Sutton and Barto uses -1). */
  readonly leftReward?: number
  /** The reward for leaving on the right (default 1). */
  readonly rightReward?: number
}

const walkParameters = (o: RandomWalkOptions): Required<RandomWalkOptions> => ({
  n: o.n ?? 19,
  jump: o.jump ?? 1,
  leftReward: o.leftReward ?? 0,
  rightReward: o.rightReward ?? 1,
})

function renderWalk(n: number, s: number): Scene {
  const w = Math.min(60, Math.max(4, 900 / n))
  const nodes: SceneNode[] = []
  for (let i = 0; i < n; i++) {
    const r = Math.max(2, w / 2 - 2)
    nodes.push(circle(w / 2 + i * w, 40, r, { fill: i === s ? 'tab1' : 'surface', stroke: 'fg', strokeWidth: 1 }))
    if (n <= 25) nodes.push(text(w / 2 + i * w, 40, String(i), { anchor: 'middle', baseline: 'middle', size: 12 }))
    if (i + 1 < n) nodes.push(line(w / 2 + i * w + r, 40, w / 2 + (i + 1) * w - r, 40, { stroke: 'muted' }))
  }
  return scene(n * w, 80, nodes, { title: 'RandomWalk' })
}

/**
 * Sutton and Barto's random walk: a chain of `n` states from which the two actions (left, right) move one state, or up
 * to `jump` states when `jump` exceeds 1, uniformly at random; leaving the chain ends the episode with the reward of
 * that side. The walk starts in the middle. With `n = 5` and rewards 0 and 1 this is Example 6.2, with `n = 19` and
 * rewards -1 and 1 the walk of the lambda-return chapter, with `n = 1000` and `jump = 100` the walk of state
 * aggregation. Its model is exact, so {@link randomWalkValues} is the truth the learners are scored against.
 *
 * @param options - The size, jump and edge rewards.
 * @returns The environment, with its model; observations are the state index.
 * @example
 * ```ts
 * import { makeRandomWalk } from 'mdml-engine/environments'
 * makeRandomWalk({ n: 5 }).model?.transitions(0, 0) // => [{ probability: 1, state: 0, reward: 0, terminated: true }]
 * ```
 */
export function makeRandomWalk(options: RandomWalkOptions = {}): Env<number, number, number> {
  const { n, jump, leftReward, rightReward } = walkParameters(options)
  const transitions = (s: number, a: number): Outcome<number>[] => {
    const direction = a === 0 ? -1 : 1
    const out: Outcome<number>[] = []
    for (let k = 1; k <= jump; k++) {
      const to = s + direction * k
      const probability = 1 / jump
      if (to < 0) out.push({ probability, state: s, reward: leftReward, terminated: true })
      else if (to >= n) out.push({ probability, state: s, reward: rightReward, terminated: true })
      else out.push({ probability, state: to, reward: 0, terminated: false })
    }
    return out
  }
  const initial = new Array<number>(n).fill(0)
  initial[Math.floor(n / 2)] = 1
  return tabularEnv({
    id: 'RandomWalk-v0',
    description: 'Drift along a chain of states to one of its two ends, where the episode pays its reward.',
    nStates: n,
    nActions: 2,
    transitions,
    initial,
    render: (s) => renderWalk(n, s),
    maxEpisodeSteps: 100 * n,
    stateLabel: 'position on the chain',
    actionLabel: 'left (0) or right (1)',
  })
}

/**
 * The true state values of the random walk under the uniformly random policy, undiscounted: for the one-step walk
 * `v(i) = leftReward + (rightReward - leftReward) (i + 1) / (n + 1)` in closed form, and for a walk with jumps by
 * solving the linear system of the exact model.
 *
 * @param options - The same options as {@link makeRandomWalk}.
 * @returns One value per state.
 * @example
 * ```ts
 * import { randomWalkValues } from 'mdml-engine/environments'
 * randomWalkValues({ n: 5 }).map((v) => Math.round(v * 6)) // => [1, 2, 3, 4, 5]
 * ```
 */
export function randomWalkValues(options: RandomWalkOptions = {}): number[] {
  const { n, jump, leftReward, rightReward } = walkParameters(options)
  if (jump === 1)
    return Array.from({ length: n }, (_, i) => leftReward + ((rightReward - leftReward) * (i + 1)) / (n + 1))
  const mdp = mdpFromEnv(makeRandomWalk(options), 1)
  return policyEvaluation(mdp, new Array(n).fill([0.5, 0.5]) as number[][])
}

/**
 * RandomWalk-v0 (Sutton and Barto, Examples 6.2 and 7.1 and the 1000-state walk of Chapter 9).
 *
 * @concept environments
 * @example
 * ```ts
 * import { randomWalk } from 'mdml-engine/environments'
 * randomWalk.create({ n: 5 }).observationSpace // => { kind: 'discrete', n: 5, label: 'position on the chain' }
 * ```
 */
export const randomWalk: EnvDefinition<number, number, number> = {
  id: 'RandomWalk-v0',
  description: 'Drift along a chain of states to one of its two ends, where the episode pays its reward.',
  options: dict({
    n: int(1, 2000, { default: 19 }),
    jump: int(1, 200, { default: 1 }),
    leftReward: real(-10, 10, { default: 0 }),
    rightReward: real(-10, 10, { default: 1 }),
  }),
  create: (options) => makeRandomWalk((options ?? {}) as RandomWalkOptions),
}

// ------------------------------------------------------------------------------------------------------- Baird

/** The linear features of Baird's counterexample: seven states, eight weights (`2 w_i + w_7`, and `w_6 + 2 w_7`). */
export const BAIRD_FEATURES: readonly (readonly number[])[] = Array.from({ length: 7 }, (_, s) =>
  Array.from({ length: 8 }, (_, j) => (s < 6 ? (j === s ? 2 : j === 7 ? 1 : 0) : j === 6 ? 1 : j === 7 ? 2 : 0)),
)

/** The behaviour policy of Baird's counterexample over (dashed, solid): the dashed action six times in seven. */
export const BAIRD_BEHAVIOUR: readonly number[] = [6 / 7, 1 / 7]

/** The target policy of Baird's counterexample: the solid action, always. */
export const BAIRD_TARGET: readonly number[] = [0, 1]

/** The initial weights of the usual demonstration. */
export const BAIRD_INITIAL_WEIGHTS: readonly number[] = [1, 1, 1, 1, 1, 1, 10, 1]

/** The discount of the usual demonstration. */
export const BAIRD_GAMMA = 0.99

/** What Baird's counterexample is run to show. */
export interface BairdTruth {
  /** The true values of the target policy: zero in every state (every reward is zero). */
  readonly values: readonly number[]
  /** The weights that represent them exactly: all zero. */
  readonly exactWeights: readonly number[]
  /** The discount. */
  readonly gamma: number
  /** Semi-gradient off-policy TD(0) diverges from {@link BAIRD_INITIAL_WEIGHTS}, though the values are representable. */
  readonly divergesUnderSemiGradientTd: true
}

/** The truth carried by Baird's counterexample. */
export const BAIRD_TRUTH: BairdTruth = {
  values: new Array<number>(7).fill(0),
  exactWeights: new Array<number>(8).fill(0),
  gamma: BAIRD_GAMMA,
  divergesUnderSemiGradientTd: true,
}

function renderBaird(s: number): Scene {
  const nodes: SceneNode[] = []
  for (let i = 0; i < 7; i++) {
    const x = 60 + (i < 6 ? i * 70 : 175)
    const y = i < 6 ? 40 : 140
    nodes.push(circle(x, y, 24, { fill: i === s ? 'tab1' : 'surface', stroke: 'fg', strokeWidth: 1 }))
    nodes.push(text(x, y, i < 6 ? `2w${i}+w7` : 'w6+2w7', { anchor: 'middle', baseline: 'middle', size: 11 }))
  }
  return scene(480, 180, nodes, { title: "Baird's counterexample" })
}

/**
 * Baird's counterexample (Baird 1995; Sutton and Barto, Example 11.1): seven states and two actions, the dashed action
 * (0) moving to one of the six upper states uniformly and the solid action (1) to the lower state, with every reward
 * zero and no termination. Used with the linear features {@link BAIRD_FEATURES}, the behaviour policy
 * {@link BAIRD_BEHAVIOUR} and the target {@link BAIRD_TARGET}, off-policy semi-gradient TD(0) diverges for any positive
 * step size although the true values (all zero) are representable; gradient-TD and emphatic-TD methods converge.
 * Episodes start in a uniformly random state and are cut after 1000 steps.
 *
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeBairdCounterexample } from 'mdml-engine/environments'
 * makeBairdCounterexample().model?.transitions(0, 1) // => [{ probability: 1, state: 6, reward: 0, terminated: false }]
 * ```
 */
export function makeBairdCounterexample(): Env<number, number, number> {
  const transitions = (_s: number, a: number): Outcome<number>[] =>
    a === 1
      ? [{ probability: 1, state: 6, reward: 0, terminated: false }]
      : Array.from({ length: 6 }, (_, i) => ({ probability: 1 / 6, state: i, reward: 0, terminated: false }))
  return tabularEnv({
    id: 'BairdCounterexample-v0',
    description: 'Seven states with linear features on which off-policy semi-gradient TD diverges.',
    nStates: 7,
    nActions: 2,
    transitions,
    initial: new Array<number>(7).fill(1 / 7),
    render: renderBaird,
    maxEpisodeSteps: 1000,
    stateLabel: 'one of six upper states or the lower state',
    actionLabel: 'dashed (0) or solid (1)',
  })
}

/**
 * BairdCounterexample-v0.
 *
 * @concept environments
 * @example
 * ```ts
 * import { bairdCounterexample } from 'mdml-engine/environments'
 * bairdCounterexample.create().actionSpace.kind // => 'discrete'
 * ```
 */
export const bairdCounterexample: EnvDefinition<number, number, number> = {
  id: 'BairdCounterexample-v0',
  description: 'Seven states with linear features on which off-policy semi-gradient TD diverges.',
  create: () => makeBairdCounterexample(),
}

// ------------------------------------------------------------------------------------- infinite variance process

/** The true value of the start state of the infinite-variance process under the always-left target policy. */
export const INFINITE_VARIANCE_VALUE = 1

/**
 * The one-state process of Sutton and Barto's Example 5.5: from its only state the left action (0) returns to the state
 * with probability 0.9 and ends the episode with reward 1 with probability 0.1, and the right action (1) ends the
 * episode with reward 0. The target policy that always goes left has value 1; the behaviour that picks uniformly makes
 * the ordinary importance-sampling estimate of that value converge to 1 with infinite variance, and the weighted one
 * converge with finite variance.
 *
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeInfiniteVariance } from 'mdml-engine/environments'
 * makeInfiniteVariance().model?.transitions(0, 1) // => [{ probability: 1, state: 0, reward: 0, terminated: true }]
 * ```
 */
export function makeInfiniteVariance(): Env<number, number, number> {
  return tabularEnv({
    id: 'InfiniteVariance-v0',
    description: 'One state whose always-left policy is worth 1 but whose importance ratios have infinite variance.',
    nStates: 1,
    nActions: 2,
    transitions: (_s, a): Outcome<number>[] =>
      a === 0
        ? [
            { probability: 0.9, state: 0, reward: 0, terminated: false },
            { probability: 0.1, state: 0, reward: 1, terminated: true },
          ]
        : [{ probability: 1, state: 0, reward: 0, terminated: true }],
    initial: [1],
    render: () => scene(80, 80, [circle(40, 40, 24, { fill: 'tab1', stroke: 'fg' })], { title: 'InfiniteVariance' }),
    maxEpisodeSteps: 10_000,
    stateLabel: 'the only state',
    actionLabel: 'left (0) or right (1)',
  })
}

/**
 * InfiniteVariance-v0 (Sutton and Barto, Example 5.5).
 *
 * @concept environments
 * @example
 * ```ts
 * import { infiniteVariance } from 'mdml-engine/environments'
 * infiniteVariance.create().observationSpace.kind // => 'discrete'
 * ```
 */
export const infiniteVariance: EnvDefinition<number, number, number> = {
  id: 'InfiniteVariance-v0',
  description: 'One state whose always-left policy is worth 1 but whose importance ratios have infinite variance.',
  create: () => makeInfiniteVariance(),
}
