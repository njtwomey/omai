/**
 * A goal-conditioned reaching task for hindsight experience replay: a point in the unit square moves in small steps
 * towards a goal drawn anew each episode, and the reward is sparse (0 on reaching it, -1 otherwise). The observation
 * carries the position, then the goal, so a goal-conditioned agent reads both; the environment also exports what
 * relabelling needs (the achieved goal of an observation, the reward a goal would have paid, the success test) and
 * the truth (the fewest steps from a start to a goal).
 */

import { clamp } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import type { Env, EnvDefinition } from '../rl/index.ts'
import { circle, line, rect, scene } from '../scene/index.ts'
import { box, dict, discrete, int, real } from '../space/index.ts'

/** Options of {@link makeReachGoal}. */
export interface ReachGoalOptions {
  /** The length of one move (default 0.1). */
  readonly stepSize?: number
  /** The distance from the goal that counts as reaching it (default 0.075, a little over half the diagonal of a step so that every goal can be reached). */
  readonly tolerance?: number
  /** The steps allowed per episode (default 30). */
  readonly horizon?: number
  /** The least distance between the start and the goal (default 0): large values make the reward rare. */
  readonly minDistance?: number
}

/** The state of the reaching task: the position and the goal. */
export interface ReachGoalState {
  readonly position: readonly [number, number]
  readonly goal: readonly [number, number]
  readonly t: number
}

const MOVES: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]

/**
 * The achieved goal of an observation `[x, y, gx, gy]`: the position, the part of the observation a goal can name.
 *
 * @param observation - The observation.
 * @returns `[x, y]`.
 * @example
 * ```ts
 * import { achievedGoal } from 'omai-engine/environments'
 * achievedGoal([0.1, 0.2, 0.9, 0.8]) // => [0.1, 0.2]
 * ```
 */
export function achievedGoal(observation: readonly number[]): [number, number] {
  return [observation[0] ?? 0, observation[1] ?? 0]
}

/**
 * The desired goal of an observation `[x, y, gx, gy]`.
 *
 * @param observation - The observation.
 * @returns `[gx, gy]`.
 * @example
 * ```ts
 * import { desiredGoal } from 'omai-engine/environments'
 * desiredGoal([0.1, 0.2, 0.9, 0.8]) // => [0.9, 0.8]
 * ```
 */
export function desiredGoal(observation: readonly number[]): [number, number] {
  return [observation[2] ?? 0, observation[3] ?? 0]
}

/**
 * Whether an achieved goal is within the tolerance of a desired one.
 *
 * @param achieved - Where the point is.
 * @param desired - The goal.
 * @param tolerance - The distance that counts (default 0.075, a little over half the diagonal of a step so that every goal can be reached).
 * @returns Whether the goal is reached.
 * @example
 * ```ts
 * import { goalReached } from 'omai-engine/environments'
 * goalReached([0.5, 0.5], [0.52, 0.5]) // => true
 * ```
 */
export function goalReached(achieved: readonly number[], desired: readonly number[], tolerance = 0.075): boolean {
  return Math.hypot((achieved[0] ?? 0) - (desired[0] ?? 0), (achieved[1] ?? 0) - (desired[1] ?? 0)) <= tolerance
}

/**
 * The sparse reward a goal would have paid for an achieved position: 0 when reached, -1 otherwise. Hindsight
 * relabelling recomputes rewards with this function after swapping the goal.
 *
 * @param achieved - Where the point is.
 * @param desired - The goal.
 * @param tolerance - The distance that counts (default 0.075, a little over half the diagonal of a step so that every goal can be reached).
 * @returns 0 or -1.
 * @example
 * ```ts
 * import { goalReward } from 'omai-engine/environments'
 * goalReward([0, 0], [1, 1]) // => -1
 * ```
 */
export function goalReward(achieved: readonly number[], desired: readonly number[], tolerance = 0.075): number {
  return goalReached(achieved, desired, tolerance) ? 0 : -1
}

/**
 * The fewest moves that take a point from a start to within the tolerance of a goal on the lattice of moves (the
 * truth a reaching policy is scored against; walls of the square are ignored, which only matters at the edge).
 *
 * @param from - The start.
 * @param to - The goal.
 * @param options - The step size and tolerance.
 * @returns The number of moves.
 * @example
 * ```ts
 * import { reachGoalOptimalSteps } from 'omai-engine/environments'
 * reachGoalOptimalSteps([0.5, 0.5], [0.8, 0.5]) // => 3
 * ```
 */
export function reachGoalOptimalSteps(
  from: readonly number[],
  to: readonly number[],
  options: ReachGoalOptions = {},
): number {
  const h = options.stepSize ?? 0.1
  const tol = options.tolerance ?? 0.075
  const dx = ((to[0] ?? 0) - (from[0] ?? 0)) / h
  const dy = ((to[1] ?? 0) - (from[1] ?? 0)) / h
  const reach = tol / h
  let best = Number.POSITIVE_INFINITY
  const span = Math.ceil(Math.abs(dx) + Math.abs(dy) + reach) + 1
  for (let i = -span; i <= span; i++) {
    for (let j = -span; j <= span; j++) {
      if (Math.hypot(dx - i, dy - j) <= reach + 1e-9) best = Math.min(best, Math.abs(i) + Math.abs(j))
    }
  }
  return best
}

/**
 * The reaching task as an environment: the observation `[x, y, gx, gy]`, four moves (up, right, down, left) of one step
 * length, reward 0 on reaching the goal (which ends the episode) and -1 otherwise, and a horizon after which the episode
 * is truncated. Start and goal are drawn uniformly from the unit square at each reset, so a policy must be conditioned
 * on the goal, and a random walk reaches a distant goal rarely: the setting in which relabelling failed episodes with the
 * goals they did reach teaches what the sparse reward cannot.
 *
 * @param options - Step size, tolerance, horizon and the least start-goal distance.
 * @returns The environment.
 * @example
 * ```ts
 * import { makeReachGoal } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * makeReachGoal().reset(key(0)).observation.length // => 4
 * ```
 */
export function makeReachGoal(options: ReachGoalOptions = {}): Env<ReachGoalState, readonly number[], number> {
  const h = options.stepSize ?? 0.1
  const tol = options.tolerance ?? 0.075
  const horizon = options.horizon ?? 30
  const minDistance = options.minDistance ?? 0
  const observe = (s: ReachGoalState): readonly number[] => [...s.position, ...s.goal]
  return {
    spec: {
      id: 'ReachGoal-v0',
      description: 'Move a point to a goal drawn at each reset; the reward is 0 on reaching it and -1 otherwise.',
      maxEpisodeSteps: horizon,
    },
    observationSpace: box(0, 1, [4], { label: 'position then goal' }),
    actionSpace: discrete(4, { label: 'up (0), right (1), down (2), left (3)' }),
    reset: (k: Key) => {
      // Redraw the goal until it is far enough from the start (the far corner after 100 tries, which always is).
      let u = uniform(k, [4]).data
      for (
        let attempt = 1;
        attempt < 100 && Math.hypot((u[2] ?? 0) - (u[0] ?? 0), (u[3] ?? 0) - (u[1] ?? 0)) < minDistance;
        attempt++
      ) {
        const v = uniform(foldIn(k, attempt), [2]).data
        u = Float64Array.of(u[0] ?? 0, u[1] ?? 0, v[0] ?? 0, v[1] ?? 0)
      }
      const state: ReachGoalState = {
        position: [u[0] ?? 0, u[1] ?? 0],
        goal: [u[2] ?? 0, u[3] ?? 0],
        t: 0,
      }
      return { state, observation: observe(state), info: {} }
    },
    step: (s, action) => {
      const [dx, dy] = MOVES[action] ?? [0, 0]
      const position: [number, number] = [clamp(s.position[0] + h * dx, 0, 1), clamp(s.position[1] + h * dy, 0, 1)]
      const state: ReachGoalState = { position, goal: s.goal, t: s.t + 1 }
      const success = goalReached(position, s.goal, tol)
      return {
        state,
        observation: observe(state),
        reward: success ? 0 : -1,
        terminated: success,
        truncated: false,
        info: { success: success ? 1 : 0 },
      }
    },
    render: (s) =>
      scene(
        200,
        200,
        [
          rect(0, 0, 200, 200, { fill: 'surface', stroke: 'fg' }),
          circle(s.goal[0] * 200, (1 - s.goal[1]) * 200, tol * 200, { fill: 'tab2', stroke: 'fg' }),
          circle(s.position[0] * 200, (1 - s.position[1]) * 200, 5, { fill: 'tab1', stroke: 'fg' }),
          line(s.position[0] * 200, (1 - s.position[1]) * 200, s.goal[0] * 200, (1 - s.goal[1]) * 200, {
            stroke: 'muted',
          }),
        ],
        { title: 'ReachGoal' },
      ),
    observe,
  }
}

/**
 * ReachGoal-v0.
 *
 * @concept environments
 * @example
 * ```ts
 * import { reachGoal } from 'omai-engine/environments'
 * reachGoal.create().spec.maxEpisodeSteps // => 30
 * ```
 */
export const reachGoal: EnvDefinition<ReachGoalState, readonly number[], number> = {
  id: 'ReachGoal-v0',
  description: 'Move a point to a goal drawn at each reset; the reward is 0 on reaching it and -1 otherwise.',
  options: dict({
    stepSize: real(0.01, 0.5, { default: 0.1 }),
    tolerance: real(0.01, 0.5, { default: 0.075 }),
    horizon: int(1, 10_000, { default: 30 }),
    minDistance: real(0, 1.4, { default: 0 }),
  }),
  create: (options) => makeReachGoal((options ?? {}) as ReachGoalOptions),
}
