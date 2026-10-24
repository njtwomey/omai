// Helpers for offline reinforcement-learning tests: a tiny continuous control task, and noisy experts from exact
// solutions of tabular environments. Node only; not published.
import { argmaxOf } from '../lib/base/index.ts'
import { foldIn, type Key, normal, uniform } from '../lib/random/index.ts'
import { type Env, type EnvDefinition, mdpFromEnv, type Policy, valueIteration } from '../lib/rl/index.ts'
import { circle, line, scene } from '../lib/scene/index.ts'
import { box } from '../lib/space/index.ts'

/** The state of {@link reach1d}: a position on a line. */
export interface ReachState {
  readonly x: number
}

const STEP = 0.25

/**
 * A 1-D reach task: the position starts uniformly in [-1, 1], the action in [-1, 1] moves it by `0.25 a` (clipped to
 * [-2, 2]), and each step pays `-|x'|`, so the best policy drives to the origin and stays (`a = clip(-4 x, -1, 1)`).
 * Ten steps per episode.
 */
export const reach1d: EnvDefinition<ReachState, readonly number[], readonly number[]> = {
  id: 'Reach1D-v0',
  description: 'Move a point on a line to the origin and keep it there.',
  create: (): Env<ReachState, readonly number[], readonly number[]> => ({
    spec: { id: 'Reach1D-v0', description: 'Move a point on a line to the origin.', maxEpisodeSteps: 10 },
    observationSpace: box(-2, 2, [1]),
    actionSpace: box(-1, 1, [1]),
    reset: (k: Key) => {
      const x = 2 * (uniform(k, []).data[0] ?? 0.5) - 1
      return { state: { x }, observation: [x], info: {} }
    },
    step: (s, a) => {
      const u = Math.min(1, Math.max(-1, a[0] ?? 0))
      const x = Math.min(2, Math.max(-2, s.x + STEP * u))
      return { state: { x }, observation: [x], reward: -Math.abs(x), terminated: false, truncated: false, info: {} }
    },
    observe: (s) => [s.x],
    render: (s) =>
      scene(200, 40, [line(0, 20, 200, 20, { stroke: 'muted' }), circle(100 + 50 * s.x, 20, 6, { fill: 'tab1' })]),
  }),
}

/** The expert of {@link reach1d} with Gaussian action noise of standard deviation `sigma`. */
export function reachExpert(sigma: number): Policy<readonly number[], readonly number[]> {
  return (o, k) => {
    const a = -4 * (o[0] ?? 0) + sigma * (normal(foldIn(k, 0), []).data[0] ?? 0)
    return [Math.min(1, Math.max(-1, a))]
  }
}

/**
 * An epsilon-greedy expert for a tabular environment seen through one-hot observations: optimal actions from value
 * iteration of its model, and with probability `epsilon` a uniformly random action.
 */
export function tabularExpert(
  env: Env<unknown, unknown, unknown>,
  gamma: number,
  epsilon: number,
): Policy<readonly number[], number> {
  let inner: Env<unknown, unknown, unknown> = env
  while (inner.model === undefined && inner.wrapped !== undefined) inner = inner.wrapped
  const solution = valueIteration(mdpFromEnv(inner, gamma))
  const nA = solution.Q[0]?.length ?? 1
  return (o, k) => {
    if ((uniform(foldIn(k, 0), []).data[0] ?? 1) < epsilon)
      return Math.min(nA - 1, Math.floor((uniform(foldIn(k, 1), []).data[0] ?? 0) * nA))
    return solution.policy[argmaxOf(o)] ?? 0
  }
}
