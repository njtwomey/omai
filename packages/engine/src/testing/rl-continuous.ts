// Tiny tasks for testing policy-gradient, actor-critic, safe and imitation agents. Node only; not published.

import { uniform } from '../lib/random/index.ts'
import { type Env, type Outcome, tabularEnv } from '../lib/rl/index.ts'
import { circle, scene } from '../lib/scene/index.ts'
import { box } from '../lib/space/index.ts'

const u1 = (k: Parameters<typeof uniform>[0]): number => uniform(k, []).data[0] ?? 0
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))

/**
 * A one-step contextual continuous bandit: the observation is a context `c ~ U(-1, 1)`, the action `a` in `[-1, 1]`,
 * and the reward `-(a - 0.8 c)^2`; every episode is one step. The best policy `a = 0.8 c` earns 0, `a = 0` earns about
 * -0.21 and uniformly random actions about -0.55.
 */
export function continuousBandit(): Env<number, readonly number[], readonly number[]> {
  return {
    spec: { id: 'ContinuousBandit-v0', description: 'match 0.8 times the context', maxEpisodeSteps: 1 },
    observationSpace: box(-1, 1, [1]),
    actionSpace: box(-1, 1, [1]),
    reset: (k) => {
      const c = 2 * u1(k) - 1
      return { state: c, observation: [c], info: {} }
    },
    step: (c, action) => {
      const a = clamp(action[0] ?? 0, -1, 1)
      return { state: c, observation: [c], reward: -((a - 0.8 * c) ** 2), terminated: true, truncated: false, info: {} }
    },
    render: (c) => scene(100, 20, [circle(50 + 45 * c, 10, 4)]),
    observe: (c) => [c],
  }
}

/**
 * A 1-D reach task: the position `x0 ~ U(-1, 1)` moves by `0.25 a` for an action `a` in `[-1, 1]` (clipped), within
 * `[-2, 2]`, and every step pays `-|x|` of the new position. Episodes never terminate; `make` truncates them after 10
 * steps (so learners must bootstrap). Heading straight for 0 earns about -0.7 on average; random actions about -5.
 */
export function reach(): Env<number, readonly number[], readonly number[]> {
  return {
    spec: { id: 'Reach1D-v0', description: 'move to the origin', maxEpisodeSteps: 10, deterministic: true },
    observationSpace: box(-2, 2, [1]),
    actionSpace: box(-1, 1, [1]),
    reset: (k) => {
      const x = 2 * u1(k) - 1
      return { state: x, observation: [x], info: {} }
    },
    step: (x, action) => {
      const next = clamp(x + 0.25 * clamp(action[0] ?? 0, -1, 1), -2, 2)
      return {
        state: next,
        observation: [next],
        reward: -Math.abs(next),
        terminated: false,
        truncated: false,
        info: {},
      }
    },
    render: (x) => scene(100, 20, [circle(50 + 22 * x, 10, 4)]),
    observe: (x) => [x],
  }
}

/** The default map of {@link hazardGrid}: `S` start, `H` hazard (passable, costly), `G` goal, `.` free. */
export const HAZARD_MAP: readonly string[] = ['SHHG', '....']

// Moves up, right, down, left, as in makeGridWorld.
const MOVES: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
]

/**
 * A deterministic grid world with hazard cells for safe-RL tests: the short path from `S` to `G` crosses the hazards
 * (`H`), a longer one goes around them. Each step pays -0.1, reaching the goal pays 1 and ends the episode; hazards do
 * not end it. Actions are up, right, down, left; episodes are truncated after 20 steps.
 */
export function hazardGrid(map: readonly string[] = HAZARD_MAP): Env<number, number, number> {
  const nrow = map.length
  const ncol = map[0]?.length ?? 0
  const at = (s: number): string => map[Math.floor(s / ncol)]?.[s % ncol] ?? '#'
  const transitions = (s: number, a: number): Outcome<number>[] => {
    if (at(s) === 'G') return [{ probability: 1, state: s, reward: 0, terminated: true }]
    const [dr, dc] = MOVES[a] ?? [0, 0]
    const r = Math.floor(s / ncol) + dr
    const c = (s % ncol) + dc
    const s2 = r < 0 || r >= nrow || c < 0 || c >= ncol ? s : r * ncol + c
    const goal = at(s2) === 'G'
    return [{ probability: 1, state: s2, reward: goal ? 1 : -0.1, terminated: goal }]
  }
  const cells = map.join('')
  return tabularEnv({
    id: 'HazardGrid-v0',
    description: 'reach the goal; the short path crosses hazards',
    nStates: nrow * ncol,
    nActions: 4,
    transitions,
    initial: cells.split('').map((ch) => (ch === 'S' ? 1 : 0)),
    render: (s) => scene(ncol * 10, nrow * 10, [circle((s % ncol) * 10 + 5, Math.floor(s / ncol) * 10 + 5, 3)]),
    maxEpisodeSteps: 20,
  })
}

/** The hazard cells of a {@link hazardGrid} map. */
export function hazardCells(map: readonly string[] = HAZARD_MAP): Set<number> {
  const out = new Set<number>()
  for (const [i, ch] of map.join('').split('').entries()) if (ch === 'H') out.add(i)
  return out
}
