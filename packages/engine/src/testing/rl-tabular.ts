// Helpers for tests of tabular, bandit and hierarchical agents. Node only; not published.
import { foldIn, key as makeKey, uniform } from '../lib/random/index.ts'
import type { Agent, Env, Outcome, Transition } from '../lib/rl/index.ts'
import { tabularEnv } from '../lib/rl/index.ts'
import { scene } from '../lib/scene/index.ts'

/** The Dyna maze of Sutton and Barto (Example 8.1, Figure 8.2): start S, goal G, walls #. */
export const DYNA_MAZE: readonly string[] = [
  '.......#G',
  '..#....#.',
  'S.#....#.',
  '..#......',
  '.....#...',
  '.........',
]

/**
 * A small random MDP as an environment: every action leads to two or three random successors, rewards are uniform in
 * [-1, 1], and with probability `pTerminal` an action ends the episode.
 */
export function randomMdpEnv(seed: number, nStates = 4, nActions = 2, pTerminal = 0.1): Env<number, number, number> {
  const k = makeKey(seed)
  const table: Outcome<number>[][] = []
  for (let s = 0; s < nStates; s++) {
    for (let a = 0; a < nActions; a++) {
      const u = uniform(foldIn(k, s * nActions + a), [6]).data
      const s1 = Math.floor((u[0] ?? 0) * nStates)
      const s2 = Math.floor((u[1] ?? 0) * nStates)
      const p = 0.2 + 0.6 * (u[2] ?? 0)
      const r = 2 * (u[3] ?? 0) - 1
      const keep = 1 - pTerminal
      table.push([
        { probability: keep * p, state: s1, reward: r, terminated: false },
        { probability: keep * (1 - p), state: s2, reward: r + 0.5, terminated: false },
        ...(pTerminal > 0 ? [{ probability: pTerminal, state: s, reward: 2 * (u[4] ?? 0) - 1, terminated: true }] : []),
      ])
    }
  }
  return tabularEnv({
    id: 'RandomMdp-v0',
    description: 'a random MDP',
    nStates,
    nActions,
    transitions: (s, a) => table[s * nActions + a] ?? [],
    initial: new Array<number>(nStates).fill(1 / nStates),
    render: () => scene(1, 1, []),
  })
}

/**
 * The random walk of Sutton and Barto (Example 6.2): states 1 to n between two terminal states 0 and n + 1, start in
 * the middle, step left or right with probability 1/2 whatever the action, reward 1 on reaching the right end. The
 * true values are `s / (n + 1)`.
 */
export function randomWalkEnv(n = 5): Env<number, number, number> {
  const initial = new Array<number>(n + 2).fill(0)
  initial[(n + 1) >> 1] = 1
  return tabularEnv({
    id: 'RandomWalk-v0',
    description: 'a random walk',
    nStates: n + 2,
    nActions: 1,
    transitions: (s) =>
      s === 0 || s === n + 1
        ? [{ probability: 1, state: s, reward: 0, terminated: true }]
        : [
            { probability: 0.5, state: s - 1, reward: 0, terminated: s - 1 === 0 },
            { probability: 0.5, state: s + 1, reward: s + 1 === n + 1 ? 1 : 0, terminated: s + 1 === n + 1 },
          ],
    initial,
    render: () => scene(1, 1, []),
  })
}

/** Feed a list of transitions through an agent's `learn` with keys `key(seed + i)`; the states after each one. */
export function replay<G, O, A>(agent: Agent<G, O, A>, transitions: readonly Transition<O, A>[], seed = 0): G[] {
  let s = agent.init(makeKey(seed))
  const out: G[] = []
  for (const [i, t] of transitions.entries()) {
    s = agent.learn(s, t, makeKey(seed + 1 + i))
    out.push(s)
  }
  return out
}

/** Transitions from fixture rows `[s, a, r, s2, terminated, truncated]`. */
export function transitionsOf(rows: readonly (readonly number[])[]): Transition<number, number>[] {
  return rows.map((r) => ({
    observation: r[0] ?? 0,
    action: r[1] ?? 0,
    reward: r[2] ?? 0,
    nextObservation: r[3] ?? 0,
    terminated: r[4] === 1,
    truncated: r[5] === 1,
  }))
}
