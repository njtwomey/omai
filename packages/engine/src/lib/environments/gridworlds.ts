/**
 * Grid-world environments, small tabular MDPs with known models on which planners and tabular learners can be compared:
 * gymnasium's FrozenLake (a slippery lake with holes, on 4 x 4, 8 x 8 or random maps) and CliffWalking (with its
 * slippery variant), Russell and Norvig's 4 x 3 grid world, Sutton and Barto's windy grid world, a maze with a key,
 * locked doors and traps, and the four-rooms domain used to study options.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { type Env, type EnvDefinition, type Outcome, tabularEnv } from '../rl/index.ts'
import { type Scene, type SceneNode, scene, text } from '../scene/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import {
  aimaGridWorld,
  cliffWalkingWorld,
  type FrozenLakeOptions,
  fourRoomsWorld,
  fromAscii,
  frozenLakeWorld,
  KEY_MAZE,
  reachable,
  terrainOf,
  toScene,
  transitionModel,
  WINDY_STRENGTHS,
  type WorldTask,
  windyGridWorld as windyWorld,
} from '../worlds/index.ts'

export type { FrozenLakeOptions } from '../worlds/index.ts'

/** How a grid environment is described and drawn: its spec fields and the picture's title, scale and extras. */
interface GridEnvSpec {
  readonly id: string
  readonly description: string
  readonly title: string
  /** Pixels per cell (default 64). */
  readonly cell?: number
  readonly maxEpisodeSteps?: number
  readonly rewardThreshold?: number
  readonly stateLabel: string
  readonly actionLabel: string
  /** Extra nodes drawn over the world. */
  readonly overlay?: readonly SceneNode[]
}

/**
 * A tabular environment from a world task: its transition model (`lib/worlds`, outcome for outcome gymnasium's `P`
 * table) as the model, drawn by `toScene` with the agent in its cell.
 */
function worldEnv(t: WorldTask, spec: GridEnvSpec): Env<number, number, number> {
  const m = transitionModel(t.world, t.rules, t.options)
  const cell = spec.cell ?? 64
  const render = (s: number): Scene => {
    const pic = toScene(t.world, { agent: m.state(s).cell, cell })
    return scene(pic.width, pic.height, [...pic.children, ...(spec.overlay ?? [])], { title: spec.title })
  }
  return tabularEnv({
    id: spec.id,
    description: spec.description,
    nStates: m.nStates,
    nActions: m.nActions,
    transitions: (s, a): Outcome<number>[] =>
      m.outcomes(s, a).map((o) => ({
        probability: o.probability,
        state: o.state,
        reward: o.reward,
        terminated: o.terminated,
      })),
    initial: Array.from(m.initial),
    render,
    ...(spec.maxEpisodeSteps === undefined ? {} : { maxEpisodeSteps: spec.maxEpisodeSteps }),
    ...(spec.rewardThreshold === undefined ? {} : { rewardThreshold: spec.rewardThreshold }),
    stateLabel: spec.stateLabel,
    actionLabel: spec.actionLabel,
  })
}

// ------------------------------------------------------------------------------------------------------- FrozenLake

/**
 * A random valid FrozenLake map (gymnasium's `generate_random_map`): each tile is frozen with probability `p`, the start
 * is the top-left and the goal the bottom-right corner, and maps are redrawn until a path joins them.
 *
 * @param key - The key.
 * @param size - The side length (default 8).
 * @param p - The probability that a tile is frozen (default 0.8).
 * @returns The map rows.
 * @example
 * ```ts
 * import { randomFrozenLakeMap } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * randomFrozenLakeMap(key(0), 5)[4]?.[4] // => 'G'
 * ```
 */
export function randomFrozenLakeMap(key: Key, size = 8, p = 0.8): string[] {
  for (let attempt = 0; ; attempt++) {
    const u = uniform(foldIn(key, attempt), [size * size]).data
    const board: string[] = []
    for (let r = 0; r < size; r++) {
      let row = ''
      for (let c = 0; c < size; c++) {
        const ch =
          r === 0 && c === 0
            ? 'S'
            : r === size - 1 && c === size - 1
              ? 'G'
              : (u[r * size + c] ?? 0) < Math.min(1, p)
                ? 'F'
                : 'H'
        row += ch
      }
      board.push(row)
    }
    const w = frozenLakeWorld({ desc: board }).world
    const open = reachable(w, w.starts, { blocked: (c) => terrainOf(w, c).hazard === true })
    if (w.goals.some((g) => open[g] === 1)) return board
  }
}

/**
 * Build a FrozenLake environment exactly as gymnasium's FrozenLakeEnv: on a slippery lake the intended move succeeds
 * with probability `successRate` (default 1/3) and slips to either perpendicular direction with the rest; holes and
 * the goal are terminal. Actions are left, down, right, up.
 *
 * @param options - Map, slipperiness, success rate and reward schedule.
 * @param id - The environment id.
 * @returns The environment, with its `P` table as the model.
 * @example
 * ```ts
 * import { makeFrozenLake } from 'omai-engine/environments'
 * makeFrozenLake({ mapName: '8x8' }).observationSpace // => { kind: 'discrete', n: 64, label: 'cell (row * ncol + col)' }
 * ```
 */
export function makeFrozenLake(options: FrozenLakeOptions = {}, id = 'FrozenLake-v1'): Env<number, number, number> {
  return worldEnv(frozenLakeWorld(options), {
    id,
    description: 'Cross a frozen lake from start to goal without falling into a hole; the ice may be slippery.',
    title: 'FrozenLake',
    // gymnasium registers the limits and thresholds per id, whatever the map.
    maxEpisodeSteps: id === 'FrozenLake8x8-v1' ? 200 : 100,
    rewardThreshold: id === 'FrozenLake8x8-v1' ? 0.85 : 0.7,
    stateLabel: 'cell (row * ncol + col)',
    actionLabel: 'left (0), down (1), right (2), up (3)',
  })
}

const frozenOptions = dict({
  mapName: choice(['4x4', '8x8'], { default: '4x4' }),
  isSlippery: bool({ default: true }),
  successRate: real(0, 1, { default: 1 / 3, description: 'probability that the intended move succeeds' }),
})

/**
 * FrozenLake-v1 (gymnasium): the 4 x 4 slippery lake (options select the 8 x 8 map, a custom `desc`, slipperiness, the
 * success rate and the reward schedule). `make` truncates at 100 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { frozenLake } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * frozenLake.create().reset(key(0)).observation // => 0
 * ```
 */
export const frozenLake: EnvDefinition<number, number, number> = {
  id: 'FrozenLake-v1',
  description: 'Cross a frozen lake from start to goal without falling into a hole.',
  options: frozenOptions,
  create: (options) => makeFrozenLake((options ?? {}) as FrozenLakeOptions),
}

/**
 * FrozenLake8x8-v1 (gymnasium): the 8 x 8 slippery lake; `make` truncates at 200 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { frozenLake8x8 } from 'omai-engine/environments'
 * frozenLake8x8.create().spec.maxEpisodeSteps // => 200
 * ```
 */
export const frozenLake8x8: EnvDefinition<number, number, number> = {
  id: 'FrozenLake8x8-v1',
  description: 'Cross an 8 x 8 frozen lake from start to goal without falling into a hole.',
  options: frozenOptions,
  create: (options) =>
    makeFrozenLake({ mapName: '8x8', ...((options ?? {}) as FrozenLakeOptions) }, 'FrozenLake8x8-v1'),
}

// ------------------------------------------------------------------------------------------------------ CliffWalking

/**
 * Build CliffWalking exactly as gymnasium's CliffWalkingEnv (Sutton and Barto, Example 6.6): a 4 x 12 grid, start at
 * the bottom left, goal at the bottom right, and a cliff between them that costs -100 and sends the agent back to
 * the start. Every other step costs -1. With `isSlippery` the move goes in the intended or either perpendicular
 * direction with probability 1/3 each. Actions are up, right, down, left.
 *
 * @param options - Slipperiness.
 * @param id - The environment id.
 * @returns The environment, with its `P` table as the model.
 * @example
 * ```ts
 * import { makeCliffWalking } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * makeCliffWalking().step(36, 1, key(0)).reward // => -100
 * ```
 */
export function makeCliffWalking(
  options: { readonly isSlippery?: boolean } = {},
  id = 'CliffWalking-v1',
): Env<number, number, number> {
  return worldEnv(cliffWalkingWorld(options), {
    id,
    description: 'Walk along a cliff edge from start to goal; stepping off costs -100 and restarts.',
    title: 'CliffWalking',
    stateLabel: 'cell (row * 12 + col)',
    actionLabel: 'up (0), right (1), down (2), left (3)',
  })
}

/**
 * CliffWalking-v1 (gymnasium).
 *
 * @concept environments
 * @example
 * ```ts
 * import { cliffWalking } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * cliffWalking.create().reset(key(0)).observation // => 36
 * ```
 */
export const cliffWalking: EnvDefinition<number, number, number> = {
  id: 'CliffWalking-v1',
  description: 'Walk along a cliff edge from start to goal; stepping off costs -100 and restarts.',
  options: dict({ isSlippery: bool({ default: false }) }),
  create: (options) => makeCliffWalking({ isSlippery: options?.['isSlippery'] === true }),
}

/**
 * CliffWalkingSlippery-v1 (gymnasium): cliff walking where every move may slip sideways.
 *
 * @concept environments
 * @example
 * ```ts
 * import { cliffWalkingSlippery } from 'omai-engine/environments'
 * cliffWalkingSlippery.create().model?.transitions(0, 0).length // => 3
 * ```
 */
export const cliffWalkingSlippery: EnvDefinition<number, number, number> = {
  id: 'CliffWalkingSlippery-v1',
  description: 'Cliff walking where every move may slip sideways.',
  create: () => makeCliffWalking({ isSlippery: true }, 'CliffWalkingSlippery-v1'),
}

// --------------------------------------------------------------------------------------------------------- GridWorld

/** Options of {@link makeGridWorld}. */
export interface GridWorldOptions {
  /** The map: `.` free, `#` wall, `S` start (several allowed), `G` goal, `X` pit (default worlds' `AIMA_GRID`). */
  readonly map?: readonly string[]
  /** Reward for every step that does not end the episode (default -0.04). */
  readonly stepReward?: number
  readonly goalReward?: number
  readonly pitReward?: number
  /** Probability of moving to each side of the intended direction (default 0.1, AIMA's noise). */
  readonly slip?: number
}

/**
 * A configurable grid world (Russell and Norvig's 4 x 3 world by default): moves up, right, down, left succeed with
 * probability `1 - 2 slip` and go sideways otherwise; walls and edges block; goals and pits end the episode with
 * their rewards; every other step pays `stepReward`.
 *
 * @param options - The map, rewards and slip probability.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeGridWorld } from 'omai-engine/environments'
 * makeGridWorld().observationSpace // => { kind: 'discrete', n: 12, label: 'cell (row * ncol + col)' }
 * ```
 */
export function makeGridWorld(options: GridWorldOptions = {}): Env<number, number, number> {
  return worldEnv(aimaGridWorld(options), {
    id: 'GridWorld-v0',
    description: 'Reach the goal and avoid the pit in a grid world with noisy moves.',
    title: 'GridWorld',
    maxEpisodeSteps: 100,
    stateLabel: 'cell (row * ncol + col)',
    actionLabel: 'up (0), right (1), down (2), left (3)',
  })
}

/**
 * GridWorld-v0: Russell and Norvig's 4 x 3 grid world (options set the map, rewards and slip).
 *
 * @concept environments
 * @example
 * ```ts
 * import { gridWorld } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * gridWorld.create().reset(key(0)).observation // => 8
 * ```
 */
export const gridWorld: EnvDefinition<number, number, number> = {
  id: 'GridWorld-v0',
  description: 'Reach the goal and avoid the pit in a grid world with noisy moves.',
  options: dict({
    stepReward: real(-1, 0, { default: -0.04 }),
    slip: real(0, 0.5, { default: 0.1, description: 'probability of slipping to each side' }),
  }),
  create: (options) => makeGridWorld((options ?? {}) as GridWorldOptions),
}

// --------------------------------------------------------------------------------------------------- WindyGridWorld

/** Options of {@link makeWindyGridWorld}. */
export interface WindyGridWorldOptions {
  /** Eight king's moves instead of four (Exercise 6.9). */
  readonly kingsMoves?: boolean
  /** With king's moves, also allow staying put (a ninth action). */
  readonly stay?: boolean
  /** The wind varies by one up or down with probability 1/3 each (Exercise 6.10). */
  readonly stochasticWind?: boolean
}

/**
 * The windy grid world of Sutton and Barto (Example 6.5): a 7 x 10 grid whose columns push the agent upward by the
 * wind strengths 0 0 0 1 1 1 2 2 1 0; reward -1 per step until the goal. Options add king's moves, a stay action and
 * stochastic wind.
 *
 * @param options - Moves and wind.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeWindyGridWorld } from 'omai-engine/environments'
 * makeWindyGridWorld({ kingsMoves: true }).actionSpace // => { kind: 'discrete', n: 8, label: 'up, right, down, left, then diagonals' }
 * ```
 */
export function makeWindyGridWorld(options: WindyGridWorldOptions = {}): Env<number, number, number> {
  const nA = options.kingsMoves === true ? (options.stay === true ? 9 : 8) : 4
  return worldEnv(windyWorld(options), {
    id: 'WindyGridWorld-v0',
    description: 'Reach the goal across columns of upward wind.',
    title: 'WindyGridWorld',
    overlay: WINDY_STRENGTHS.map((w, c) =>
      text(c * 64 + 32, 7 * 64 - 8, String(w), { anchor: 'middle', size: 14, fill: 'muted' }),
    ),
    maxEpisodeSteps: 1000,
    stateLabel: 'cell (row * 10 + col)',
    actionLabel: nA === 4 ? 'up (0), right (1), down (2), left (3)' : 'up, right, down, left, then diagonals',
  })
}

/**
 * WindyGridWorld-v0 (Sutton and Barto, Example 6.5; options for Exercises 6.9 and 6.10).
 *
 * @concept environments
 * @example
 * ```ts
 * import { windyGridWorld } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * windyGridWorld.create().step(33, 1, key(0)).observation // => 24
 * ```
 */
export const windyGridWorld: EnvDefinition<number, number, number> = {
  id: 'WindyGridWorld-v0',
  description: 'Reach the goal across columns of upward wind.',
  options: dict({
    kingsMoves: bool({ default: false }),
    stay: bool({ default: false }),
    stochasticWind: bool({ default: false }),
  }),
  create: (options) => makeWindyGridWorld((options ?? {}) as WindyGridWorldOptions),
}

// ----------------------------------------------------------------------------------------------- Mazes with keys

/** Moves in the order up, right, down, left. */
const URDL: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
]

/**
 * A maze with a key, locked doors and traps: the agent must fetch the key before it can pass a door; traps end the
 * episode with `trapReward`, the goal with `goalReward`, and every step costs `stepReward`. The state is the cell plus
 * whether a key is held (`cell + nCells * hasKey`; any `K` gives the key, so every map has `2 nCells` states, where
 * worlds' `keyMazeWorld` tracks each key present separately).
 *
 * @param options - The map and rewards.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeKeyMaze } from 'omai-engine/environments'
 * makeKeyMaze().observationSpace // => { kind: 'discrete', n: 108, label: 'cell + cells * hasKey' }
 * ```
 */
export function makeKeyMaze(
  options: {
    readonly map?: readonly string[]
    readonly stepReward?: number
    readonly goalReward?: number
    readonly trapReward?: number
  } = {},
): Env<number, number, number> {
  const map = options.map ?? KEY_MAZE
  const nrow = map.length
  const ncol = map[0]?.length ?? 0
  const cells = nrow * ncol
  const step = options.stepReward ?? -0.01
  const goal = options.goalReward ?? 1
  const trap = options.trapReward ?? -1
  const at = (cell: number): string => map[Math.floor(cell / ncol)]?.[cell % ncol] ?? '#'
  const transitions = (s: number, a: number): Outcome<number>[] => {
    const cell = s % cells
    const key = s >= cells
    const here = at(cell)
    if (here === 'G' || here === 'T' || here === '#') return [{ probability: 1, state: s, reward: 0, terminated: true }]
    const [dr, dc] = URDL[a] ?? [0, 0]
    const r = Math.floor(cell / ncol) + dr
    const c = (cell % ncol) + dc
    let next = r * ncol + c
    const ch = r < 0 || r >= nrow || c < 0 || c >= ncol ? '#' : at(next)
    if (ch === '#' || (ch === 'D' && !key)) next = cell
    const nowKey = key || at(next) === 'K'
    const s2 = next + (nowKey ? cells : 0)
    const landed = at(next)
    if (landed === 'G') return [{ probability: 1, state: s2, reward: goal, terminated: true }]
    if (landed === 'T') return [{ probability: 1, state: s2, reward: trap, terminated: true }]
    return [{ probability: 1, state: s2, reward: step, terminated: false }]
  }
  const initial = new Array<number>(2 * cells).fill(0)
  const starts = map
    .join('')
    .split('')
    .flatMap((ch, i) => (ch === 'S' ? [i] : []))
  if (starts.length === 0) throw new OmaiError('value', 'keyMaze: the map has no start cell S')
  for (const s of starts) initial[s] = 1 / starts.length
  return tabularEnv({
    id: 'KeyMaze-v0',
    description: 'Fetch the key, open the door and reach the goal while avoiding traps.',
    nStates: 2 * cells,
    nActions: 4,
    transitions,
    initial,
    render: (s) => {
      const cell = s % cells
      const held = s >= cells
      const rows = held ? map.map((row) => row.replaceAll('K', '.')) : map
      const pic = toScene(fromAscii(rows), { agent: cell, cell: 64 })
      return scene(pic.width, pic.height, pic.children, { title: held ? 'KeyMaze (key held)' : 'KeyMaze' })
    },
    maxEpisodeSteps: 200,
    stateLabel: 'cell + cells * hasKey',
    actionLabel: 'up (0), right (1), down (2), left (3)',
  })
}

/**
 * KeyMaze-v0: a small maze with a key, a locked door and a trap.
 *
 * @concept environments
 * @example
 * ```ts
 * import { keyMaze } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * keyMaze.create().reset(key(0)).observation // => 10
 * ```
 */
export const keyMaze: EnvDefinition<number, number, number> = {
  id: 'KeyMaze-v0',
  description: 'Fetch the key, open the door and reach the goal while avoiding traps.',
  options: dict({ stepReward: real(-1, 0, { default: -0.01 }) }),
  create: (options) => makeKeyMaze((options ?? {}) as { readonly stepReward?: number }),
}

// --------------------------------------------------------------------------------------------------------- FourRooms

/**
 * The four-rooms domain of Sutton, Precup and Singh (1999): four rooms joined by hallways; each move succeeds with
 * probability 2/3 and otherwise goes in one of the other three directions (1/9 each); reaching the goal (by default
 * the east hallway) pays 1 and ends the episode. Episodes start in a uniformly random free cell.
 *
 * @param options - The goal cell (row and column) and the success probability.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeFourRooms } from 'omai-engine/environments'
 * makeFourRooms().observationSpace.kind // => 'discrete'
 * ```
 */
export function makeFourRooms(
  options: { readonly goal?: readonly [number, number]; readonly success?: number } = {},
): Env<number, number, number> {
  return worldEnv(fourRoomsWorld(options), {
    id: 'FourRooms-v0',
    description: 'Navigate between four rooms joined by hallways to reach the goal.',
    title: 'FourRooms',
    cell: 32,
    maxEpisodeSteps: 500,
    stateLabel: 'free cell index',
    actionLabel: 'up (0), right (1), down (2), left (3)',
  })
}

/**
 * FourRooms-v0 (Sutton, Precup and Singh 1999).
 *
 * @concept environments
 * @example
 * ```ts
 * import { fourRooms } from 'omai-engine/environments'
 * fourRooms.create().observationSpace // => { kind: 'discrete', n: 104, label: 'free cell index' }
 * ```
 */
export const fourRooms: EnvDefinition<number, number, number> = {
  id: 'FourRooms-v0',
  description: 'Navigate between four rooms joined by hallways to reach the goal.',
  options: dict({
    goalRow: int(1, 11, { default: 7 }),
    goalColumn: int(1, 11, { default: 9 }),
  }),
  create: (options) => {
    const r = options?.['goalRow']
    const c = options?.['goalColumn']
    return makeFourRooms(typeof r === 'number' && typeof c === 'number' ? { goal: [r, c] } : {})
  },
}
