import { WhiteboxError } from '../../../lib/base/index.ts'
import { foldIn, type Key, uniform } from '../../../lib/random/index.ts'
import { type Env, type EnvDefinition, type Outcome, tabularEnv } from '../../../lib/rl/index.ts'
import { circle, type Paint, rect, type Scene, type SceneNode, scene, text } from '../../../lib/scene/index.ts'
import { bool, choice, dict, int, real } from '../../../lib/space/index.ts'

/** Moves on a grid: row and column offsets of left, down, right, up (FrozenLake's action order). */
const LDRU: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
]

/** Moves in the order up, right, down, left (CliffWalking's and the grid world's action order). */
const URDL: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
]

const CELL_COLOURS: Readonly<Record<string, Paint>> = {
  S: 'surface',
  F: 'surface',
  '.': 'surface',
  ' ': 'surface',
  H: '#3a3a8c',
  X: 'tab3',
  C: 'tab3',
  T: 'tab3',
  G: 'tab2',
  '#': 'fg',
  w: 'fg',
  K: '#cccc00',
  D: 'tab5',
}

/**
 * Draw a grid of cells from rows of characters, with an agent marker: the shared picture of every grid world (cells
 * coloured by kind, start, goal, holes, walls, keys and doors labelled).
 *
 * @param rows - The map, one string per row.
 * @param agent - The agent's row and column.
 * @param options - Cell size, title and extra nodes drawn on top.
 * @returns The scene.
 * @example
 * ```ts
 * import { renderGrid } from 'whitebox-engine/methods/rl'
 * renderGrid(['SF', 'FG'], [0, 0]).width // => 128
 * ```
 */
export function renderGrid(
  rows: readonly string[],
  agent: readonly [number, number] | undefined,
  options: { readonly cell?: number; readonly title?: string; readonly overlay?: readonly SceneNode[] } = {},
): Scene {
  const cell = options.cell ?? 64
  const nrow = rows.length
  const ncol = rows[0]?.length ?? 0
  const nodes: SceneNode[] = []
  for (const [r, row] of rows.entries()) {
    for (let c = 0; c < row.length; c++) {
      const ch = row[c] ?? ' '
      nodes.push(
        rect(c * cell, r * cell, cell, cell, { fill: CELL_COLOURS[ch] ?? 'surface', stroke: 'grid', strokeWidth: 1 }),
      )
      if ('SGKDHXT'.includes(ch) && ch !== ' ') {
        nodes.push(
          text(c * cell + cell / 2, r * cell + cell / 2, ch, {
            anchor: 'middle',
            baseline: 'middle',
            size: cell / 3,
            fill: ch === 'H' ? 'bg' : 'fg',
          }),
        )
      }
    }
  }
  if (agent !== undefined) {
    nodes.push(
      circle(agent[1] * cell + cell / 2, agent[0] * cell + cell / 2, cell / 3, {
        fill: 'tab1',
        id: 'agent',
        title: `agent at row ${agent[0]}, column ${agent[1]}`,
      }),
    )
  }
  nodes.push(...(options.overlay ?? []))
  return scene(ncol * cell, nrow * cell, nodes, options.title === undefined ? {} : { title: options.title })
}

// ------------------------------------------------------------------------------------------------------- FrozenLake

/** gymnasium's FrozenLake maps. */
export const FROZEN_LAKE_MAPS: Readonly<Record<'4x4' | '8x8', readonly string[]>> = {
  '4x4': ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  '8x8': ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'],
}

function pathExists(board: readonly string[]): boolean {
  const n = board.length
  const frontier: [number, number][] = [[0, 0]]
  const seen = new Set<number>()
  while (frontier.length > 0) {
    const [r, c] = frontier.pop() as [number, number]
    if (seen.has(r * n + c)) continue
    seen.add(r * n + c)
    for (const [dr, dc] of [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ] as const) {
      const r2 = r + dr
      const c2 = c + dc
      if (r2 < 0 || r2 >= n || c2 < 0 || c2 >= n) continue
      const ch = board[r2]?.[c2]
      if (ch === 'G') return true
      if (ch !== 'H') frontier.push([r2, c2])
    }
  }
  return false
}

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
 * import { randomFrozenLakeMap } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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
    if (pathExists(board)) return board
  }
}

/** Options of FrozenLake. */
export interface FrozenLakeOptions {
  readonly mapName?: '4x4' | '8x8'
  readonly desc?: readonly string[]
  readonly isSlippery?: boolean
  readonly successRate?: number
  /** Rewards for reaching the goal, a hole and a frozen tile (default `[1, 0, 0]`). */
  readonly rewardSchedule?: readonly [number, number, number]
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
 * import { makeFrozenLake } from 'whitebox-engine/methods/rl'
 * makeFrozenLake({ mapName: '8x8' }).observationSpace // => { kind: 'discrete', n: 64, label: 'cell (row * ncol + col)' }
 * ```
 */
export function makeFrozenLake(options: FrozenLakeOptions = {}, id = 'FrozenLake-v1'): Env<number, number, number> {
  const desc = options.desc ?? FROZEN_LAKE_MAPS[options.mapName ?? '4x4']
  const nrow = desc.length
  const ncol = desc[0]?.length ?? 0
  const slippery = options.isSlippery ?? true
  const success = options.successRate ?? 1 / 3
  const fail = (1 - success) / 2
  const schedule = options.rewardSchedule ?? [1, 0, 0]
  const letter = (r: number, c: number): string => desc[r]?.[c] ?? 'F'
  const move = (r: number, c: number, a: number): Outcome<number> => {
    const [dr, dc] = LDRU[a] ?? [0, 0]
    const r2 = Math.min(Math.max(r + dr, 0), nrow - 1)
    const c2 = Math.min(Math.max(c + dc, 0), ncol - 1)
    const l = letter(r2, c2)
    const reward = schedule['GHF'.indexOf('GHF'.includes(l) ? l : 'F')] ?? 0
    return { probability: 1, state: r2 * ncol + c2, reward, terminated: l === 'G' || l === 'H' }
  }
  const table: Outcome<number>[][][] = []
  for (let s = 0; s < nrow * ncol; s++) {
    const r = Math.floor(s / ncol)
    const c = s % ncol
    const row: Outcome<number>[][] = []
    for (let a = 0; a < 4; a++) {
      const l = letter(r, c)
      if (l === 'G' || l === 'H') row.push([{ probability: 1, state: s, reward: 0, terminated: true }])
      else if (slippery) {
        row.push(
          [(a + 3) % 4, a, (a + 1) % 4].map((b) => ({ ...move(r, c, b), probability: b === a ? success : fail })),
        )
      } else row.push([move(r, c, a)])
    }
    table.push(row)
  }
  const starts = desc
    .join('')
    .split('')
    .map((ch): number => (ch === 'S' ? 1 : 0))
  const nStarts = starts.reduce((a, b) => a + b, 0)
  if (nStarts === 0) throw new WhiteboxError('value', 'FrozenLake: the map has no start tile S')
  return tabularEnv({
    id,
    description: 'Cross a frozen lake from start to goal without falling into a hole; the ice may be slippery.',
    nStates: nrow * ncol,
    nActions: 4,
    transitions: (s, a) => table[s]?.[a] ?? [],
    initial: starts.map((x) => x / nStarts),
    render: (s) => renderGrid(desc, [Math.floor(s / ncol), s % ncol], { title: 'FrozenLake' }),
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
 * import { frozenLake } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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
 * import { frozenLake8x8 } from 'whitebox-engine/methods/rl'
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

const CLIFF_ROWS = ['............', '............', '............', 'SCCCCCCCCCCG']

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
 * import { makeCliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makeCliffWalking().step(36, 1, key(0)).reward // => -100
 * ```
 */
export function makeCliffWalking(
  options: { readonly isSlippery?: boolean } = {},
  id = 'CliffWalking-v1',
): Env<number, number, number> {
  const slippery = options.isSlippery ?? false
  const nrow = 4
  const ncol = 12
  const start = 3 * ncol
  const outcomes = (s: number, a: number): Outcome<number>[] => {
    const r = Math.floor(s / ncol)
    const c = s % ncol
    const moves = slippery ? [(a + 3) % 4, a, (a + 1) % 4] : [a]
    return moves.map((m) => {
      const [dr, dc] = URDL[m] ?? [0, 0]
      const r2 = Math.max(Math.min(r + dr, nrow - 1), 0)
      const c2 = Math.max(Math.min(c + dc, ncol - 1), 0)
      const p = 1 / moves.length
      if (r2 === 3 && c2 >= 1 && c2 <= 10) return { probability: p, state: start, reward: -100, terminated: false }
      return { probability: p, state: r2 * ncol + c2, reward: -1, terminated: r2 === 3 && c2 === 11 }
    })
  }
  const initial = new Array<number>(nrow * ncol).fill(0)
  initial[start] = 1
  return tabularEnv({
    id,
    description: 'Walk along a cliff edge from start to goal; stepping off costs -100 and restarts.',
    nStates: nrow * ncol,
    nActions: 4,
    transitions: outcomes,
    initial,
    render: (s) => renderGrid(CLIFF_ROWS, [Math.floor(s / ncol), s % ncol], { title: 'CliffWalking' }),
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
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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
 * import { cliffWalkingSlippery } from 'whitebox-engine/methods/rl'
 * cliffWalkingSlippery.create().model?.transitions(0, 0).length // => 3
 * ```
 */
export const cliffWalkingSlippery: EnvDefinition<number, number, number> = {
  id: 'CliffWalkingSlippery-v1',
  description: 'Cliff walking where every move may slip sideways.',
  create: () => makeCliffWalking({ isSlippery: true }, 'CliffWalkingSlippery-v1'),
}

// --------------------------------------------------------------------------------------------------------- GridWorld

/** The 4 x 3 world of Russell and Norvig (AIMA, Figure 17.1): `#` wall, `G` +1, `X` -1, `S` start. */
export const AIMA_GRID: readonly string[] = ['...G', '.#.X', 'S...']

/** Options of {@link makeGridWorld}. */
export interface GridWorldOptions {
  /** The map: `.` free, `#` wall, `S` start (several allowed), `G` goal, `X` pit (default {@link AIMA_GRID}). */
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
 * import { makeGridWorld } from 'whitebox-engine/methods/rl'
 * makeGridWorld().observationSpace // => { kind: 'discrete', n: 12, label: 'cell (row * ncol + col)' }
 * ```
 */
export function makeGridWorld(options: GridWorldOptions = {}): Env<number, number, number> {
  const map = options.map ?? AIMA_GRID
  const nrow = map.length
  const ncol = map[0]?.length ?? 0
  const step = options.stepReward ?? -0.04
  const goal = options.goalReward ?? 1
  const pit = options.pitReward ?? -1
  const slip = options.slip ?? 0.1
  const at = (r: number, c: number): string => map[r]?.[c] ?? '#'
  const target = (s: number, m: number): number => {
    const [dr, dc] = URDL[m] ?? [0, 0]
    const r = Math.floor(s / ncol) + dr
    const c = (s % ncol) + dc
    return r < 0 || r >= nrow || c < 0 || c >= ncol || at(r, c) === '#' ? s : r * ncol + c
  }
  const outcome = (s2: number, p: number): Outcome<number> => {
    const ch = at(Math.floor(s2 / ncol), s2 % ncol)
    if (ch === 'G') return { probability: p, state: s2, reward: goal, terminated: true }
    if (ch === 'X') return { probability: p, state: s2, reward: pit, terminated: true }
    return { probability: p, state: s2, reward: step, terminated: false }
  }
  const transitions = (s: number, a: number): Outcome<number>[] => {
    const ch = at(Math.floor(s / ncol), s % ncol)
    if (ch === 'G' || ch === 'X' || ch === '#') return [{ probability: 1, state: s, reward: 0, terminated: true }]
    const parts: [number, number][] = [
      [a, 1 - 2 * slip],
      [(a + 1) % 4, slip],
      [(a + 3) % 4, slip],
    ]
    return parts.filter(([, p]) => p > 0).map(([m, p]) => outcome(target(s, m), p))
  }
  const starts = map
    .join('')
    .split('')
    .map((ch): number => (ch === 'S' ? 1 : 0))
  const n = starts.reduce((x, y) => x + y, 0)
  if (n === 0) throw new WhiteboxError('value', 'gridWorld: the map has no start cell S')
  return tabularEnv({
    id: 'GridWorld-v0',
    description: 'Reach the goal and avoid the pit in a grid world with noisy moves.',
    nStates: nrow * ncol,
    nActions: 4,
    transitions,
    initial: starts.map((x) => x / n),
    render: (s) => renderGrid(map, [Math.floor(s / ncol), s % ncol], { title: 'GridWorld' }),
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
 * import { gridWorld } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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

const WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
const KING: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 1],
  [1, 1],
  [1, -1],
  [-1, -1],
  [0, 0],
]

/**
 * The windy grid world of Sutton and Barto (Example 6.5): a 7 x 10 grid whose columns push the agent upward by the
 * wind strengths 0 0 0 1 1 1 2 2 1 0; reward -1 per step until the goal. Options add king's moves, a stay action and
 * stochastic wind.
 *
 * @param options - Moves and wind.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeWindyGridWorld } from 'whitebox-engine/methods/rl'
 * makeWindyGridWorld({ kingsMoves: true }).actionSpace // => { kind: 'discrete', n: 8, label: 'up, right, down, left, then diagonals' }
 * ```
 */
export function makeWindyGridWorld(options: WindyGridWorldOptions = {}): Env<number, number, number> {
  const nrow = 7
  const ncol = 10
  const nA = options.kingsMoves === true ? (options.stay === true ? 9 : 8) : 4
  const start = 3 * ncol
  const goal = 3 * ncol + 7
  const transitions = (s: number, a: number): Outcome<number>[] => {
    if (s === goal) return [{ probability: 1, state: s, reward: 0, terminated: true }]
    const r = Math.floor(s / ncol)
    const c = s % ncol
    const [dr, dc] = KING[a] ?? [0, 0]
    const w = WIND[c] ?? 0
    const gusts = options.stochasticWind === true && w > 0 ? [-1, 0, 1] : [0]
    const out = new Map<number, number>()
    for (const g of gusts) {
      const r2 = Math.min(Math.max(r + dr - (w + g), 0), nrow - 1)
      const c2 = Math.min(Math.max(c + dc, 0), ncol - 1)
      const s2 = r2 * ncol + c2
      out.set(s2, (out.get(s2) ?? 0) + 1 / gusts.length)
    }
    return [...out.entries()].map(([s2, p]) => ({ probability: p, state: s2, reward: -1, terminated: s2 === goal }))
  }
  const initial = new Array<number>(nrow * ncol).fill(0)
  initial[start] = 1
  const rows = Array.from({ length: nrow }, (_, r) =>
    Array.from({ length: ncol }, (_, c) => (r * ncol + c === start ? 'S' : r * ncol + c === goal ? 'G' : '.')).join(''),
  )
  return tabularEnv({
    id: 'WindyGridWorld-v0',
    description: 'Reach the goal across columns of upward wind.',
    nStates: nrow * ncol,
    nActions: nA,
    transitions,
    initial,
    render: (s) =>
      renderGrid(rows, [Math.floor(s / ncol), s % ncol], {
        title: 'WindyGridWorld',
        overlay: WIND.map((w, c) =>
          text(c * 64 + 32, 7 * 64 - 8, String(w), { anchor: 'middle', size: 14, fill: 'muted' }),
        ),
      }),
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
 * import { windyGridWorld } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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

/** The default maze of {@link makeKeyMaze}: `#` wall, `S` start, `K` key, `D` door (needs the key), `T` trap, `G` goal. */
export const KEY_MAZE: readonly string[] = [
  '#########',
  '#S..#..G#',
  '#.#.#.#.#',
  '#.#..D..#',
  '#K#T#####',
  '#########',
]

/**
 * A maze with a key, locked doors and traps: the agent must fetch the key before it can pass a door; traps end the
 * episode with `trapReward`, the goal with `goalReward`, and every step costs `stepReward`. The state is the cell plus
 * whether the key is held (`cell + nCells * hasKey`).
 *
 * @param options - The map and rewards.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeKeyMaze } from 'whitebox-engine/methods/rl'
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
  if (starts.length === 0) throw new WhiteboxError('value', 'keyMaze: the map has no start cell S')
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
      return renderGrid(rows, [Math.floor(cell / ncol), cell % ncol], {
        title: held ? 'KeyMaze (key held)' : 'KeyMaze',
      })
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
 * import { keyMaze } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
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

/** The four-rooms layout of Sutton, Precup and Singh (1999): `w` wall. */
export const FOUR_ROOMS: readonly string[] = [
  'wwwwwwwwwwwww',
  'w     w     w',
  'w     w     w',
  'w           w',
  'w     w     w',
  'w     w     w',
  'ww wwww     w',
  'w     www www',
  'w     w     w',
  'w     w     w',
  'w           w',
  'w     w     w',
  'wwwwwwwwwwwww',
]

/**
 * The four-rooms domain of Sutton, Precup and Singh (1999): four rooms joined by hallways; each move succeeds with
 * probability 2/3 and otherwise goes in one of the other three directions (1/9 each); reaching the goal (by default
 * the east hallway) pays 1 and ends the episode. Episodes start in a uniformly random free cell.
 *
 * @param options - The goal cell (row and column) and the success probability.
 * @returns The environment, with its model.
 * @example
 * ```ts
 * import { makeFourRooms } from 'whitebox-engine/methods/rl'
 * makeFourRooms().observationSpace.kind // => 'discrete'
 * ```
 */
export function makeFourRooms(
  options: { readonly goal?: readonly [number, number]; readonly success?: number } = {},
): Env<number, number, number> {
  const [gr, gc] = options.goal ?? [7, 9]
  const success = options.success ?? 2 / 3
  const nrow = FOUR_ROOMS.length
  const ncol = FOUR_ROOMS[0]?.length ?? 0
  const free = (r: number, c: number): boolean => r >= 0 && r < nrow && c >= 0 && c < ncol && FOUR_ROOMS[r]?.[c] !== 'w'
  const cells: [number, number][] = []
  for (let r = 0; r < nrow; r++) for (let c = 0; c < ncol; c++) if (free(r, c)) cells.push([r, c])
  const index = new Map(cells.map(([r, c], i) => [r * ncol + c, i]))
  const goal = index.get(gr * ncol + gc)
  if (goal === undefined) throw new WhiteboxError('value', 'fourRooms: the goal is not a free cell')
  const transitions = (s: number, a: number): Outcome<number>[] => {
    if (s === goal) return [{ probability: 1, state: s, reward: 0, terminated: true }]
    const [r, c] = cells[s] ?? [0, 0]
    const out = new Map<number, number>()
    for (let m = 0; m < 4; m++) {
      const p = m === a ? success : (1 - success) / 3
      const [dr, dc] = URDL[m] ?? [0, 0]
      const s2 = free(r + dr, c + dc) ? (index.get((r + dr) * ncol + c + dc) ?? s) : s
      out.set(s2, (out.get(s2) ?? 0) + p)
    }
    return [...out.entries()].map(([s2, p]) => ({
      probability: p,
      state: s2,
      reward: s2 === goal ? 1 : 0,
      terminated: s2 === goal,
    }))
  }
  const initial = cells.map((_, i) => (i === goal ? 0 : 1 / (cells.length - 1)))
  const rows = FOUR_ROOMS.map((row, r) =>
    row
      .split('')
      .map((ch, c) => (r === gr && c === gc ? 'G' : ch))
      .join(''),
  )
  return tabularEnv({
    id: 'FourRooms-v0',
    description: 'Navigate between four rooms joined by hallways to reach the goal.',
    nStates: cells.length,
    nActions: 4,
    transitions,
    initial,
    render: (s) => renderGrid(rows, cells[s] ?? [0, 0], { title: 'FourRooms', cell: 32 }),
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
 * import { fourRooms } from 'whitebox-engine/methods/rl'
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
