import { WhiteboxError } from '../base/index.ts'
import { uniformStream } from '../random/_unit.ts'
import { foldIn, type Key } from '../random/index.ts'
import { fromAscii, type Legend } from './ascii.ts'
import {
  ACTIONS,
  type ActionSet,
  lockedDoors,
  type ModelOptions,
  pickups,
  type Rule,
  rewardOn,
  rules,
  slip,
  slipUniform,
  stepReward,
  teleport,
  terminateOn,
  wind,
} from './rules.ts'
import { terrainOf, type World, world } from './world.ts'

/**
 * A world with the rules and model options that make it a task: pass `transitionModel(t.world, t.rules, t.options)`
 * (or `deterministicModel`) to get its dynamics.
 *
 * @concept grid-worlds
 */
export interface WorldTask {
  readonly world: World
  readonly rules: readonly Rule[]
  readonly options: ModelOptions
}

/** gymnasium's FrozenLake maps. */
export const FROZEN_LAKE_MAPS: Readonly<Record<'4x4' | '8x8', readonly string[]>> = {
  '4x4': ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  '8x8': ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'],
}

/** FrozenLake's legend: `S` start, `F` frozen, `H` hole, `G` goal. */
export const FROZEN_LAKE_LEGEND: Legend = {
  F: { terrain: 'ice' },
  S: { terrain: 'ice', start: true },
  G: { terrain: 'ice', goal: true },
}

const ICE = { name: 'ice', char: 'F', paint: 'tab9' as const }

/** Options of {@link frozenLakeWorld} (gymnasium's keyword arguments). */
export interface FrozenLakeOptions {
  readonly mapName?: '4x4' | '8x8'
  readonly desc?: readonly string[]
  readonly isSlippery?: boolean
  /** The probability that the intended move happens on a slippery lake (default 1/3). */
  readonly successRate?: number
  /** Rewards for reaching the goal, a hole and anything else (default `[1, 0, 0]`). */
  readonly rewardSchedule?: readonly [number, number, number]
}

/**
 * FrozenLake as gymnasium's FrozenLakeEnv: cross the ice from the start to the goal without falling into a hole. On
 * a slippery lake the move goes left of, along or right of the intended direction (gymnasium's order), the intended
 * one with probability `successRate`; holes and the goal end the episode and absorb. Actions are left, down, right,
 * up; the transition model reproduces gymnasium's `P` table outcome for outcome.
 *
 * @param options - Map, slipperiness, success rate and reward schedule.
 * @returns The task.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { frozenLakeWorld, toAscii, transitionModel } from 'whitebox-engine/worlds'
 * const t = frozenLakeWorld()
 * toAscii(t.world, { legend: { F: { terrain: 'ice' }, S: { terrain: 'ice', start: true }, G: { terrain: 'ice', goal: true } } }) // => ['SFFF', 'FHFH', 'FFFH', 'HFFG']
 * transitionModel(t.world, t.rules, t.options).outcomes(14, 2)[1] // => { probability: 1 / 3, state: 15, reward: 1, terminated: true, cost: 1 }
 * ```
 */
export function frozenLakeWorld(options: FrozenLakeOptions = {}): WorldTask {
  const desc = options.desc ?? FROZEN_LAKE_MAPS[options.mapName ?? '4x4']
  const w = fromAscii(desc, { legend: FROZEN_LAKE_LEGEND, terrains: [ICE], name: 'FrozenLake' })
  const success = options.successRate ?? 1 / 3
  const fail = (1 - success) / 2
  const [goal, hole, other] = options.rewardSchedule ?? [1, 0, 0]
  const plain = (c: number, x: World): boolean => !x.goals.includes(c) && !terrainOf(x, c).hazard
  return {
    world: w,
    rules: rules(
      options.isSlippery === false
        ? []
        : [
            slip([
              [-1, fail],
              [0, success],
              [1, fail],
            ]),
          ],
      rewardOn('goals', goal, { replace: true }),
      rewardOn('hazards', hole, { replace: true }),
      rewardOn(plain, other, { replace: true }),
      terminateOn('goals'),
      terminateOn('hazards'),
    ),
    options: { actions: ACTIONS.ldru },
  }
}

/** CliffWalking's map: a 4 x 12 grid with the cliff along the bottom between the start and the goal. */
export const CLIFF_WALKING_MAP: readonly string[] = ['............', '............', '............', 'SCCCCCCCCCCG']

/**
 * CliffWalking as gymnasium's CliffWalkingEnv (Sutton and Barto, Example 6.6): every move costs -1; stepping onto the
 * cliff costs -100 and sends the agent back to the start; reaching the goal ends the episode. With `isSlippery` a
 * move goes left of, along or right of the intended direction with probability 1/3 each. Actions are up, right,
 * down, left; no state absorbs, as in gymnasium's table.
 *
 * @param options - Slipperiness.
 * @returns The task.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { cliffWalkingWorld, transitionModel } from 'whitebox-engine/worlds'
 * const t = cliffWalkingWorld()
 * transitionModel(t.world, t.rules, t.options).outcomes(36, 1) // => [{ probability: 1, state: 36, reward: -100, terminated: false, cost: 1 }]
 * ```
 */
export function cliffWalkingWorld(options: { readonly isSlippery?: boolean } = {}): WorldTask {
  const w = fromAscii(CLIFF_WALKING_MAP, { name: 'CliffWalking' })
  const third = 1 / 3
  return {
    world: w,
    rules: rules(
      options.isSlippery === true
        ? [
            slip([
              [-1, third],
              [0, third],
              [1, third],
            ]),
          ]
        : [],
      stepReward(-1),
      rewardOn({ terrain: 'cliff' }, -100, { replace: true }),
      teleport({ terrain: 'cliff' }, 'start'),
      terminateOn('goals'),
    ),
    options: { actions: ACTIONS.urdl, absorbing: false },
  }
}

/** The 4 x 3 world of Russell and Norvig (AIMA, Figure 17.1): `#` wall, `G` +1, `X` the -1 pit, `S` start. */
export const AIMA_GRID: readonly string[] = ['...G', '.#.X', 'S...']

/**
 * The AIMA grid world (Russell and Norvig, Figure 17.1): moves succeed with probability `1 - 2 slip` and veer to
 * either side otherwise; walls and edges block; the goal pays `goalReward` and the pit `pitReward`, both ending the
 * episode; every other move pays `stepReward`. Actions are up, right, down, left.
 *
 * @param options - The map, rewards and slip probability (defaults: AIMA's map, -0.04, +1, -1, 0.1).
 * @returns The task.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { aimaGridWorld, transitionModel } from 'whitebox-engine/worlds'
 * const t = aimaGridWorld()
 * transitionModel(t.world, t.rules, t.options).outcomes(2, 1).map((o) => o.state) // => [3, 6, 2]
 * ```
 */
export function aimaGridWorld(
  options: {
    readonly map?: readonly string[]
    readonly stepReward?: number
    readonly goalReward?: number
    readonly pitReward?: number
    readonly slip?: number
  } = {},
): WorldTask {
  const s = options.slip ?? 0.1
  return {
    world: fromAscii(options.map ?? AIMA_GRID, { name: 'AIMA grid' }),
    rules: rules(
      slip([
        [0, 1 - 2 * s],
        [1, s],
        [-1, s],
      ]),
      stepReward(options.stepReward ?? -0.04),
      rewardOn('goals', options.goalReward ?? 1, { replace: true }),
      rewardOn('hazards', options.pitReward ?? -1, { replace: true }),
      terminateOn('goals'),
      terminateOn('hazards'),
    ),
    options: { actions: ACTIONS.urdl },
  }
}

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
 * probability `success` (2/3) and otherwise goes one of the other three ways (1/9 each); the goal (the east hallway by
 * default) pays 1 and ends the episode. States are the free cells; episodes start uniformly in any free cell but the
 * goal; outcomes reaching the same cell are merged.
 *
 * @param options - The goal `[row, col]` and the success probability.
 * @returns The task.
 * @throws WhiteboxError with code `value` when the goal is not a free cell.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { fourRoomsWorld, transitionModel } from 'whitebox-engine/worlds'
 * const t = fourRoomsWorld()
 * transitionModel(t.world, t.rules, t.options).nStates // => 104
 * ```
 */
export function fourRoomsWorld(
  options: { readonly goal?: readonly [number, number]; readonly success?: number } = {},
): WorldTask {
  const [gr, gc] = options.goal ?? [7, 9]
  const base = fromAscii(FOUR_ROOMS, { legend: { w: { terrain: 'wall' } }, name: 'FourRooms' })
  const goal = gr * 13 + gc
  if (!(gr >= 0 && gr < 13 && gc >= 0 && gc < 13) || FOUR_ROOMS[gr]?.[gc] === 'w') {
    throw new WhiteboxError('value', 'fourRoomsWorld: the goal is not a free cell')
  }
  return {
    world: { ...base, goals: [goal] },
    rules: rules(slipUniform(options.success ?? 2 / 3), rewardOn('goals', 1), terminateOn('goals')),
    options: { actions: ACTIONS.urdl, cells: 'passable', merge: true, initial: 'uniform' },
  }
}

/** Upward wind of each column of the windy grid world. */
export const WINDY_STRENGTHS: readonly number[] = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

/**
 * The windy grid world of Sutton and Barto (Example 6.5): a 7 x 10 grid whose columns push the agent up by their wind
 * strength, -1 per step until the goal. Options add king's moves (Exercise 6.9), a stay action and stochastic wind
 * (Exercise 6.10). Moves are clamped to the grid; outcomes reaching the same cell are merged.
 *
 * @param options - King's moves, stay and stochastic wind.
 * @returns The task.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { transitionModel, windyGridWorld } from 'whitebox-engine/worlds'
 * const t = windyGridWorld()
 * transitionModel(t.world, t.rules, t.options).outcomes(33, 1)[0]?.state // => 24
 * ```
 */
export function windyGridWorld(
  options: { readonly kingsMoves?: boolean; readonly stay?: boolean; readonly stochasticWind?: boolean } = {},
): WorldTask {
  const rows = Array.from({ length: 7 }, (_, r) => (r === 3 ? 'S......G..' : '..........'))
  const actions: ActionSet =
    options.kingsMoves === true ? (options.stay === true ? ACTIONS.kingStay : ACTIONS.king) : ACTIONS.urdl
  return {
    world: fromAscii(rows, { name: 'WindyGridWorld' }),
    rules: rules(
      wind(WINDY_STRENGTHS, { stochastic: options.stochasticWind === true }),
      stepReward(-1),
      terminateOn('goals'),
    ),
    options: { actions, edges: 'clamp', merge: true },
  }
}

/** The default key maze: `#` wall, `S` start, `K` key, `D` door (needs the key), `T` trap, `G` goal. */
export const KEY_MAZE: readonly string[] = [
  '#########',
  '#S..#..G#',
  '#.#.#.#.#',
  '#.#..D..#',
  '#K#T#####',
  '#########',
]

/**
 * A maze with a key, a locked door and a trap: the door opens only while the key is held (picked up by walking onto
 * it); the trap ends the episode with `trapReward`, the goal with `goalReward`; every other step pays `stepReward`.
 * States are `cell + cells * hasKey`.
 *
 * @param options - The map and rewards (defaults: {@link KEY_MAZE}, -0.01, +1, -1).
 * @returns The task.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { keyMazeWorld, transitionModel } from 'whitebox-engine/worlds'
 * const t = keyMazeWorld()
 * transitionModel(t.world, t.rules, t.options).nStates // => 108
 * ```
 */
export function keyMazeWorld(
  options: {
    readonly map?: readonly string[]
    readonly stepReward?: number
    readonly goalReward?: number
    readonly trapReward?: number
  } = {},
): WorldTask {
  return {
    world: fromAscii(options.map ?? KEY_MAZE, { name: 'KeyMaze' }),
    rules: rules(
      lockedDoors(),
      pickups(),
      stepReward(options.stepReward ?? -0.01),
      rewardOn('goals', options.goalReward ?? 1, { replace: true }),
      rewardOn('hazards', options.trapReward ?? -1, { replace: true }),
      terminateOn('goals'),
      terminateOn('hazards'),
    ),
    options: { actions: ACTIONS.urdl },
  }
}

/** gymnasium's Taxi map: `|` walls between cells, `:` open, the four stands R, G, Y and B. */
export const TAXI_MAP: readonly string[] = [
  '+---------+',
  '|R: | : :G|',
  '| : | : : |',
  '| : : : : |',
  '| | : | : |',
  '|Y| : |B: |',
  '+---------+',
]

/** Taxi's movement actions in gymnasium's order: south, north, east, west. */
export const TAXI_ACTIONS: ActionSet = {
  names: ['south', 'north', 'east', 'west'],
  actions: [
    { kind: 'move', direction: 2 },
    { kind: 'move', direction: 0 },
    { kind: 'move', direction: 1 },
    { kind: 'move', direction: 3 },
  ],
}

/**
 * Taxi's 5 x 5 map (gymnasium's Taxi): thin walls between cells and the four stands as landmarks `R`, `G`, `Y` and
 * `B`, with the taxi's four moves. Pickup, drop-off and the passenger belong to the environment; this is the world
 * they move through.
 *
 * @returns The task (the world, no rules, Taxi's movement actions).
 * @concept grid-worlds
 * @example
 * ```ts
 * import { destination, taxiWorld } from 'whitebox-engine/worlds'
 * const t = taxiWorld()
 * [t.world.landmarks, destination(t.world, 1, 1)] // => [{ R: [0], G: [4], Y: [20], B: [23] }, -1]
 * ```
 */
export function taxiWorld(): WorldTask {
  const drawing = TAXI_MAP.flatMap((row, i) => (i > 0 && i < TAXI_MAP.length - 2 ? [row, '+ + + + + +'] : [row]))
  const legend: Legend = Object.fromEntries(['R', 'G', 'Y', 'B'].map((k) => [k, { terrain: 'floor', landmark: k }]))
  return {
    world: fromAscii(drawing, { format: 'walls', legend, name: 'Taxi' }),
    rules: [],
    options: { actions: TAXI_ACTIONS },
  }
}

// ---------------------------------------------------------------------------------------------- MiniGrid layouts

/** The MiniGrid-style layouts. */
export type MiniGridLayout = 'empty' | 'doorKey' | 'fourRooms' | 'lavaGap' | 'crossing'

/** Options of {@link miniGridWorld}. */
export interface MiniGridOptions {
  /** The side, walls included (defaults: 8 empty and door-key, 19 four rooms, 7 lava gap, 9 crossing). */
  readonly size?: number
  /** River crossings: the obstacle (`'lava'` for LavaCrossing, `'wall'` for SimpleCrossing) and how many rivers. */
  readonly obstacle?: 'lava' | 'wall'
  readonly crossings?: number
}

/**
 * MiniGrid-style layouts as worlds (Chevalier-Boisvert et al.): a walled square with the agent starting facing east
 * and a goal; `empty` (start top-left, goal bottom-right), `doorKey` (a wall splits the room; a yellow key on the
 * agent's side opens a locked door in it), `fourRooms` (four rooms with one gap in each inner wall; random start and
 * goal), `lavaGap` (a lava column with one gap) and `crossing` (lava or wall rivers across the room, opened along a
 * monotone path). Layouts that MiniGrid randomises are drawn from the key; every one is solvable. The rules end
 * episodes at the goal (reward 1) and in lava, pick keys up on entry and open doors with them; actions turn left,
 * turn right and move forward. MiniGrid's step-discounted reward and explicit pickup and toggle actions belong to the
 * environment built on this world.
 *
 * @param layout - The layout.
 * @param key - The key for the randomised parts.
 * @param options - Size and river crossings.
 * @returns The task.
 * @throws WhiteboxError with code `value` for a size too small for the layout.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { miniGridWorld, toAscii } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * toAscii(miniGridWorld('empty', key(0), { size: 5 }).world) // => ['#####', '#S..#', '#...#', '#..G#', '#####']
 * ```
 */
export function miniGridWorld(layout: MiniGridLayout, key: Key, options: MiniGridOptions = {}): WorldTask {
  const size = options.size ?? (layout === 'fourRooms' ? 19 : layout === 'lavaGap' ? 7 : layout === 'crossing' ? 9 : 8)
  const minimum = layout === 'fourRooms' ? 7 : layout === 'crossing' ? 5 : layout === 'empty' ? 3 : 5
  if (!Number.isInteger(size) || size < minimum) {
    throw new WhiteboxError('value', `miniGridWorld: '${layout}' needs a size of at least ${minimum}`)
  }
  const next = uniformStream(key, foldIn)
  const int = (lo: number, hi: number): number => lo + Math.min(hi - lo - 1, Math.floor(next() * (hi - lo)))
  const cell = (r: number, c: number): number => r * size + c
  const terrain = new Array<string>(size * size).fill('floor')
  for (let i = 0; i < size; i++) {
    for (const c of [cell(0, i), cell(size - 1, i), cell(i, 0), cell(i, size - 1)]) terrain[c] = 'wall'
  }
  const items: { kind: 'key' | 'door'; cell: number; colour: string }[] = []
  let start = cell(1, 1)
  let goal = cell(size - 2, size - 2)
  const freeIn = (r0: number, c0: number, r1: number, c1: number, avoid: readonly number[]): number => {
    for (let attempt = 0; attempt < 256; attempt++) {
      const c = cell(int(r0, r1), int(c0, c1))
      if (terrain[c] === 'floor' && !avoid.includes(c)) return c
    }
    for (let r = r0; r < r1; r++)
      for (let c = c0; c < c1; c++)
        if (terrain[cell(r, c)] === 'floor' && !avoid.includes(cell(r, c))) return cell(r, c)
    return cell(r0, c0)
  }
  if (layout === 'doorKey') {
    const split = int(2, size - 2)
    for (let r = 1; r < size - 1; r++) terrain[cell(r, split)] = 'wall'
    const door = cell(int(1, size - 2), split)
    terrain[door] = 'floor'
    items.push({ kind: 'door', cell: door, colour: 'yellow' })
    start = freeIn(1, 1, size - 1, split, [])
    items.push({ kind: 'key', cell: freeIn(1, 1, size - 1, split, [start]), colour: 'yellow' })
  } else if (layout === 'fourRooms') {
    const half = Math.floor(size / 2)
    for (let i = 1; i < size - 1; i++) {
      terrain[cell(i, half)] = 'wall'
      terrain[cell(half, i)] = 'wall'
    }
    // One gap in each of the four inner wall segments.
    terrain[cell(int(1, half), half)] = 'floor'
    terrain[cell(int(half + 1, size - 1), half)] = 'floor'
    terrain[cell(half, int(1, half))] = 'floor'
    terrain[cell(half, int(half + 1, size - 1))] = 'floor'
    start = freeIn(1, 1, size - 1, size - 1, [])
    goal = freeIn(1, 1, size - 1, size - 1, [start])
  } else if (layout === 'lavaGap') {
    const col = int(2, size - 2)
    for (let r = 1; r < size - 1; r++) terrain[cell(r, col)] = 'lava'
    terrain[cell(int(1, size - 1), col)] = 'floor'
  } else if (layout === 'crossing') {
    const obstacle = options.obstacle === 'wall' ? 'wall' : 'lava'
    const lines: ['h' | 'v', number][] = []
    for (let i = 2; i < size - 2; i += 2) lines.push(['h', i], ['v', i])
    // A keyed shuffle of the candidate rivers, keeping the first `crossings`.
    const order = lines.map((l) => [next(), l] as const).sort((a, b) => a[0] - b[0])
    const chosen = order.slice(0, Math.max(1, options.crossings ?? 1)).map(([, l]) => l)
    const hs = chosen
      .filter(([d]) => d === 'h')
      .map(([, i]) => i)
      .sort((a, b) => a - b)
    const vs = chosen
      .filter(([d]) => d === 'v')
      .map(([, i]) => i)
      .sort((a, b) => a - b)
    for (const r of hs) for (let c = 1; c < size - 1; c++) terrain[cell(r, c)] = obstacle
    for (const c of vs) for (let r = 1; r < size - 1; r++) terrain[cell(r, c)] = obstacle
    // Walk room to room towards the goal, opening one cell of the river crossed at each step.
    const rowsAt = [0, ...hs, size - 1]
    const colsAt = [0, ...vs, size - 1]
    const path = [...hs.map(() => 'down' as const), ...vs.map(() => 'right' as const)]
      .map((d) => [next(), d] as const)
      .sort((a, b) => a[0] - b[0])
      .map(([, d]) => d)
    let ri = 0
    let ci = 0
    for (const d of path) {
      if (d === 'down') {
        terrain[cell(rowsAt[ri + 1] ?? 0, int((colsAt[ci] ?? 0) + 1, colsAt[ci + 1] ?? 1))] = 'floor'
        ri++
      } else {
        terrain[cell(int((rowsAt[ri] ?? 0) + 1, rowsAt[ri + 1] ?? 1), colsAt[ci + 1] ?? 0)] = 'floor'
        ci++
      }
    }
  }
  const w = world({
    topology: { kind: 'square', rows: size, cols: size },
    terrain,
    items,
    starts: [start],
    goals: [goal],
    name: `MiniGrid-${layout}`,
  })
  return {
    world: w,
    rules: rules(
      lockedDoors(),
      pickups(),
      rewardOn('goals', 1, { replace: true }),
      terminateOn('goals'),
      terminateOn('hazards'),
    ),
    options: { actions: ACTIONS.turning, heading: 1 },
  }
}
