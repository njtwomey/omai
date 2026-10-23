import type { Json } from '../../../lib/base/index.ts'
import { foldIn, type Key, uniform } from '../../../lib/random/index.ts'
import type { Env, EnvDefinition } from '../../../lib/rl/index.ts'
import { type Paint, rect, type Scene, type SceneNode, scene } from '../../../lib/scene/index.ts'
import { bool, box, dict, discrete, real } from '../../../lib/space/index.ts'

// MinAtar (Young and Tian 2019, github.com/kenjyoung/MinAtar): five Atari games reduced to a 10 x 10 grid with one
// boolean channel per kind of object. The rules below follow MinAtar's game modules line by line; MinAtar draws its
// random elements from numpy, these draw them from keys, so episodes match in law, not in draws.

/** MinAtar's full action set, in its order: no-op, left, up, right, down, fire. */
export const MINATAR_ACTIONS: readonly string[] = ['n', 'l', 'u', 'r', 'd', 'f']

type Move = 'n' | 'l' | 'u' | 'r' | 'd' | 'f'
const MOVES: readonly Move[] = ['n', 'l', 'u', 'r', 'd', 'f']

/** The state of a MinAtar environment: the game's own state, the last executed action (for sticky actions) and time. */
export interface MinAtarState<G> {
  readonly game: G
  /** The last executed action, as an index into the full action set {@link MINATAR_ACTIONS}. */
  readonly lastAction: number
  readonly terminal: boolean
  readonly t: number
}

/** A MinAtar environment: an {@link Env} with flat `[10, 10, channels]` 0/1 observations and its channel names. */
export interface MinAtarEnv<G> extends Env<MinAtarState<G>, readonly number[], number> {
  readonly channelNames: readonly string[]
  /** The actions available to the agent, as indices into the full action set. */
  readonly actionSet: readonly number[]
}

/** A MinAtar environment definition (what the five games export). */
export interface MinAtarDefinition<G> extends EnvDefinition<MinAtarState<G>, readonly number[], number> {
  readonly channelNames: readonly string[]
  create(options?: Readonly<Record<string, unknown>>): MinAtarEnv<G>
}

/** Options of every MinAtar game. */
export interface MinAtarOptions {
  /** Repeat the previous action with this probability (default 0.1, MinAtar's `sticky_action_prob`). */
  readonly stickyActionProb?: number
  /** Speed the game up over time (default true, MinAtar's `difficulty_ramping`). */
  readonly difficultyRamping?: boolean
  /** Act in the game's minimal action set (default true, as the v1 gym ids) or in all six actions. */
  readonly useMinimalActionSet?: boolean
}

/** A source of uniform numbers drawn lazily from a key. */
type Draw = () => number

function draws(k: Key): Draw {
  let block = 0
  let u: ArrayLike<number> = []
  let i = 0
  return () => {
    if (i >= u.length) {
      u = uniform(foldIn(k, block++), [16]).data
      i = 0
    }
    return u[i++] ?? 0
  }
}

/** The rules of one game: its channels, minimal actions and pure-in-effect update on a private copy of its state. */
interface Rules<G> {
  readonly name: string
  readonly description: string
  readonly channels: readonly string[]
  readonly minimal: readonly Move[]
  init(draw: Draw): G
  clone(g: G): G
  /** Apply one action to `g` in place; returns the reward and whether the game ended. */
  act(g: G, a: Move, draw: Draw, ramping: boolean): { readonly reward: number; readonly terminal: boolean }
  /** The set cells, as `[row, column, channel]`. */
  cells(g: G): [number, number, number][]
  /** The current level of the difficulty ramp (null when the game does not ramp). */
  difficulty(g: G): number | null
}

const PALETTE: readonly Paint[] = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6', 'tab7', 'tab8', 'tab9']

function observation(cells: readonly [number, number, number][], nc: number): number[] {
  const out = new Array<number>(100 * nc).fill(0)
  for (const [y, x, c] of cells) if (y >= 0 && y < 10 && x >= 0 && x < 10) out[(y * 10 + x) * nc + c] = 1
  return out
}

/**
 * The observation of a MinAtar environment as nested booleans `[row][column][channel]` (MinAtar's `state()` array).
 *
 * @param observation - A flat observation of length `100 channels`.
 * @param channels - The number of channels.
 * @returns The `[10][10][channels]` grid.
 * @example
 * ```ts
 * import { breakoutMinAtar, minatarGrid } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const obs = breakoutMinAtar.create().reset(key(0)).observation
 * minatarGrid(obs, 4)[9]?.[4] // => [true, false, false, false]
 * ```
 */
export function minatarGrid(observation: readonly number[], channels: number): boolean[][][] {
  return Array.from({ length: 10 }, (_, y) =>
    Array.from({ length: 10 }, (_, x) =>
      Array.from({ length: channels }, (_, c) => (observation[(y * 10 + x) * channels + c] ?? 0) === 1),
    ),
  )
}

/**
 * Draw a MinAtar grid as coloured cells: each cell takes the colour of its highest set channel (as MinAtar's display
 * does), and its tooltip lists every set channel.
 *
 * @param cells - The set cells, as `[row, column, channel]`.
 * @param channels - The channel names.
 * @param title - The scene title.
 * @returns The scene.
 * @example
 * ```ts
 * import { renderMinAtar } from 'whitebox-engine/methods/rl'
 * renderMinAtar([[0, 0, 1]], ['a', 'b'], 'Demo').children.length // => 102
 * ```
 */
export function renderMinAtar(
  cells: readonly [number, number, number][],
  channels: readonly string[],
  title: string,
): Scene {
  const cell = 32
  const top = new Map<number, number>()
  const names = new Map<number, string[]>()
  for (const [y, x, c] of cells) {
    if (y < 0 || y >= 10 || x < 0 || x >= 10) continue
    const i = y * 10 + x
    top.set(i, Math.max(top.get(i) ?? -1, c))
    names.set(i, [...(names.get(i) ?? []), channels[c] ?? String(c)])
  }
  const nodes: SceneNode[] = []
  for (let i = 0; i < 100; i++) {
    const c = top.get(i)
    const y = Math.floor(i / 10)
    const x = i % 10
    nodes.push(
      rect(x * cell, y * cell, cell, cell, {
        fill: c === undefined ? 'surface' : (PALETTE[c % PALETTE.length] ?? 'fg'),
        stroke: 'grid',
        strokeWidth: 1,
        ...(c === undefined ? {} : { title: `row ${y}, column ${x}: ${(names.get(i) ?? []).join(', ')}` }),
      }),
    )
  }
  // A legend row of channel swatches below the grid.
  for (const [c, name] of channels.entries()) {
    nodes.push(rect(c * cell, 10 * cell + 4, cell, 12, { fill: PALETTE[c % PALETTE.length] ?? 'fg', title: name }))
  }
  return scene(10 * cell, 10 * cell + 16, nodes, { title })
}

function makeMinAtar<G>(rules: Rules<G>, id: string, options: MinAtarOptions): MinAtarEnv<G> {
  const sticky = options.stickyActionProb ?? 0.1
  const ramping = options.difficultyRamping ?? true
  const minimal = options.useMinimalActionSet ?? true
  const actionSet = (minimal ? rules.minimal : MOVES).map((m) => MOVES.indexOf(m))
  const nc = rules.channels.length
  const observe = (s: MinAtarState<G>): readonly number[] => observation(rules.cells(s.game), nc)
  return {
    spec: { id, description: rules.description, deterministic: false },
    observationSpace: box(0, 1, [10, 10, nc], { label: `${rules.channels.join(', ')} (row, column, channel)` }),
    actionSpace: discrete(actionSet.length, {
      label: actionSet.map((a) => MINATAR_ACTIONS[a] ?? '?').join(', '),
    }),
    channelNames: rules.channels,
    actionSet,
    reset: (k) => {
      const s: MinAtarState<G> = { game: rules.init(draws(foldIn(k, 1))), lastAction: 0, terminal: false, t: 0 }
      return { state: s, observation: observe(s), info: { difficulty: rules.difficulty(s.game) } }
    },
    step: (s, action, k) => {
      if (s.terminal) {
        return { state: s, observation: observe(s), reward: 0, terminated: true, truncated: false, info: {} }
      }
      const chosen = actionSet[action] ?? 0
      // MinAtar's Environment.act: with probability sticky_action_prob the last executed action repeats.
      const executed = sticky > 0 && (uniform(foldIn(k, 0), []).data[0] ?? 1) < sticky ? s.lastAction : chosen
      const game = rules.clone(s.game)
      const r = rules.act(game, MOVES[executed] ?? 'n', draws(foldIn(k, 1)), ramping)
      const next: MinAtarState<G> = { game, lastAction: executed, terminal: r.terminal, t: s.t + 1 }
      const info: Record<string, Json> = {
        difficulty: rules.difficulty(game),
        action: MINATAR_ACTIONS[executed] ?? 'n',
      }
      return {
        state: next,
        observation: observe(next),
        reward: r.reward,
        terminated: r.terminal,
        truncated: false,
        info,
      }
    },
    render: (s) => renderMinAtar(rules.cells(s.game), rules.channels, `${rules.name} (MinAtar)`),
    observe,
  }
}

const minatarOptions = dict({
  stickyActionProb: real(0, 1, { default: 0.1, description: 'probability of repeating the previous action' }),
  difficultyRamping: bool({ default: true, description: 'speed the game up over time' }),
  useMinimalActionSet: bool({ default: true, description: "act in the game's minimal action set" }),
})

function definition<G>(rules: Rules<G>): MinAtarDefinition<G> {
  const id = `${rules.name}-MinAtar-v1`
  return {
    id,
    description: rules.description,
    options: minatarOptions,
    channelNames: rules.channels,
    create: (options) => makeMinAtar(rules, id, (options ?? {}) as MinAtarOptions),
  }
}

const step = (right: boolean): number => (right ? 1 : -1)
const inside = (x: number): boolean => x >= 0 && x <= 9

// ---------------------------------------------------------------------------------------------------------- Breakout

/**
 * The state of Breakout: the ball (position, direction 0 up-left, 1 up-right, 2 down-right, 3 down-left, and its
 * previous position, drawn as a trail), the paddle column, the bricks (`bricks[row * 10 + column]`), and whether the
 * ball struck a brick on the previous step.
 */
export interface BreakoutState {
  ballX: number
  ballY: number
  ballDir: number
  lastX: number
  lastY: number
  pos: number
  bricks: number[]
  strike: boolean
}

const BREAKOUT: Rules<BreakoutState> = {
  name: 'Breakout',
  description: 'Bounce a ball off a paddle to break three rows of bricks; missing the ball ends the game.',
  channels: ['paddle', 'ball', 'trail', 'brick'],
  minimal: ['n', 'l', 'r'],
  init: (draw) => {
    const right = draw() < 0.5
    const [ballX, ballDir] = right ? [0, 2] : [9, 3]
    const bricks = new Array<number>(100).fill(0)
    for (let i = 10; i < 40; i++) bricks[i] = 1
    return { ballX, ballY: 3, ballDir, lastX: ballX, lastY: 3, pos: 4, bricks, strike: false }
  },
  clone: (g) => ({ ...g, bricks: [...g.bricks] }),
  act: (g, a) => {
    let reward = 0
    let terminal = false
    if (a === 'l') g.pos = Math.max(0, g.pos - 1)
    else if (a === 'r') g.pos = Math.min(9, g.pos + 1)
    g.lastX = g.ballX
    g.lastY = g.ballY
    let x = g.ballX + (g.ballDir === 0 || g.ballDir === 3 ? -1 : 1)
    let y = g.ballY + (g.ballDir < 2 ? -1 : 1)
    let strikeToggle = false
    if (x < 0 || x > 9) {
      x = Math.min(Math.max(x, 0), 9)
      g.ballDir = [1, 0, 3, 2][g.ballDir] ?? 0
    }
    if (y < 0) {
      y = 0
      g.ballDir = [3, 2, 1, 0][g.ballDir] ?? 0
    } else if (g.bricks[y * 10 + x] === 1) {
      strikeToggle = true
      if (!g.strike) {
        reward += 1
        g.strike = true
        g.bricks[y * 10 + x] = 0
        y = g.lastY
        g.ballDir = [3, 2, 1, 0][g.ballDir] ?? 0
      }
    } else if (y === 9) {
      if (!g.bricks.some((b) => b === 1)) for (let i = 10; i < 40; i++) g.bricks[i] = 1
      if (g.ballX === g.pos) {
        g.ballDir = [3, 2, 1, 0][g.ballDir] ?? 0
        y = g.lastY
      } else if (x === g.pos) {
        g.ballDir = [2, 3, 0, 1][g.ballDir] ?? 0
        y = g.lastY
      } else terminal = true
    }
    if (!strikeToggle) g.strike = false
    g.ballX = x
    g.ballY = y
    return { reward, terminal }
  },
  cells: (g) => {
    const out: [number, number, number][] = [
      [g.ballY, g.ballX, 1],
      [9, g.pos, 0],
      [g.lastY, g.lastX, 2],
    ]
    for (const [i, b] of g.bricks.entries()) if (b === 1) out.push([Math.floor(i / 10), i % 10, 3])
    return out
  },
  difficulty: () => null,
}

/**
 * Breakout-MinAtar-v1: a paddle on the bottom row, a ball moving diagonally and three rows of bricks. Breaking a brick
 * pays 1 and bounces the ball; hitting the paddle bounces it back (straight up when the ball is above the paddle,
 * deflected when it lands next to it); the bricks reappear when all are gone and the ball reaches the bottom; missing
 * the ball ends the episode. Channels: paddle, ball, trail, brick. Actions: no-op, left, right.
 *
 * @concept environments
 * @reference Young and Tian (2019), MinAtar: an Atari-inspired testbed for thorough and reproducible reinforcement
 *   learning experiments.
 * @example
 * ```ts
 * import { breakoutMinAtar } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = breakoutMinAtar.create()
 * env.reset(key(0)).observation.length // => 400
 * env.channelNames // => ['paddle', 'ball', 'trail', 'brick']
 * ```
 */
export const breakoutMinAtar: MinAtarDefinition<BreakoutState> = definition(BREAKOUT)

// ----------------------------------------------------------------------------------------------------------- Freeway

/** A car of Freeway: column, lane row, frames until it next moves, and signed speed (frames per move, sign = way). */
export interface FreewayCar {
  x: number
  y: number
  timer: number
  speed: number
}

/** The state of Freeway: the chicken's row (column 4), the cars, the move cooldown and the frames left. */
export interface FreewayState {
  pos: number
  cars: FreewayCar[]
  moveTimer: number
  terminateTimer: number
}

const FREEWAY_PLAYER_SPEED = 3
const FREEWAY_TIME_LIMIT = 2500

function randomizeCars(draw: Draw, cars: FreewayCar[] | undefined): FreewayCar[] {
  const speeds = Array.from({ length: 8 }, () => 1 + Math.floor(draw() * 5))
  const signs = Array.from({ length: 8 }, () => (draw() < 0.5 ? -1 : 1))
  return Array.from({ length: 8 }, (_, i) => {
    const speed = (speeds[i] ?? 1) * (signs[i] ?? 1)
    const old = cars?.[i]
    return { x: old?.x ?? 0, y: old?.y ?? i + 1, timer: Math.abs(speed), speed }
  })
}

const FREEWAY: Rules<FreewayState> = {
  name: 'Freeway',
  description: 'Guide a chicken across eight lanes of traffic; each crossing pays 1, and a car sends it back.',
  channels: ['chicken', 'car', 'speed1', 'speed2', 'speed3', 'speed4', 'speed5'],
  minimal: ['n', 'u', 'd'],
  init: (draw) => ({
    pos: 9,
    cars: randomizeCars(draw, undefined),
    moveTimer: FREEWAY_PLAYER_SPEED,
    terminateTimer: FREEWAY_TIME_LIMIT,
  }),
  clone: (g) => ({ ...g, cars: g.cars.map((c) => ({ ...c })) }),
  act: (g, a, draw) => {
    let reward = 0
    if (a === 'u' && g.moveTimer === 0) {
      g.moveTimer = FREEWAY_PLAYER_SPEED
      g.pos = Math.max(0, g.pos - 1)
    } else if (a === 'd' && g.moveTimer === 0) {
      g.moveTimer = FREEWAY_PLAYER_SPEED
      g.pos = Math.min(9, g.pos + 1)
    }
    if (g.pos === 0) {
      reward += 1
      g.cars = randomizeCars(draw, g.cars)
      g.pos = 9
    }
    for (const car of g.cars) {
      if (car.x === 4 && car.y === g.pos) g.pos = 9
      if (car.timer === 0) {
        car.timer = Math.abs(car.speed)
        car.x += car.speed > 0 ? 1 : -1
        if (car.x < 0) car.x = 9
        else if (car.x > 9) car.x = 0
        if (car.x === 4 && car.y === g.pos) g.pos = 9
      } else car.timer -= 1
    }
    if (g.moveTimer > 0) g.moveTimer -= 1
    g.terminateTimer -= 1
    return { reward, terminal: g.terminateTimer < 0 }
  },
  cells: (g) => {
    const out: [number, number, number][] = [[g.pos, 4, 0]]
    for (const car of g.cars) {
      out.push([car.y, car.x, 1])
      let back = car.speed > 0 ? car.x - 1 : car.x + 1
      if (back < 0) back = 9
      else if (back > 9) back = 0
      out.push([car.y, back, 1 + Math.abs(car.speed)])
    }
    return out
  },
  difficulty: () => null,
}

/**
 * Freeway-MinAtar-v1: a chicken in column 4 crosses eight lanes, each with one car of random speed (1 to 5 frames per
 * move) and direction. The chicken moves at most every three frames; reaching the top row pays 1, redraws the cars'
 * speeds and returns it to the bottom; a car that touches the chicken sends it back to the bottom. The episode ends
 * after 2500 frames. Channels: chicken, car and five speed trails. Actions: no-op, up, down.
 *
 * @concept environments
 * @example
 * ```ts
 * import { freewayMinAtar } from 'whitebox-engine/methods/rl'
 * freewayMinAtar.create().actionSpace // => { kind: 'discrete', n: 3, label: 'n, u, d' }
 * ```
 */
export const freewayMinAtar: MinAtarDefinition<FreewayState> = definition(FREEWAY)

// ----------------------------------------------------------------------------------------------------------- Asterix

/** An entity of Asterix: column, row, direction (moving right) and whether it is treasure (gold) or an enemy. */
export interface AsterixEntity {
  x: number
  y: number
  right: boolean
  gold: boolean
}

/** The state of Asterix: the player, one entity slot per row 1 to 8, and the spawn, move and ramp timers. */
export interface AsterixState {
  playerX: number
  playerY: number
  entities: (AsterixEntity | null)[]
  spawnSpeed: number
  spawnTimer: number
  moveSpeed: number
  moveTimer: number
  rampTimer: number
  rampIndex: number
}

const ASTERIX_RAMP_INTERVAL = 100
const ASTERIX_SPAWN_SPEED = 10
const ASTERIX_MOVE_INTERVAL = 5

function asterixSpawn(g: AsterixState, draw: Draw): void {
  const right = draw() < 0.5
  const gold = draw() < 1 / 3
  const slots = g.entities.flatMap((e, i) => (e === null ? [i] : []))
  if (slots.length === 0) return
  const slot = slots[Math.floor(draw() * slots.length)] ?? 0
  g.entities[slot] = { x: right ? 0 : 9, y: slot + 1, right, gold }
}

const ASTERIX: Rules<AsterixState> = {
  name: 'Asterix',
  description: 'Collect treasure and dodge enemies that cross the screen in eight lanes; touching an enemy ends it.',
  channels: ['player', 'enemy', 'trail', 'gold'],
  minimal: ['n', 'l', 'u', 'r', 'd'],
  init: () => ({
    playerX: 5,
    playerY: 5,
    entities: new Array<AsterixEntity | null>(8).fill(null),
    spawnSpeed: ASTERIX_SPAWN_SPEED,
    spawnTimer: ASTERIX_SPAWN_SPEED,
    moveSpeed: ASTERIX_MOVE_INTERVAL,
    moveTimer: ASTERIX_MOVE_INTERVAL,
    rampTimer: ASTERIX_RAMP_INTERVAL,
    rampIndex: 0,
  }),
  clone: (g) => ({ ...g, entities: g.entities.map((e) => (e === null ? null : { ...e })) }),
  act: (g, a, draw, ramping) => {
    let reward = 0
    let terminal = false
    if (g.spawnTimer === 0) {
      asterixSpawn(g, draw)
      g.spawnTimer = g.spawnSpeed
    }
    if (a === 'l') g.playerX = Math.max(0, g.playerX - 1)
    else if (a === 'r') g.playerX = Math.min(g.playerX + 1, 9)
    else if (a === 'u') g.playerY = Math.max(g.playerY - 1, 1)
    else if (a === 'd') g.playerY = Math.min(g.playerY + 1, 8)
    const touch = (i: number, e: AsterixEntity): void => {
      if (e.x === g.playerX && e.y === g.playerY) {
        if (e.gold) {
          g.entities[i] = null
          reward += 1
        } else terminal = true
      }
    }
    for (const [i, e] of g.entities.entries()) if (e !== null) touch(i, e)
    if (g.moveTimer === 0) {
      g.moveTimer = g.moveSpeed
      for (const [i, e] of g.entities.entries()) {
        if (e === null) continue
        e.x += step(e.right)
        if (!inside(e.x)) g.entities[i] = null
        else touch(i, e)
      }
    }
    g.spawnTimer -= 1
    g.moveTimer -= 1
    if (ramping && (g.spawnSpeed > 1 || g.moveSpeed > 1)) {
      if (g.rampTimer >= 0) g.rampTimer -= 1
      else {
        if (g.moveSpeed > 1 && g.rampIndex % 2 === 1) g.moveSpeed -= 1
        if (g.spawnSpeed > 1) g.spawnSpeed -= 1
        g.rampIndex += 1
        g.rampTimer = ASTERIX_RAMP_INTERVAL
      }
    }
    return { reward, terminal }
  },
  cells: (g) => {
    const out: [number, number, number][] = [[g.playerY, g.playerX, 0]]
    for (const e of g.entities) {
      if (e === null) continue
      out.push([e.y, e.x, e.gold ? 3 : 1])
      const back = e.right ? e.x - 1 : e.x + 1
      if (inside(back)) out.push([e.y, back, 2])
    }
    return out
  },
  difficulty: (g) => g.rampIndex,
}

/**
 * Asterix-MinAtar-v1: the player moves in rows 1 to 8 while enemies and treasure (one in three) spawn at either edge
 * of a free lane and cross the screen. Collecting treasure pays 1; touching an enemy ends the episode. With difficulty
 * ramping, every 100 frames entities spawn one frame sooner, and every second ramp they also move faster. Channels:
 * player, enemy, trail, gold. Actions: no-op, left, up, right, down.
 *
 * @concept environments
 * @example
 * ```ts
 * import { asterixMinAtar } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * asterixMinAtar.create().reset(key(0)).info // => { difficulty: 0 }
 * ```
 */
export const asterixMinAtar: MinAtarDefinition<AsterixState> = definition(ASTERIX)

// ------------------------------------------------------------------------------------------------------ Space Invaders

/**
 * The state of Space Invaders: the cannon column, the alien, friendly-bullet and enemy-bullet maps
 * (`map[row * 10 + column]`), the aliens' direction (-1 left, 1 right), their move interval and the timers.
 */
export interface SpaceInvadersState {
  pos: number
  aliens: number[]
  friendlyBullets: number[]
  enemyBullets: number[]
  alienDir: number
  enemyMoveInterval: number
  alienMoveTimer: number
  alienShotTimer: number
  rampIndex: number
  shotTimer: number
}

const SI_SHOT_COOL_DOWN = 5
const SI_ENEMY_MOVE_INTERVAL = 12
const SI_ENEMY_SHOT_INTERVAL = 10

function alienBlock(): number[] {
  const m = new Array<number>(100).fill(0)
  for (let y = 0; y < 4; y++) for (let x = 2; x < 8; x++) m[y * 10 + x] = 1
  return m
}

/** numpy.roll of a 10 x 10 map by `dy` rows and `dx` columns (cells wrap around). */
function roll(m: readonly number[], dy: number, dx: number): number[] {
  const out = new Array<number>(100).fill(0)
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) out[((y + dy + 10) % 10) * 10 + ((x + dx + 10) % 10)] = m[y * 10 + x] ?? 0
  }
  return out
}

const column = (m: readonly number[], x: number): number[] => Array.from({ length: 10 }, (_, y) => m[y * 10 + x] ?? 0)

const SPACE_INVADERS: Rules<SpaceInvadersState> = {
  name: 'SpaceInvaders',
  description: 'Shoot down a descending block of aliens that fires back; reaching the bottom or being hit ends it.',
  channels: ['cannon', 'alien', 'alien_left', 'alien_right', 'friendly_bullet', 'enemy_bullet'],
  minimal: ['n', 'l', 'r', 'f'],
  init: () => ({
    pos: 5,
    aliens: alienBlock(),
    friendlyBullets: new Array<number>(100).fill(0),
    enemyBullets: new Array<number>(100).fill(0),
    alienDir: -1,
    enemyMoveInterval: SI_ENEMY_MOVE_INTERVAL,
    alienMoveTimer: SI_ENEMY_MOVE_INTERVAL,
    alienShotTimer: SI_ENEMY_SHOT_INTERVAL,
    rampIndex: 0,
    shotTimer: 0,
  }),
  clone: (g) => ({
    ...g,
    aliens: [...g.aliens],
    friendlyBullets: [...g.friendlyBullets],
    enemyBullets: [...g.enemyBullets],
  }),
  act: (g, a, _draw, ramping) => {
    let reward = 0
    let terminal = false
    if (a === 'f' && g.shotTimer === 0) {
      g.friendlyBullets[90 + g.pos] = 1
      g.shotTimer = SI_SHOT_COOL_DOWN
    } else if (a === 'l') g.pos = Math.max(0, g.pos - 1)
    else if (a === 'r') g.pos = Math.min(9, g.pos + 1)
    g.friendlyBullets = roll(g.friendlyBullets, -1, 0)
    for (let x = 0; x < 10; x++) g.friendlyBullets[90 + x] = 0
    g.enemyBullets = roll(g.enemyBullets, 1, 0)
    for (let x = 0; x < 10; x++) g.enemyBullets[x] = 0
    if (g.enemyBullets[90 + g.pos] === 1) terminal = true
    if (g.aliens[90 + g.pos] === 1) terminal = true
    const count = (): number => g.aliens.reduce((s, v) => s + v, 0)
    if (g.alienMoveTimer === 0) {
      g.alienMoveTimer = Math.min(count(), g.enemyMoveInterval)
      const atLeft = column(g.aliens, 0).some((v) => v === 1)
      const atRight = column(g.aliens, 9).some((v) => v === 1)
      if ((atLeft && g.alienDir < 0) || (atRight && g.alienDir > 0)) {
        g.alienDir = -g.alienDir
        if (g.aliens.slice(90).some((v) => v === 1)) terminal = true
        g.aliens = roll(g.aliens, 1, 0)
      } else g.aliens = roll(g.aliens, 0, g.alienDir)
      if (g.aliens[90 + g.pos] === 1) terminal = true
    }
    if (g.alienShotTimer === 0) {
      g.alienShotTimer = SI_ENEMY_SHOT_INTERVAL
      // The nearest column with an alien (ties to the left), and its lowest alien.
      const order = Array.from({ length: 10 }, (_, i) => i).sort((p, q) => Math.abs(p - g.pos) - Math.abs(q - g.pos))
      for (const x of order) {
        const col = column(g.aliens, x)
        const y = col.lastIndexOf(1)
        if (y >= 0) {
          g.enemyBullets[y * 10 + x] = 1
          break
        }
      }
    }
    for (let i = 0; i < 100; i++) {
      if (g.aliens[i] === 1 && g.friendlyBullets[i] === 1) {
        reward += 1
        g.aliens[i] = 0
        g.friendlyBullets[i] = 0
      }
    }
    if (g.shotTimer > 0) g.shotTimer -= 1
    g.alienMoveTimer -= 1
    g.alienShotTimer -= 1
    if (count() === 0) {
      if (g.enemyMoveInterval > 6 && ramping) {
        g.enemyMoveInterval -= 1
        g.rampIndex += 1
      }
      g.aliens = alienBlock()
    }
    return { reward, terminal }
  },
  cells: (g) => {
    const out: [number, number, number][] = [[9, g.pos, 0]]
    for (let i = 0; i < 100; i++) {
      const y = Math.floor(i / 10)
      const x = i % 10
      if (g.aliens[i] === 1) out.push([y, x, 1], [y, x, g.alienDir < 0 ? 2 : 3])
      if (g.friendlyBullets[i] === 1) out.push([y, x, 4])
      if (g.enemyBullets[i] === 1) out.push([y, x, 5])
    }
    return out
  },
  difficulty: (g) => g.rampIndex,
}

/**
 * SpaceInvaders-MinAtar-v1: a cannon on the bottom row faces a 4 x 6 block of aliens that marches sideways, steps down
 * at the edges and speeds up as it thins out; every ten frames the alien nearest the cannon's column fires. Each alien
 * shot pays 1 (the cannon reloads in five frames); a cleared wave returns faster (with difficulty ramping); being hit
 * or letting an alien reach the cannon's row ends the episode. Channels: cannon, alien, alien_left, alien_right,
 * friendly_bullet, enemy_bullet. Actions: no-op, left, right, fire.
 *
 * @concept environments
 * @example
 * ```ts
 * import { spaceInvadersMinAtar } from 'whitebox-engine/methods/rl'
 * spaceInvadersMinAtar.channelNames.length // => 6
 * ```
 */
export const spaceInvadersMinAtar: MinAtarDefinition<SpaceInvadersState> = definition(SPACE_INVADERS)

// ---------------------------------------------------------------------------------------------------------- Seaquest

/** A moving object of Seaquest: column, row, direction, and its move (and, for enemy subs, shot) countdowns. */
export interface SeaquestMover {
  x: number
  y: number
  right: boolean
  timer: number
  shot?: number
}

/** A bullet of Seaquest: column, row and direction. */
export interface SeaquestBullet {
  x: number
  y: number
  right: boolean
}

/**
 * The state of Seaquest: the submarine (position, facing right or not), oxygen, rescued divers, the bullets, enemy
 * fish, enemy subs and divers, the spawn and shot timers, the difficulty ramp, and whether the sub is surfaced.
 */
export interface SeaquestState {
  oxygen: number
  diverCount: number
  subX: number
  subY: number
  subOr: boolean
  friendlyBullets: SeaquestBullet[]
  enemyBullets: SeaquestBullet[]
  fish: SeaquestMover[]
  subs: SeaquestMover[]
  divers: SeaquestMover[]
  enemySpawnSpeed: number
  enemySpawnTimer: number
  diverSpawnTimer: number
  moveSpeed: number
  rampIndex: number
  shotTimer: number
  surface: boolean
}

const SQ_MAX_OXYGEN = 200
const SQ_INIT_SPAWN_SPEED = 20
const SQ_DIVER_SPAWN_SPEED = 30
const SQ_INIT_MOVE_INTERVAL = 5
const SQ_SHOT_COOL_DOWN = 5
const SQ_ENEMY_SHOT_INTERVAL = 10
const SQ_DIVER_MOVE_INTERVAL = 5

const remove = <T>(list: T[], item: T): void => {
  const i = list.indexOf(item)
  if (i >= 0) list.splice(i, 1)
}

const at = (o: { readonly x: number; readonly y: number }, x: number, y: number): boolean => o.x === x && o.y === y

function seaquestSpawnEnemy(g: SeaquestState, draw: Draw): void {
  const right = draw() < 0.5
  const isSub = draw() < 1 / 3
  const x = right ? 0 : 9
  const y = 1 + Math.floor(draw() * 8)
  // No enemy spawns in a row that already has one.
  if ([...g.subs, ...g.fish].some((z) => z.y === y)) return
  if (isSub) g.subs.push({ x, y, right, timer: g.moveSpeed, shot: SQ_ENEMY_SHOT_INTERVAL })
  else g.fish.push({ x, y, right, timer: g.moveSpeed })
}

function seaquestSpawnDiver(g: SeaquestState, draw: Draw): void {
  const right = draw() < 0.5
  const y = 1 + Math.floor(draw() * 8)
  g.divers.push({ x: right ? 0 : 9, y, right, timer: SQ_DIVER_MOVE_INTERVAL })
}

/** Surfacing with divers: six divers pay the remaining oxygen in tenths and are all released; else one is used up. */
function seaquestSurface(g: SeaquestState, ramping: boolean): number {
  g.surface = true
  let reward = 0
  if (g.diverCount === 6) {
    g.diverCount = 0
    reward = Math.floor((g.oxygen * 10) / SQ_MAX_OXYGEN)
  } else g.diverCount -= 1
  g.oxygen = SQ_MAX_OXYGEN
  if (ramping && (g.enemySpawnSpeed > 1 || g.moveSpeed > 2)) {
    if (g.moveSpeed > 2 && g.rampIndex % 2 === 1) g.moveSpeed -= 1
    if (g.enemySpawnSpeed > 1) g.enemySpawnSpeed -= 1
    g.rampIndex += 1
  }
  return reward
}

/** Move an enemy (fish or sub) one step on its timer; returns the reward from a friendly bullet it runs into. */
function seaquestMoveEnemy(g: SeaquestState, list: SeaquestMover[], e: SeaquestMover, end: () => void): number {
  if (at(e, g.subX, g.subY)) end()
  if (e.timer !== 0) {
    e.timer -= 1
    return 0
  }
  e.timer = g.moveSpeed
  e.x += step(e.right)
  if (!inside(e.x)) remove(list, e)
  else if (at(e, g.subX, g.subY)) end()
  else {
    for (const b of g.friendlyBullets) {
      if (at(e, b.x, b.y)) {
        remove(list, e)
        remove(g.friendlyBullets, b)
        return 1
      }
    }
  }
  return 0
}

const SEAQUEST: Rules<SeaquestState> = {
  name: 'Seaquest',
  description: 'Pilot a submarine that shoots enemy fish and subs, rescues divers and surfaces for oxygen.',
  channels: [
    'sub_front',
    'sub_back',
    'friendly_bullet',
    'trail',
    'enemy_bullet',
    'enemy_fish',
    'enemy_sub',
    'oxygen_guage',
    'diver_guage',
    'diver',
  ],
  minimal: ['n', 'l', 'u', 'r', 'd', 'f'],
  init: () => ({
    oxygen: SQ_MAX_OXYGEN,
    diverCount: 0,
    subX: 5,
    subY: 0,
    subOr: false,
    friendlyBullets: [],
    enemyBullets: [],
    fish: [],
    subs: [],
    divers: [],
    enemySpawnSpeed: SQ_INIT_SPAWN_SPEED,
    enemySpawnTimer: SQ_INIT_SPAWN_SPEED,
    diverSpawnTimer: SQ_DIVER_SPAWN_SPEED,
    moveSpeed: SQ_INIT_MOVE_INTERVAL,
    rampIndex: 0,
    shotTimer: 0,
    surface: true,
  }),
  clone: (g) => ({
    ...g,
    friendlyBullets: g.friendlyBullets.map((b) => ({ ...b })),
    enemyBullets: g.enemyBullets.map((b) => ({ ...b })),
    fish: g.fish.map((f) => ({ ...f })),
    subs: g.subs.map((s) => ({ ...s })),
    divers: g.divers.map((d) => ({ ...d })),
  }),
  act: (g, a, draw, ramping) => {
    let reward = 0
    let terminal = false
    const end = (): void => {
      terminal = true
    }
    if (g.enemySpawnTimer === 0) {
      seaquestSpawnEnemy(g, draw)
      g.enemySpawnTimer = g.enemySpawnSpeed
    }
    if (g.diverSpawnTimer === 0) {
      seaquestSpawnDiver(g, draw)
      g.diverSpawnTimer = SQ_DIVER_SPAWN_SPEED
    }
    if (a === 'f' && g.shotTimer === 0) {
      g.shotTimer = SQ_SHOT_COOL_DOWN
      g.friendlyBullets.push({ x: g.subX, y: g.subY, right: g.subOr })
    } else if (a === 'l') {
      g.subX = Math.max(0, g.subX - 1)
      g.subOr = false
    } else if (a === 'r') {
      g.subX = Math.min(9, g.subX + 1)
      g.subOr = true
    } else if (a === 'u') g.subY = Math.max(0, g.subY - 1)
    else if (a === 'd') g.subY = Math.min(8, g.subY + 1)
    // Friendly bullets fly one cell and destroy the first fish (else sub) they meet.
    for (const b of [...g.friendlyBullets].reverse()) {
      b.x += step(b.right)
      if (!inside(b.x)) {
        remove(g.friendlyBullets, b)
        continue
      }
      const fish = g.fish.find((f) => at(f, b.x, b.y))
      const target = fish ?? g.subs.find((s) => at(s, b.x, b.y))
      if (target !== undefined) {
        remove(fish === undefined ? g.subs : g.fish, target)
        remove(g.friendlyBullets, b)
        reward += 1
      }
    }
    // Divers drift and are picked up (up to six) when the sub meets them.
    for (const d of [...g.divers].reverse()) {
      if (at(d, g.subX, g.subY) && g.diverCount < 6) {
        remove(g.divers, d)
        g.diverCount += 1
      } else if (d.timer === 0) {
        d.timer = SQ_DIVER_MOVE_INTERVAL
        d.x += step(d.right)
        if (!inside(d.x)) remove(g.divers, d)
        else if (at(d, g.subX, g.subY) && g.diverCount < 6) {
          remove(g.divers, d)
          g.diverCount += 1
        }
      } else d.timer -= 1
    }
    // Enemy subs move, collide, are shot, and fire on their own timers (even on the frame they are destroyed).
    for (const s of [...g.subs].reverse()) {
      reward += seaquestMoveEnemy(g, g.subs, s, end)
      if (s.shot === 0) {
        s.shot = SQ_ENEMY_SHOT_INTERVAL
        g.enemyBullets.push({ x: s.x, y: s.y, right: s.right })
      } else s.shot = (s.shot ?? SQ_ENEMY_SHOT_INTERVAL) - 1
    }
    for (const b of [...g.enemyBullets].reverse()) {
      if (at(b, g.subX, g.subY)) terminal = true
      b.x += step(b.right)
      if (!inside(b.x)) remove(g.enemyBullets, b)
      else if (at(b, g.subX, g.subY)) terminal = true
    }
    for (const f of [...g.fish].reverse()) reward += seaquestMoveEnemy(g, g.fish, f, end)
    if (g.enemySpawnTimer > 0) g.enemySpawnTimer -= 1
    if (g.diverSpawnTimer > 0) g.diverSpawnTimer -= 1
    if (g.shotTimer > 0) g.shotTimer -= 1
    if (g.oxygen < 0) terminal = true
    if (g.subY > 0) {
      g.oxygen -= 1
      g.surface = false
    } else if (!g.surface) {
      if (g.diverCount === 0) terminal = true
      else reward += seaquestSurface(g, ramping)
    }
    return { reward, terminal }
  },
  cells: (g) => {
    const out: [number, number, number][] = [
      [g.subY, g.subX, 0],
      [g.subY, g.subOr ? g.subX - 1 : g.subX + 1, 1],
    ]
    const oxygen = Math.floor((g.oxygen * 10) / SQ_MAX_OXYGEN)
    for (let x = 0; x < oxygen; x++) out.push([9, x, 7])
    for (let x = 9 - g.diverCount; x < 9; x++) out.push([9, x, 8])
    for (const b of g.friendlyBullets) out.push([b.y, b.x, 2])
    for (const b of g.enemyBullets) out.push([b.y, b.x, 4])
    const movers: [readonly SeaquestMover[], number][] = [
      [g.fish, 5],
      [g.subs, 6],
      [g.divers, 9],
    ]
    for (const [list, c] of movers) {
      for (const m of list) {
        out.push([m.y, m.x, c])
        const back = m.right ? m.x - 1 : m.x + 1
        if (inside(back)) out.push([m.y, back, 3])
      }
    }
    return out
  },
  difficulty: (g) => g.rampIndex,
}

/**
 * Seaquest-MinAtar-v1: a submarine shoots enemy fish and enemy subs (which fire back) that spawn in free lanes, and
 * rescues divers that drift across. Oxygen runs down while submerged; surfacing with no divers ends the episode,
 * surfacing with one to five divers uses one up and refills the oxygen, and surfacing with six releases them all for
 * a bonus of the remaining oxygen in tenths. Each kill pays 1; collisions with enemies or their bullets, and running
 * out of oxygen, end the episode. With difficulty ramping each surfacing makes enemies spawn sooner (and, every second
 * time, move faster). Channels: sub_front, sub_back, friendly_bullet, trail, enemy_bullet, enemy_fish, enemy_sub,
 * oxygen_guage, diver_guage, diver (MinAtar's names). Actions: all six.
 *
 * @concept environments
 * @example
 * ```ts
 * import { seaquestMinAtar } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = seaquestMinAtar.create({ stickyActionProb: 0 })
 * const s = env.reset(key(0)).state
 * env.step(s, 4, key(1)).state.game.oxygen // => 199
 * ```
 */
export const seaquestMinAtar: MinAtarDefinition<SeaquestState> = definition(SEAQUEST)
