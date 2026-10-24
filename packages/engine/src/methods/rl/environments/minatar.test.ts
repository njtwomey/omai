import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../../../lib/random/index.ts'
import { contains } from '../../../lib/space/index.ts'
import {
  type AsterixState,
  asterixMinAtar,
  type BreakoutState,
  breakoutMinAtar,
  type FreewayState,
  freewayMinAtar,
  type MinAtarDefinition,
  type MinAtarEnv,
  type MinAtarState,
  minatarGrid,
  type SeaquestState,
  type SpaceInvadersState,
  seaquestMinAtar,
  spaceInvadersMinAtar,
} from './minatar.ts'

// Full action indices: no-op, left, up, right, down, fire.
const N = 0
const L = 1
const U = 2
const R = 3
const D = 4
const F = 5

const planted = <G>(game: G): MinAtarState<G> => ({ game, lastAction: 0, terminal: false, t: 0 })
const envOf = <G>(def: MinAtarDefinition<G>, options: Record<string, unknown> = {}) =>
  def.create({ stickyActionProb: 0, useMinimalActionSet: false, ...options })
const play = <G>(def: MinAtarDefinition<G>, game: G, action: number, k = key(0), options = {}) =>
  envOf(def, options).step(planted(game), action, k)
const startOf = <G>(def: MinAtarDefinition<G>): G => envOf(def).reset(key(0)).state.game

const emptyMap = (): number[] => new Array<number>(100).fill(0)

describe('MinAtar environments', () => {
  it('have boxed [10, 10, channels] observations, MinAtar ids, minimal action sets and coloured renders', () => {
    const games = [
      [breakoutMinAtar, 4, 3],
      [asterixMinAtar, 4, 5],
      [freewayMinAtar, 7, 3],
      [seaquestMinAtar, 10, 6],
      [spaceInvadersMinAtar, 6, 4],
    ] as const
    for (const [def, channels, actions] of games) {
      const env = def.create() as unknown as MinAtarEnv<unknown>
      expect(def.id).toMatch(/-MinAtar-v1$/)
      expect(env.channelNames).toHaveLength(channels)
      expect(env.observationSpace).toMatchObject({ kind: 'box', shape: [10, 10, channels] })
      expect(env.actionSpace).toMatchObject({ kind: 'discrete', n: actions })
      expect(def.create({ useMinimalActionSet: false }).actionSpace).toMatchObject({ n: 6 })
      const r = env.reset(key(1))
      expect(contains(env.observationSpace, r.observation)).toBe(true)
      expect(env.render(r.state).children.length).toBe(100 + channels)
    }
    expect(spaceInvadersMinAtar.create().actionSet).toEqual([N, L, R, F])
  })

  it('run random episodes to termination reproducibly from keys', () => {
    for (const def of [breakoutMinAtar, asterixMinAtar, seaquestMinAtar, spaceInvadersMinAtar] as const) {
      const env = def.create() as unknown as ReturnType<typeof breakoutMinAtar.create>
      const run = (seed: number): number[] => {
        let s = env.reset(key(seed)).state
        const rewards: number[] = []
        for (let t = 0; t < 3000; t++) {
          const a = Math.floor((((t * 7919 + seed) % 97) / 97) * env.actionSet.length)
          const r = env.step(s, a, foldIn(key(seed), t))
          rewards.push(r.reward)
          s = r.state
          if (r.terminated) break
        }
        return rewards
      }
      const a = run(3)
      expect(run(3)).toEqual(a)
      expect(a.length).toBeLessThan(3000)
    }
  })

  it('apply sticky actions with the given probability, starting from a no-op', () => {
    const always = breakoutMinAtar.create({ stickyActionProb: 1 })
    const s = always.reset(key(0)).state
    // The minimal set is no-op, left, right: pressing right repeats the no-op.
    expect(always.step(s, 2, key(1)).state.game.pos).toBe(4)
    const env = breakoutMinAtar.create()
    let repeats = 0
    for (let i = 0; i < 2000; i++) {
      const r = env.step({ ...s, lastAction: L }, 2, key(i))
      if (r.state.lastAction === L) repeats++
    }
    expect(repeats / 2000).toBeGreaterThan(0.08)
    expect(repeats / 2000).toBeLessThan(0.12)
  })
})

describe('Breakout', () => {
  const base = startOf(breakoutMinAtar)
  const game = (patch: Partial<BreakoutState>): BreakoutState => ({ ...base, bricks: [...base.bricks], ...patch })

  it('starts the ball in row 3 at either wall, heading down and inwards', () => {
    const starts = new Set<string>()
    for (let i = 0; i < 20; i++) {
      const g = breakoutMinAtar.create().reset(key(i)).state.game
      starts.add(`${g.ballX},${g.ballY},${g.ballDir}`)
    }
    expect(starts).toEqual(new Set(['0,3,2', '9,3,3']))
  })

  it('bounces off the side walls and the ceiling', () => {
    const side = play(breakoutMinAtar, game({ ballX: 0, ballY: 6, ballDir: 0 }), N).state.game
    expect([side.ballX, side.ballY, side.ballDir]).toEqual([0, 5, 1])
    const top = play(breakoutMinAtar, game({ ballX: 5, ballY: 0, ballDir: 1, bricks: emptyMap() }), N).state.game
    expect([top.ballX, top.ballY, top.ballDir]).toEqual([6, 0, 2])
  })

  it('scores a brick, removes it and bounces back, but not twice in a row', () => {
    const r = play(breakoutMinAtar, game({ ballX: 3, ballY: 4, ballDir: 0 }), N)
    expect(r.reward).toBe(1)
    expect(r.state.game.bricks[3 * 10 + 2]).toBe(0)
    expect([r.state.game.ballX, r.state.game.ballY, r.state.game.ballDir]).toEqual([2, 4, 3])
    // Straight after a strike the ball passes into the next brick without scoring.
    const again = play(breakoutMinAtar, game({ ballX: 3, ballY: 4, ballDir: 0, strike: true }), N)
    expect(again.reward).toBe(0)
    expect([again.state.game.ballX, again.state.game.ballY]).toEqual([2, 3])
  })

  it('bounces off the paddle, deflects off its edge, and ends when the ball is missed', () => {
    const straight = play(breakoutMinAtar, game({ ballX: 4, ballY: 8, ballDir: 3, pos: 4 }), N).state.game
    expect([straight.ballX, straight.ballY, straight.ballDir]).toEqual([3, 8, 0])
    const edge = play(breakoutMinAtar, game({ ballX: 5, ballY: 8, ballDir: 3, pos: 4 }), N).state.game
    expect([edge.ballX, edge.ballY, edge.ballDir]).toEqual([4, 8, 1])
    const miss = play(breakoutMinAtar, game({ ballX: 7, ballY: 8, ballDir: 2, pos: 4 }), N)
    expect(miss.terminated).toBe(true)
    expect(miss.reward).toBe(0)
    // The paddle moves left and right before the ball moves.
    expect(play(breakoutMinAtar, game({ ballX: 4, ballY: 8, ballDir: 2, pos: 4 }), R).terminated).toBe(false)
  })

  it('restores the bricks when all are gone and the ball reaches the bottom', () => {
    const r = play(breakoutMinAtar, game({ ballX: 4, ballY: 8, ballDir: 3, pos: 4, bricks: emptyMap() }), N)
    expect(r.state.game.bricks.filter((b) => b === 1)).toHaveLength(30)
  })

  it('observes paddle, ball, trail and bricks', () => {
    const r = play(breakoutMinAtar, game({ ballX: 5, ballY: 6, ballDir: 2 }), N)
    const grid = minatarGrid(r.observation, 4)
    expect(grid[9]?.[4]?.[0]).toBe(true)
    expect(grid[7]?.[6]?.[1]).toBe(true)
    expect(grid[6]?.[5]?.[2]).toBe(true)
    expect(grid[2]?.[0]?.[3]).toBe(true)
  })
})

describe('Freeway', () => {
  const base = startOf(freewayMinAtar)
  const game = (patch: Partial<FreewayState>): FreewayState => ({
    ...base,
    cars: base.cars.map((c) => ({ ...c, x: 0, timer: 5, speed: 5 })),
    ...patch,
  })

  it('starts eight cars in column 0, one per lane, with speeds 1 to 5 either way', () => {
    for (const [i, car] of base.cars.entries()) {
      expect(car).toMatchObject({ x: 0, y: i + 1 })
      expect(Math.abs(car.speed)).toBeGreaterThanOrEqual(1)
      expect(Math.abs(car.speed)).toBeLessThanOrEqual(5)
      expect(car.timer).toBe(Math.abs(car.speed))
    }
    expect(base.pos).toBe(9)
  })

  it('moves cars on their timers and wraps them around', () => {
    const g = game({})
    g.cars[0] = { x: 9, y: 1, timer: 0, speed: 2 }
    g.cars[1] = { x: 0, y: 2, timer: 0, speed: -3 }
    g.cars[2] = { x: 5, y: 3, timer: 2, speed: 4 }
    const cars = play(freewayMinAtar, g, N).state.game.cars
    expect(cars[0]).toEqual({ x: 0, y: 1, timer: 2, speed: 2 })
    expect(cars[1]).toEqual({ x: 9, y: 2, timer: 3, speed: -3 })
    expect(cars[2]).toEqual({ x: 5, y: 3, timer: 1, speed: 4 })
  })

  it('pays 1 for crossing, sends the chicken back and redraws the speeds', () => {
    const r = play(freewayMinAtar, game({ pos: 1, moveTimer: 0 }), U, key(4))
    expect(r.reward).toBe(1)
    expect(r.state.game.pos).toBe(9)
    expect(r.state.game.cars.map((c) => c.x)).toEqual(new Array<number>(8).fill(0))
    expect(r.state.game.cars.some((c) => c.speed !== 5)).toBe(true)
  })

  it('moves the chicken at most every three frames and knocks it back on a collision', () => {
    const moved = play(freewayMinAtar, game({ pos: 5, moveTimer: 0 }), U).state.game
    expect([moved.pos, moved.moveTimer]).toEqual([4, 2])
    expect(play(freewayMinAtar, game({ pos: 5, moveTimer: 2 }), U).state.game.pos).toBe(5)
    const g = game({ pos: 6, moveTimer: 1 })
    g.cars[5] = { x: 3, y: 6, timer: 0, speed: 1 }
    expect(play(freewayMinAtar, g, N).state.game.pos).toBe(9)
  })

  it('ends after 2500 frames and draws each car with a trail on its speed channel', () => {
    expect(play(freewayMinAtar, game({ terminateTimer: 1 }), N).terminated).toBe(false)
    expect(play(freewayMinAtar, game({ terminateTimer: 0 }), N).terminated).toBe(true)
    const g = game({})
    g.cars[0] = { x: 5, y: 1, timer: 3, speed: -3 }
    const grid = minatarGrid(play(freewayMinAtar, g, N).observation, 7)
    expect(grid[1]?.[5]?.[1]).toBe(true)
    expect(grid[1]?.[6]?.[4]).toBe(true)
    expect(grid[9]?.[4]?.[0]).toBe(true)
  })
})

describe('Asterix', () => {
  const base = startOf(asterixMinAtar)
  const game = (patch: Partial<AsterixState>): AsterixState => ({
    ...base,
    entities: new Array(8).fill(null),
    ...patch,
  })

  it('spawns enemies and treasure (one in three) at an edge of a free lane', () => {
    let gold = 0
    const n = 600
    for (let i = 0; i < n; i++) {
      const g = game({ spawnTimer: 0 })
      const entities = play(asterixMinAtar, g, N, key(i)).state.game.entities.filter((e) => e !== null)
      expect(entities).toHaveLength(1)
      const e = entities[0]
      if (e === undefined || e === null) throw new Error('no spawn')
      expect(e.x).toBe(e.right ? 0 : 9)
      if (e.gold) gold++
    }
    expect(gold / n).toBeGreaterThan(0.27)
    expect(gold / n).toBeLessThan(0.4)
    // With every lane occupied nothing spawns.
    const full = game({
      spawnTimer: 0,
      moveTimer: 3,
      entities: Array.from({ length: 8 }, (_, i) => ({ x: 2, y: i + 1, right: true, gold: false })),
    })
    full.playerY = 1
    full.playerX = 7
    expect(play(asterixMinAtar, full, N).state.game.entities.every((e) => e !== null && e.x === 2)).toBe(true)
  })

  it('pays 1 for treasure and ends on an enemy, before and after entities move', () => {
    const treasure = game({
      entities: [null, null, null, null, { x: 5, y: 5, right: true, gold: true }, null, null, null],
    })
    const r = play(asterixMinAtar, treasure, N)
    expect(r.reward).toBe(1)
    expect(r.state.game.entities[4]).toBeNull()
    const enemy = game({
      moveTimer: 0,
      entities: [null, null, null, null, { x: 4, y: 5, right: true, gold: false }, null, null, null],
    })
    expect(play(asterixMinAtar, enemy, N).terminated).toBe(true)
  })

  it('moves entities on the move timer, removes those that leave, and keeps the player in rows 1 to 8', () => {
    const g = game({
      moveTimer: 0,
      entities: [
        { x: 9, y: 1, right: true, gold: false },
        { x: 3, y: 2, right: false, gold: true },
        null,
        null,
        null,
        null,
        null,
        null,
      ],
    })
    const next = play(asterixMinAtar, g, N).state.game
    expect(next.entities[0]).toBeNull()
    expect(next.entities[1]).toMatchObject({ x: 2, y: 2 })
    expect(next.moveTimer).toBe(next.moveSpeed - 1)
    expect(play(asterixMinAtar, game({ playerY: 1 }), U).state.game.playerY).toBe(1)
    expect(play(asterixMinAtar, game({ playerY: 8 }), D).state.game.playerY).toBe(8)
  })

  it('ramps difficulty every 100 frames: spawns sooner, and moves faster every second ramp', () => {
    const first = play(asterixMinAtar, game({ rampTimer: -1, spawnTimer: 5, moveTimer: 3 }), N).state.game
    expect([first.spawnSpeed, first.moveSpeed, first.rampIndex, first.rampTimer]).toEqual([9, 5, 1, 100])
    const second = play(asterixMinAtar, { ...first, rampTimer: -1 }, N).state.game
    expect([second.spawnSpeed, second.moveSpeed, second.rampIndex]).toEqual([8, 4, 2])
    const flat = play(asterixMinAtar, game({ rampTimer: -1 }), N, key(0), { difficultyRamping: false }).state.game
    expect(flat.spawnSpeed).toBe(10)
  })
})

describe('Space Invaders', () => {
  const base = startOf(spaceInvadersMinAtar)
  const game = (patch: Partial<SpaceInvadersState>): SpaceInvadersState => ({
    ...base,
    aliens: [...base.aliens],
    friendlyBullets: emptyMap(),
    enemyBullets: emptyMap(),
    ...patch,
  })
  const cellsOf = (m: readonly number[]): number[] => m.flatMap((v, i) => (v === 1 ? [i] : []))

  it('marches the aliens sideways, then down and back at the edges, faster as they thin out', () => {
    const left = play(spaceInvadersMinAtar, game({ alienMoveTimer: 0 }), N).state.game
    expect(cellsOf(left.aliens)[0]).toBe(1)
    expect(left.alienMoveTimer).toBe(11)
    const edge = emptyMap()
    edge[20] = 1
    const down = play(spaceInvadersMinAtar, game({ alienMoveTimer: 0, aliens: edge }), N).state.game
    expect(cellsOf(down.aliens)).toEqual([30])
    expect(down.alienDir).toBe(1)
    // One alien left: it moves every frame.
    expect(down.alienMoveTimer).toBe(0)
  })

  it('fires from the cannon with a cooldown and scores when a bullet meets an alien', () => {
    const fired = play(spaceInvadersMinAtar, game({ pos: 3 }), F).state.game
    expect(cellsOf(fired.friendlyBullets)).toEqual([83])
    expect(fired.shotTimer).toBe(4)
    expect(cellsOf(play(spaceInvadersMinAtar, game({ shotTimer: 2 }), F).state.game.friendlyBullets)).toEqual([])
    const bullets = emptyMap()
    bullets[44] = 1
    const hit = play(spaceInvadersMinAtar, game({ friendlyBullets: bullets }), N)
    expect(hit.reward).toBe(1)
    expect(hit.state.game.aliens[34]).toBe(0)
    expect(hit.state.game.friendlyBullets[34]).toBe(0)
  })

  it('shoots from the lowest alien of the nearest column, and an enemy bullet on the cannon ends the game', () => {
    const shot = play(spaceInvadersMinAtar, game({ pos: 0, alienShotTimer: 0 }), N).state.game
    expect(cellsOf(shot.enemyBullets)).toEqual([32])
    const bullets = emptyMap()
    bullets[85] = 1
    expect(play(spaceInvadersMinAtar, game({ pos: 5, enemyBullets: bullets }), N).terminated).toBe(true)
    expect(play(spaceInvadersMinAtar, game({ pos: 5, enemyBullets: bullets }), L).terminated).toBe(false)
  })

  it('ends when aliens reach the bottom row, and a cleared wave returns faster', () => {
    const low = emptyMap()
    low[95] = 1
    expect(play(spaceInvadersMinAtar, game({ pos: 5, aliens: low }), N).terminated).toBe(true)
    const last = emptyMap()
    last[45] = 1
    const bullets = emptyMap()
    bullets[55] = 1
    const cleared = play(spaceInvadersMinAtar, game({ aliens: last, friendlyBullets: bullets }), N)
    expect(cleared.reward).toBe(1)
    expect(cleared.state.game.aliens).toEqual(base.aliens)
    expect([cleared.state.game.enemyMoveInterval, cleared.state.game.rampIndex]).toEqual([11, 1])
  })

  it('observes the aliens on their direction channel', () => {
    const grid = minatarGrid(play(spaceInvadersMinAtar, game({}), N).observation, 6)
    expect(grid[0]?.[2]?.slice(1, 4)).toEqual([true, true, false])
  })
})

describe('Seaquest', () => {
  const base = startOf(seaquestMinAtar)
  const game = (patch: Partial<SeaquestState>): SeaquestState => ({
    ...base,
    friendlyBullets: [],
    enemyBullets: [],
    fish: [],
    subs: [],
    divers: [],
    ...patch,
  })

  it('uses oxygen while submerged and ends when it runs out', () => {
    const r = play(seaquestMinAtar, game({ subY: 4, surface: false, oxygen: 50 }), N)
    expect(r.state.game.oxygen).toBe(49)
    expect(play(seaquestMinAtar, game({ subY: 4, surface: false, oxygen: 0 }), N).terminated).toBe(false)
    expect(play(seaquestMinAtar, game({ subY: 4, surface: false, oxygen: -1 }), N).terminated).toBe(true)
  })

  it('ends on surfacing with no divers, and uses one diver to refill the oxygen otherwise', () => {
    expect(play(seaquestMinAtar, game({ subY: 1, surface: false, diverCount: 0 }), U).terminated).toBe(true)
    const r = play(seaquestMinAtar, game({ subY: 1, surface: false, diverCount: 3, oxygen: 80 }), U)
    expect(r.terminated).toBe(false)
    expect(r.reward).toBe(0)
    expect([r.state.game.diverCount, r.state.game.oxygen, r.state.game.surface]).toEqual([2, 200, true])
    // Each surfacing ramps the difficulty: enemies spawn sooner, and every second time move faster.
    expect([r.state.game.enemySpawnSpeed, r.state.game.moveSpeed, r.state.game.rampIndex]).toEqual([19, 5, 1])
    const again = play(seaquestMinAtar, { ...r.state.game, subY: 1, surface: false }, U).state.game
    expect([again.enemySpawnSpeed, again.moveSpeed, again.rampIndex]).toEqual([18, 4, 2])
    // Staying on the surface does nothing more.
    expect(play(seaquestMinAtar, r.state.game, N).state.game.diverCount).toBe(2)
  })

  it('pays the remaining oxygen in tenths for surfacing with six divers', () => {
    const r = play(seaquestMinAtar, game({ subY: 1, surface: false, diverCount: 6, oxygen: 130 }), U)
    // Surfacing costs no oxygen: floor(130 * 10 / 200) = 6.
    expect(r.reward).toBe(6)
    expect([r.state.game.diverCount, r.state.game.oxygen]).toEqual([0, 200])
  })

  it('picks up divers it meets, at most six', () => {
    const diver = { x: 4, y: 3, right: true, timer: 2 }
    const r = play(seaquestMinAtar, game({ subX: 4, subY: 3, surface: false, divers: [diver] }), N)
    expect([r.state.game.diverCount, r.state.game.divers.length]).toEqual([1, 0])
    const full = play(seaquestMinAtar, game({ subX: 4, subY: 3, surface: false, diverCount: 6, divers: [diver] }), N)
    expect([full.state.game.diverCount, full.state.game.divers.length]).toEqual([6, 1])
    // Divers drift on their own timer and leave at the edges.
    const drift = play(
      seaquestMinAtar,
      game({ subY: 5, surface: false, divers: [{ x: 9, y: 2, right: true, timer: 0 }] }),
      N,
    )
    expect(drift.state.game.divers).toEqual([])
  })

  it('shoots fish and subs, and enemy subs fire back', () => {
    const fish = { x: 6, y: 4, right: false, timer: 3 }
    const r = play(seaquestMinAtar, game({ subX: 4, subY: 4, subOr: true, surface: false, fish: [fish] }), F)
    // The bullet starts on the sub, flies one cell to x = 5, then the fish (on its timer) waits: no hit yet.
    expect(r.reward).toBe(0)
    const hit = play(seaquestMinAtar, r.state.game, N)
    expect(hit.reward).toBe(1)
    expect(hit.state.game.fish).toEqual([])
    const sub = { x: 2, y: 6, right: true, timer: 4, shot: 0 }
    const fired = play(seaquestMinAtar, game({ subX: 8, subY: 3, surface: false, subs: [sub] }), N).state.game
    expect(fired.enemyBullets).toEqual([{ x: 3, y: 6, right: true }])
    expect(fired.subs[0]?.shot).toBe(10)
  })

  it('ends on touching an enemy or an enemy bullet', () => {
    const fish = { x: 5, y: 4, right: false, timer: 0 }
    expect(play(seaquestMinAtar, game({ subX: 4, subY: 4, surface: false, fish: [fish] }), N).terminated).toBe(true)
    const bullet = { x: 3, y: 4, right: true }
    expect(
      play(seaquestMinAtar, game({ subX: 4, subY: 4, surface: false, enemyBullets: [bullet] }), N).terminated,
    ).toBe(true)
  })

  it('spawns enemies (subs one time in three) at the edges of free lanes, and divers on their own timer', () => {
    let subs = 0
    let spawned = 0
    for (let i = 0; i < 300; i++) {
      const g = play(seaquestMinAtar, game({ enemySpawnTimer: 0, diverSpawnTimer: 0 }), N, key(i)).state.game
      const enemies = [...g.subs, ...g.fish]
      expect(g.divers).toHaveLength(1)
      expect(g.divers[0]?.y).toBeGreaterThanOrEqual(1)
      expect(g.divers[0]?.y).toBeLessThanOrEqual(8)
      expect(enemies).toHaveLength(1)
      const e = enemies[0]
      if (e === undefined) throw new Error('no spawn')
      expect(e.x).toBe(e.right ? 0 : 9)
      spawned++
      subs += g.subs.length
      expect(g.enemySpawnTimer).toBe(19)
    }
    expect(subs / spawned).toBeGreaterThan(0.25)
    expect(subs / spawned).toBeLessThan(0.42)
    // A lane with an enemy gets no second one.
    const lanes = Array.from({ length: 8 }, (_, i) => ({ x: 0, y: i + 1, right: true, timer: 9 }))
    const g = play(seaquestMinAtar, game({ enemySpawnTimer: 0, fish: lanes }), N, key(3)).state.game
    expect(g.fish).toHaveLength(8)
    expect(g.subs).toHaveLength(0)
  })

  it('observes the sub, its oxygen and diver gauges', () => {
    const r = play(
      seaquestMinAtar,
      game({ subX: 3, subY: 2, subOr: true, surface: false, oxygen: 101, diverCount: 2 }),
      N,
    )
    const grid = minatarGrid(r.observation, 10)
    expect(grid[2]?.[3]?.[0]).toBe(true)
    expect(grid[2]?.[2]?.[1]).toBe(true)
    expect(grid[9]?.filter((c) => c[7]).length).toBe(5)
    expect(grid[9]?.filter((c) => c[8]).length).toBe(2)
  })
})
