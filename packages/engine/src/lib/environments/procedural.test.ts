import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../random/index.ts'
import { make, mdpFromEnv, runEpisode, valueIteration } from '../rl/index.ts'
import { contains } from '../space/index.ts'
import {
  braidMaze,
  deadEnds,
  generateMaze,
  gridDistances,
  keyDoor,
  levelSplit,
  type Maze,
  type MazeAlgorithm,
  makeProcedural,
  type ProceduralLevel,
  type ProceduralState,
  proceduralLevel,
  proceduralMaze,
} from './procedural.ts'

const ALGORITHMS: readonly MazeAlgorithm[] = ['backtracker', 'prim', 'kruskal']

/** Open walls between neighbouring cells. */
function passages(maze: Maze): number {
  let n = 0
  for (const [y, row] of maze.grid.entries()) {
    for (let x = 0; x < row.length; x++) if (row[x] === '.' && (y % 2 === 1) !== (x % 2 === 1)) n++
  }
  return n
}

function connected(maze: Maze): boolean {
  const d = gridDistances(maze.grid, [1, 1])
  const w = maze.grid[0]?.length ?? 0
  for (let r = 0; r < maze.rows; r++)
    for (let c = 0; c < maze.cols; c++) if ((d[(2 * r + 1) * w + 2 * c + 1] ?? -1) < 0) return false
  return true
}

/** Shortest number of moves to the goal through keys and doors, by breadth-first search over (position, keys). */
function shortestSolution(level: ProceduralLevel): number {
  const map = level.map
  const w = map[0]?.length ?? 0
  const start = level.start[0] * w + level.start[1]
  const seen = new Map<string, number>([[`${start},0`, 0]])
  const queue: [number, number][] = [[start, 0]]
  for (let q = 0; q < queue.length; q++) {
    const [pos, keys] = queue[q] ?? [0, 0]
    const d = seen.get(`${pos},${keys}`) ?? 0
    for (const [dr, dc] of [
      [-1, 0],
      [0, 1],
      [1, 0],
      [0, -1],
    ] as const) {
      const r = Math.floor(pos / w) + dr
      const c = (pos % w) + dc
      const ch = map[r]?.[c] ?? '#'
      if (ch === '#') continue
      if (ch >= 'A' && ch <= 'D' && (keys & (1 << (ch.charCodeAt(0) - 65))) === 0) continue
      if (ch === 'G') return d + 1
      const k2 = ch >= 'a' && ch <= 'd' ? keys | (1 << (ch.charCodeAt(0) - 97)) : keys
      const id = `${r * w + c},${k2}`
      if (seen.has(id)) continue
      seen.set(id, d + 1)
      queue.push([r * w + c, k2])
    }
  }
  return -1
}

describe('maze generation', () => {
  it('carves perfect mazes (spanning trees: connected, cells - 1 passages) with every algorithm', () => {
    for (const algorithm of ALGORITHMS) {
      for (let seed = 0; seed < 15; seed++) {
        const rows = 2 + (seed % 5)
        const cols = 3 + (seed % 4)
        const maze = generateMaze(key(seed), { rows, cols, algorithm })
        expect(maze.grid).toHaveLength(2 * rows + 1)
        expect(connected(maze)).toBe(true)
        expect(passages(maze)).toBe(rows * cols - 1)
        // The outer wall is intact.
        expect(maze.grid[0]).toBe('#'.repeat(2 * cols + 1))
      }
    }
  })

  it('is a pure function of the key, and the algorithms differ in texture', () => {
    expect(generateMaze(key(4), { algorithm: 'prim' })).toEqual(generateMaze(key(4), { algorithm: 'prim' }))
    expect(generateMaze(key(4)).grid).not.toEqual(generateMaze(key(5)).grid)
    // Prim's mazes have many more dead ends than the backtracker's long corridors.
    let prim = 0
    let dfs = 0
    for (let s = 0; s < 10; s++) {
      prim += deadEnds(generateMaze(key(s), { rows: 8, cols: 8, algorithm: 'prim' })).length
      dfs += deadEnds(generateMaze(key(s), { rows: 8, cols: 8 })).length
    }
    expect(prim).toBeGreaterThan(dfs)
  })

  it('braids loops into mazes while keeping them connected', () => {
    for (let seed = 0; seed < 10; seed++) {
      const perfect = generateMaze(key(seed), { rows: 6, cols: 6 })
      const braided = braidMaze(perfect, key(100 + seed), 1)
      expect(connected(braided)).toBe(true)
      expect(deadEnds(braided)).toEqual([])
      expect(passages(braided)).toBeGreaterThan(35)
      const half = generateMaze(key(seed), { rows: 6, cols: 6, braid: 0.5 })
      expect(passages(half)).toBeGreaterThanOrEqual(35)
    }
    expect(() => generateMaze(key(0), { rows: 0 })).toThrow(/positive/)
  })
})

describe('key-door levels', () => {
  it('need every key for its door, and every key is reachable before its door', () => {
    for (const keys of [1, 2, 3]) {
      for (let level = 0; level < 20; level++) {
        for (const braid of [0, 0.7]) {
          const L = proceduralLevel(level, { rows: 6, cols: 6, keys, braid })
          expect(L.keys).toHaveLength(keys)
          expect(L.doors).toHaveLength(keys)
          for (let i = 0; i < keys; i++) {
            // With doors i and later shut, the goal and the later keys are out of reach, but key i is not.
            const shut = L.map.map((row) =>
              row
                .split('')
                .map((ch) => (ch >= 'A' && ch <= 'D' && ch.charCodeAt(0) - 65 >= i ? '#' : ch))
                .join(''),
            )
            const d = gridDistances(shut, L.start)
            const w = shut[0]?.length ?? 0
            const [kr, kc] = L.keys[i] ?? [0, 0]
            expect(d[kr * w + kc]).toBeGreaterThan(0)
            expect(d[L.goal[0] * w + L.goal[1]]).toBe(-1)
          }
          expect(shortestSolution(L)).toBeGreaterThan(0)
        }
      }
    }
  })

  it('opens a door only with its key', () => {
    const L = proceduralLevel(0, { rows: 4, cols: 4, keys: 1 })
    const env = makeProcedural({ rows: 4, cols: 4, keys: 1, numLevels: 1 }, 'KeyDoor-v0')
    const w = L.map[0]?.length ?? 0
    const [dr, dc] = L.doors[0] ?? [0, 0]
    // Stand next to the door on the start side and step into it, without and with the key.
    const side = (
      [
        [-1, 0, 2],
        [0, 1, 3],
        [1, 0, 0],
        [0, -1, 1],
      ] as const
    ).find(([r, c]) => {
      const d = gridDistances(
        L.map.map((row) => row.replace(/[A-D]/g, '#')),
        L.start,
      )
      return (d[(dr + r) * w + dc + c] ?? -1) >= 0
    })
    if (side === undefined) throw new Error('no approach to the door')
    const at = (dr + side[0]) * w + dc + side[1]
    const locked = env.step({ level: 0, pos: at, keys: 0 }, side[2], key(0))
    expect(locked.state.pos).toBe(at)
    const open = env.step({ level: 0, pos: at, keys: 1 }, side[2], key(0))
    expect(open.state.pos).toBe(dr * w + dc)
  })
})

describe('procedural environments', () => {
  it('solve every level by value iteration, in the fewest moves', () => {
    for (const [def, keys] of [
      [proceduralMaze, 0],
      [keyDoor, 1],
      [keyDoor, 2],
    ] as const) {
      for (let level = 0; level < 6; level++) {
        const options = { rows: 4, cols: 4, keys, numLevels: 1, startLevel: level, observation: 'index' }
        const env = def.create(options)
        const L = env.level(level)
        const moves = shortestSolution(L)
        const sol = valueIteration(mdpFromEnv(env, 0.9))
        const start = env.reset(key(0))
        expect(start.info['level']).toBe(level)
        expect(sol.V[start.observation as number]).toBeCloseTo(0.9 ** (moves - 1), 10)
        const episode = runEpisode(make(def, options), (o) => sol.policy[o as number] ?? 0, key(1))
        expect(episode.return).toBe(1)
        expect(episode.length).toBe(moves)
      }
    }
  })

  it('model a whole level range, whose value is the mean over its levels', () => {
    const options = { rows: 3, cols: 3, keys: 1, numLevels: 3, startLevel: 4, observation: 'index' }
    const env = keyDoor.create(options)
    const sol = valueIteration(mdpFromEnv(env, 0.9))
    const expected = [4, 5, 6].map((l) => 0.9 ** (shortestSolution(env.level(l)) - 1))
    const starts = env.model?.initial.map((i) => sol.V[env.model?.stateIndex(i.state) ?? 0] ?? 0) ?? []
    expect(starts).toHaveLength(3)
    for (const [i, v] of starts.entries()) expect(v).toBeCloseTo(expected[i] ?? 0, 10)
  })

  it('draw training and test levels from disjoint, reproducible ranges', () => {
    const split = levelSplit(20, 10, 100)
    const train = proceduralMaze.create({ rows: 3, cols: 3, ...split.train })
    const test = proceduralMaze.create({ rows: 3, cols: 3, ...split.test })
    const seen = (env: typeof train): Set<number> => {
      const out = new Set<number>()
      for (let i = 0; i < 300; i++) out.add(env.reset(foldIn(key(9), i)).info['level'] as number)
      return out
    }
    const a = seen(train)
    const b = seen(test)
    expect(a).toEqual(new Set(Array.from({ length: 20 }, (_, i) => 100 + i)))
    expect(b).toEqual(new Set(Array.from({ length: 10 }, (_, i) => 120 + i)))
    expect(seen(train)).toEqual(a)
    // The same level is the same maze in every environment; levels and level seeds change it.
    const other = makeProcedural({ rows: 3, cols: 3 })
    expect(other.level(105)).toEqual(train.level(105))
    expect(train.level(105).map).not.toEqual(train.level(106).map)
    expect(makeProcedural({ rows: 3, cols: 3, levelSeed: 7 }).level(105).map).not.toEqual(train.level(105).map)
    // Unlimited levels draw from the whole range.
    expect(proceduralMaze.create().reset(key(3)).info['level']).toBeGreaterThan(1000)
  })

  it('observe full grids, local views or state indices, with held keys', () => {
    const grid = keyDoor.create({ rows: 3, cols: 3, keys: 1, numLevels: 1 })
    const r = grid.reset(key(0))
    // Channels: wall, agent, goal, key_a, door_A; then one held-key flag.
    expect(grid.channelNames).toEqual(['wall', 'agent', 'goal', 'key_a', 'door_A'])
    expect(r.observation).toHaveLength(7 * 7 * 5 + 1)
    expect(contains(grid.observationSpace, r.observation as readonly number[])).toBe(true)
    const obs = r.observation as readonly number[]
    const s = r.state as ProceduralState
    expect(obs[s.pos * 5 + 1]).toBe(1)
    expect(obs.filter((_, i) => i % 5 === 1 && i < 245).reduce((x, y) => x + y, 0)).toBe(1)
    const view = keyDoor.create({ rows: 3, cols: 3, keys: 1, numLevels: 1, observation: 'view', viewRadius: 1 })
    const v = view.reset(key(0)).observation as readonly number[]
    expect(v).toHaveLength(9 * 4 + 1)
    // The centre of the view is the agent's open cell; a corner beyond the grid edge reads as wall.
    expect(v[4 * 4]).toBe(0)
    const L = view.level(0)
    const w = L.map[0]?.length ?? 0
    const [kr, kc] = L.keys[0] ?? [0, 0]
    const held = view.observe?.({ level: 0, pos: kr * w + kc, keys: 1 }) as readonly number[]
    expect(held[held.length - 1]).toBe(1)
    expect(held.filter((_, i) => i % 4 === 2 && i < 36)).toEqual(new Array<number>(9).fill(0))
    const index = keyDoor.create({ rows: 3, cols: 3, numLevels: 2, observation: 'index' })
    expect(index.observationSpace).toMatchObject({ kind: 'discrete', n: 2 * 49 * 2 })
    expect(() => keyDoor.create({ observation: 'index' })).toThrow(/finite numLevels/)
  })

  it('render the level with its keys and doors', () => {
    const env = keyDoor.create({ rows: 3, cols: 3, keys: 2, numLevels: 1 })
    const scene = env.render(env.reset(key(0)).state)
    expect(scene.title).toBe('KeyDoor-v0 level 0')
    expect(scene.children.filter((n) => n.title?.startsWith('door'))).toHaveLength(2)
    expect(scene.children.filter((n) => n.title?.startsWith('key'))).toHaveLength(2)
  })
})
