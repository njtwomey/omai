/**
 * Small worlds for path search: hand-drawn ASCII maps with walls, water and forest of different costs, which show where
 * each search shines or fails, and generated mazes, both as grid worlds.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import type { Key } from '../../lib/random/index.ts'
import {
  braid,
  fromAscii,
  generateMaze,
  type MazeAlgorithmName,
  type World,
  wallsToBlocks,
} from '../../lib/worlds/index.ts'

/** The hand-drawn search maps, as ASCII (`S` start, `G` goal, `#` wall, `~` water costing 5, `%` forest costing 2). */
export const SEARCH_MAPS: Readonly<Record<'open' | 'detour' | 'river' | 'rooms', readonly string[]>> = {
  // An empty room: breadth-first search floods it, A* walks almost straight to the goal.
  open: [
    '................',
    '................',
    '..S.............',
    '................',
    '.............G..',
    '................',
    '................',
  ],
  // A wall in the way, cupped towards the start: greedy best-first walks into the cup, A* pays for its optimality.
  detour: [
    '..................',
    '..................',
    '......#####.......',
    '..........#.......',
    '...S......#....G..',
    '..........#.......',
    '......#####.......',
    '..................',
  ],
  // Water costs 5 to enter: breadth-first search wades straight across, uniform-cost search walks to the bridge.
  river: [
    '.........~~~........',
    '.........~~~........',
    '..S......~~~......G.',
    '.........~~~........',
    '.........~~~........',
    '....................',
    '.........~~~........',
    '.........~~~........',
  ],
  // Four rooms joined by doors.
  rooms: [
    '.......#........',
    '.S.....#........',
    '...............',
    '.......#........',
    '####.#####.#####',
    '.......#........',
    '.......#........',
    '...............G',
    '.......#........',
  ].map((r) => r.padEnd(16, '.')),
}

/** The names of the hand-drawn maps. */
export type SearchMapName = keyof typeof SEARCH_MAPS

/**
 * A hand-drawn search map as a world.
 *
 * @param name - The map: `open`, `detour`, `river` or `rooms`.
 * @returns The world, with its start and goal.
 */
export function searchMap(name: SearchMapName): World {
  const rows = SEARCH_MAPS[name]
  if (rows === undefined) throw new WhiteboxError('value', `searchMap: unknown map '${name}'`)
  return fromAscii(rows, { name })
}

/** Options of {@link mazeWorld}. */
export interface MazeWorldOptions {
  /** Maze cells per column and per row (default 8 by 12); the blocked-cell world is twice that plus one. */
  readonly rows?: number
  readonly cols?: number
  readonly algorithm?: MazeAlgorithmName
  /** The probability of opening each dead end (default 0: a perfect maze, one path between any two cells). */
  readonly braid?: number
}

/**
 * A generated maze as a world of blocked cells (walls become cells), started at the top left and ending at the bottom
 * right.
 *
 * @param key - The key that carves it.
 * @param options - Size, algorithm and braiding.
 * @returns The world.
 */
export function mazeWorld(key: Key, options: MazeWorldOptions = {}): World {
  const rows = options.rows ?? 8
  const cols = options.cols ?? 12
  const maze = generateMaze(key, { rows, cols, algorithm: options.algorithm ?? 'backtracker' })
  const loose = (options.braid ?? 0) > 0 ? braid(maze, key, options.braid) : maze
  const blocks = wallsToBlocks(loose)
  const width = 2 * cols + 1
  return { ...blocks, starts: [width + 1], goals: [(2 * rows - 1) * width + 2 * cols - 1], name: 'maze' }
}
