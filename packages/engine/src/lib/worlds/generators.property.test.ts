import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { iterate } from '../trace/index.ts'
import {
  braid,
  bridges,
  deadEnds,
  generateMaze,
  isPerfectMaze,
  type MazeAlgorithmName,
  mazeAlgorithm,
  regions,
  world,
} from './index.ts'

const ANY: readonly MazeAlgorithmName[] = ['backtracker', 'prim', 'kruskal', 'wilson', 'aldousBroder']
const SQUARE: readonly MazeAlgorithmName[] = [...ANY, 'eller', 'binaryTree', 'sidewinder', 'division']

describe('maze generators (properties)', () => {
  it('every algorithm carves a perfect maze: a spanning tree of the grid, connected, every passage a bridge', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SQUARE),
        fc.integer({ min: 1, max: 7 }),
        fc.integer({ min: 1, max: 7 }),
        fc.nat(10_000),
        fc.constantFrom('NE', 'NW', 'SE', 'SW'),
        (algorithm, rows, cols, seed, bias) => {
          const w = generateMaze(key(seed), { rows, cols, algorithm, bias: bias as 'NE' })
          expect(isPerfectMaze(w)).toBe(true)
          expect(regions(w).count).toBe(1)
          expect(bridges(w)).toHaveLength(w.n - 1)
          // The same key gives the same maze.
          expect(Array.from(generateMaze(key(seed), { rows, cols, algorithm, bias: bias as 'NE' }).walls)).toEqual(
            Array.from(w.walls),
          )
        },
      ),
      { numRuns: 150 },
    )
  })

  it('the topology-free algorithms carve perfect mazes on hexagonal grids and graphs', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...ANY),
        fc.integer({ min: 1, max: 6 }),
        fc.integer({ min: 1, max: 6 }),
        fc.nat(10_000),
        (algorithm, rows, cols, seed) => {
          expect(isPerfectMaze(generateMaze(key(seed), { rows, cols, algorithm, topology: 'hex' }))).toBe(true)
          // A wheel graph: a hub joined to every node of a cycle.
          const m = rows + cols + 1
          const neighbours = Array.from({ length: m + 1 }, (_, v) =>
            v === 0 ? Array.from({ length: m }, (_, i) => i + 1) : [0, (v % m) + 1, ((v - 2 + m) % m) + 1],
          )
          const base = world({ topology: { kind: 'graph', n: m + 1, neighbours } })
          expect(isPerfectMaze(generateMaze(key(seed), { base, algorithm }))).toBe(true)
        },
      ),
      { numRuns: 80 },
    )
  })

  it('traced runs only ever open passages (or, for division, only add walls), and finish', () => {
    fc.assert(
      fc.property(fc.constantFrom(...SQUARE), fc.nat(1000), (algorithm, seed) => {
        let previous: Uint32Array | undefined
        let steps = 0
        for (const { state } of iterate(mazeAlgorithm({ rows: 4, cols: 5, algorithm }), { seed, maxSteps: 5000 })) {
          if (previous !== undefined) {
            for (let c = 0; c < state.walls.length; c++) {
              const before = previous[c] ?? 0
              const after = state.walls[c] ?? 0
              if (algorithm === 'division') expect(after & before).toBe(before)
              else expect(after & before).toBe(after)
            }
          }
          previous = state.walls
          steps++
          if (state.done) break
        }
        expect(steps).toBeLessThan(5000)
      }),
      { numRuns: 60 },
    )
  })

  it('braiding removes dead ends and keeps the maze connected', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SQUARE),
        fc.nat(1000),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (algorithm, seed, p) => {
          const m = generateMaze(key(seed), { rows: 5, cols: 6, algorithm })
          const b = braid(m, key(seed + 1), p)
          expect(regions(b).count).toBe(1)
          expect(deadEnds(b).length).toBeLessThanOrEqual(deadEnds(m).length)
          expect(deadEnds(braid(m, key(seed + 1), 1))).toEqual([])
        },
      ),
      { numRuns: 60 },
    )
  })
})
