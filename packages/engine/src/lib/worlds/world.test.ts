import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import {
  cellAtPoint,
  cellCentre,
  cellCoords,
  cellCorners,
  cellIndex,
  cliffWalkingWorld,
  destination,
  exits,
  fromAscii,
  greedyPath,
  hasWall,
  neighbours,
  passable,
  setTerrain,
  setWalls,
  terrainOf,
  toAscii,
  toScene,
  toStructure,
  wallsToBlocks,
  world,
  worldSize,
} from './index.ts'

const throwsValue = (f: () => unknown): void => {
  try {
    f()
  } catch (e) {
    expect(isOmaiError(e, 'value')).toBe(true)
    return
  }
  throw new Error('expected a value error')
}

describe('world construction', () => {
  it('validates sizes, terrains, cells and walls', () => {
    throwsValue(() => world({ topology: { kind: 'square', rows: 0, cols: 2 } }))
    throwsValue(() => world({ topology: { kind: 'square', rows: 2, cols: 2 }, terrain: ['floor'] }))
    throwsValue(() => world({ topology: { kind: 'square', rows: 1, cols: 2 }, terrain: ['floor', 'lava?'] }))
    throwsValue(() => world({ topology: { kind: 'square', rows: 2, cols: 2 }, walls: [[0, 3]] }))
    throwsValue(() => world({ topology: { kind: 'square', rows: 2, cols: 2 }, starts: [4] }))
    throwsValue(() => world({ topology: { kind: 'graph', n: 2, neighbours: [[1], []] } }))
    throwsValue(() => setTerrain(fromAscii(['..']), [0], 'nothing'))
  })

  it('keeps walls symmetric and reopens them', () => {
    const w = setWalls(fromAscii(['...', '...']), [
      [1, 4],
      [0, 1],
    ])
    expect([hasWall(w, 1, 2), hasWall(w, 4, 0), hasWall(w, 0, 1), hasWall(w, 1, 3)]).toEqual([true, true, true, true])
    const open = setWalls(w, [[4, 1]], false)
    expect([hasWall(open, 1, 2), hasWall(open, 4, 0)]).toEqual([false, false])
  })
})

describe('coordinates', () => {
  it('converts between cells, rows and columns, and points on squares', () => {
    const w = fromAscii(['....', '....', '....'])
    for (let c = 0; c < w.n; c++) {
      const [r, col] = cellCoords(w, c)
      expect(cellIndex(w, r, col)).toBe(c)
      const [x, y] = cellCentre(w, c)
      expect(cellAtPoint(w, x, y)).toBe(c)
    }
    expect([cellIndex(w, -1, 0), cellAtPoint(w, 4.2, 0), cellAtPoint(w, -0.1, 1)]).toEqual([-1, -1, -1])
    expect(worldSize(w)).toEqual([4, 3])
  })

  it('places hexagons one unit apart, inverts centres and gives six corners', () => {
    const w = fromAscii(['....', '....', '....'], { topology: 'hex' })
    for (let c = 0; c < w.n; c++) {
      const [x, y] = cellCentre(w, c)
      expect(cellAtPoint(w, x, y)).toBe(c)
      expect(cellCorners(w, c)).toHaveLength(6)
      for (const nb of neighbours(w, c)) {
        const [x2, y2] = cellCentre(w, nb)
        expect(Math.hypot(x2 - x, y2 - y)).toBeCloseTo(1, 12)
      }
    }
    // Interior hexagons have six neighbours; the opposite of each side faces back.
    expect(neighbours(w, 5)).toHaveLength(6)
    expect(neighbours(w, 6).sort((a, b) => a - b)).toEqual([2, 3, 5, 7, 10, 11])
  })

  it('throws for row-based queries on graph worlds', () => {
    const g = world({ topology: { kind: 'graph', n: 3, neighbours: [[1], [0, 2], [1]] } })
    throwsValue(() => cellCoords(g, 0))
    throwsValue(() => cellIndex(g, 0, 0))
    throwsValue(() => toAscii(g))
    expect(neighbours(g, 1)).toEqual([0, 2])
    expect(cellAtPoint(g, 0, 0)).toBe(-1)
  })
})

describe('moves', () => {
  it('respects walls, terrain, one-way cells and blocked cells', () => {
    const w = fromAscii(['.>.', '.#.'])
    expect(destination(w, 0, 1)).toBe(1)
    expect(destination(w, 2, 3)).toBe(-1)
    expect(destination(w, 0, 2)).toBe(3)
    expect(destination(w, 1, 2)).toBe(-1)
    expect(destination(w, 0, 1, { blocked: (c) => c === 1 })).toBe(-1)
    expect(destination(setWalls(w, [[0, 3]]), 0, 2)).toBe(-1)
    expect(destination(w, 0, 7)).toBe(-1)
  })

  it('lists exits under 8-connectivity with the corner rules', () => {
    const w = fromAscii(['...', '.#.', '...'])
    expect(exits(w, 0, { connectivity: 8 })).toEqual([
      [1, 1],
      [2, 3],
    ])
    expect(neighbours(w, 0, { connectivity: 8, corners: 'loose' })).toEqual([1, 3])
    const open = fromAscii(['...', '...', '...'])
    expect(neighbours(open, 0, { connectivity: 8 })).toEqual([1, 3, 4])
  })

  it('reports passability and terrain', () => {
    const w = fromAscii(['.#~H'])
    expect([0, 1, 2, 3, 4, -1].map((c) => passable(w, c))).toEqual([true, false, true, true, false, false])
    expect([terrainOf(w, 2).name, terrainOf(w, 3).hazard]).toEqual(['water', true])
    throwsValue(() => terrainOf(w, 9))
  })
})

describe('blocks, structures and scenes', () => {
  it('turns thin walls into blocking cells', () => {
    const rows = ['#######', '#S..#.#', '#.###.#', '#....G#', '#######']
    const thin = fromAscii(rows, { format: 'walls' })
    expect(thin.n).toBe(6)
    expect(toAscii(wallsToBlocks(thin))).toEqual(rows)
    throwsValue(() => wallsToBlocks(fromAscii(['..'], { topology: 'hex' })))
  })

  it('describes one-way moves as directed edges', () => {
    const s = toStructure(fromAscii(['.>.']), { title: 'conveyor' })
    expect(s.title).toBe('conveyor')
    expect(s.edges).toEqual([
      { from: 'c0', to: 'c1', directed: false },
      { from: 'c1', to: 'c2', directed: true },
    ])
  })

  it('draws squares, hexagons and graphs with walls, items, values and paths', () => {
    const w = fromAscii(['#####', '#S.K#', '#.###', '#D.$#', '#####'], {
      format: 'walls',
      legend: { $: { terrain: 'floor', goal: true, item: { kind: 'pickup' } } },
    })
    expect([w.items.map((it) => it.kind), w.goals]).toEqual([['key', 'door', 'pickup'], [3]])
    const s = toScene(w, { values: [1, 2, 3, 0], path: [0, 2], agent: { x: 0.5, y: 0.5, heading: 0 }, title: 'maze' })
    expect(s.title).toBe('maze')
    const walls = s.children.find((n) => n.kind === 'group' && n.id === 'walls')
    expect(walls?.kind === 'group' && walls.children.filter((n) => n.kind === 'line').length).toBeGreaterThan(0)
    expect(s.children.filter((n) => n.kind === 'circle' && n.fill === 'accent')).toHaveLength(4)
    const hex = toScene(fromAscii(['..', '..'], { topology: 'hex' }))
    expect(hex.children.filter((n) => n.kind === 'path')).toHaveLength(4)
    const g = toScene(world({ topology: { kind: 'graph', n: 3, neighbours: [[1], [0, 2], [1]] }, walls: [[0, 1]] }))
    expect(g.children.filter((n) => n.kind === 'circle')).toHaveLength(3)
    expect(g.children.filter((n) => n.kind === 'line')).toHaveLength(2)
  })

  it('shades values by magnitude, ignores a NaN when scaling, and takes a paint, a ramp and a fixed range', () => {
    const w = fromAscii(['S.G'])
    const discs = (options: Parameters<typeof toScene>[1]): { opacity?: number; fill?: string }[] =>
      toScene(w, options).children.flatMap((n) => (n.kind === 'circle' && n.opacity !== undefined ? [n] : []))
    const plain = discs({ values: [0, 1, 2] })
    expect(plain.map((d) => d.opacity)).toEqual([0, 0.375, 0.75])
    // One NaN no longer switches the shading off.
    expect(discs({ values: [Number.NaN, 1, 2] }).map((d) => d.opacity)).toEqual([0.375, 0.75])
    // A paint, and a ramp (constant opacity, colour by the share of the range).
    expect(discs({ values: [1, 2, 4], valueStyle: { fill: 'tab0', maxOpacity: 0.5 } }).map((d) => d.fill)).toEqual([
      'tab0',
      'tab0',
      'tab0',
    ])
    const ramp = discs({
      values: [1, 3],
      valueStyle: { fill: (_v, share) => (share > 0.5 ? 'tab3' : 'tab0'), range: [0, 4], maxOpacity: 1 },
    })
    expect(ramp.map((d) => [d.fill, d.opacity])).toEqual([
      ['tab0', 1],
      ['tab3', 1],
    ])
  })

  it('follows the greedy path up the values to a goal, a local maximum or the step limit', () => {
    const w = fromAscii(['S..G'])
    expect(greedyPath(w, [0, 1, 2, 3], 0)).toEqual([0, 1, 2, 3])
    expect(greedyPath(w, [0, 1, 0, 3], 0)).toEqual([0, 1])
    expect(greedyPath(w, [0, 1, 2, 3], 0, { maxSteps: 2 })).toEqual([0, 1, 2])
    expect(greedyPath(w, [0, 1, 2, 3], 3)).toEqual([3])
  })

  it('does not step onto hazardous cells when walking up the values, however high a placeholder value there is', () => {
    const w = cliffWalkingWorld().world
    const goal = w.goals[0] ?? 0
    // Negative number of moves to the goal over the open cells (breadth first, never through the cliff); the cliff cells
    // hold 0, the highest value in the table.
    const moves = new Map<number, number>([[goal, 0]])
    for (const cell of moves.keys())
      for (const next of neighbours(w, cell))
        if (!moves.has(next) && !terrainOf(w, next).hazard) moves.set(next, (moves.get(cell) ?? 0) + 1)
    const values = Array.from({ length: w.n }, (_, cell) => (terrainOf(w, cell).hazard ? 0 : -(moves.get(cell) ?? 99)))
    const path = greedyPath(w, values, w.starts[0] ?? 0)
    expect(path.some((cell) => terrainOf(w, cell).hazard)).toBe(false)
    expect(path.at(-1)).toBe(goal)
    // Walking anywhere, as before, falls into the cliff at once.
    const rash = greedyPath(w, values, w.starts[0] ?? 0, { skip: () => false })
    expect(terrainOf(w, rash[1] ?? 0).hazard).toBe(true)
  })

  it('names the path, the walls as one group, so a scene can be restyled by id', () => {
    const w = fromAscii(['S#G', '...'], { format: 'cells' } as never)
    const scene = toScene(w, { path: [0, 3, 4, 5, 2] })
    const ids = scene.children.map((n) => n.id)
    expect(ids).toContain('path')
    const walled = toScene(fromAscii(['#####', '#S#G#', '#####'], { format: 'walls' }))
    expect(walled.children.some((n) => n.kind === 'group' && n.id === 'walls')).toBe(true)
  })
})
