import { describe, expect, it } from 'vitest'
import { isMdmlError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { policies, recorder } from '../trace/index.ts'
import {
  braid,
  deadEnds,
  fromAscii,
  generateMaze,
  isPerfectMaze,
  mazeAlgorithm,
  setWalls,
  toAscii,
  world,
} from './index.ts'

describe('maze generators', () => {
  it('refuse square-only algorithms elsewhere and keep a base world’s contents', () => {
    for (const algorithm of ['eller', 'binaryTree', 'sidewinder', 'division'] as const) {
      try {
        mazeAlgorithm({ algorithm, topology: 'hex' })
        expect.unreachable()
      } catch (e) {
        expect(isMdmlError(e, 'value')).toBe(true)
      }
    }
    const base = fromAscii(['S~.', '..G'])
    const m = generateMaze(key(1), { base, algorithm: 'kruskal' })
    expect([m.starts, m.goals, Array.from(m.terrain)]).toEqual([[0], [5], Array.from(base.terrain)])
    expect(isPerfectMaze(m)).toBe(true)
  })

  it('record carving as a traced run with metrics, wall tensors and frames', () => {
    const rec = recorder()
    generateMaze(key(2), {
      rows: 3,
      cols: 3,
      algorithm: 'aldousBroder',
      tracer: rec,
      policy: policies.teaching,
      runId: 'maze',
    })
    const types = new Set(rec.events.map((e) => e.type))
    expect(['run', 'metric', 'tensor', 'frame'].filter((t) => !types.has(t as never))).toEqual([])
    expect(mazeAlgorithm({ rows: 2, cols: 2, algorithm: 'binaryTree', bias: 'SW' }).config).toEqual({
      algorithm: 'binaryTree',
      n: 4,
      topology: 'square',
      bias: 'SW',
    })
  })

  it('draws the binary tree and sidewinder textures: an open corridor along one side', () => {
    const tree = toAscii(generateMaze(key(5), { rows: 4, cols: 5, algorithm: 'binaryTree' }), { format: 'walls' })
    // Carving north or east leaves the top row and the east column as straight corridors.
    expect(tree[1]).toBe('#S........#')
    expect(
      tree.filter((_, i) => i > 1 && i < tree.length - 1 && i % 2 === 0).every((row) => row[row.length - 2] === '.'),
    ).toBe(true)
    const winder = toAscii(generateMaze(key(5), { rows: 4, cols: 5, algorithm: 'sidewinder' }), { format: 'walls' })
    expect(winder[1]).toBe('#S........#')
  })

  it('detects mazes that are not perfect, and braids on any topology', () => {
    const open = world({ topology: { kind: 'square', rows: 2, cols: 2 } })
    expect(isPerfectMaze(open)).toBe(false)
    expect(
      isPerfectMaze(
        setWalls(open, [
          [0, 1],
          [2, 3],
          [0, 2],
        ]),
      ),
    ).toBe(false)
    const hex = generateMaze(key(3), { rows: 4, cols: 4, topology: 'hex', algorithm: 'backtracker' })
    expect(deadEnds(braid(hex, key(4))).length).toBe(0)
    const kept = braid(generateMaze(key(3), { rows: 4, cols: 4 }), key(4), 1, { allowed: () => false })
    expect(isPerfectMaze(kept)).toBe(true)
  })
})
