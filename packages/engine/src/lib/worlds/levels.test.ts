import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { iterate, recorder, runAlgorithm } from '../trace/index.ts'
import {
  bspDungeon,
  cave,
  caveAlgorithm,
  noiseTerrain,
  passable,
  perlinNoise,
  regions,
  terrainOf,
  valueNoise,
} from './index.ts'

describe('BSP dungeons', () => {
  it('carve disjoint rooms inside the walls, all joined by corridors, deterministically per key', () => {
    fc.assert(
      fc.property(
        fc.nat(10_000),
        fc.integer({ min: 8, max: 40 }),
        fc.integer({ min: 8, max: 40 }),
        fc.integer({ min: 3, max: 8 }),
        (seed, rows, cols, minLeaf) => {
          const d = bspDungeon(key(seed), { rows, cols, minLeaf, maxDepth: 4 })
          const w = d.world
          expect(regions(w).count).toBe(1)
          expect(d.rooms.length).toBeGreaterThanOrEqual(1)
          expect(d.corridors).toHaveLength(d.rooms.length - 1)
          const seen = new Set<number>()
          for (const room of d.rooms) {
            expect(
              room.row >= 1 && room.col >= 1 && room.row + room.height <= rows - 1 && room.col + room.width <= cols - 1,
            ).toBe(true)
            for (let r = room.row; r < room.row + room.height; r++) {
              for (let c = room.col; c < room.col + room.width; c++) {
                expect(seen.has(r * cols + c)).toBe(false)
                seen.add(r * cols + c)
                expect(passable(w, r * cols + c)).toBe(true)
              }
            }
          }
          expect([passable(w, w.starts[0] ?? -1), passable(w, w.goals[0] ?? -1)]).toEqual([true, true])
          expect(Array.from(bspDungeon(key(seed), { rows, cols, minLeaf, maxDepth: 4 }).world.terrain)).toEqual(
            Array.from(w.terrain),
          )
        },
      ),
      { numRuns: 60 },
    )
    expect(() => bspDungeon(key(0), { rows: 4, cols: 40 })).toThrow()
  })
})

describe('cellular-automaton caves', () => {
  it('apply the birth and survival rule each generation, with the border always wall', () => {
    const alg = caveAlgorithm({ rows: 12, cols: 15, fill: 0.5, generations: 3 })
    const [first, second] = Array.from(iterate(alg, { seed: 4, maxSteps: 1 }), (x) => x.state)
    if (first === undefined || second === undefined) throw new Error('no generation')
    const [s0, s1] = [first, second]
    for (let r = 0; r < 12; r++) {
      for (let c = 0; c < 15; c++) {
        let n = 0
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue
            const r2 = r + dr
            const c2 = c + dc
            n += r2 < 0 || c2 < 0 || r2 >= 12 || c2 >= 15 ? 1 : (s0.wall[r2 * 15 + c2] ?? 1)
          }
        }
        const border = r === 0 || c === 0 || r === 11 || c === 14
        const expected = border || n >= ((s0.wall[r * 15 + c] ?? 0) === 1 ? 4 : 5) ? 1 : 0
        expect(s1.wall[r * 15 + c]).toBe(expected)
      }
    }
  })

  it('are traced, reproducible, and keep one region between their start and goal', () => {
    const rec = recorder()
    const run = runAlgorithm(caveAlgorithm({ rows: 16, cols: 20 }), { seed: 1, maxSteps: 50, tracer: rec })
    expect([run.steps, run.status]).toEqual([5, 'converged'])
    for (let seed = 0; seed < 20; seed++) {
      const w = cave(key(seed), { rows: 18, cols: 26 })
      const r = regions(w)
      expect(r.count).toBeLessThanOrEqual(1)
      if (r.count === 1) {
        expect(r.labels[w.starts[0] ?? 0]).toBe(0)
        expect(r.labels[w.goals[0] ?? 0]).toBe(0)
      }
      expect(Array.from(cave(key(seed), { rows: 18, cols: 26 }).terrain)).toEqual(Array.from(w.terrain))
    }
    const all = cave(key(3), { rows: 18, cols: 26, keepLargest: false })
    expect(regions(all).count).toBeGreaterThanOrEqual(1)
  })
})

describe('noise', () => {
  it('Perlin noise vanishes on the lattice and stays within sqrt(2) / 2; value noise stays within [0, 1]', () => {
    fc.assert(
      fc.property(fc.nat(10_000), fc.integer({ min: 1, max: 6 }), (seed, scale) => {
        const p = perlinNoise(key(seed), 13, 17, { scale })
        for (let r = 0; r < 13; r++) {
          for (let c = 0; c < 17; c++) {
            const v = p[r * 17 + c] ?? 0
            expect(Math.abs(v)).toBeLessThanOrEqual(Math.SQRT1_2 + 1e-12)
            if (r % scale === 0 && c % scale === 0) expect(v).toBe(0)
          }
        }
        const v = valueNoise(key(seed), 13, 17, { scale, octaves: 3 })
        for (const x of v) expect(x >= 0 && x <= 1).toBe(true)
        expect(Array.from(valueNoise(key(seed), 13, 17, { scale, octaves: 3 }))).toEqual(Array.from(v))
      }),
      { numRuns: 40 },
    )
  })

  it('is smooth: neighbouring cells of one octave differ by at most a Lipschitz bound over the scale', () => {
    const scale = 8
    const p = perlinNoise(key(1), 32, 32, { scale })
    const v = valueNoise(key(1), 32, 32, { scale })
    for (let r = 0; r < 32; r++) {
      for (let c = 0; c + 1 < 32; c++) {
        // Per lattice unit: Perlin's slope is below 6 (fade slope 1.875 times corner terms up to sqrt 2, plus the
        // gradients), value noise's below 1.875 (the fade's slope times lattice differences of at most 1).
        expect(Math.abs((p[r * 32 + c + 1] ?? 0) - (p[r * 32 + c] ?? 0))).toBeLessThan(6 / scale)
        expect(Math.abs((v[r * 32 + c + 1] ?? 0) - (v[r * 32 + c] ?? 0))).toBeLessThan(1.875 / scale + 1e-12)
      }
    }
    expect(() => perlinNoise(key(0), 4, 4, { scale: 0 })).toThrow()
    expect(() => valueNoise(key(0), 0, 4)).toThrow()
  })

  it('makes terrain bands with a start and goal on passable land', () => {
    for (const noise of ['perlin', 'value'] as const) {
      const w = noiseTerrain(key(5), { rows: 20, cols: 20, noise })
      const names = new Set(Array.from({ length: w.n }, (_, c) => terrainOf(w, c).name))
      expect([...names].every((n) => ['water', 'grass', 'forest', 'mountain'].includes(n))).toBe(true)
      for (const c of [...w.starts, ...w.goals]) {
        expect(passable(w, c)).toBe(true)
        expect(terrainOf(w, c).name).not.toBe('water')
      }
    }
    const custom = noiseTerrain(key(5), {
      rows: 6,
      cols: 6,
      bands: [{ below: Infinity, terrain: 'sand' }],
      terrains: [{ name: 'sand', char: ':', cost: 2 }],
    })
    expect(terrainOf(custom, 0).name).toBe('sand')
  })
})
