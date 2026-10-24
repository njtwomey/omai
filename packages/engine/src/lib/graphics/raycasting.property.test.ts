import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { castRay, fromAscii, worldGeometry } from '../worlds/index.ts'
import { ddaCast } from './index.ts'

// Random square maps of floor and walls, with a floor cell to start from.
const mapArb = fc
  .tuple(fc.integer({ min: 3, max: 9 }), fc.integer({ min: 3, max: 9 }))
  .chain(([rows, cols]) =>
    fc.tuple(
      fc.array(fc.array(fc.boolean(), { minLength: cols, maxLength: cols }), { minLength: rows, maxLength: rows }),
      fc.integer({ min: 0, max: rows * cols - 1 }),
      fc.double({ min: 0.05, max: 0.95, noNaN: true }),
      fc.double({ min: 0.05, max: 0.95, noNaN: true }),
      fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
    ),
  )
  .map(([grid, start, fx, fy, angle]) => {
    const rows = grid.length
    const cols = grid[0]?.length ?? 0
    const lines = grid.map((row, r) => row.map((wall, c) => (wall && r * cols + c !== start ? '#' : '.')).join(''))
    return {
      lines,
      rows,
      cols,
      origin: [(start % cols) + fx, Math.floor(start / cols) + fy] as [number, number],
      angle,
    }
  })

describe('DDA ray casting', () => {
  it('visits exactly the cells a fine brute-force march visits', () => {
    fc.assert(
      fc.property(mapArb, ({ lines, rows, cols, origin, angle }) => {
        const w = fromAscii(lines)
        const r = ddaCast(w, origin, angle)
        const dx = Math.cos(angle)
        const dy = Math.sin(angle)
        const end = r.hit?.distance ?? Math.hypot(rows, cols) + 1
        // Skip rays that pass within 1e-3 of a grid corner: there either neighbour order is right, and a march with
        // steps of 1e-4 could step over a corner cell the ray only clips (it stays in it for less than the distance).
        // Elsewhere the ray stays in every cell it enters for longer than a step.
        for (const s of r.steps) {
          const gx = Math.abs(s.point[0] - Math.round(s.point[0]))
          const gy = Math.abs(s.point[1] - Math.round(s.point[1]))
          if (gx < 1e-3 && gy < 1e-3) return
        }
        const march: number[] = []
        const h = 1e-4
        const n = Math.ceil(end / h) + 1
        for (let k = 0; k <= n; k++) {
          const t = Math.min(k * h, end + 1e-9)
          const x = origin[0] + t * dx
          const y = origin[1] + t * dy
          const c = Math.floor(x)
          const rr = Math.floor(y)
          if (c < 0 || rr < 0 || c >= cols || rr >= rows) break
          const cell = rr * cols + c
          if (march[march.length - 1] !== cell) march.push(cell)
        }
        expect(r.cells).toEqual(march)
      }),
      { numRuns: 60 },
    )
  })

  it('stops at the same wall as the continuous ray cast of worlds', () => {
    fc.assert(
      fc.property(mapArb, ({ lines, origin, angle }) => {
        const w = fromAscii(lines)
        const r = ddaCast(w, origin, angle)
        const c = castRay(worldGeometry(w), origin, angle)
        expect(r.hit).not.toBeNull()
        expect(r.hit?.distance ?? 0).toBeCloseTo(c?.distance ?? Number.NaN, 9)
        expect(r.hit?.normal[0]).toBeCloseTo(c?.normal[0] ?? Number.NaN, 12)
        expect(r.hit?.normal[1]).toBeCloseTo(c?.normal[1] ?? Number.NaN, 12)
      }),
      { numRuns: 100 },
    )
  })
})
