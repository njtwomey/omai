import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { minOf } from '../base/index.ts'
import { orient, segmentIntersection } from '../numerics/geometry/index.ts'
import {
  castRay,
  cellCentre,
  distanceToWalls,
  fromAscii,
  lineOfSight,
  moveCircle,
  passable,
  setWalls,
  sweepSegment,
  type World,
  wallDistanceField,
  wallSegments,
  worldGeometry,
  worldSize,
} from './index.ts'

type P = readonly [number, number]

const pointSegment = (p: P, a: P, b: P): number => {
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const len2 = ex * ex + ey * ey
  const u = len2 === 0 ? 0 : Math.min(1, Math.max(0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2))
  return Math.hypot(p[0] - a[0] - u * ex, p[1] - a[1] - u * ey)
}

const segmentSegment = (p: P, q: P, a: P, b: P): number =>
  segmentIntersection(p, q, a, b) !== null
    ? 0
    : Math.min(pointSegment(p, a, b), pointSegment(q, a, b), pointSegment(a, p, q), pointSegment(b, p, q))

const crosses = (a: P, b: P, c: P, d: P): boolean => {
  const m = 1e-9
  const o = [orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)]
  return o.every((x) => Math.abs(x) > m) && (o[0] ?? 0) * (o[1] ?? 0) < 0 && (o[2] ?? 0) * (o[3] ?? 0) < 0
}

// Random square or hexagonal worlds with blocking cells and, on squares, random thin walls.
const worldArb: fc.Arbitrary<World> = fc
  .record({
    rows: fc.integer({ min: 1, max: 5 }),
    cols: fc.integer({ min: 2, max: 6 }),
    hex: fc.boolean(),
    walls: fc.array(fc.tuple(fc.nat(40), fc.boolean()), { maxLength: 8 }),
  })
  .chain(({ rows, cols, hex, walls }) =>
    fc
      .array(fc.constantFrom('.', '.', '.', '#', '%'), { minLength: rows * cols, maxLength: rows * cols })
      .map((cells) => {
        const lines = Array.from({ length: rows }, (_, r) => cells.slice(r * cols, (r + 1) * cols).join(''))
        lines[0] = `.${lines[0]?.slice(1)}`
        const w = fromAscii(lines, hex ? { topology: 'hex' } : {})
        if (hex) return w
        const pairs: [number, number][] = []
        for (const [k, down] of walls) {
          const a = k % w.n
          if (down && a + cols < w.n) pairs.push([a, a + cols])
          if (!down && (a % cols) + 1 < cols) pairs.push([a, a + 1])
        }
        return setWalls(w, pairs)
      }),
  )

describe('continuous motion and sight (properties)', () => {
  it('never passes through walls: the swept path keeps the radius from every solid segment', () => {
    fc.assert(
      fc.property(
        worldArb,
        fc.double({ min: 0, max: 0.45, noNaN: true }),
        fc.array(fc.tuple(fc.double({ min: -3, max: 3, noNaN: true }), fc.double({ min: -3, max: 3, noNaN: true })), {
          minLength: 1,
          maxLength: 6,
        }),
        fc.boolean(),
        (w, radius, moves, slide) => {
          const g = worldGeometry(w)
          const walls = wallSegments(w)
          let p: P = cellCentre(w, 0)
          for (const d of moves) {
            const m = moveCircle(g, p, d, radius, { slide })
            expect(m.path[0]).toEqual(p)
            expect(m.path[m.path.length - 1]).toEqual(m.position)
            // Every leg of the path (the moves between contacts) keeps the radius from every wall.
            for (let i = 0; i + 1 < m.path.length; i++) {
              const from = m.path[i] ?? p
              const to = m.path[i + 1] ?? p
              for (const [a, b] of walls) expect(segmentSegment(from, to, a, b)).toBeGreaterThanOrEqual(radius - 1e-7)
            }
            expect(m.travelled).toBeLessThanOrEqual(Math.hypot(d[0], d[1]) + 1e-9)
            p = m.position
          }
        },
      ),
      { numRuns: 200 },
    )
  })

  it('line of sight is symmetric and agrees with a brute-force check of every opaque segment', () => {
    fc.assert(
      fc.property(
        worldArb,
        fc.array(fc.tuple(fc.double({ min: 0, max: 1, noNaN: true }), fc.double({ min: 0, max: 1, noNaN: true })), {
          minLength: 2,
          maxLength: 6,
        }),
        fc.boolean(),
        (w, unit, centres) => {
          const g = worldGeometry(w)
          const [width, height] = worldSize(w)
          const pts: P[] = centres
            ? Array.from({ length: w.n }, (_, c) => cellCentre(w, c)).filter((_, c) => passable(w, c))
            : unit.map(([x, y]) => [x * width, y * height] as const)
          const opaque = wallSegments(w, 'opaque')
          for (const a of pts) {
            for (const b of pts) {
              const ab = lineOfSight(g, a, b)
              expect(lineOfSight(g, b, a)).toBe(ab)
              // Clear by a margin: visible. Properly crossed by a margin: hidden. (Grazes are left to the robust test.)
              if (minOf(opaque.map(([c, d]) => segmentSegment(a, b, c, d))) > 1e-9) expect(ab).toBe(true)
              if (opaque.some(([c, d]) => crosses(a, b, c, d))) expect(ab).toBe(false)
            }
          }
        },
      ),
      { numRuns: 120 },
    )
  })

  it('rays hit the nearest wall and distances match a brute-force minimum', () => {
    fc.assert(
      fc.property(
        worldArb,
        // Off the world's boundary: a ray that starts 1e-15 inside an edge and runs within 1e-14 radians of parallel to
        // it grazes the wall, and the cast and the sweep round to different answers (a counterexample fast-check found).
        fc.double({ min: 1e-6, max: 1 - 1e-6, noNaN: true }),
        fc.double({ min: 1e-6, max: 1 - 1e-6, noNaN: true }),
        fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
        (w, ux, uy, angle) => {
          const g = worldGeometry(w)
          const [width, height] = worldSize(w)
          const o: P = [ux * width, uy * height]
          const solid = wallSegments(w)
          const nearest = minOf(solid.map(([a, b]) => pointSegment(o, a, b)))
          expect(distanceToWalls(g, o)).toBeCloseTo(nearest, 12)
          // Rays starting on a wall graze it; the comparison is for rays from open space.
          if (nearest < 1e-6) return
          const hit = castRay(g, o, angle, { kind: 'solid' })
          const d: P = [Math.cos(angle), Math.sin(angle)]
          // A swept segment of length 2 meets the same wall as the ray, when the ray's hit is within it.
          const sweep = sweepSegment(g, o, [o[0] + 2 * d[0], o[1] + 2 * d[1]])
          if (hit !== null && hit.distance < 2 - 1e-9) expect(2 * (sweep?.t ?? Number.NaN)).toBeCloseTo(hit.distance, 9)
          // A sweep that only touches a wall's end (the ray runs exactly through a corner) is a graze, where rounding
          // decides between the two answers, as in the brute force below.
          const touchesEnd =
            sweep !== null &&
            solid.some(([a, b]) => [a, b].some((e) => Math.hypot(sweep.point[0] - e[0], sweep.point[1] - e[1]) < 1e-9))
          if ((hit === null || hit.distance > 2 + 1e-9) && !touchesEnd) expect(sweep).toBeNull()
          // The brute force: the nearest crossing of a long segment along the ray. A crossing within 1e-9 of a wall's end
          // is a graze (the ray runs past a corner), where rounding decides; such walls are left out.
          let best = Number.POSITIVE_INFINITY
          let grazed = false
          for (const [a, b] of solid) {
            const far: P = [o[0] + 100 * d[0], o[1] + 100 * d[1]]
            const x = segmentIntersection(o, far, a, b)
            if (x === null) continue
            if (Math.min(Math.hypot(x[0] - a[0], x[1] - a[1]), Math.hypot(x[0] - b[0], x[1] - b[1])) < 1e-9)
              grazed = true
            else best = Math.min(best, Math.hypot(x[0] - o[0], x[1] - o[1]))
          }
          if (Number.isFinite(best) && !grazed) expect(hit?.distance).toBeCloseTo(best, 9)
          if (Number.isFinite(best)) expect(hit?.distance ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(best + 1e-9)
        },
      ),
      { numRuns: 200 },
    )
  })

  it('Euclidean wall distance fields equal the brute-force distance to the nearest wall cell and are 1-Lipschitz', () => {
    fc.assert(
      fc.property(worldArb, fc.boolean(), (w, border) => {
        if (w.topology.kind !== 'square') return
        const cols = w.topology.cols
        const rows = w.topology.rows
        const f = wallDistanceField(w, { border })
        for (let c = 0; c < w.n; c++) {
          const [r0, c0] = [Math.floor(c / cols), c % cols]
          let best = Number.POSITIVE_INFINITY
          for (let b = 0; b < w.n; b++) {
            if (passable(w, b)) continue
            best = Math.min(best, Math.hypot(Math.floor(b / cols) - r0, (b % cols) - c0))
          }
          if (border) best = Math.min(best, r0 + 1, c0 + 1, rows - r0, cols - c0)
          if (Number.isFinite(best)) expect(f[c]).toBeCloseTo(best, 12)
          else expect(f[c]).toBe(best)
          if (c + 1 < w.n && (c % cols) + 1 < cols && Number.isFinite(f[c] ?? 0)) {
            expect(Math.abs((f[c] ?? 0) - (f[c + 1] ?? 0))).toBeLessThanOrEqual(1 + 1e-12)
          }
        }
      }),
      { numRuns: 150 },
    )
  })
})
