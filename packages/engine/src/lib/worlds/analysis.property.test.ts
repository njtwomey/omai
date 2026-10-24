import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  bridges,
  chokePoints,
  distanceField,
  exits,
  fromAscii,
  neighbours,
  passable,
  reachable,
  regions,
  shortestPath,
  terrainOf,
  type World,
} from './index.ts'

const worldArb = (
  chars = '...#~>',
): fc.Arbitrary<{ readonly w: World; readonly hex: boolean; readonly eight: boolean }> =>
  fc
    .record({
      rows: fc.integer({ min: 1, max: 6 }),
      cols: fc.integer({ min: 1, max: 7 }),
      hex: fc.boolean(),
      eight: fc.boolean(),
    })
    .chain(({ rows, cols, hex, eight }) =>
      fc.array(fc.constantFrom(...chars.split('')), { minLength: rows * cols, maxLength: rows * cols }).map((cells) => {
        const lines = Array.from({ length: rows }, (_, r) => cells.slice(r * cols, (r + 1) * cols).join(''))
        return { w: fromAscii(lines, hex ? { topology: 'hex' } : {}), hex, eight: eight && !hex }
      }),
    )

describe('analysis (properties)', () => {
  it('breadth-first fields satisfy d(v) = 1 + min over predecessors, and Dijkstra fields their weighted form', () => {
    fc.assert(
      fc.property(worldArb(), fc.nat(), ({ w, eight, hex }, seed) => {
        const options = eight ? { connectivity: 8 as const } : {}
        const free = Array.from({ length: w.n }, (_, c) => c).filter((c) => passable(w, c))
        const source = free[seed % Math.max(1, free.length)]
        if (source === undefined) return
        const hops = distanceField(w, source, options)
        const cost = distanceField(w, source, { ...options, weighted: true })
        const bestHop = new Float64Array(w.n).fill(Number.POSITIVE_INFINITY)
        const bestCost = new Float64Array(w.n).fill(Number.POSITIVE_INFINITY)
        for (let u = 0; u < w.n; u++) {
          for (const [d, v] of exits(w, u, options)) {
            bestHop[v] = Math.min(bestHop[v] ?? 0, (hops[u] ?? 0) + 1)
            const step = terrainOf(w, v).cost * (!hex && d >= 4 ? Math.SQRT2 : 1)
            bestCost[v] = Math.min(bestCost[v] ?? 0, (cost[u] ?? 0) + step)
          }
        }
        for (let v = 0; v < w.n; v++) {
          if (v === source) continue
          expect(hops[v]).toBe(bestHop[v])
          if (Number.isFinite(cost[v] ?? 0)) expect(cost[v]).toBeCloseTo(bestCost[v] ?? 0, 9)
          else expect(bestCost[v]).toBe(Number.POSITIVE_INFINITY)
          // Reachability and finite distance agree, and a path of the right length exists.
          expect(reachable(w, source, options)[v] === 1).toBe(Number.isFinite(hops[v] ?? 0))
        }
        const far = hops.findLastIndex((d) => Number.isFinite(d))
        const route = shortestPath(w, source, far, options)
        expect(route?.length).toBe((hops[far] ?? 0) + 1)
        for (let i = 0; i + 1 < (route?.length ?? 0); i++) {
          expect(neighbours(w, route?.[i] ?? 0, options)).toContain(route?.[i + 1])
        }
      }),
      { numRuns: 120 },
    )
  })

  it('choke points and bridges are exactly the cells and passages whose removal splits a region', () => {
    fc.assert(
      fc.property(worldArb('...#'), ({ w, eight }) => {
        const options = eight ? { connectivity: 8 as const } : {}
        const base = regions(w, options).count
        const cuts = chokePoints(w, options)
        for (let c = 0; c < w.n; c++) {
          if (!passable(w, c)) continue
          const without = regions(w, { ...options, blocked: (x) => x === c }).count
          const isolated = neighbours(w, c, options).length === 0
          expect(cuts.includes(c)).toBe(without > base - (isolated ? 1 : 0))
        }
        const bs = bridges(w, options)
        for (const [a, b] of bs) expect(neighbours(w, a, options)).toContain(b)
        // A bridge's ends fall into different regions once the passage between them is cut.
        for (const [a, b] of bs) {
          const seen = new Uint8Array(w.n)
          const queue = [a]
          seen[a] = 1
          for (let q = 0; q < queue.length; q++) {
            const u = queue[q] ?? 0
            for (const v of neighbours(w, u, options)) {
              if ((u === a && v === b) || (u === b && v === a) || seen[v] === 1) continue
              seen[v] = 1
              queue.push(v)
            }
          }
          expect(seen[b]).toBe(0)
        }
      }),
      { numRuns: 120 },
    )
  })
})
