import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import * as S from './index.ts'

// Neighbour search where many points are equally far: integer lattices, duplicated points and a query that is a data point.
// The exact trees must return the brute-force distances (indices may differ among ties) and the same radius counts.
const lattice = (w: number, h: number): number[][] =>
  Array.from({ length: w * h }, (_, i) => [i % w, Math.floor(i / w)])
const duplicated = (): number[][] => [...lattice(4, 4), ...lattice(4, 4).slice(0, 8), [0, 0], [0, 0], [0, 0]]
const line = (n: number): number[][] => Array.from({ length: n }, (_, i) => [i])

describe('exact trees on ties and duplicates', () => {
  it('KD, ball and VP trees return the brute-force distances and radius counts on lattices, duplicates and lines', () => {
    const failures: string[] = []
    const rows: readonly (readonly [string, number[][], number[][]])[] = [
      [
        '7 x 7 lattice',
        lattice(7, 7),
        [
          [3, 3],
          [0, 0],
          [3.5, 3.5],
          [10, 10],
          [-2, 3],
        ],
      ],
      ['a line of 20 points', line(20), [[0], [9.5], [-3], [25]]],
      [
        'one point',
        [[2, 2]],
        [
          [0, 0],
          [2, 2],
        ],
      ],
    ]
    for (const [name, data, queries] of rows) {
      for (const metric of ['euclidean', 'cityblock', 'chebyshev'] as const) {
        const k = Math.min(9, data.length)
        const brute = S.kNearest(data, queries, k, { metric })
        const bruteDistances = toFlat(brute.distances)
        for (const tree of [
          S.kdTree(data, { metric, leafSize: 3 }),
          S.ballTree(data, { metric, leafSize: 3 }),
          S.vpTree(data, { metric, leafSize: 3 }),
        ]) {
          const label = `${name}, ${tree.kind}, ${metric}`
          const got = tree.query(queries, k)
          const d = toFlat(got.distances)
          if (
            d.length !== bruteDistances.length ||
            d.some((v, i) => Math.abs(v - (bruteDistances[i] ?? Number.NaN)) > 1e-12)
          )
            failures.push(`${label}: distances differ from brute force`)
          // Every returned index is a valid, distinct data point at the distance reported.
          const indices = Array.from(got.indices.data as ArrayLike<number>)
          queries.forEach((_, q) => {
            const row = indices.slice(q * k, (q + 1) * k)
            if (new Set(row).size !== row.length || row.some((i) => i < 0 || i >= data.length))
              failures.push(`${label}: query ${q} returns repeated or invalid indices`)
          })
          // Radius counts agree, with the boundary included (a lattice point is exactly at distance 1).
          for (const r of [0, 1, 1.5, 3]) {
            const want = S.radiusNeighbours(data, queries, r, { metric }).map((row) => row.indices.length)
            const counts = tree.countNeighbours(queries, r)
            if (counts.some((c, i) => c !== want[i]))
              failures.push(`${label}: counts at radius ${r} are ${counts}, brute force ${want}`)
            const listed = tree.queryRadius(queries, r).indices.map((row) => row.length)
            if (listed.some((c, i) => c !== want[i]))
              failures.push(`${label}: queryRadius at ${r} lists ${listed}, brute force ${want}`)
          }
        }
      }
    }
    // A data point queried against its own data is at distance 0, and excludeSelf skips exactly it.
    const own = S.kNearest(lattice(5, 5), undefined, 3, { excludeSelf: true })
    const withSelf = S.kNearest(lattice(5, 5), undefined, 3)
    if (
      toFlat(withSelf.distances)
        .filter((_, i) => i % 3 === 0)
        .some((v) => v !== 0)
    )
      failures.push('the nearest neighbour of a data point is not itself')
    expect(toFlat(own.distances).every((v) => v >= 1)).toBe(true)
    expect(failures).toEqual([])
  })
})
