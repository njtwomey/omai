import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { fromAscii, setWalls, toAscii, type World } from './index.ts'

// The characters toAscii writes for the default legend (a space reads as floor and is written as '.').
const CANONICAL = '.#~,%MHLCXT^>v<SGKD$'.split('')

const rowsArb = (maxRows = 6, maxCols = 7): fc.Arbitrary<string[]> =>
  fc
    .tuple(fc.integer({ min: 1, max: maxRows }), fc.integer({ min: 1, max: maxCols }))
    .chain(([r, c]) =>
      fc.array(fc.array(fc.constantFrom(...CANONICAL), { minLength: c, maxLength: c }), { minLength: r, maxLength: r }),
    )
    .map((rows) => rows.map((row) => row.join('')))

const same = (a: World, b: World): void => {
  expect(Array.from(a.terrain)).toEqual(Array.from(b.terrain))
  expect(Array.from(a.walls)).toEqual(Array.from(b.walls))
  expect([a.starts, a.goals, a.items, a.topology]).toEqual([b.starts, b.goals, b.items, b.topology])
}

describe('ASCII maps (properties)', () => {
  it('round-trips one-character-per-cell maps, square and hexagonal', () => {
    fc.assert(
      fc.property(rowsArb(), fc.boolean(), (rows, hex) => {
        const w = fromAscii(rows, hex ? { topology: 'hex' } : {})
        expect(toAscii(w)).toEqual(rows)
        same(fromAscii(toAscii(w), hex ? { topology: 'hex' } : {}), w)
      }),
      { numRuns: 150 },
    )
  })

  it('round-trips thin-wall drawings with any walls', () => {
    fc.assert(
      fc.property(rowsArb(4, 5), fc.array(fc.tuple(fc.nat(30), fc.boolean())), (rows, picks) => {
        const base = fromAscii(rows)
        const t = base.topology as { readonly rows: number; readonly cols: number }
        const pairs: [number, number][] = []
        for (const [k, down] of picks) {
          const a = k % base.n
          const b = down ? a + t.cols : a + 1
          if (down ? b < base.n : (a % t.cols) + 1 < t.cols) pairs.push([a, b])
        }
        const w = setWalls(base, pairs)
        const drawn = toAscii(w, { format: 'walls' })
        expect(drawn).toHaveLength(2 * t.rows + 1)
        same(fromAscii(drawn, { format: 'walls' }), w)
      }),
      { numRuns: 150 },
    )
  })

  it('reads spaces as floor and writes them back as dots', () => {
    fc.assert(
      fc.property(rowsArb(), (rows) => {
        const spaced = rows.map((r) => r.replaceAll('.', ' '))
        expect(toAscii(fromAscii(spaced))).toEqual(rows)
      }),
      { numRuns: 50 },
    )
  })
})
