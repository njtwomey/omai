import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { interleavingCredit, interleavingPreference, teamDraftInterleave } from './interleaving.ts'

describe('team-draft interleaving', () => {
  const a = [1, 2, 3, 4, 5, 6]
  const b = [3, 7, 1, 8, 2, 9]
  for (const seed of [0, 1, 2, 3, 4, 5]) {
    it(`shows each item once, keeps each ranker's order and balances the teams (key ${seed})`, () => {
      const r = teamDraftInterleave(a, b, key(seed))
      expect(new Set(r.items).size).toBe(r.items.length)
      expect(r.items.length).toBe(new Set([...a, ...b]).size)
      for (const [team, list] of [
        ['a', a],
        ['b', b],
      ] as const) {
        const mine = r.items.filter((_, i) => r.teams[i] === team)
        const positions = mine.map((x) => list.indexOf(x))
        expect(positions).toEqual([...positions].sort((x, y) => x - y))
      }
      for (let n = 2; n <= r.items.length; n += 2) {
        const as = r.teams.slice(0, n).filter((t) => t === 'a').length
        expect(as).toBe(n / 2)
      }
    })
  }

  it('flips the coin each round: both rankers pick first over many draws', () => {
    let aFirst = 0
    for (let s = 0; s < 200; s++) if (teamDraftInterleave([1], [2], key(s)).teams[0] === 'a') aFirst++
    expect(aFirst).toBeGreaterThan(70)
    expect(aFirst).toBeLessThan(130)
  })

  it('credits clicks to the ranker that supplied them and finds a preference', () => {
    expect(interleavingCredit(['a', 'b', 'a', 'b'], [1, 0, 1, 0])).toEqual({ a: 2, b: 0, winner: 'a' })
    expect(interleavingCredit(['a', 'b'], [0, 0]).winner).toBe('tie')
    const p = interleavingPreference(['a', 'a', 'a', 'b', 'tie'])
    expect([p.wins, p.losses, p.ties, p.preference]).toEqual([3, 1, 1, 0.75])
    expect(p.standardError).toBeCloseTo(Math.sqrt((0.75 * 0.25) / 4), 12)
    expect(() => interleavingCredit(['a'], [1, 0])).toThrow()
  })

  it('prefers the better ranker when users click the relevant items', () => {
    // Items 0-2 are relevant. Ranker A puts them first, ranker B last.
    const A = [0, 1, 2, 3, 4, 5, 6, 7]
    const B = [7, 6, 5, 4, 3, 2, 1, 0]
    const winners = Array.from({ length: 300 }, (_, s) => {
      const r = teamDraftInterleave(A, B, key(s), 6)
      return interleavingCredit(
        r.teams,
        r.items.map((x) => (x <= 2 ? 1 : 0)),
      ).winner
    })
    expect(interleavingPreference(winners).preference).toBeGreaterThan(0.9)
  })
})
