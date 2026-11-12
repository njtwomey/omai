import { describe, expect, it } from 'vitest'
import { checkpoints, fit, rate } from './compute.ts'
import { PLAYERS, shuffled, tournament } from './data.ts'
import { EARLY, readResult, tally } from './reading.ts'

describe('the who-is-better computation', () => {
  it('has a tournament of forty games in which every pair meets four times', () => {
    const g = tournament()
    expect(g.a).toHaveLength(40)
    expect(g.b).toHaveLength(40)
    expect(g.result.every((r) => r === 0 || r === 1 || r === 2)).toBe(true)
    expect(g.a.every((a, i) => a !== g.b[i])).toBe(true)
    const t = tally(g)
    // every game has a win and a loss or two draws, so the totals match
    expect(t.reduce((s, x) => s + x.wins, 0)).toBe(t.reduce((s, x) => s + x.losses, 0))
    expect(t.every((x) => x.wins + x.draws + x.losses === 16)).toBe(true)
  })

  it('rates the players after the first game, the first dozen and every fifth game', () => {
    expect(checkpoints(40)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 35, 40])
    expect(checkpoints(10)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  })

  it('returns plain data of the right shapes on a small tournament', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), { rounds: 1, record: true })
    expect(r.games.a).toHaveLength(10)
    expect(r.ratings).toHaveLength(PLAYERS.length)
    expect(r.history.map((h) => h.after)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expect(r.history.every((h) => h.ratings.length === PLAYERS.length)).toBe(true)
    expect(r.history[0]?.ratings.every((x) => x.mean === 0 && x.sd === 1)).toBe(true)
    // the spreads shrink as games arrive
    const last = r.history.at(-1)?.ratings ?? []
    expect(last.every((x) => x.sd < 1)).toBe(true)
    expect(r.settled).toBe(true)
    expect(r.record).toBeDefined()
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('gives the same final ratings whatever order the games arrive in', () => {
    const a = rate(shuffled(tournament(), 1)).ratings
    const b = rate(shuffled(tournament(), 2)).ratings
    a.forEach((x, i) => {
      expect(x.mean).toBeCloseTo(b[i]?.mean ?? Number.NaN, 4)
      expect(x.sd).toBeCloseTo(b[i]?.sd ?? Number.NaN, 4)
    })
  }, 30000)

  it('on the real tournament puts the players in the order of their true skills, and reads the first games', () => {
    const r = fit(1, undefined, { rounds: 4, record: false })
    const reading = readResult(r)
    expect(reading.ladder.map((x) => x.name)).toEqual([...PLAYERS])
    expect(reading.inTrueOrder).toBe(true)
    expect(reading.moves.length).toBeGreaterThan(3)
    expect(reading.moves.every((m) => m.game <= EARLY)).toBe(true)
    expect(reading.spreadAfter(10)).toBeLessThan(1)
    expect(reading.spreadAfter(40)).toBeLessThan(reading.spreadAfter(10) as number)
  }, 60000)
})
