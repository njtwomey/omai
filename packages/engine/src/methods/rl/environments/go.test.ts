import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { playAec } from '../../../lib/rl/index.ts'
import { type GoState, go, goArea, goState } from './go.ts'

const env5 = go.create({ size: 5 })
const PASS = 25
const step = (env: typeof env5, s: GoState, a: number): GoState => env.step(s, a, key(0)).state

describe('Go captures and legality', () => {
  it('captures a group without liberties', () => {
    const s = goState(['.....', '..X..', '.XOX.', '.....', '.....'])
    const after = step(env5, s, 17)
    expect(after.board[12]).toBe(0)
    expect(after.board[17]).toBe(1)
    expect(after.toMove).toBe(1)
  })

  it('forbids suicide unless the stone captures', () => {
    const s = goState(['.X...', 'X....', '.....', '.....', '.....'], 1)
    expect(env5.legalActions?.(s, 'player_1')).not.toContain(0)
    expect(env5.step(s, 0, key(0)).state.forfeit).toBe(1)
    const capture = goState(['.XO..', 'XO...', 'O....', '.....', '.....'], 1)
    expect(env5.legalActions?.(capture, 'player_1')).toContain(0)
    const after = step(env5, capture, 0)
    expect(after.board[1]).toBe(0)
    expect(after.board[5]).toBe(0)
  })

  it('forbids the immediate recapture of a ko under both rules', () => {
    const ko = goState(['.XO..', 'XO.O.', '.XO..', '.....', '.....'])
    for (const rule of ['positional', 'simple'] as const) {
      const env = go.create({ size: 5, ko: rule })
      const taken = step(env, ko, 7)
      expect(taken.board[6]).toBe(0)
      expect(env.legalActions?.(taken, 'player_1')).not.toContain(6)
      // After an exchange elsewhere the recapture is allowed again.
      const later = step(env, step(env, taken, 24), 20)
      expect(env.legalActions?.(later, 'player_1')).toContain(6)
    }
    expect(step(go.create({ size: 5, ko: 'simple' }), ko, 7).ko).toBe(6)
  })

  it('forbids recreating any earlier position under positional superko only', () => {
    const s = goState(['.....', '.X...', '.....', '...O.', '.....'])
    const simple = go.create({ size: 5, ko: 'simple' })
    const repeated = { ...s, history: [...s.history, step(simple, s, 12).hash] }
    expect(simple.legalActions?.(repeated, 'player_0')).toContain(12)
    expect(env5.legalActions?.(repeated, 'player_0')).not.toContain(12)
  })
})

describe('Go scoring and the end of the game', () => {
  it('scores by area (Tromp-Taylor) with komi', () => {
    const rows = ['.X.O.', '.X.O.', '.X.O.', '.X.O.', '.X.O.']
    expect(goArea(goState(rows).board, 5)).toEqual([10, 10])
    const even = go.create({ size: 5, komi: 0 })
    const r = even.step(step(even, goState(rows), PASS), PASS, key(0))
    expect(r.rewards).toEqual({ player_0: 0, player_1: 0 })
    expect(r.state.score).toEqual([10, 10])
    const komi = go.create({ size: 5, komi: 0.5 })
    expect(komi.step(step(komi, goState(rows), PASS), PASS, key(0)).rewards).toEqual({ player_0: -1, player_1: 1 })
    expect(goArea(goState(['X....', '.....', '.....', '.....', '.....']).board, 5)).toEqual([25, 0])
  })

  it('ends after two consecutive passes or the move cap', () => {
    const s = step(env5, step(env5, step(env5, env5.reset(key(0)), PASS), 12), PASS)
    expect(s.over).toBe(false)
    expect(s.passes).toBe(1)
    const capped = go.create({ size: 5, maxMoves: 2 })
    const ep = playAec(capped, { player_0: () => 0, player_1: () => 24 }, key(0))
    expect(ep.length).toBe(2)
    expect(ep.states[2]?.score).toEqual([1, 1 + 7.5])
    expect(ep.states[2]?.forfeit).toBeUndefined()
    expect(() => go.create({ size: 6 })).toThrow(/size/)
  })

  it('observes own, opponent and empty planes plus the colour plane, and renders', () => {
    const s = goState(['X....', '.....', '.....', '.....', '....O'])
    const black = env5.observe(s, 'player_0')
    const white = env5.observe(s, 'player_1')
    expect(black.length).toBe(100)
    expect([black[0], black[25 + 24], black[75]]).toEqual([1, 1, 1])
    expect([white[24], white[25], white[75]]).toEqual([1, 1, 0])
    expect(env5.legalActions?.(s, 'player_0').length).toBe(24)
    expect(env5.render(s).children.length).toBeGreaterThan(10)
  })
})
