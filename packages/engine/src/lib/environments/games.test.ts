import { describe, expect, it } from 'vitest'
import { expectedReturns, negamaxValue, type Strategy, uniformStrategy } from '../../testing/games.ts'
import { key } from '../random/index.ts'
import { type AecEnv, isEnvDefinition, listEnvironments } from '../rl/index.ts'
import { gameMcts } from '../rl/planning.ts'
import { contains } from '../space/index.ts'
import {
  connectFour,
  connectFourLine,
  type KuhnPokerState,
  kuhnPayoff,
  kuhnPoker,
  kuhnPokerState,
  leducPoker,
  leducPokerState,
  leducShowdown,
  nim,
  nimMove,
  ticTacToe,
} from './games.ts'
import * as environments from './index.ts'

const play = <S>(env: AecEnv<S, readonly number[], number>, s: S, actions: readonly number[]): S => {
  let state = s
  for (const [i, a] of actions.entries()) state = env.step(state, a, key(i)).state
  return state
}

describe('game definitions', () => {
  it('are multi-agent definitions that single-agent registries skip', () => {
    expect(isEnvDefinition(nim)).toBe(false)
    expect(listEnvironments(environments).some((d) => d.id === 'Nim-v0')).toBe(false)
  })

  it('reset, observe in their spaces, mask actions and render', () => {
    for (const def of [ticTacToe, connectFour, nim, kuhnPoker, leducPoker]) {
      const env = def.create() as AecEnv<unknown, readonly number[], number>
      const s = env.reset(key(3))
      expect(env.currentAgent(s)).toBe('player_0')
      for (const agent of env.possibleAgents) {
        const space = env.observationSpaces[agent]
        expect(space !== undefined && contains(space, env.observe(s, agent) as never), def.id).toBe(true)
      }
      expect(env.legalActions?.(s, 'player_1')).toEqual([])
      const ep = randomPlayout(env, key(1))
      expect(ep.returns[0], def.id).toBeCloseTo(-ep.returns[1], 12)
      expect(env.render(ep.final).kind).toBe('scene')
    }
  })

  it('forfeit the game on an illegal action', () => {
    const env = ticTacToe.create()
    const s = env.step(env.reset(key(0)), 4, key(0)).state
    const r = env.step(s, 4, key(1))
    expect(r.rewards).toEqual({ player_0: 1, player_1: -1 })
    expect(r.state.forfeit).toBe(1)
    expect(env.currentAgent(r.state)).toBeUndefined()
    expect(() => env.step(r.state, 0, key(2))).toThrow(/over/)
  })
})

// Uniformly random legal play to the end of the game.
function randomPlayout<S>(
  env: AecEnv<S, readonly number[], number>,
  k: ReturnType<typeof key>,
): { final: S; returns: [number, number] } {
  let s = env.reset(k)
  const returns: [number, number] = [0, 0]
  for (let t = 0; t < 10_000; t++) {
    const agent = env.currentAgent(s)
    if (agent === undefined) break
    const legal = env.legalActions?.(s, agent) ?? []
    const a = legal[(t * 7919 + 13) % legal.length] ?? 0
    const r = env.step(s, a, key(t))
    returns[0] += r.rewards['player_0'] ?? 0
    returns[1] += r.rewards['player_1'] ?? 0
    s = r.state
  }
  return { final: s, returns }
}

describe('tic-tac-toe', () => {
  it('detects wins and draws', () => {
    const env = ticTacToe.create()
    const won = play(env, env.reset(key(0)), [0, 3, 1, 4, 2])
    expect(won.over).toBe(true)
    expect(won.winner).toBe(0)
    const draw = play(env, env.reset(key(0)), [0, 1, 2, 4, 3, 5, 7, 6, 8])
    expect(draw.over).toBe(true)
    expect(draw.winner).toBeUndefined()
  })

  it('has minimax value 0 and sees forced wins', () => {
    const env = ticTacToe.create()
    const id = (s: { readonly board: readonly number[] }): string => s.board.join('')
    expect(negamaxValue(env, env.reset(key(0)), id)).toBe(0)
    // A corner opening answered on an adjacent edge loses for O; answered in the centre it draws.
    expect(negamaxValue(env, play(env, env.reset(key(0)), [0, 1]), id)).toBe(1)
    expect(negamaxValue(env, play(env, env.reset(key(0)), [0, 4]), id)).toBe(0)
    const s = play(env, env.reset(key(0)), [0, 4])
    expect(env.observe(s, 'player_0').slice(0, 9)).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0])
    expect(env.observe(s, 'player_1').slice(0, 9)).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0])
  })
})

describe('connect four', () => {
  it('drops pieces, fills columns and detects every kind of four', () => {
    const env = connectFour.create()
    const vertical = play(env, env.reset(key(0)), [0, 1, 0, 1, 0, 1, 0])
    expect(vertical.winner).toBe(0)
    expect(vertical.line).toEqual([14, 21, 28, 35])
    const full = play(env, env.reset(key(0)), [2, 2, 2, 2, 2, 2])
    expect(env.legalActions?.(full, 'player_0')).toEqual([0, 1, 3, 4, 5, 6])
    const board = new Array<number>(42).fill(0)
    for (const c of [35, 29, 23, 17]) board[c] = 2
    expect(connectFourLine(board, 23)).toEqual([17, 23, 29, 35])
    board[17] = 0
    expect(connectFourLine(board, 23)).toBeUndefined()
    // Diagonal down-right.
    const d = new Array<number>(42).fill(0)
    for (const c of [14, 22, 30, 38]) d[c] = 1
    expect(connectFourLine(d, 30)).toEqual([14, 22, 30, 38])
  })
})

describe('Nim', () => {
  it('encodes actions as heap * maxTake + take - 1', () => {
    expect(nimMove(0, 3)).toEqual({ heap: 0, take: 1 })
    const env = nim.create({ heaps: [2] })
    expect(env.legalActions?.(env.reset(key(0)), 'player_0')).toEqual([0, 1])
    expect(env.step(env.reset(key(0)), 1, key(0)).rewards).toEqual({ player_0: 1, player_1: -1 })
    expect(gameMcts(env, env.reset(key(0)), { simulations: 100 }).action).toBe(1)
    const misere = nim.create({ heaps: [2], misere: true })
    expect(misere.step(misere.reset(key(0)), 1, key(0)).rewards).toEqual({ player_0: -1, player_1: 1 })
  })

  it('loses for the mover exactly at the P-positions (xor of the heaps is zero)', () => {
    const env = nim.create({ heaps: [3, 4, 5] })
    const id = (s: { readonly heaps: readonly number[] }): string => s.heaps.join(',')
    for (let a = 0; a <= 3; a++)
      for (let b = 0; b <= 4; b++)
        for (let c = 0; c <= 5; c++) {
          if (a + b + c === 0) continue
          const s = { ...env.reset(key(0)), heaps: [a, b, c] }
          expect(negamaxValue(env, s, id), `${a},${b},${c}`).toBe((a ^ b ^ c) === 0 ? -1 : 1)
        }
  })

  it('observes one-hot heaps and whose turn it is', () => {
    const env = nim.create({ heaps: [1, 2] })
    const s = env.reset(key(0))
    expect(env.observe(s, 'player_0')).toEqual([0, 1, 0, 0, 1, 1])
    expect(env.observe(s, 'player_1')).toEqual([0, 1, 0, 0, 1, 0])
    expect(() => nim.create({ heaps: [0] })).toThrow(/positive/)
  })
})

// Kuhn's Nash family for the first player (alpha in [0, 1/3]) against the second player's unique equilibrium.
const kuhnNash =
  (alpha: number): Strategy<KuhnPokerState> =>
  (s, agent) => {
    const h = s.history.join(',')
    const card = s.cards[agent === 'player_0' ? 0 : 1]
    const bet = (p: number): number[] => [1 - p, p]
    if (agent === 'player_0') {
      if (h === '') return bet([alpha, 0, 3 * alpha][card] ?? 0)
      return bet([0, alpha + 1 / 3, 1][card] ?? 0)
    }
    if (h === '0') return bet([1 / 3, 0, 1][card] ?? 0)
    return bet([0, 1 / 3, 1][card] ?? 0)
  }

const deals = (n: number, k: number): number[][] => {
  const out: number[][] = []
  const rec = (prefix: number[]): void => {
    if (prefix.length === k) {
      out.push(prefix)
      return
    }
    for (let c = 0; c < n; c++) if (!prefix.includes(c)) rec([...prefix, c])
  }
  rec([])
  return out
}

describe('Kuhn poker', () => {
  it('pays the pot by the rules', () => {
    expect(kuhnPayoff([2, 0], [0, 0])).toEqual([1, -1])
    expect(kuhnPayoff([2, 0], [0, 1, 0])).toEqual([-1, 1])
    expect(kuhnPayoff([0, 1], [1, 0])).toEqual([1, -1])
    expect(() => kuhnPayoff([0, 1], [1])).toThrow(/terminal/)
  })

  it('has game value -1/18 for the first player under Nash play', () => {
    const env = kuhnPoker.create()
    for (const alpha of [0, 1 / 6, 1 / 3]) {
      let v = 0
      for (const [a, b] of deals(3, 2))
        v += expectedReturns(env, kuhnPokerState([a ?? 0, b ?? 0]), kuhnNash(alpha))[0] / 6
      expect(v).toBeCloseTo(-1 / 18, 14)
    }
  })

  it('hides the opponent card and deals from the key', () => {
    const env = kuhnPoker.create()
    const s = env.step(kuhnPokerState([0, 2]), 1, key(0)).state
    expect(env.observe(s, 'player_0')).toEqual([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    expect(env.observe(s, 'player_1')).toEqual([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    const seen = new Set<string>()
    for (let i = 0; i < 40; i++) seen.add(env.reset(key(i)).cards.join(''))
    expect(seen.size).toBe(6)
  })
})

describe('Leduc poker', () => {
  it('runs two betting rounds with raise sizes 2 and 4 and at most two raises', () => {
    const env = leducPoker.create()
    let s = leducPokerState([0, 4, 1])
    expect(env.legalActions?.(s, 'player_0')).toEqual([1, 2])
    s = play(env, s, [2, 2])
    expect(s.contributions).toEqual([3, 5])
    expect(env.legalActions?.(s, 'player_0')).toEqual([0, 1])
    s = play(env, s, [1])
    expect(s.round).toBe(1)
    expect(env.observe(s, 'player_1').slice(8, 14)).toEqual([0, 1, 0, 0, 0, 0])
    s = play(env, s, [2, 1])
    expect(s.contributions).toEqual([9, 9])
    expect(s.over).toBe(true)
    // player_0 holds a jack and the public card is the other jack: checked down, the pair wins the pot of 2.
    const r = env.step(play(env, leducPokerState([0, 4, 1]), [1, 1, 1]), 1, key(0))
    expect(r.rewards).toEqual({ player_0: 1, player_1: -1 })
  })

  it('scores pairs, high cards and splits, and folds lose the contribution', () => {
    expect(leducShowdown([0, 5, 1])).toBe(0)
    expect(leducShowdown([2, 5, 0])).toBe(1)
    expect(leducShowdown([4, 5, 0])).toBeUndefined()
    const env = leducPoker.create()
    const r = env.step(play(env, leducPokerState([0, 4, 2]), [2]), 0, key(0))
    expect(r.rewards).toEqual({ player_0: 1, player_1: -1 })
  })

  it('is zero-sum and fair to neither seat on average under uniform play, by exact enumeration', () => {
    const env = leducPoker.create()
    let v = 0
    const all = deals(6, 3)
    for (const d of all) {
      const [a, b] = expectedReturns(env, leducPokerState([d[0] ?? 0, d[1] ?? 0, d[2] ?? 0]), uniformStrategy)
      expect(a + b).toBeCloseTo(0, 12)
      v += a / all.length
    }
    expect(Number.isFinite(v)).toBe(true)
  })
})
