import { describe, expect, it } from 'vitest'
import { boardGame } from '../../testing/search.ts'
import { messages, policies, recorder, validateAgainstSchema } from '../trace/index.ts'
import {
  alphaBeta,
  alphaBetaAlgorithm,
  expectimax,
  mcts,
  mctsAlgorithm,
  minimax,
  negamax,
  type StochasticGame,
} from './index.ts'

describe('expectimax', () => {
  it('averages chance outcomes (AIMA figure 5.11 style)', () => {
    // MAX picks a or b; each leads to a chance node over MIN choices.
    const children: Record<string, string[]> = {
      root: ['a', 'b'],
      a1: ['x1', 'x2'],
      a2: ['x3', 'x4'],
      b1: ['y1', 'y2'],
      b2: ['y3', 'y4'],
    }
    const outcomes: Record<string, { state: string; probability: number }[]> = {
      a: [
        { state: 'a1', probability: 0.9 },
        { state: 'a2', probability: 0.1 },
      ],
      b: [
        { state: 'b1', probability: 0.9 },
        { state: 'b2', probability: 0.1 },
      ],
    }
    const leaf: Record<string, number> = { x1: 2, x2: 3, x3: 1, x4: 4, y1: 1, y2: 20, y3: 4, y4: 40 }
    const game: StochasticGame<string, string> = {
      initial: 'root',
      player: (s) => (s === 'root' ? 'MAX' : 'MIN'),
      actions: (s) => children[s] ?? [],
      result: (_s, a) => a,
      isTerminal: (s) => s in leaf,
      utility: (s, p) => (p === 'MAX' ? (leaf[s] ?? 0) : -(leaf[s] ?? 0)),
      isChance: (s) => s in outcomes,
      outcomes: (s) => outcomes[s] ?? [],
    }
    const r = expectimax(game, 'root')
    // a: 0.9 * min(2, 3) + 0.1 * min(1, 4) = 1.9; b: 0.9 * 1 + 0.1 * 4 = 1.3.
    expect(r.action).toBe('a')
    expect(r.value).toBeCloseTo(1.9, 12)
    expect(() =>
      expectimax({ ...game, isChance: undefined } as unknown as StochasticGame<string, string>, 'root'),
    ).toThrow()
    expect(expectimax(game, 'root', { depth: 1, evaluate: () => 7 }).value).toBe(7)
  })
})

describe('tic-tac-toe', () => {
  const game = boardGame()
  it('takes a win and blocks a loss', () => {
    // X: (1,1), (1,2); O: (2,2), (3,3); X to move wins at (1,3).
    const win = game.play([
      [1, 1],
      [2, 2],
      [1, 2],
      [3, 3],
    ])
    expect(minimax(game, win).action).toEqual([1, 3])
    expect(alphaBeta(game, win).value).toBe(1)
    expect(negamax(game, win).value).toBe(1)
    expect(mcts(game, win, { simulations: 400, seed: 3 }).action).toEqual([1, 3])
    // O to move must block (1,3).
    const block = game.play([
      [1, 1],
      [2, 2],
      [1, 2],
    ])
    expect(alphaBeta(game, block).action).toEqual([1, 3])
  })

  it('values the empty board as a draw with alpha-beta and finds the same with move ordering', () => {
    const plain = alphaBeta(game, game.initial)
    expect(plain.value).toBe(0)
    const centre = (_s: unknown, moves: readonly (readonly [number, number])[]) =>
      [...moves].sort((a, b) => Math.abs(a[0] - 2) + Math.abs(a[1] - 2) - (Math.abs(b[0] - 2) + Math.abs(b[1] - 2)))
    const ordered = alphaBeta(game, game.initial, { order: centre })
    expect(ordered.value).toBe(0)
    expect(ordered.nodes).toBeLessThan(plain.nodes)
  })

  it('stops on its node budget and reports the best move so far', () => {
    const r = minimax(game, game.initial, { maxNodes: 500 })
    expect(r.status).toBe('budget')
    expect(r.nodes).toBeLessThanOrEqual(500)
  })

  it('records visits, values and cutoffs as declared messages', () => {
    const rec = recorder()
    const algorithm = alphaBetaAlgorithm(game, game.play([[2, 2]]))
    alphaBeta(game, game.play([[2, 2]]), { tracer: rec, policy: policies.teaching })
    const run = rec.run()
    expect(messages(run, 'visit').length).toBe(2316)
    expect(messages(run, 'prune').length).toBeGreaterThan(0)
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
  })

  it('handles terminal roots', () => {
    const over = game.play([
      [1, 1],
      [2, 1],
      [1, 2],
      [2, 2],
      [1, 3],
    ])
    expect(game.isTerminal(over)).toBe(true)
    const r = minimax(game, over)
    expect(r.action).toBeUndefined()
    expect(r.value).toBe(-1)
    expect(() => mctsAlgorithm(game, over)).toThrow()
    expect(() => mcts(game, game.initial, { simulations: 0 })).toThrow()
  })
})

describe('Monte Carlo tree search', () => {
  it('uses priors and value estimates (PUCT) and is reproducible by seed', () => {
    const game = boardGame()
    const pos = game.play([
      [1, 1],
      [2, 2],
      [1, 2],
      [3, 3],
    ])
    const uniform = (_s: unknown, moves: readonly unknown[]) => moves.map(() => 1 / moves.length)
    const r = mcts(game, pos, { simulations: 300, prior: uniform, value: () => 0, seed: 1 })
    expect(r.action).toEqual([1, 3])
    expect(r.visits.reduce((a, b) => a + b, 0)).toBe(300)
    const a = mcts(game, game.initial, { simulations: 50, seed: 9 })
    const b = mcts(game, game.initial, { simulations: 50, seed: 9 })
    expect(a.visits).toEqual(b.visits)
    const rollout = mcts(game, pos, {
      simulations: 100,
      rollout: (_s, moves) => moves[0] as readonly [number, number],
      seed: 2,
    })
    expect(rollout.simulations).toBe(100)
  })
})
