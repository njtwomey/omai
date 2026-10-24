import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { alphaBeta, expectimax, type Game, minimax, negamax, type StochasticGame } from './index.ts'

/** A random game tree: inner nodes list children; leaves hold MAX's utility; players alternate MAX, MIN. */
interface Tree {
  readonly children: readonly Tree[]
  readonly utility: number
}

const tree: fc.Arbitrary<Tree> = fc.letrec<{ node: Tree }>((tie) => ({
  node: fc.oneof(
    { depthSize: 'small', withCrossShrink: true },
    fc.record({ children: fc.constant([] as Tree[]), utility: fc.integer({ min: -9, max: 9 }) }),
    fc.record({ children: fc.array(tie('node'), { minLength: 1, maxLength: 4 }), utility: fc.constant(0) }),
  ),
})).node

/** The game over a tree: states are paths of child indices. */
function treeGame(root: Tree): Game<readonly number[], number> {
  const at = (path: readonly number[]): Tree => path.reduce((t, i) => t.children[i] ?? t, root)
  return {
    initial: [],
    player: (p) => (p.length % 2 === 0 ? 'MAX' : 'MIN'),
    actions: (p) => at(p).children.map((_, i) => i),
    result: (p, i) => [...p, i],
    isTerminal: (p) => at(p).children.length === 0,
    utility: (p, who) => (who === 'MAX' ? at(p).utility : 0 - at(p).utility),
    key: (p) => p.join('.'),
  }
}

describe('game-tree search properties', () => {
  it('alpha-beta and negamax equal minimax, visiting no more nodes', () => {
    fc.assert(
      fc.property(tree, fc.integer({ min: 1, max: 6 }), (t, depth) => {
        const game = treeGame(t)
        const full = minimax(game, [])
        const ab = alphaBeta(game, [])
        expect(ab.value).toBe(full.value)
        expect(ab.action).toEqual(full.action)
        expect(ab.nodes).toBeLessThanOrEqual(full.nodes)
        expect(negamax(game, []).value).toBe(full.value)
        expect(negamax(game, [], { alphaBeta: false }).value).toBe(full.value)
        // With a depth limit and an evaluation the equalities still hold.
        const evaluate = (p: readonly number[], who: string): number => (who === 'MAX' ? p.length : -p.length)
        const cut = minimax(game, [], { depth, evaluate })
        expect(alphaBeta(game, [], { depth, evaluate }).value).toBe(cut.value)
        expect(negamax(game, [], { depth, evaluate }).value).toBe(cut.value)
        // Move ordering changes the effort, never the value.
        expect(alphaBeta(game, [], { order: 'evaluate', evaluate }).value).toBe(full.value)
        expect(alphaBeta(game, [], { order: (_s, moves) => [...moves].reverse() }).value).toBe(full.value)
      }),
      { numRuns: 200 },
    )
  })

  it('expectimax without chance nodes is minimax, and the principal variation reaches the value', () => {
    fc.assert(
      fc.property(tree, (t) => {
        const game = treeGame(t)
        const stochastic: StochasticGame<readonly number[], number> = {
          ...game,
          isChance: () => false,
          outcomes: () => [],
        }
        const full = minimax(game, [])
        expect(expectimax(stochastic, []).value).toBe(full.value)
        let state: readonly number[] = []
        for (const a of full.principalVariation) state = game.result(state, a)
        expect(game.isTerminal(state)).toBe(true)
        expect(game.utility(state, 'MAX')).toBe(full.value)
      }),
      { numRuns: 200 },
    )
  })
})
