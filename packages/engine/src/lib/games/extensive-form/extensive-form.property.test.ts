import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import {
  backwardInduction,
  cfrPlus,
  exploitability,
  gameTree,
  payoffMatrices,
  randomGame,
  solveSequenceForm,
  solveZeroSum,
  type TreeSpec,
  toNormalForm,
  treeGame,
} from '../index.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 - 1 })

/** A zero-sum matrix game played in sequence: player 1 moves without seeing player 0's action. */
function hidden(a: readonly (readonly number[])[]): TreeSpec {
  return {
    player: 0,
    infoset: 'row',
    actions: a.map((row, i) => ({
      label: `r${i}`,
      child: {
        player: 1,
        infoset: 'column',
        actions: row.map((v, j) => ({ label: `c${j}`, child: { returns: [v, -v] } })),
      },
    })),
  }
}

/** A random two-player zero-sum tree with perfect information and chance nodes. */
const perfect: fc.Arbitrary<TreeSpec> = fc.letrec<{ node: TreeSpec }>((tie) => ({
  node: fc.oneof(
    { depthSize: 'small', withCrossShrink: true },
    fc.integer({ min: -5, max: 5 }).map((v): TreeSpec => ({ returns: [v, -v] })),
    fc
      .tuple(fc.integer({ min: 0, max: 1 }), fc.array(tie('node'), { minLength: 1, maxLength: 3 }))
      .map(([player, kids]): TreeSpec => ({ player, actions: kids.map((child, i) => ({ label: `a${i}`, child })) })),
    fc
      .array(tie('node'), { minLength: 2, maxLength: 2 })
      .map((kids): TreeSpec => ({ chance: kids.map((child) => ({ p: 0.5, child })) })),
  ),
})).node

describe('extensive-form solvers', () => {
  it('a hidden-move matrix game has the matrix as normal form and the LP value in sequence form', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 4 }), fc.integer({ min: 1, max: 4 }), seed, (m, n, s) => {
        const [a] = payoffMatrices(randomGame(key(s), [m, n], { low: -1, high: 1 }))
        const tree = gameTree(treeGame('hidden', 2, hidden(a)))
        expect(payoffMatrices(toNormalForm(tree))[0]).toEqual(a)
        const value = solveZeroSum(a).value
        const sf = solveSequenceForm(tree)
        expect(sf.values[0]).toBeCloseTo(value, 8)
        expect(exploitability(tree, sf.policy)).toBeLessThan(1e-8)
        expect(exploitability(tree, cfrPlus(tree, { iterations: 400 }).averagePolicy)).toBeLessThan(0.02)
      }),
      { numRuns: 25 },
    )
  })

  it('backward induction is an equilibrium of perfect-information games, worth the sequence-form value', () => {
    fc.assert(
      fc.property(perfect, (root) => {
        const tree = gameTree(treeGame('random', 2, root))
        const bi = backwardInduction(tree)
        expect(exploitability(tree, bi.policy)).toBeLessThan(1e-9)
        expect(solveSequenceForm(tree).values[0]).toBeCloseTo(bi.values[0] ?? 0, 7)
      }),
      { numRuns: 40 },
    )
  })
})
