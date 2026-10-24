import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { tensor } from '../../tensor/index.ts'
import { recorder } from '../../trace/index.ts'
import {
  bestResponse,
  CHANCE,
  cfr,
  cfrAlgorithm,
  type ExtensiveGame,
  exploitability,
  extensiveNashConv,
  gameTree,
  kuhnPoker,
  leducPoker,
  solveSequenceForm,
  TERMINAL,
  uniformPolicy,
} from '../index.ts'

const GAMES: Readonly<Record<string, () => ExtensiveGame<unknown>>> = { kuhn_poker: kuhnPoker, leduc_poker: leducPoker }
const TREES = { kuhn_poker: gameTree(kuhnPoker()), leduc_poker: gameTree(leducPoker()) }

describe('poker rules against OpenSpiel', () => {
  for (const c of loadFixture('games/extensive_form/poker').cases) {
    it(c.name, () => {
      const tree = TREES[c.name as keyof typeof TREES]
      expect(tree.nodes.length).toBe(c.params['nodes'])
      expect(tree.infosets.length).toBe(c.params['infosets'])
      const leaves = tree.nodes.filter((n) => n.player === TERMINAL)
      expect(leaves.length).toBe(c.params['terminals'])
      let total = 0
      for (const n of leaves) total += Math.abs(n.returns[0] ?? 0)
      expect(total).toBeCloseTo(c.params['absoluteReturns'] as number, 9)
      const game = (GAMES[c.name] ?? kuhnPoker)()
      for (const [history, value] of c.params['sample'] as [number[], number][]) {
        let s = game.initial()
        for (const a of history) {
          const player = game.currentPlayer(s)
          const legal = player === CHANCE ? game.chanceOutcomes(s).map(([x]) => x) : game.legalActions(s)
          expect(legal).toContain(a)
          s = game.apply(s, a)
        }
        expect(game.currentPlayer(s)).toBe(TERMINAL)
        expect(game.returns(s)[0]).toBe(value)
      }
    })
  }
})

describe('CFR against OpenSpiel', () => {
  for (const c of loadFixture('games/extensive_form/cfr').cases) {
    it(c.name, () => {
      const tree = TREES[c.params['game'] as keyof typeof TREES]
      const rec = recorder()
      const variant = c.params['variant'] as 'cfr' | 'cfr+' | 'linear' | 'discounted'
      const r = cfr(tree, { iterations: c.params['iterations'] as number, variant, tracer: rec })
      const trace = rec
        .run()
        .events.flatMap((e) => (e.type === 'metric' && e.name === 'exploitability' && e.step > 0 ? [e.value] : []))
      expectMatches(tensor(trace), c.output, c.rtol, c.atol, c.name)
      for (const [key, probs] of Object.entries(c.params['policy'] as Record<string, number[]>))
        for (const [a, p] of probs.entries())
          expect(Math.abs((r.averagePolicy[key]?.[a] ?? Number.NaN) - p)).toBeLessThanOrEqual(1e-12 + 1e-9 * p)
      expect(r.exploitability).toBeCloseTo(trace[trace.length - 1] ?? Number.NaN, 12)
      expect(cfrAlgorithm(tree, { variant }).name).toBe(`cfr-${variant}`)
    })
  }
})

describe('best responses against OpenSpiel', () => {
  for (const c of loadFixture('games/extensive_form/best_response').cases) {
    it(c.name, () => {
      const tree = TREES[c.params['game'] as keyof typeof TREES]
      const given = c.params['policy'] as Record<string, number[]>
      const policy = Object.keys(given).length === 0 ? uniformPolicy(tree) : given
      const b0 = bestResponse(tree, policy, 0)
      const b1 = bestResponse(tree, policy, 1)
      expectMatches(
        tensor([b0.value, b1.value, extensiveNashConv(tree, policy), exploitability(tree, policy)]),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
      const improvements = c.params['improvements'] as number[]
      expect(b0.value - b0.onPolicyValue).toBeCloseTo(improvements[0] ?? Number.NaN, 10)
      expect(b1.value - b1.onPolicyValue).toBeCloseTo(improvements[1] ?? Number.NaN, 10)
    })
  }
})

describe('sequence-form LP against OpenSpiel', () => {
  for (const c of loadFixture('games/extensive_form/sequence_form').cases) {
    it(c.name, () => {
      const r = solveSequenceForm(TREES.kuhn_poker)
      expectMatches(tensor([...r.values]), c.output, c.rtol, c.atol, c.name)
      expect(exploitability(TREES.kuhn_poker, r.policy)).toBeLessThan(1e-9)
      expect(c.params['exploitability'] as number).toBeLessThan(1e-6)
    })
  }
})
